// gemv_p40.cu -- decode (batch = 1) PQ2_0 mat-vec, the bandwidth-critical
// kernel of the whole engine.
//
// At batch 1 every generated token must stream all ~7.2 GB of PQ2_0 weights
// once.  On the P40 that is a hard 7.21 GB / 346 GB/s = 20.8 ms floor, i.e.
// ~48 tok/s ceiling.  Everything below exists to get as close to that floor as
// the GP102 memory system allows:
//
//   * one warp per output row, lane-strided over the 128-weight blocks, so the
//     weight stream is fully coalesced and each lane issues an independent,
//     in-flight 2-byte-aligned code fetch (no shared memory, no __syncthreads);
//   * INT8 DP4A (native on sm_61) against activations pre-quantised into the
//     PQ2_0-native permuted Q8_1 layout -- no per-weight bit unpacking;
//   * __launch_bounds__(256, 8) forces <= 32 registers so 64 warps/SM (the
//     GP102 maximum) stay resident and hide the GDDR5 latency;
//   * activations are ~5 KB and stay hot in the 3 MB L2, so they cost nothing.
#include "../pq2_0.h"
#include "../p40_common.h"
#include "bonsai_p40.h"

#include <cuda_runtime.h>

using namespace bonsai;

namespace {

constexpr int GEMV_WARPS_PER_BLOCK = 8;   // 256 threads

// y[row] = W[row,:] . x   (x already in the permuted Q8_1 layout)
__global__ void __launch_bounds__(GEMV_WARPS_PER_BLOCK * 32, 8)
gemv_pq2_0_p40_kernel(const block_pq2_0 * __restrict__ W,
                      const block_q8_1  * __restrict__ X,
                      float * __restrict__ Y,
                      const int n_out, const int nblocks) {
    const int lane = threadIdx.x & 31;
    const int row  = blockIdx.x * GEMV_WARPS_PER_BLOCK + (threadIdx.x >> 5);
    if (row >= n_out) return;

    const block_pq2_0 * wrow = W + (size_t)row * (size_t)nblocks;

    float acc = 0.0f;
    for (int b = lane; b < nblocks; b += 32) {
        const block_pq2_0 * wb = wrow + b;
        const block_q8_1  * ab = X + b * QI_PQ2_0;
        const float dw = f16_to_f32(wb->d);

#pragma unroll
        for (int iqs = 0; iqs < QI_PQ2_0; ++iqs) {
            // The PQ2_0 block is only 2-byte aligned inside the row, so fetch
            // the two 16-code words as pairs of uint16.
            const uint16_t * qp = (const uint16_t *)(wb->qs + iqs * 8);
            const uint32_t w0 = (uint32_t)qp[0] | ((uint32_t)qp[1] << 16);
            const uint32_t w1 = (uint32_t)qp[2] | ((uint32_t)qp[3] << 16);

            const int4 a0 = *(const int4 *)(ab[iqs].qs + 0);
            const int4 a1 = *(const int4 *)(ab[iqs].qs + 16);

            int sumi = 0;
            sumi = dp4a((int)((w0 >> 0) & 0x03030303u), a0.x, sumi);
            sumi = dp4a((int)((w0 >> 2) & 0x03030303u), a0.y, sumi);
            sumi = dp4a((int)((w0 >> 4) & 0x03030303u), a0.z, sumi);
            sumi = dp4a((int)((w0 >> 6) & 0x03030303u), a0.w, sumi);
            sumi = dp4a((int)((w1 >> 0) & 0x03030303u), a1.x, sumi);
            sumi = dp4a((int)((w1 >> 2) & 0x03030303u), a1.y, sumi);
            sumi = dp4a((int)((w1 >> 4) & 0x03030303u), a1.z, sumi);
            sumi = dp4a((int)((w1 >> 6) & 0x03030303u), a1.w, sumi);

            // value = code - 1  ->  subtract the activation sum once per chunk.
            sumi -= q8_1_sum(&ab[iqs]);
            acc += dw * f16_to_f32(ab[iqs].d) * (float)sumi;
        }
    }

#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) Y[row] = acc;
}

// One warp per 32-element chunk: amax, quantise, and store transposed.
__global__ void __launch_bounds__(GEMV_WARPS_PER_BLOCK * 32, 8)
quant_q8_1_pq2_kernel(const float * __restrict__ x, block_q8_1 * __restrict__ y,
                      const int nchunk) {
    const int lane = threadIdx.x & 31;
    const int warp = blockIdx.x * GEMV_WARPS_PER_BLOCK + (threadIdx.x >> 5);
    const int nwarp = gridDim.x * GEMV_WARPS_PER_BLOCK;

    for (int c = warp; c < nchunk; c += nwarp) {
        const float v = x[c * QK8_1 + lane];

        float amax = fabsf(v);
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        const float d  = amax / 127.0f;
        const float id = d > 0.0f ? 1.0f / d : 0.0f;

        int q = __float2int_rn(v * id);
        q = max(-127, min(127, q));
        int sum = q;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);

        y[c].d = f32_to_f16(d);
        y[c].s = (uint16_t)(int16_t)sum;   // raw int16 activation sum
        // Transposed 4x4 store inside each 16-element half: element `lane`
        // goes to position 16*(lane>>4) + 4*(lane&3) + ((lane>>2)&3).
        y[c].qs[16 * (lane >> 4) + 4 * (lane & 3) + ((lane >> 2) & 3)] = (int8_t)q;
    }
}

} // namespace

int bonsai_p40_quant_activations(const float * x, void * out, int64_t n, bonsai_stream stream) {
    if (n % QK8_1 != 0) return 1;
    const int nchunk = (int)(n / QK8_1);
    const int warps  = (nchunk + 3) / 4;                 // ~4 chunks per warp
    const int blocks = (warps + GEMV_WARPS_PER_BLOCK - 1) / GEMV_WARPS_PER_BLOCK;
    quant_q8_1_pq2_kernel<<<blocks, GEMV_WARPS_PER_BLOCK * 32, 0, (cudaStream_t)stream>>>(
        x, (block_q8_1 *)out, nchunk);
    return (int)cudaGetLastError();
}

int bonsai_p40_gemv(const void * W, const void * xq, float * y,
                    int64_t n_out, int64_t n_in, bonsai_stream stream) {
    if (n_in % QK_PQ2_0 != 0) return 1;
    const int nblocks = (int)(n_in / QK_PQ2_0);
    const int blocks  = (int)((n_out + GEMV_WARPS_PER_BLOCK - 1) / GEMV_WARPS_PER_BLOCK);
    gemv_pq2_0_p40_kernel<<<blocks, GEMV_WARPS_PER_BLOCK * 32, 0, (cudaStream_t)stream>>>(
        (const block_pq2_0 *)W, (const block_q8_1 *)xq, y, (int)n_out, nblocks);
    return (int)cudaGetLastError();
}
