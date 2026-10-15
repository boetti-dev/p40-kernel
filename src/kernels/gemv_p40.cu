// gemv_p40.cu -- decode (batch = 1) PQ2_0 mat-vec, the bandwidth-critical
// kernel of the whole engine.
//
// COST MODEL (why the kernel looks like this)
// -------------------------------------------
// Per token the whole ~7.2 GB PQ2_0 weight set must cross the GDDR5:
//   7.21e9 B / 346e9 B/s = 20.8 ms  ->  48 tok/s ceiling.
// Everything else is noise: activations are ~5 KB, Hadamard/RMSNorm/GDN are
// O(10^7) ops, and the whole token needs ~0.45 G warp-instructions (~2.4 ms of
// issue against the P40's 184 G warp-inst/s).  The only thing worth tuning is
// memory-level parallelism on the weight stream.
//
// Three levers, all applied here:
//
//  1. 4 output rows per warp.  The activation bytes of a 128-weight block are
//     fetched once and reused by all four rows, and the warp keeps four
//     independent weight streams in flight instead of one.  At ~600 ns GDDR5
//     latency a warp needs several outstanding cache lines: with 30 SMs x 24
//     warps x 4 lines x 128 B ~ 3.7 MB in flight, versus the ~208 KB the loop
//     actually has to keep in flight to reach 346 GB/s.
//  2. The +1 code bias is folded into one `bias = -sum(a)` per 32-element chunk
//     (one int16 load + negate) and reused by all four rows, instead of a
//     fetch + subtract per row.
//  3. The weight scale `d` is loaded once per row per block, outside the chunk
//     loop.
//
// The weight fetch stays 4x uint16 per 8 code bytes: `block_pq2_0` is a 34-byte
// struct, so `qs` is only 2-byte aligned and a uint32/int4 load would be a
// misaligned access on Pascal.
#include "../pq2_0.h"
#include "../p40_common.h"
#include "../p40_ptx.h"
#include "bonsai_p40.h"

#include <cuda_runtime.h>

using namespace bonsai;

namespace {

constexpr int GEMV_WARPS_PER_BLOCK = 8;   // 256 threads
constexpr int GEMV_ROWS_PER_WARP   = 4;   // weight streams + activation reuse
constexpr int GEMV_ROWS_PER_BLOCK  = GEMV_WARPS_PER_BLOCK * GEMV_ROWS_PER_WARP;

// __launch_bounds__(256, 4): 32 warps/SM and 64 registers/thread, so the
// four-row body (4 pointers, 4 scales, 4 accumulators, 8 activation words,
// 8 code words) stays in registers without spilling.
__global__ void __launch_bounds__(GEMV_WARPS_PER_BLOCK * 32, 4)
gemv_pq2_0_p40_kernel(const block_pq2_0 * __restrict__ W,
                      const block_q8_1  * __restrict__ X,
                      float * __restrict__ Y,
                      const int n_out, const int nblocks) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row0 = (blockIdx.x * GEMV_WARPS_PER_BLOCK + warp) * GEMV_ROWS_PER_WARP;
    if (row0 >= n_out) return;

    // Rows past the end are clamped (computed but never stored) so the loop
    // body stays branch-free.
    const int last = n_out - 1;
    const block_pq2_0 * wr[GEMV_ROWS_PER_WARP];
#pragma unroll
    for (int r = 0; r < GEMV_ROWS_PER_WARP; ++r)
        wr[r] = W + (size_t)min(row0 + r, last) * (size_t)nblocks;

    float acc[GEMV_ROWS_PER_WARP];
#pragma unroll
    for (int r = 0; r < GEMV_ROWS_PER_WARP; ++r) acc[r] = 0.0f;

    for (int b = lane; b < nblocks; b += 32) {
        const block_q8_1 * ab = X + (size_t)b * BONSAI_QI_PQ2_0;

        // Keep the weight stream ahead of the dp4a consumers.  All four rows
        // are prefetched (each is a separate stream), clamped to the row so the
        // hint can never point outside the weight buffer.  The activation side
        // lives in L2 already (a few KB) and needs no help.
        const int pf = min(b + 32, nblocks - 1);
#pragma unroll
        for (int r = 0; r < GEMV_ROWS_PER_WARP; ++r) p40_prefetch_l2(wr[r] + pf);

        float dw[GEMV_ROWS_PER_WARP];
#pragma unroll
        for (int r = 0; r < GEMV_ROWS_PER_WARP; ++r)
            dw[r] = f16_to_f32(p40_ldg_u16(&wr[r][b].d));

#pragma unroll
        for (int iqs = 0; iqs < BONSAI_QI_PQ2_0; ++iqs) {
            // Activations: 32 bytes at 4-byte alignment -> eight int32 loads.
            const int * ap = (const int *)ab[iqs].qs;
            const int a0 = ap[0], a1 = ap[1], a2 = ap[2], a3 = ap[3];
            const int a4 = ap[4], a5 = ap[5], a6 = ap[6], a7 = ap[7];
            const int bias = -q8_1_sum(&ab[iqs]);   // value = code - 1
            const float da = f16_to_f32(ab[iqs].d);

            int s[GEMV_ROWS_PER_WARP];
#pragma unroll
            for (int r = 0; r < GEMV_ROWS_PER_WARP; ++r) s[r] = bias;

#pragma unroll
            for (int r = 0; r < GEMV_ROWS_PER_WARP; ++r) {
                const uint16_t * qp = (const uint16_t *)(wr[r][b].qs + iqs * 8);
                const uint32_t w0 = p40_ldg_u16x2(qp + 0);
                const uint32_t w1 = p40_ldg_u16x2(qp + 2);
                s[r] = p40_dp4a_us((w0 >> 0) & 0x03030303u, a0, s[r]);
                s[r] = p40_dp4a_us((w0 >> 2) & 0x03030303u, a1, s[r]);
                s[r] = p40_dp4a_us((w0 >> 4) & 0x03030303u, a2, s[r]);
                s[r] = p40_dp4a_us((w0 >> 6) & 0x03030303u, a3, s[r]);
                s[r] = p40_dp4a_us((w1 >> 0) & 0x03030303u, a4, s[r]);
                s[r] = p40_dp4a_us((w1 >> 2) & 0x03030303u, a5, s[r]);
                s[r] = p40_dp4a_us((w1 >> 4) & 0x03030303u, a6, s[r]);
                s[r] = p40_dp4a_us((w1 >> 6) & 0x03030303u, a7, s[r]);
            }

#pragma unroll
            for (int r = 0; r < GEMV_ROWS_PER_WARP; ++r)
                acc[r] += dw[r] * da * (float)s[r];
        }
    }

#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
#pragma unroll
        for (int r = 0; r < GEMV_ROWS_PER_WARP; ++r)
            acc[r] += __shfl_down_sync(0xffffffffu, acc[r], o);
    }
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < GEMV_ROWS_PER_WARP; ++r)
            if (row0 + r < n_out) Y[row0 + r] = acc[r];
    }
}

// One warp per 32-element chunk: amax, quantise, and store transposed.
__global__ void __launch_bounds__(GEMV_WARPS_PER_BLOCK * 32, 8)
quant_q8_1_pq2_kernel(const float * __restrict__ x, block_q8_1 * __restrict__ y,
                      const int nchunk) {
    const int lane = threadIdx.x & 31;
    const int warp = blockIdx.x * GEMV_WARPS_PER_BLOCK + (threadIdx.x >> 5);
    const int nwarp = gridDim.x * GEMV_WARPS_PER_BLOCK;

    for (int c = warp; c < nchunk; c += nwarp) {
        const float v = x[c * BONSAI_QK8_1 + lane];

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
    if (n <= 0 || n % BONSAI_QK8_1 != 0) return 1;
    const int nchunk = (int)(n / BONSAI_QK8_1);
    const int warps  = (nchunk + 3) / 4;                 // ~4 chunks per warp
    const int blocks = (warps + GEMV_WARPS_PER_BLOCK - 1) / GEMV_WARPS_PER_BLOCK;
    quant_q8_1_pq2_kernel<<<blocks, GEMV_WARPS_PER_BLOCK * 32, 0, (cudaStream_t)stream>>>(
        x, (block_q8_1 *)out, nchunk);
    return (int)cudaGetLastError();
}

int bonsai_p40_gemv(const void * W, const void * xq, float * y,
                    int64_t n_out, int64_t n_in, bonsai_stream stream) {
    if (n_in <= 0 || n_in % BONSAI_QK_PQ2_0 != 0) return 1;
    if (n_out <= 0) return 0;
    const int nblocks = (int)(n_in / BONSAI_QK_PQ2_0);
    const int blocks  = (int)((n_out + GEMV_ROWS_PER_BLOCK - 1) / GEMV_ROWS_PER_BLOCK);
    gemv_pq2_0_p40_kernel<<<blocks, GEMV_WARPS_PER_BLOCK * 32, 0, (cudaStream_t)stream>>>(
        (const block_pq2_0 *)W, (const block_q8_1 *)xq, y, (int)n_out, nblocks);
    return (int)cudaGetLastError();
}