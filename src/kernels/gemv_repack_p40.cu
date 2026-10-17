// gemv_repack_p40.cu -- decode GEMV over the ALIGNED repacked layout.
//
// Why this exists (all measured, docs/NEXT_STEPS.md):
//   aligned uint4 stream         251.8 GB/s
//   PQ2_0 addressing, pure read  183.6 GB/s   <- the 34-byte stride costs 37%
//   stock decode (achieves)      157.5 GB/s
//   my earlier kernels          124.7 / 81.9  (occupancy, L2 activation re-reads)
//
// Four constraints, each from a measurement:
//  1. weight loads 8-byte aligned and warp-contiguous -> the repacked layout
//  2. activations from shared memory                  -> quantised once per block
//  3. registers <= 32 for full occupancy              -> small per-lane state
//  4. quantisation fused, no extra launch             -> done in this kernel
//
// THE OPEN TRADE-OFF, and why RP_ROWS is a build knob:
// a lane processing one 8-byte unit needs 32 bytes of activation for it, so the
// activation:weight byte ratio is 4:1 at RP_ROWS=1.  Processing R rows with the
// same unit amortises that to 4:R -- but costs R row pointers and R
// accumulators.  An earlier measurement says registers matter a lot (64 regs ->
// 50% occupancy -> -35%).  So R decides, and only a measurement can settle it:
//
//   -DBONSAI_REPACK_ROWS=1   <= 32 regs, 64 warps/SM, activation:weight 4:1
//   -DBONSAI_REPACK_ROWS=2   <= 42 regs, 48 warps/SM, 2:1
//   -DBONSAI_REPACK_ROWS=4   <= 64 regs, 32 warps/SM, 1:1
#include "../pq2_0.h"
#include "../pq2_0_repack.h"
#include "../p40_common.h"
#include "bonsai_p40.h"

#include <cuda_runtime.h>

#include <algorithm>

using namespace bonsai;

namespace {

constexpr int RP_WARPS = 8;   // 256 threads

#ifndef BONSAI_REPACK_ROWS
#define BONSAI_REPACK_ROWS 2
#endif
constexpr int RP_ROWS = BONSAI_REPACK_ROWS;

#if BONSAI_REPACK_ROWS >= 4
#define RP_MIN_BLOCKS 4       // <= 64 regs, 32 warps/SM
#elif BONSAI_REPACK_ROWS == 2
#define RP_MIN_BLOCKS 6       // <= 42 regs, 48 warps/SM
#else
#define RP_MIN_BLOCKS 8       // <= 32 regs, 64 warps/SM
#endif

// Quantise x (n_in floats) into shared memory in the PQ2_0-native permuted
// Q8_1 layout.  8 lanes per 32-element chunk, 4 elements per lane: the same
// arrangement the stock kernel uses, and the one that keeps shared-memory
// accesses bank-conflict free.
__device__ inline void quantise_act_to_smem(const float * __restrict__ x, block_q8_1 * act,
                                            int nchunk, int nthreads) {
    const int ngroup = nthreads / 8;              // chunks per pass
    for (int c = threadIdx.x / 8; c < nchunk; c += ngroup) {
        const int s = threadIdx.x & 7;            // lane inside the 8-lane group
        const float * xc = x + (size_t)c * QK8_1 + s * 4;
        float v[4];
        float amax = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) { v[i] = xc[i]; amax = fmaxf(amax, fabsf(v[i])); }
#pragma unroll
        for (int o = 1; o < 8; o <<= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o, 8));
        const float d  = amax / 127.0f;
        const float id = d > 0.0f ? 1.0f / d : 0.0f;
        int sum = 0;
        int8_t q[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            int t = __float2int_rn(v[i] * id);
            t = max(-127, min(127, t));
            q[i] = (int8_t)t; sum += t;
        }
#pragma unroll
        for (int o = 1; o < 8; o <<= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o, 8);
        if (s == 0) { act[c].d = f32_to_f16(d); act[c].s = (uint16_t)(int16_t)sum; }
#pragma unroll

__global__ void __launch_bounds__(RP_WARPS * 32, RP_MIN_BLOCKS)
gemv_repack_p40_kernel(const uint8_t * __restrict__ W, const float * __restrict__ x,
                       float * __restrict__ y, const int n_out, const int nblocks) {
    extern __shared__ block_q8_1 act_smem[];
    const int nchunk = nblocks * 4;

    quantise_act_to_smem(x, act_smem, nchunk, RP_WARPS * 32);
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row0 = (blockIdx.x * RP_WARPS + warp) * RP_ROWS;
    if (row0 >= n_out) return;

    const int stride = repack_row_stride(nblocks);
    const int last   = n_out - 1;
    const uint8_t * rp[RP_ROWS];
#pragma unroll
    for (int r = 0; r < RP_ROWS; ++r)
        rp[r] = W + (size_t)min(row0 + r, last) * (size_t)stride;

    float acc[RP_ROWS];
#pragma unroll
    for (int r = 0; r < RP_ROWS; ++r) acc[r] = 0.0f;

    const int nunits = repack_n_units(nblocks);
    const int coff   = repack_codes_offset(nblocks);

    for (int u = lane; u < nunits; u += 32) {
        // the activation chunk this unit needs: read ONCE, reused by all rows
        const block_q8_1 * a = act_smem + repack_act_chunk(u, nblocks);
        const int * ap = (const int *)a->qs;
        const int a0 = ap[0], a1 = ap[1], a2 = ap[2], a3 = ap[3];
        const int a4 = ap[4], a5 = ap[5], a6 = ap[6], a7 = ap[7];
        const float da = f16_to_f32(a->d);
        const int bias = -q8_1_sum(a);

        const int b = repack_block_of_unit(u, nblocks);
        const size_t uoff = (size_t)coff + (size_t)u * 8;

#pragma unroll
        for (int r = 0; r < RP_ROWS; ++r) {
            const uint8_t * q = rp[r] + uoff;
            // 8-byte aligned unit -> two aligned 32-bit loads (the whole point)
            const uint32_t w0 = *(const uint32_t *)(q + 0);
            const uint32_t w1 = *(const uint32_t *)(q + 4);
            int s = bias;
            s = dp4a((int)((w0 >> 0) & 0x03030303u), a0, s);
            s = dp4a((int)((w0 >> 2) & 0x03030303u), a1, s);
            s = dp4a((int)((w0 >> 4) & 0x03030303u), a2, s);
            s = dp4a((int)((w0 >> 6) & 0x03030303u), a3, s);
            s = dp4a((int)((w1 >> 0) & 0x03030303u), a4, s);
            s = dp4a((int)((w1 >> 2) & 0x03030303u), a5, s);
            s = dp4a((int)((w1 >> 4) & 0x03030303u), a6, s);
            s = dp4a((int)((w1 >> 6) & 0x03030303u), a7, s);
            const uint16_t d = *(const uint16_t *)(rp[r] + 2 * b);
            acc[r] += f16_to_f32(d) * da * (float)s;
        }
    }

#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
#pragma unroll
        for (int r = 0; r < RP_ROWS; ++r)
            acc[r] += __shfl_down_sync(0xffffffffu, acc[r], o);
    }
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < RP_ROWS; ++r)
            if (row0 + r < n_out) y[row0 + r] = acc[r];
    }
}

// --- repack kernel: pure byte permutation, one unit per thread -------------
__global__ void __launch_bounds__(256)
repack_pq2_0_kernel(const block_pq2_0 * __restrict__ src, uint8_t * __restrict__ dst,
                    const int n_out, const int nblocks) {
    const int stride = repack_row_stride(nblocks);
    const int coff   = repack_codes_offset(nblocks);
    const int nunits = repack_n_units(nblocks);
    const long total = (long)n_out * nunits;

    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (long)gridDim.x * blockDim.x) {
        const int row = (int)(i / nunits);
        const int u   = (int)(i % nunits);
        const int p   = repack_plane_of_unit(u, nblocks);
        const int b   = repack_block_of_unit(u, nblocks);
        const block_pq2_0 * s = src + (size_t)row * nblocks + b;
        uint8_t * d = dst + (size_t)row * stride + coff + (size_t)u * 8;
        // source is only 2-byte aligned, destination is 8-byte aligned:
        // copy as four uint16 (valid on both sides), never as a 64-bit load.
        const uint16_t * sp = (const uint16_t *)(s->qs + p * 8);
        uint16_t * dp = (uint16_t *)d;
#pragma unroll
        for (int i = 0; i < 4; ++i) dp[i] = sp[i];
        if (p == 0) {
            uint16_t * dd = (uint16_t *)(dst + (size_t)row * stride + 2 * b);
            *dd = s->d;
        }
    }
}

} // namespace

int bonsai_p40_repack(const void * src, void * dst, int64_t n_out, int64_t n_in,
                      bonsai_stream stream) {
    if (n_in % 128 != 0 || n_out <= 0) return 1;
    const int nblocks = (int)(n_in / 128);
    if (!repack_requires(nblocks)) return 1;
    const long total = (long)n_out * repack_n_units(nblocks);
    const long nb = (total + 255) / 256;
    const int blocks = (int)(nb > 65535 ? 65535 : nb);
    repack_pq2_0_kernel<<<blocks, 256, 0, (cudaStream_t)stream>>>(
        (const block_pq2_0 *)src, (uint8_t *)dst, (int)n_out, nblocks);
    return (int)cudaGetLastError();
}

int bonsai_p40_gemv_repack(const void * W, const float * x, float * y,
                           int64_t n_out, int64_t n_in, bonsai_stream stream) {
    if (n_in % 128 != 0 || n_out <= 0) return 1;
    const int nblocks = (int)(n_in / 128);
    if (!repack_requires(nblocks)) return 1;
    const int smem = nblocks * 4 * (int)sizeof(block_q8_1);
    if (smem > 48 * 1024) return 1;                    // P40: 48 KiB per block max
    const int blocks = (int)((n_out + RP_WARPS * RP_ROWS - 1) / (RP_WARPS * RP_ROWS));
    gemv_repack_p40_kernel<<<blocks, RP_WARPS * 32, smem, (cudaStream_t)stream>>>(
        (const uint8_t *)W, x, y, (int)n_out, nblocks);
    return (int)cudaGetLastError();
}

        for (int i = 0; i < 4; ++i) {
            const int j = s * 4 + i;                            // element index
            const int p = 16 * (j >> 4) + 4 * (j & 3) + ((j >> 2) & 3);
            act[c].qs[p] = q[i];
        }
    }
}
