// gemm_p40.cu -- prefill (batch > 1) PQ2_0 mat-mul.
//
// STATUS: this is a STANDALONE FALLBACK, not the Pascal MMQ path.
//
// The PrismML fork already ships a Pascal-tuned PQ2_0 MMQ
// (mmq-config-pascal.cuh: nthreads=256, occupancy=2, I=64 rows, J in
// {8..64}, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K) and
// ggml_cuda_mmq_get_config() selects it for __CUDA_ARCH__ < VOLTA.  When
// building against the fork, use that and do NOT wire this file in -- see
// docs/MMQ.md and docs/INTEGRATION.md.
//
// What this file is for: a self-contained prefill GEMM so the stack can be
// built and tested without pulling in ggml's MMQ machinery.  It is deliberately
// simple: one output row per lane, no opaque SRAM layout, no K-pipelining
// engine.  Expect the fork's MMQ to be faster.
//
// Decode is a pure bandwidth problem; prefill is a compute problem, and the
// P40 has no tensor cores.  The only usable ALU for low-bit matmul on GP102 is
// INT8 DP4A (128/SM/cycle), so this is a DP4A tile GEMM:
//
//   block tile  : 256 output rows x 16 token columns
//   warp        : 32 output rows (one per lane)
//   k-step      : one PQ2_0 block (128 weights) = 4 Q8_1 chunks
//
// The 16-column activation slab for the current k-block is staged in shared
// memory (2 KB) once and read by every lane, so weights dominate DRAM traffic
// exactly once per tile and the DP4A pipe stays fed.
#include "../pq2_0.h"
#include "../p40_common.h"
#include "../p40_ptx.h"
#include "bonsai_p40.h"

#include <cuda_runtime.h>

using namespace bonsai;

namespace {

constexpr int GEMM_WARPS   = 8;    // 256 threads
constexpr int GEMM_BM      = 32;   // rows per warp
constexpr int GEMM_BN      = 16;   // token columns per block
constexpr int GEMM_ROWS    = GEMM_WARPS * GEMM_BM;   // 256 rows per block

__global__ void __launch_bounds__(GEMM_WARPS * 32, 2)
gemm_pq2_0_p40_kernel(const block_pq2_0 * __restrict__ W,
                                      const block_q8_1  * __restrict__ X,
                                      float * __restrict__ Y,
                                      const int n_out, const int n_tok,
                                      const int nblocks) {
    __shared__ __align__(16) int8_t X_s[QI_PQ2_0 * GEMM_BN * 32];
    __shared__ float   D_s[QI_PQ2_0 * GEMM_BN];
    __shared__ int     S_s[QI_PQ2_0 * GEMM_BN];

    const int row0 = blockIdx.x * GEMM_ROWS;
    const int col0 = blockIdx.y * GEMM_BN;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row  = row0 + warp * GEMM_BM + lane;   // 0..255
    const int ncol = min(GEMM_BN, n_tok - col0);

    float acc[GEMM_BN];
#pragma unroll
    for (int j = 0; j < GEMM_BN; ++j) acc[j] = 0.0f;

    const bool row_ok = row < n_out;
    const block_pq2_0 * wrow = W + (size_t)(row_ok ? row : 0) * (size_t)nblocks;

    for (int b = 0; b < nblocks; ++b) {
        // ---- stage the activation slab for k-block b --------------------
        __syncthreads();
        for (int i = threadIdx.x; i < QI_PQ2_0 * GEMM_BN; i += blockDim.x) {
            const int iqs = i / GEMM_BN;
            const int j   = i % GEMM_BN;
            if (j < ncol) {
                const block_q8_1 * src = X + (size_t)(b * QI_PQ2_0 + iqs) * n_tok + (col0 + j);
                // 32 bytes: copy as eight int32 (`qs` is only 4-byte aligned).
                const uint32_t * sp = (const uint32_t *)src->qs;
                uint32_t * dp = (uint32_t *)(X_s + i * 32);
#pragma unroll
                for (int t = 0; t < 8; ++t) dp[t] = sp[t];
                D_s[i] = f16_to_f32(src->d);
                S_s[i] = q8_1_sum(src);
            } else {
                uint32_t * dp = (uint32_t *)(X_s + i * 32);
#pragma unroll
                for (int t = 0; t < 8; ++t) dp[t] = 0u;
                D_s[i] = 0.0f;
                S_s[i] = 0;
            }
        }
        __syncthreads();

        if (!row_ok) continue;

        const float dw = f16_to_f32(wrow[b].d);
        const uint16_t * qp = (const uint16_t *)wrow[b].qs;

#pragma unroll
        for (int iqs = 0; iqs < QI_PQ2_0; ++iqs) {
            const uint32_t w0 = (uint32_t)qp[iqs * 4 + 0] | ((uint32_t)qp[iqs * 4 + 1] << 16);
            const uint32_t w1 = (uint32_t)qp[iqs * 4 + 2] | ((uint32_t)qp[iqs * 4 + 3] << 16);
            const int t0 = (int)((w0 >> 0) & 0x03030303u);
            const int t1 = (int)((w0 >> 2) & 0x03030303u);
            const int t2 = (int)((w0 >> 4) & 0x03030303u);
            const int t3 = (int)((w0 >> 6) & 0x03030303u);
            const int t4 = (int)((w1 >> 0) & 0x03030303u);
            const int t5 = (int)((w1 >> 2) & 0x03030303u);
            const int t6 = (int)((w1 >> 4) & 0x03030303u);
            const int t7 = (int)((w1 >> 6) & 0x03030303u);

#pragma unroll
            for (int j = 0; j < GEMM_BN; ++j) {
                const int idx = iqs * GEMM_BN + j;
                const int4 a0 = ((const int4 *)X_s)[idx * 2 + 0];
                const int4 a1 = ((const int4 *)X_s)[idx * 2 + 1];
                int sumi = 0;
                sumi = p40_dp4a_us((uint32_t)t0, a0.x, sumi);
                sumi = p40_dp4a_us((uint32_t)t1, a0.y, sumi);
                sumi = p40_dp4a_us((uint32_t)t2, a0.z, sumi);
                sumi = p40_dp4a_us((uint32_t)t3, a0.w, sumi);
                sumi = p40_dp4a_us((uint32_t)t4, a1.x, sumi);
                sumi = p40_dp4a_us((uint32_t)t5, a1.y, sumi);
                sumi = p40_dp4a_us((uint32_t)t6, a1.z, sumi);
                sumi = p40_dp4a_us((uint32_t)t7, a1.w, sumi);
                sumi -= S_s[idx];
                // One FMA instead of two multiply + add: dw is invariant over j,
                // so fold it into the activation scale once per element.
                acc[j] += (dw * D_s[idx]) * (float)sumi;
            }
        }
    }

    if (row_ok) {
#pragma unroll
        for (int j = 0; j < GEMM_BN; ++j) {
            if (j < ncol) Y[(size_t)row * n_tok + (col0 + j)] = acc[j];
        }
    }
}

} // namespace

int bonsai_p40_gemm(const void * W, const void * Xq, float * Y,
                    int64_t n_out, int64_t n_in, int64_t n_tok, bonsai_stream stream) {
    if (n_in <= 0 || n_in % QK_PQ2_0 != 0) return 1;
    if (n_out <= 0 || n_tok <= 0) return 0;
    const int nblocks = (int)(n_in / QK_PQ2_0);
    dim3 grid((unsigned)((n_out + GEMM_ROWS - 1) / GEMM_ROWS),
              (unsigned)((n_tok + GEMM_BN - 1) / GEMM_BN));
    gemm_pq2_0_p40_kernel<<<grid, GEMM_WARPS * 32, 0, (cudaStream_t)stream>>>(
        (const block_pq2_0 *)W, (const block_q8_1 *)Xq, Y,
        (int)n_out, (int)n_tok, nblocks);
    return (int)cudaGetLastError();
}
