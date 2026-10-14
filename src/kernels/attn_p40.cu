// attn_p40.cu -- the two attention paths of qwen35 on the P40.
//
//  * 48 layers use Gated DeltaNet (linear attention): a per-head [128,128]
//    recurrent state updated once per token.  Columns of the state are
//    independent, so the kernel tiles them freely.
//  * 16 layers use full GQA attention (24 q heads / 4 kv heads, head_dim 256).
//    Decode is one query against the KV cache; prefill is causal.
//
// Pascal has no FP16 ALU worth using, so every accumulation here is FP32.
#include "../ops.h"
#include "../p40_common.h"
#include "bonsai_p40.h"

#include <cuda_runtime.h>

using namespace bonsai;

namespace {

constexpr int GDN_VTILE = 32;    // state columns per block (smem 128*32*4 = 16 KiB)

// One block per (value head, V tile of 32).  One warp = 32 columns.
__global__ void gdn_step_p40_kernel(float * __restrict__ state,
                                    const float * __restrict__ q,
                                    const float * __restrict__ k,
                                    const float * __restrict__ v,
                                    const float * __restrict__ g,
                                    const float * __restrict__ beta,
                                    float * __restrict__ out,
                                    const int n_k_heads, const int n_v_heads,
                                    const int K, const int V) {
    __shared__ float S[128 * GDN_VTILE];   // [K][GDN_VTILE]

    const int lane = threadIdx.x & 31;     // column within the tile
    const int vh   = blockIdx.x / (V / GDN_VTILE);
    const int vt   = blockIdx.x % (V / GDN_VTILE);
    const int vcol = vt * GDN_VTILE + lane;         // global value column
    const int kh   = vh / (n_v_heads / n_k_heads);  // grouped-query mapping

    float * Sg = state + ((size_t)vh * K) * V + vcol;
    float * Sh = S + lane;                          // S[i*GDN_VTILE + lane]

    for (int i = 0; i < K; ++i) Sh[i * GDN_VTILE] = Sg[(size_t)i * V];

    const float decay = expf(g[vh]);
    for (int i = 0; i < K; ++i) Sh[i * GDN_VTILE] *= decay;

    const float * kk = k + (size_t)kh * K;
    const float * qq = q + (size_t)kh * K;

    float kv = 0.0f;
    for (int i = 0; i < K; ++i) kv += kk[i] * Sh[i * GDN_VTILE];

    const float b  = beta[vh];
    const float vv = v[(size_t)vh * V + vcol];
    for (int i = 0; i < K; ++i) Sh[i * GDN_VTILE] += b * kk[i] * (vv - kv);

    float o = 0.0f;
    for (int i = 0; i < K; ++i) o += qq[i] * Sh[i * GDN_VTILE];

    out[(size_t)vh * V + vcol] = o;
    for (int i = 0; i < K; ++i) Sg[(size_t)i * V] = Sh[i * GDN_VTILE];
}

// ---------------------------------------------------------------------------
// Full GQA attention.  head_dim is fixed at 256; one warp per (head, query),
// each lane owning 8 head dimensions, online (flash) softmax in registers.
// ---------------------------------------------------------------------------
constexpr int ATT_DIMS_PER_LANE = 8;   // 256 / 32

__global__ void attn_p40_kernel(const float * __restrict__ q,
                                const float * __restrict__ k,
                                const float * __restrict__ v,
                                float * __restrict__ out,
                                const int n_tok, const int n_head, const int n_head_kv,
                                const int head_dim, const float scale) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int gid  = blockIdx.x * (blockDim.x >> 5) + warp;
    const int nq   = n_head * n_tok;
    if (gid >= nq) return;

    const int head = gid % n_head;
    const int qi   = gid / n_head;
    const int kvh  = head / (n_head / n_head_kv);
    const int D    = head_dim;
    const int nper = D / 32;

    float qreg[ATT_DIMS_PER_LANE];
    const float * qp = q + ((size_t)qi * n_head + head) * D + lane * nper;
    for (int i = 0; i < nper; ++i) qreg[i] = qp[i] * scale;

    float oreg[ATT_DIMS_PER_LANE];
    for (int i = 0; i < nper; ++i) oreg[i] = 0.0f;
    float m = -INFINITY, l = 0.0f;

    for (int j = 0; j <= qi; ++j) {                 // causal
        const float * kp = k + ((size_t)j * n_head_kv + kvh) * D + lane * nper;
        float part = 0.0f;
        for (int i = 0; i < nper; ++i) part += qreg[i] * kp[i];
        for (int o = 16; o > 0; o >>= 1) part += __shfl_xor_sync(0xffffffffu, part, o);

        const float m_new = fmaxf(m, part);
        const float alpha = expf(m - m_new);
        const float p     = expf(part - m_new);
        l = l * alpha + p;

        const float * vp = v + ((size_t)j * n_head_kv + kvh) * D + lane * nper;
        for (int i = 0; i < nper; ++i) oreg[i] = oreg[i] * alpha + p * vp[i];
        m = m_new;
    }

    const float inv = 1.0f / l;
    float * op = out + ((size_t)qi * n_head + head) * D + lane * nper;
    for (int i = 0; i < nper; ++i) op[i] = oreg[i] * inv;
}

} // namespace

int bonsai_p40_gdn_step(float * state, const float * q, const float * k, const float * v,
                        const float * g, const float * beta, float * out,
                        int n_k_heads, int n_v_heads, int K, int V, bonsai_stream stream) {
    if (V % GDN_VTILE != 0 || K > 128) return 1;
    const int blocks = n_v_heads * (V / GDN_VTILE);
    gdn_step_p40_kernel<<<blocks, 32, 0, (cudaStream_t)stream>>>(
        state, q, k, v, g, beta, out, n_k_heads, n_v_heads, K, V);
    return (int)cudaGetLastError();
}

int bonsai_p40_attn(const float * q, const float * k, const float * v, float * out,
                    int n_tok, int n_head, int n_head_kv, int head_dim, float scale,
                    bonsai_stream stream) {
    if (head_dim != 256 || (n_head % n_head_kv) != 0) return 1;
    const int warps = 8;
    const int blocks = (n_head * n_tok + warps - 1) / warps;
    attn_p40_kernel<<<blocks, warps * 32, 0, (cudaStream_t)stream>>>(
        q, k, v, out, n_tok, n_head, n_head_kv, head_dim, scale);
    return (int)cudaGetLastError();
}
