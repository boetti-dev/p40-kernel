// kv_p40.cu -- the KV cache of the Bonsai-P40 engine.
//
// See src/kv_cache.h for the sizing table and the layout rationale, and
// docs/KV_CACHE.md for the plan.  In short:
//
//   * only the 16 full-attention layers carry a growing cache; the 48 Gated
//     DeltaNet layers carry a constant 3 MiB state each and are allocated
//     together with it;
//   * K and V are Q8_0, head-major, so streaming positions is a contiguous
//     DRAM run and the score is a dp4a dot;
//   * decode uses a split-K flash pass, because at 262K the cache read
//     (26.4 ms) overtakes the weight stream (20.8 ms) and only parallelism
//     over the sequence can keep 346 GB/s busy.
#include "../kv_cache.h"
#include "../p40_common.h"
#include "../p40_ptx.h"
#include "bonsai_kv.h"

#include <cuda_runtime.h>
#include <cstdlib>

using namespace bonsai;

// ---------------------------------------------------------------------------
// Host-side cache object.  Opaque to callers.
// ---------------------------------------------------------------------------
struct bonsai_kv_cache {
    kv_geometry geo;
    block_q8_0 * d_k;      // growing keys
    block_q8_0 * d_v;      // growing values
    float      * d_gdn;    // constant linear-attention state
    float      * d_wsplit; // split-K workspace: n_head * n_split * (D + 2)
    block_q8_0 * d_q8;     // quantised query heads: n_head * (D/32)
    int          n_split;
};

namespace {

// Lane mapping shared by every kernel in this file: 4 lanes per Q8_0 block,
// 8 blocks per 256-wide head => lane owns block (lane>>2) and the four dims
// 4*(lane&3) .. +3 of that block.  That makes the score one dp4a per lane and
// every 32-byte block read a fully coalesced warp access.
__device__ __forceinline__ int kv_blk_of_lane(int lane) { return lane >> 2; }
__device__ __forceinline__ int kv_sub_of_lane(int lane) { return lane & 3; }

// --- append: quantise one (layer, pos, kv_head) K/V row ---------------------
__global__ void __launch_bounds__(256)
kv_quantize_append_kernel(block_q8_0 * __restrict__ K, block_q8_0 * __restrict__ V,
                          const float * __restrict__ ksrc, const float * __restrict__ vsrc,
                          const int pos0, const int n_tok,
                          const int n_kv_head, const int head_dim,
                          const int capacity, const int layer) {
    const int lane = threadIdx.x & 31;
    const int row  = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);   // token*head
    const int nrow = n_tok * n_kv_head;
    if (row >= nrow) return;

    const int t   = row / n_kv_head;
    const int h   = row % n_kv_head;
    const int pos = pos0 + t;
    const int blk = kv_blk_of_lane(lane);      // 0..7
    const int sub = kv_sub_of_lane(lane);

    const int64_t bidx = (((int64_t)layer * n_kv_head + h) * capacity + pos) * (head_dim / QK8_0) + blk;
    const float * kr = ksrc + ((int64_t)row * head_dim);
    const float * vr = vsrc + ((int64_t)row * head_dim);

    // this lane owns dims blk*32 + sub*4 .. +3
    const int d0 = blk * 32 + sub * 4;

    float kamax = 0.0f, vamax = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        kamax = fmaxf(kamax, fabsf(kr[d0 + i]));
        vamax = fmaxf(vamax, fabsf(vr[d0 + i]));
    }
    // reduce over the 4 lanes that share this block (width-4 segments)
    kamax = fmaxf(kamax, __shfl_xor_sync(0xffffffffu, kamax, 1, 4));
    kamax = fmaxf(kamax, __shfl_xor_sync(0xffffffffu, kamax, 2, 4));
    vamax = fmaxf(vamax, __shfl_xor_sync(0xffffffffu, vamax, 1, 4));
    vamax = fmaxf(vamax, __shfl_xor_sync(0xffffffffu, vamax, 2, 4));

    const float kd  = kamax / 127.0f;
    const float kid = kd > 0.0f ? 1.0f / kd : 0.0f;
    const float vd  = vamax / 127.0f;
    const float vid = vd > 0.0f ? 1.0f / vd : 0.0f;

    if (sub == 0) { K[bidx].d = f32_to_f16(kd); V[bidx].d = f32_to_f16(vd); }

    int8_t kq[4], vq[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        int a = (int)lroundf(kr[d0 + i] * kid);
        int b = (int)lroundf(vr[d0 + i] * vid);
        a = max(-127, min(127, a));
        b = max(-127, min(127, b));
        kq[i] = (int8_t)a; vq[i] = (int8_t)b;
    }
    // 4 consecutive dims = one aligned 4-byte store
    int32_t kw, vw;
    std::memcpy(&kw, kq, 4);
    std::memcpy(&vw, vq, 4);
    std::memcpy(K[bidx].qs + sub * 4, &kw, 4);
    std::memcpy(V[bidx].qs + sub * 4, &vw, 4);
}

// --- decode kernels --------------------------------------------------------
__global__ void __launch_bounds__(256)
kv_quantize_q_kernel(const float * __restrict__ q, block_q8_0 * __restrict__ q8,
                     const int n_head, const int head_dim) {
    const int lane = threadIdx.x & 31;
    const int head = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (head >= n_head) return;

    const int blk = kv_blk_of_lane(lane);
    const int sub = kv_sub_of_lane(lane);
    const int d0  = blk * 32 + sub * 4;
    const float * qr = q + (int64_t)head * head_dim;

    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) amax = fmaxf(amax, fabsf(qr[d0 + i]));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1, 4));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2, 4));

    const float d  = amax / 127.0f;
    const float id = d > 0.0f ? 1.0f / d : 0.0f;
    if (sub == 0) q8[(int64_t)head * (head_dim / QK8_0) + blk].d = f32_to_f16(d);

    int8_t qq[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        int a = (int)lroundf(qr[d0 + i] * id);
        qq[i] = (int8_t)max(-127, min(127, a));
    }
    int32_t qw;
    std::memcpy(&qw, qq, 4);
    std::memcpy(q8[(int64_t)head * (head_dim / QK8_0) + blk].qs + sub * 4, &qw, 4);
}

// --- decode: one split of the flash pass -----------------------------------
// grid = (n_split, n_head).  Workspace per (head, split) = [D floats acc][m][l].
__global__ void __launch_bounds__(32)
kv_attn_split_kernel(const block_q8_0 * __restrict__ q8,
                     const block_q8_0 * __restrict__ K,
                     const block_q8_0 * __restrict__ V,
                     float * __restrict__ ws,
                     const int pos, const int n_split, const int n_kv_head,
                     const int head_dim, const int capacity,
                     const int layer, const float scale, const int grp) {
    const int lane  = threadIdx.x & 31;
    const int split = blockIdx.x;
    const int head  = blockIdx.y;
    const int kvh   = head / grp;
    const int blk   = kv_blk_of_lane(lane);
    const int sub   = kv_sub_of_lane(lane);
    const int d0    = blk * 32 + sub * 4;
    const int nblk  = head_dim / QK8_0;

    // query word + scale, constant over the whole split
    const block_q8_0 * qb = q8 + (int64_t)head * nblk + blk;
    int32_t qw;
    std::memcpy(&qw, qb->qs + sub * 4, 4);
    const float dq = f16_to_f32(qb->d) * scale;

    const int n_pos   = pos + 1;
    const int chunk   = (n_pos + n_split - 1) / n_split;
    const int p_begin = split * chunk;
    const int p_end   = min(p_begin + chunk, n_pos);

    const int64_t base = ((int64_t)layer * n_kv_head + kvh) * capacity;

    float m = -INFINITY, l = 0.0f;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int p = p_begin; p < p_end; ++p) {
        const int64_t kb = (base + p) * nblk + blk;
        int32_t kw;
        std::memcpy(&kw, K[kb].qs + sub * 4, 4);
        const float dk = f16_to_f32(K[kb].d);

        int sumi = dp4a((int)qw, (int)kw, 0);
        float part = dq * dk * (float)sumi;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) part += __shfl_xor_sync(0xffffffffu, part, o);

        const float m_new = fmaxf(m, part);
        const float alpha = expf(m - m_new);
        const float p_w   = expf(part - m_new);
        l = l * alpha + p_w;

        const int64_t vb = (base + p) * nblk + blk;
        int32_t vw;
        std::memcpy(&vw, V[vb].qs + sub * 4, 4);
        const float dv = f16_to_f32(V[vb].d);
        const int8_t * vqx = (const int8_t *)&vw;
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[i] = acc[i] * alpha + p_w * ((float)vqx[i] * dv);
        m = m_new;
    }

    float * slot = ws + ((int64_t)head * n_split + split) * (head_dim + 2);
#pragma unroll
    for (int i = 0; i < 4; ++i) slot[d0 + i] = acc[i];
    if (lane == 0) { slot[head_dim + 0] = m; slot[head_dim + 1] = l; }
}

// --- decode: merge the splits and normalise --------------------------------
__global__ void __launch_bounds__(32)
kv_attn_combine_kernel(const float * __restrict__ ws, float * __restrict__ out,
                       const int n_head, const int n_split, const int head_dim) {
    const int lane = threadIdx.x & 31;
    const int head = blockIdx.x;
    if (head >= n_head) return;

    const int blk = kv_blk_of_lane(lane);
    const int sub = kv_sub_of_lane(lane);
    const int d0  = blk * 32 + sub * 4;

    float m = -INFINITY, l = 0.0f;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int s = 0; s < n_split; ++s) {
        const float * slot = ws + ((int64_t)head * n_split + s) * (head_dim + 2);
        const float sm = __shfl_sync(0xffffffffu, slot[head_dim + 0], 0);
        const float sl = __shfl_sync(0xffffffffu, slot[head_dim + 1], 0);
        const float mn = fmaxf(m, sm);
        const float wa = expf(m - mn);
        const float wb = expf(sm - mn);
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[i] = acc[i] * wa + slot[d0 + i] * wb;
        l = l * wa + sl * wb;
        m = mn;
    }
    const float inv = 1.0f / l;
    float * op = out + (int64_t)head * head_dim;
#pragma unroll
    for (int i = 0; i < 4; ++i) op[d0 + i] = acc[i] * inv;
}

// --- zero-fill helper (GDN state reset) ------------------------------------
__global__ void kv_zero_kernel(float * __restrict__ p, const int64_t n) {
    const int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i < n) p[i] = 0.0f;
}

} // namespace
// ===========================================================================
// Host API.  See include/bonsai_kv.h.
// ===========================================================================
static constexpr int KV_DEFAULT_SPLIT = 32;

int64_t bonsai_kv_bytes(int capacity) {
    if (capacity <= 0) return 0;
    if (capacity > model_27b::context_len) capacity = model_27b::context_len;
    return kv_geometry::for_27b(capacity).total_bytes();
}

int64_t bonsai_kv_gdn_bytes(void) { return gdn_state_bytes(); }

bonsai_kv_cache * bonsai_kv_create(int capacity, bonsai_stream stream) {
    if (capacity <= 0) return nullptr;
    if (capacity > model_27b::context_len) capacity = model_27b::context_len;

    bonsai_kv_cache * kv = (bonsai_kv_cache *)std::calloc(1, sizeof(bonsai_kv_cache));
    if (!kv) return nullptr;

    kv->geo     = kv_geometry::for_27b(capacity);
    kv->n_split = KV_DEFAULT_SPLIT;

    const int64_t kbytes = kv->geo.array_bytes();
    const int64_t gbytes = gdn_state_bytes();
    const int64_t wbytes = (int64_t)model_27b::n_head * kv->n_split *
                           (kv->geo.head_dim + 2) * sizeof(float);
    const int64_t qbytes = (int64_t)model_27b::n_head *
                           (kv->geo.head_dim / QK8_0) * sizeof(block_q8_0);

    if (cudaMalloc(&kv->d_k, kbytes) != cudaSuccess) { std::free(kv); return nullptr; }
    if (cudaMalloc(&kv->d_v, kbytes) != cudaSuccess) {
        cudaFree(kv->d_k); std::free(kv); return nullptr;
    }
    if (cudaMalloc(&kv->d_gdn, gbytes) != cudaSuccess) {
        cudaFree(kv->d_v); cudaFree(kv->d_k); std::free(kv); return nullptr;
    }
    if (cudaMalloc(&kv->d_wsplit, wbytes) != cudaSuccess) {
        cudaFree(kv->d_gdn); cudaFree(kv->d_v); cudaFree(kv->d_k); std::free(kv); return nullptr;
    }
    if (cudaMalloc(&kv->d_q8, qbytes) != cudaSuccess) {
        cudaFree(kv->d_wsplit); cudaFree(kv->d_gdn); cudaFree(kv->d_v);
        cudaFree(kv->d_k); std::free(kv); return nullptr;
    }

    (void)cudaMemsetAsync(kv->d_gdn, 0, gbytes, (cudaStream_t)stream);
    return kv;
}

void bonsai_kv_free(bonsai_kv_cache * kv) {
    if (!kv) return;
    cudaFree(kv->d_k);
    cudaFree(kv->d_v);
    cudaFree(kv->d_gdn);
    cudaFree(kv->d_wsplit);
    cudaFree(kv->d_q8);
    std::free(kv);
}

void bonsai_kv_reset(bonsai_kv_cache * kv, bonsai_stream stream) {
    if (!kv) return;
    const int64_t n = gdn_state_bytes() / sizeof(float);
    const int t = 256;
    kv_zero_kernel<<<(unsigned)((n + t - 1) / t), t, 0, (cudaStream_t)stream>>>(kv->d_gdn, n);
}

float * bonsai_kv_gdn_state(bonsai_kv_cache * kv, int layer) {
    if (!kv || layer < 0 || layer >= model_27b::n_linear) return nullptr;
    const int64_t per_layer = (int64_t)model_27b::ssm_v_heads *
                              model_27b::ssm_state * model_27b::ssm_state;
    return kv->d_gdn + (int64_t)layer * per_layer;
}

int bonsai_kv_append_batch(bonsai_kv_cache * kv, int layer, int pos0, int n_tok,
                           const float * k, const float * v, bonsai_stream stream) {
    if (!kv || !k || !v) return 1;
    if (layer < 0 || layer >= kv->geo.n_layers) return 1;
    if (n_tok <= 0 || pos0 < 0 || pos0 + n_tok > kv->geo.capacity) return 1;

    const int warps  = 8;
    const int nrow   = n_tok * kv->geo.n_kv_head;
    const int blocks = (nrow + warps - 1) / warps;
    kv_quantize_append_kernel<<<blocks, warps * 32, 0, (cudaStream_t)stream>>>(
        kv->d_k, kv->d_v, k, v, pos0, n_tok, kv->geo.n_kv_head, kv->geo.head_dim,
        kv->geo.capacity, layer);
    return (int)cudaGetLastError();
}

int bonsai_kv_append(bonsai_kv_cache * kv, int layer, int pos,
                     const float * k, const float * v, bonsai_stream stream) {
    return bonsai_kv_append_batch(kv, layer, pos, 1, k, v, stream);
}

int bonsai_kv_attn_decode(bonsai_kv_cache * kv, int layer, int pos,
                          const float * q, float * out, bonsai_stream stream) {
    if (!kv || !q || !out) return 1;
    if (layer < 0 || layer >= kv->geo.n_layers) return 1;
    if (pos < 0 || pos >= kv->geo.capacity) return 1;

    cudaStream_t s = (cudaStream_t)stream;
    const int n_head = model_27b::n_head;
    const int n_kv   = kv->geo.n_kv_head;
    const int D      = kv->geo.head_dim;
    const int grp    = n_head / n_kv;                        // 6 q heads per kv head
    const float scale = 1.0f / sqrtf((float)D);
    const int n_split = kv->n_split;

    // 1. quantise the query heads
    kv_quantize_q_kernel<<<(n_head + 7) / 8, 8 * 32, 0, s>>>(q, kv->d_q8, n_head, D);

    // 2. split flash pass: one block per (split, q head)
    dim3 grid(n_split, n_head);
    kv_attn_split_kernel<<<grid, 32, 0, s>>>(
        kv->d_q8, kv->d_k, kv->d_v, kv->d_wsplit, pos, n_split, n_kv, D,
        kv->geo.capacity, layer, scale, grp);

    // 3. merge splits and normalise
    kv_attn_combine_kernel<<<n_head, 32, 0, s>>>(kv->d_wsplit, out, n_head, n_split, D);

    return (int)cudaGetLastError();
}