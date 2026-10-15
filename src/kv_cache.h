// kv_cache.h -- plan and numeric core of the Bonsai-P40 KV cache.
//
// WHY THIS IS NOT A UNIFORM CACHE
// -------------------------------
// Ternary-Bonsai-2-27B (qwen35) is a hybrid-attention model: only 16 of the 64
// blocks are full attention (block index i with (i+1)%4 == 0).  The other 48
// are Gated DeltaNet and keep a *constant-size* recurrent state per head, so
// they do not grow with context at all.  Per token:
//
//   16 full layers    x 4 kv heads x 256 dim = 16384 K elems (+ 16384 V)
//   48 linear layers  x 48 v heads x 128 x 128 = fixed 3 MiB per layer
//
// So the plan has two halves with completely different behaviour, and only the
// growing half decides whether 262K context is affordable on 24 GB.
//
// SIZING ON THE P40 (16 layers, 4 kv heads, head_dim 256, capacity 262144)
// -----------------------------------------------------------------------
//   type          bytes/token   cache @262k   read/decode   +20.8ms  tok/s
//   F16 / F16          65536       16.00 GiB      49.7 ms    70.5 ms  14.2
//   Q8_0 / Q8_0        34816        8.50 GiB      26.4 ms    47.2 ms  21.2
//   Q8_0 / Q4_0        26624        6.50 GiB      20.2 ms    41.0 ms  24.4
//   Q4_0 / Q4_0        18432        4.50 GiB      14.0 ms    34.8 ms  28.8
//
//   weights 7.21 GiB + F16/F16 cache 16.00 GiB = 23.21 GiB of 24 GiB
//
// F16 is therefore NOT usable: it leaves no room for activations, the CUDA
// context, the GDN state (144 MiB) or the graph workspace.  Q8_0/Q8_0 is the
// default -- it roughly halves the KV traffic versus F16 while 8-bit K keeps
// the attention scores unbiased.  The KV read only matters at long context:
// 0.41 ms at 4K, 3.3 ms at 32K, 26.4 ms at 262K, i.e. it overtakes the 20.8 ms
// weight stream somewhere around 200K.
//
// LAYOUT
// ------
// Head-major, K and V in separate arrays so they can be quantised differently:
//
//   K[ ((layer * n_kv_head + kv_head) * capacity + pos) * blocks_per_pos + blk ]
//
// Streaming consecutive `pos` for one (layer, kv_head) is one contiguous run of
// capacity * (head_dim/32) * 34 bytes.  That makes the decode read a clean DRAM
// stream on the P40; a token-major layout would stride by n_kv_head*head_dim
// bytes per position and waste requests.
//
// Q8_0 (34 B = 1 fp16 scale + 32 int8) is used for both, which reduces the
// score to DP4A: byte i of a K block pairs with byte i of the query block with
// no shuffle -- exactly the alignment dp4a wants.
#pragma once

#include "p40_common.h"
#include "bonsai_model.h"

namespace bonsai {

// Same macro-collision rule as in pq2_0.h: the fork's ggml-common.h
// #defines QK8_0, so we must not declare that name ourselves.
#ifdef QK8_0
static constexpr int BONSAI_QK8_0 = QK8_0;           // inherit ggml's value
#else
static constexpr int BONSAI_QK8_0 = 32;
#endif

struct block_q8_0 {                              // 34 bytes
    uint16_t d;                                  // fp16 scale
    int8_t   qs[BONSAI_QK8_0];
};

// ---------------------------------------------------------------------------
// Cache geometry.  Only the 16 full-attention layers own a growing cache.
// ---------------------------------------------------------------------------
struct kv_geometry {
    int n_layers;      // full-attention layers carrying a cache (16)
    int n_kv_head;     // 4
    int head_dim;      // 256
    int capacity;      // max positions (up to 262144)

    static kv_geometry for_27b(int capacity) {
        return kv_geometry{model_27b::n_full, model_27b::n_head_kv,
                           model_27b::head_dim, capacity};
    }

    int     blocks_per_pos()    const { return head_dim / BONSAI_QK8_0; }              // 8
    int64_t blocks_per_stream() const { return (int64_t)capacity * blocks_per_pos(); }
    int64_t k_blocks()          const { return (int64_t)n_layers * n_kv_head * blocks_per_stream(); }
    int64_t array_bytes()       const { return k_blocks() * (int64_t)sizeof(block_q8_0); }
    int64_t total_bytes()       const { return 2 * array_bytes(); }

    int64_t block_index(int layer, int kv_head, int pos, int blk) const {
        return (((int64_t)layer * n_kv_head + kv_head) * capacity + pos) * blocks_per_pos() + blk;
    }
};

// Gated DeltaNet state: constant, independent of context.
//   48 layers x 48 value heads x 128 x 128 floats (FP32 on purpose: ~144 MiB
//   total and 0.87 ms read+write per token, ~4% of the weight stream -- not
//   worth halving at the cost of precision).
inline int64_t gdn_state_bytes() {
    return (int64_t)model_27b::n_linear * model_27b::ssm_v_heads *
           model_27b::ssm_state * model_27b::ssm_state * sizeof(float);
}

// ---------------------------------------------------------------------------
// Quantisation of a K or V row (head_dim contiguous floats) into block_q8_0.
// Same Q8_0 block as everywhere in llama.cpp: d = amax/127.
// ---------------------------------------------------------------------------
inline void kv_quantize_row_q8_0(const float * x, block_q8_0 * y, int n) {
    for (int b = 0; b < n / BONSAI_QK8_0; ++b) {
        const float * src = x + b * BONSAI_QK8_0;
        float amax = 0.0f;
        for (int i = 0; i < BONSAI_QK8_0; ++i) amax = fmaxf(amax, fabsf(src[i]));
        const float d  = amax / 127.0f;
        const float id = d > 0.0f ? 1.0f / d : 0.0f;
        y[b].d = f32_to_f16(d);
        for (int i = 0; i < BONSAI_QK8_0; ++i) {
            int v = (int)lroundf(src[i] * id);
            if (v < -127) { v = -127; }
            if (v >  127) { v =  127; }
            y[b].qs[i] = (int8_t)v;
        }
    }
}

inline void kv_dequantize_row_q8_0(const block_q8_0 * x, float * y, int n) {
    for (int b = 0; b < n / BONSAI_QK8_0; ++b) {
        const float d = f16_to_f32(x[b].d);
        for (int i = 0; i < BONSAI_QK8_0; ++i) y[b * BONSAI_QK8_0 + i] = (float)x[b].qs[i] * d;
    }
}

// ---------------------------------------------------------------------------
// Score core: int8 dot of two Q8_0 blocks via dp4a, scaled by both.  This is
// the exact body of the split-K attention kernel.
// ---------------------------------------------------------------------------
BONSAI_HD inline float kv_dot_q8_0_block(const block_q8_0 * a, const block_q8_0 * b) {
    int sumi = 0;
    for (int w = 0; w < BONSAI_QK8_0 / 4; ++w) {
        int av, bv;
        std::memcpy(&av, a->qs + w * 4, 4);
        std::memcpy(&bv, b->qs + w * 4, 4);
        sumi = dp4a(av, bv, sumi);
    }
    return f16_to_f32(a->d) * f16_to_f32(b->d) * (float)sumi;
}

// Whole-head score between a quantised query head and a cached key head.
BONSAI_HD inline float kv_score(const block_q8_0 * q, const block_q8_0 * k, int nblocks) {
    float acc = 0.0f;
    for (int b = 0; b < nblocks; ++b) acc += kv_dot_q8_0_block(q + b, k + b);
    return acc;
}

// ---------------------------------------------------------------------------
// Split-K bookkeeping.  Each split produces a partial (max m, sumexp l,
// accumulator acc[D]); the combiner merges them with rescaling.  n_split == 1
// degenerates to plain online softmax.
// ---------------------------------------------------------------------------
struct kv_split {
    float m;        // running max of the scores in this split
    float l;        // running sum of exp(score - m)
};

// Merge split `b` into `a`:  a := combine(a, b),  rescaling a by exp(a.m - m).
BONSAI_HD inline void kv_split_merge(float a_m, float a_l, float * a_acc,
                                     float b_m, float b_l, const float * b_acc, int D) {
    const float m  = fmaxf(a_m, b_m);
    const float wa = expf(a_m - m);
    const float wb = expf(b_m - m);
    for (int d = 0; d < D; ++d) a_acc[d] = a_acc[d] * wa + b_acc[d] * wb;
    (void)a_l; (void)b_l;
}

// Final normalisation: out = acc / l.
BONSAI_HD inline void kv_split_finish(const float * acc, float l, float * out, int D) {
    const float inv = 1.0f / l;
    for (int d = 0; d < D; ++d) out[d] = acc[d] * inv;
}

} // namespace bonsai