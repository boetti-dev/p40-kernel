// bonsai_kv.h -- KV-cache API of the Bonsai-P40 engine.
//
// The cache is split by design, because the model is hybrid:
//   * 16 full-attention layers own a growing Q8_0 K/V cache (see src/kv_cache.h)
//   * 48 Gated DeltaNet layers own a constant 3 MiB recurrent state each
// Both are allocated once from a descriptor; nothing is paged or evicted,
// because the model has a fixed 262144-token window and no sliding window.
#pragma once

#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

typedef void * bonsai_stream;

// Opaque cache handle (host side structure, device arrays inside).
typedef struct bonsai_kv_cache bonsai_kv_cache;

// --- sizing ---------------------------------------------------------------
// Bytes needed for the growing K+V cache at `capacity` positions.
int64_t bonsai_kv_bytes(int capacity);
// Bytes needed for the constant Gated DeltaNet state (independent of capacity).
int64_t bonsai_kv_gdn_bytes(void);

// --- lifecycle ------------------------------------------------------------
// Allocates the whole state (growing cache + GDN state).  Returns NULL on OOM.
// `capacity` is clamped to the model's 262144-token window.
bonsai_kv_cache * bonsai_kv_create(int capacity, bonsai_stream stream);
void              bonsai_kv_free(bonsai_kv_cache * kv);
// Resets every position counter to 0 (no device writes to the cache needed).
void              bonsai_kv_reset(bonsai_kv_cache * kv, bonsai_stream stream);

// --- append ---------------------------------------------------------------
// Quantises and stores the K and V of one full-attention layer at position
// `pos`.  `k`/`v` are [n_kv_head, head_dim] row-major floats as produced by the
// projection (already RoPE'd for K).  `layer` is the 0..15 full-layer index.
int bonsai_kv_append(bonsai_kv_cache * kv, int layer, int pos,
                     const float * k, const float * v, bonsai_stream stream);

// Batched variant for prefill: appends `n_tok` positions starting at `pos0`.
// `k`/`v` are [n_tok, n_kv_head, head_dim].
int bonsai_kv_append_batch(bonsai_kv_cache * kv, int layer, int pos0, int n_tok,
                           const float * k, const float * v, bonsai_stream stream);

// --- decode ---------------------------------------------------------------
// Attention for one full-attention layer at query position `pos` (inclusive
// causal over 0..pos).  `q` is [n_head, head_dim]; writes `out` [n_head, head_dim].
// Internally a split-K flash decode over the cache (see kv_p40.cu).
int bonsai_kv_attn_decode(bonsai_kv_cache * kv, int layer, int pos,
                          const float * q, float * out, bonsai_stream stream);

// --- Gated DeltaNet state -------------------------------------------------
// Returns the per-layer state pointer (n_v_head, state, state floats).
float * bonsai_kv_gdn_state(bonsai_kv_cache * kv, int layer);

#ifdef __cplusplus
}
#endif