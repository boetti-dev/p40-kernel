// bonsai_p40.h -- public C API of the Bonsai-P40 compute kernels.
//
// This library contains ONE forward path: Ternary-Bonsai-2-27B (qwen35) in the
// PQ2_0 ("q2_0") packing, running on an NVIDIA Tesla P40 (sm_61).  There is no
// code path for any other model, quantisation, or GPU.
#pragma once

#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque stream handle so the header stays free of CUDA types.
typedef void * bonsai_stream;

// --- decode (batch = 1) matrix-vector ---------------------------------------
// y[n_out] = W[n_out, n_in] . x, where W is PQ2_0 and x has already been
// quantised to the PQ2_0-native Q8_1 layout by bonsai_p40_quant_activations().
int bonsai_p40_gemv(const void * W, const void * xq, float * y,
                    int64_t n_out, int64_t n_in, bonsai_stream stream);

// Quantise `n` floats to the permuted Q8_1 activation layout.
int bonsai_p40_quant_activations(const float * x, void * out, int64_t n, bonsai_stream stream);

// --- prefill (batch > 1) batched matmul -------------------------------------
// Y[n_out, n_tok] = W[n_out, n_in] . X[n_in, n_tok], X in permuted Q8_1.
int bonsai_p40_gemm(const void * W, const void * Xq, float * Y,
                    int64_t n_out, int64_t n_in, int64_t n_tok, bonsai_stream stream);

// --- aligned repacked layout (docs/NEXT_STEPS.md, Weg 2) --------------------
// W in the repacked layout is size-neutral: row stride = 34 * (n_in/128).
// Requires n_in/128 to be a multiple of 8 (true for every row of this model).
int bonsai_p40_repack(const void * src, void * dst, int64_t n_out, int64_t n_in, bonsai_stream stream);
int bonsai_p40_gemv_repack(const void * W, const float * x, float * y,
                           int64_t n_out, int64_t n_in, bonsai_stream stream);

// --- fused elementwise / transform kernels ----------------------------------
// Activation Hadamard rotation (block 1024) with an explicit +-1 sign vector.
int bonsai_p40_fwht(float * x, const float * signs, int64_t n, bonsai_stream stream);

// RMSNorm over rows of length n.
int bonsai_p40_rmsnorm(const float * x, const float * w, float * y,
                       int64_t n_rows, int64_t n, float eps, bonsai_stream stream);

// Gated DeltaNet recurrent step for the linear-attention layers (GQA: n_v_heads
// value heads share n_k_heads key heads).
int bonsai_p40_gdn_step(float * state, const float * q, const float * k, const float * v,
                        const float * g, const float * beta, float * out,
                        int n_k_heads, int n_v_heads, int K, int V, bonsai_stream stream);

// Causal flash attention for the full-attention layers (GQA n_head/n_head_kv).
int bonsai_p40_attn(const float * q, const float * k, const float * v, float * out,
                    int n_tok, int n_head, int n_head_kv, int head_dim, float scale,
                    bonsai_stream stream);

#ifdef __cplusplus
}
#endif
