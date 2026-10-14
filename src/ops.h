// ops.h -- host/device elementwise and recurrent primitives shared by the P40
// kernels and the host reference.  Every function here is BONSAI_HD so the
// exact arithmetic the P40 executes can be checked on the host.
#pragma once

#include "p40_common.h"

namespace bonsai {

// ---------------------------------------------------------------------------
// RMSNorm (per row).  Qwen uses eps = 1e-6 and a learned weight.
// ---------------------------------------------------------------------------
BONSAI_HD inline float rmsnorm_scale(const float * x, int n, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < n; ++i) ss += x[i] * x[i];
    return 1.0f / sqrtf(ss / (float)n + eps);
}

BONSAI_HD inline void rmsnorm(const float * x, const float * w, float * y, int n, float eps) {
    const float s = rmsnorm_scale(x, n, eps);
    for (int i = 0; i < n; ++i) y[i] = x[i] * s * w[i];
}

// ---------------------------------------------------------------------------
// SwiGLU: out = silu(gate) * up, then the caller applies the down projection.
// ---------------------------------------------------------------------------
BONSAI_HD inline float silu(float x) { return x / (1.0f + expf(-x)); }

BONSAI_HD inline void swiglu(const float * gate, const float * up, float * y, int n) {
    for (int i = 0; i < n; ++i) y[i] = silu(gate[i]) * up[i];
}

// ---------------------------------------------------------------------------
// RoPE (GPT-NeoX / Qwen half-split convention) over `rope_dim` of every head
// of dimension `head_dim`.  Position `pos`, base `freq_base`.
// ---------------------------------------------------------------------------
BONSAI_HD inline void rope_apply(float * q, int head_dim, int rope_dim, int pos, float freq_base) {
    const int half = rope_dim / 2;
    for (int i = 0; i < half; ++i) {
        const float inv = powf(freq_base, -2.0f * (float)i / (float)rope_dim);
        const float ang = (float)pos * inv;
        const float c = cosf(ang), s = sinf(ang);
        const float x0 = q[i], x1 = q[i + half];
        q[i]        = x0 * c - x1 * s;
        q[i + half] = x0 * s + x1 * c;
    }
    (void)head_dim;
}

// ---------------------------------------------------------------------------
// Gated DeltaNet recurrent step (the linear-attention layers of qwen35).
//
//   state S : [K, V]   (per value head, K = V = state_size = 128)
//   g       : log decay for this step (scalar, per head)
//   beta    : write strength  (scalar, per head)
//   k       : [K]  key,  q : [K] query,  v : [V] value
//   out     : [V]
//
//   S  <- S * exp(g)
//   kv <- k^T S
//   S  <- S + beta * k (x) (v - kv)
//   out<- q^T S
//
// Row-major S indexed [i*V + j].
// ---------------------------------------------------------------------------
BONSAI_HD inline void gdn_step(float * S, const float * q, const float * k,
                               const float * v, float g, float beta, float * out,
                               int K, int V) {
    const float decay = expf(g);
    // S *= decay
    for (int i = 0; i < K; ++i)
        for (int j = 0; j < V; ++j) S[i * V + j] *= decay;

    // kv = k^T S  (length V)
    for (int j = 0; j < V; ++j) {
        float acc = 0.0f;
        for (int i = 0; i < K; ++i) acc += k[i] * S[i * V + j];
        out[j] = acc;                    // reuse `out` as kv scratch
    }
    // rank-1 update
    for (int i = 0; i < K; ++i) {
        const float kk = beta * k[i];
        for (int j = 0; j < V; ++j) S[i * V + j] += kk * (v[j] - out[j]);
    }
    // out = q^T S
    for (int j = 0; j < V; ++j) {
        float acc = 0.0f;
        for (int i = 0; i < K; ++i) acc += q[i] * S[i * V + j];
        out[j] = acc;
    }
}

// Depthwise causal conv1d (kernel `k`, dilation 1) used by the GDN input path.
BONSAI_HD inline float conv1d_depthwise(const float * x, const float * w, int t, int ch, int k, int nch) {
    float acc = 0.0f;
    for (int i = 0; i < k; ++i) {
        const int tt = t - (k - 1) + i;
        if (tt >= 0) acc += x[tt * nch + ch] * w[i * nch + ch];
    }
    return acc;
}

} // namespace bonsai
