// pq2_0.h -- the "q2_0" weight format of Ternary-Bonsai-2-27B and its numeric
// core, shared verbatim between the host reference/tests and the P40 kernels.
//
// The shipped GGUF file is the PQ2_0 packing (GGML_TYPE_PQ2_0 = 142):
//
//   block_pq2_0 : 128 weights, 2-bit codes, one fp16 scale per 128.
//   code c in {0,1,2,3}  ->  value = (c - 1) * d   (dequantize_row_pq2_0)
//   element j lives in byte j/4, bits (j%4)*2, little endian.
//
// The checkpoint is ternary, so only codes {0,1,2} actually occur; the kernel
// still handles code 3 because the format (and the reference quantiser) can
// produce it and the dequantiser maps it to +2.
//
// Group size 128, so block size = 2 (fp16 d) + 32 (codes) = 34 bytes.
#pragma once

#include "p40_common.h"

namespace bonsai {

static constexpr int QK_PQ2_0 = 128;             // weights per block
static constexpr int QK8_1    = 32;              // elements per q8_1 activation block
static constexpr int QI_PQ2_0 = QK_PQ2_0 / 32;   // 4 code chunks per block

struct block_pq2_0 {                             // 34 bytes, 2-byte aligned
    uint16_t d;                                  // fp16 group scale
    uint8_t  qs[QK_PQ2_0 / 4];                   // 32 bytes = 128 * 2 bit
};

struct block_q8_1 {                              // 36 bytes
    uint16_t d;                                  // fp16 scale
    uint16_t s;                                  // fp16 bits holding int16 sum(q)
    int8_t   qs[QK8_1];
};

BONSAI_HD inline float pq2_0_value(const block_pq2_0 * b, int j) {
    const int code = (b->qs[j >> 2] >> ((j & 3) * 2)) & 0x03;
    return (float)(code - 1) * f16_to_f32(b->d);
}

// Reference dequantiser -- mirrors dequantize_row_pq2_0() in the PrismML fork.
inline void dequantize_row_pq2_0(const block_pq2_0 * x, float * y, int64_t k) {
    const int64_t nb = k / QK_PQ2_0;
    for (int64_t i = 0; i < nb; ++i) {
        const float d = f16_to_f32(x[i].d);
        for (int j = 0; j < QK_PQ2_0; ++j) {
            const int code = (x[i].qs[j >> 2] >> ((j & 3) * 2)) & 0x03;
            y[i * QK_PQ2_0 + j] = (float)(code - 1) * d;
        }
    }
}

// Reference quantiser -- mirrors quantize_row_pq2_0_ref().
inline void quantize_row_pq2_0(const float * x, block_pq2_0 * y, int64_t k) {
    const int64_t nb = k / QK_PQ2_0;
    for (int64_t i = 0; i < nb; ++i) {
        float amax = 0.0f;
        for (int j = 0; j < QK_PQ2_0; ++j) amax = fmaxf(amax, fabsf(x[i * QK_PQ2_0 + j]));
        const float d  = amax;
        const float id = d > 0.0f ? 1.0f / d : 0.0f;
        y[i].d = f32_to_f16(d);
        for (int j = 0; j < QK_PQ2_0 / 4; ++j) y[i].qs[j] = 0;
        for (int j = 0; j < QK_PQ2_0; ++j) {
            int q = (int)lroundf(x[i * QK_PQ2_0 + j] * id) + 1;
            if (q < 0) q = 0; if (q > 3) q = 3;
            y[i].qs[j >> 2] |= (uint8_t)(q << ((j & 3) * 2));
        }
    }
}

// ---------------------------------------------------------------------------
// Activation quantiser.  `permute = false` gives the plain Q8_1 layout;
// `permute = true` gives the PQ2_0-native layout used by the device kernels:
// inside every 32-element chunk the four int32 words are transposed so that
// word k holds elements {k, k+4, k+8, k+12}.  This lets the kernel pair the
// extracted 2-bit codes with activation bytes without any per-weight shuffle.
// ---------------------------------------------------------------------------
inline void quantize_row_q8_1(const float * x, block_q8_1 * y, int64_t k, bool permute) {
    const int64_t nb = k / QK8_1;
    for (int64_t i = 0; i < nb; ++i) {
        float amax = 0.0f;
        for (int j = 0; j < QK8_1; ++j) amax = fmaxf(amax, fabsf(x[i * QK8_1 + j]));
        const float d  = amax / 127.0f;
        const float id = d > 0.0f ? 1.0f / d : 0.0f;
        int sum = 0;
        int8_t q[QK8_1];
        for (int j = 0; j < QK8_1; ++j) {
            int v = (int)lroundf(x[i * QK8_1 + j] * id);
            if (v < -127) v = -127; if (v > 127) v = 127;
            q[j] = (int8_t)v; sum += v;
        }
        y[i].d = f32_to_f16(d);
        // `s` carries the raw int16 activation sum in its bit pattern (the
        // PQ2_0 kernels read it back as int16, never as a float).
        y[i].s = (uint16_t)(int16_t)sum;
        // Transposed 4x4 layout inside each 16-element half: position j holds
        // element 16*(j>>4) + 4*(j&3) + ((j>>2)&3), so the kernel's extracted
        // 2-bit code word k lines up byte-for-byte with activation word k.
        for (int j = 0; j < QK8_1; ++j) {
            const int src = 16 * (j >> 4) + 4 * (j & 3) + ((j >> 2) & 3);
            y[i].qs[j] = q[permute ? src : j];
        }
    }
}

BONSAI_HD inline int q8_1_sum(const block_q8_1 * a) { return (int16_t)a->s; }

// ---------------------------------------------------------------------------
// Naive float dot of one PQ2_0 block against 128 float activations.  This is
// the definition every kernel must reproduce.
// ---------------------------------------------------------------------------
inline float pq2_0_dot_f32(const block_pq2_0 * w, const float * a) {
    const float d = f16_to_f32(w->d);
    float acc = 0.0f;
    for (int j = 0; j < QK_PQ2_0; ++j) {
        const int code = (w->qs[j >> 2] >> ((j & 3) * 2)) & 0x03;
        acc += (float)(code - 1) * a[j];
    }
    return acc * d;
}

// ---------------------------------------------------------------------------
// Integer DP4A dot of one PQ2_0 block against the single Q8_1 activation block
// that covers chunk `iqs` (32 activations).  This is the exact inner body of
// both device kernels: the GEMV passes agroup[iqs], the GEMM passes the staged
// shared-memory block directly.
BONSAI_HD inline float pq2_0_dot_q8_1_block(const block_pq2_0 * w,
                                            const block_q8_1 * a,
                                            int iqs) {
    const uint8_t * qs = w->qs + iqs * 8;     // 8 code bytes = 32 elements
    const int8_t  * as = a->qs;               // 32 activations
    int sumi = 0;
    for (int word = 0; word < 2; ++word) {
        uint32_t wv; std::memcpy(&wv, qs + word * 4, 4);
        int av[4]; std::memcpy(av, as + word * 16, 16);
        // t[k] holds codes {k, k+4, k+8, k+12} of this 16-code word as bytes.
        const int t0 = (int)((wv >> 0) & 0x03030303u);
        const int t1 = (int)((wv >> 2) & 0x03030303u);
        const int t2 = (int)((wv >> 4) & 0x03030303u);
        const int t3 = (int)((wv >> 6) & 0x03030303u);
        sumi = dp4a(t0, av[0], sumi);
        sumi = dp4a(t1, av[1], sumi);
        sumi = dp4a(t2, av[2], sumi);
        sumi = dp4a(t3, av[3], sumi);
    }
    // value = code - 1  =>  sum((c-1)*a) = sum(c*a) - sum(a)
    sumi -= q8_1_sum(a);
    const float d = f16_to_f32(w->d) * f16_to_f32(a->d);
    return d * (float)sumi;
}

// Group form: `agroup` points at the four contiguous activation blocks of one
// weight block (the GEMV layout, one token).
BONSAI_HD inline float pq2_0_dot_q8_1_perm(const block_pq2_0 * w,
                                           const block_q8_1 * agroup,
                                           int iqs) {
    return pq2_0_dot_q8_1_block(w, agroup + iqs, iqs);
}

// Whole-row dot: `n` must be a multiple of QK_PQ2_0.
inline float pq2_0_row_dot_perm(const block_pq2_0 * w, const block_q8_1 * a, int n) {
    float acc = 0.0f;
    for (int i = 0; i < n / QK_PQ2_0; ++i) {
        const block_pq2_0 * wb = w + i;
        const block_q8_1  * ab = a + i * QI_PQ2_0;
        for (int iqs = 0; iqs < QI_PQ2_0; ++iqs) acc += pq2_0_dot_q8_1_perm(wb, ab, iqs);
    }
    return acc;
}

} // namespace bonsai
