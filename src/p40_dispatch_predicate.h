// p40_dispatch_predicate.h -- the ONLY decision that lets the P40 kernel take
// over a MUL_MAT, kept free of ggml and CUDA types so it can be unit-tested.
//
// This is the safety-critical part of the PrismML integration: if it says yes
// for a shape the kernel does not implement, results are silently wrong; if it
// says no, ggml just falls through to the stock path.  Hence: pure C++,
// exhaustively tested (see tests/test_p40.cpp, section 11).
#pragma once

#include <cstdint>

namespace bonsai {

// Numeric values of the ggml_type enum we care about (ggml.h; stable ABI).
static constexpr int P40_GGML_TYPE_F32   = 0;
static constexpr int P40_GGML_TYPE_PQ2_0 = 142;

// Geometry the kernel actually supports.
static constexpr int P40_QK_PQ2_0        = 128;   // weights per PQ2_0 block
static constexpr int P40_BLOCK_PQ2_0_BYTES = 34;  // fp16 d + 32 code bytes
static constexpr int P40_MAX_ROW         = 32768; // widest row we allocate scratch for

// Everything the predicate needs, as plain numbers.
struct p40_mulmat_shape {
    int     type0, type1, type2;   // src0, src1, dst ggml types
    int64_t n_in, n_out;           // src0->ne[0], src0->ne[1]
    int64_t ne20, ne30;            // src0->ne[2], src0->ne[3]
    int64_t ne11, ne12, ne13;      // src1->ne[1..3]
    int64_t dst_ne1;               // dst->ne[1]
    int64_t nb1;                   // src0->nb[1] (row stride in bytes)
};

inline bool p40_decode_path_ok(const p40_mulmat_shape & s) {
    if (s.type0 != P40_GGML_TYPE_PQ2_0) return false;
    if (s.type1 != P40_GGML_TYPE_F32 || s.type2 != P40_GGML_TYPE_F32) return false;
    // plain 2D MUL_MAT, one output column (decode)
    if (s.ne20 != 1 || s.ne30 != 1) return false;
    if (s.ne11 != 1 || s.ne12 != 1 || s.ne13 != 1) return false;
    if (s.dst_ne1 != 1) return false;
    // dimensions the kernel handles
    if (s.n_in <= 0 || s.n_out <= 0) return false;
    if (s.n_in % P40_QK_PQ2_0 != 0) return false;
    if (s.n_in > P40_MAX_ROW) return false;
    // rows must be a contiguous run of PQ2_0 blocks
    if (s.nb1 != (s.n_in / P40_QK_PQ2_0) * P40_BLOCK_PQ2_0_BYTES) return false;
    return true;
}

} // namespace bonsai
