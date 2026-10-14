// fwht.h -- the blockwise Hadamard transform declared by the model metadata
// (prism.hadamard.*): normalized Sylvester-Walsh-Hadamard, block size 1024,
// applied along the input (last) dimension, with an explicit +-1 sign vector.
//
// The rotation is folded into the stored weights offline, so at inference the
// runtime only has to apply the matching transform to activations.  This is a
// pure elementwise butterfly: it is bandwidth-bound and maps very well to the
// P40's 128 FP32 lanes/SM.
#pragma once

#include "p40_common.h"

namespace bonsai {

static constexpr int HADAMARD_BLOCK = 1024;
static constexpr float HADAMARD_NORM = 1.0f / 32.0f; // 1/sqrt(1024)

// In-place normalised WHT of one 1024-element block.  `signs` may be null.
// The transform is orthonormal: fwht(fwht(x)) == x (up to fp32 rounding).
BONSAI_HD inline void fwht_1024(float * x, const float * signs) {
    if (signs) {
        for (int i = 0; i < HADAMARD_BLOCK; ++i) x[i] *= signs[i];
    }
    for (int h = 1; h < HADAMARD_BLOCK; h <<= 1) {
        for (int i = 0; i < HADAMARD_BLOCK; i += 2 * h) {
            for (int j = 0; j < h; ++j) {
                const float a = x[i + j];
                const float b = x[i + j + h];
                x[i + j]     = a + b;
                x[i + j + h] = a - b;
            }
        }
    }
    for (int i = 0; i < HADAMARD_BLOCK; ++i) x[i] *= HADAMARD_NORM;
}

// Applies the transform blockwise over a vector of length n (n % 1024 == 0).
BONSAI_HD inline void fwht_blocks(float * x, const float * signs, int n) {
    for (int b = 0; b < n; b += HADAMARD_BLOCK) {
        fwht_1024(x + b, signs ? signs + b : nullptr);
    }
}

} // namespace bonsai
