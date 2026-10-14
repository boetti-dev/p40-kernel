// p40_common.h -- shared host/device primitives for the Bonsai-P40 kernel.
//
// Everything in this header is compiled both by nvcc (device) and by a plain
// C++ host compiler (reference + tests).  The numeric core of every kernel is
// expressed as a BONSAI_HD inline function so the exact same code path that the
// P40 runs can be validated on the host against a naive dequantised reference.
//
// Target: NVIDIA Tesla P40 (GP102, sm_61, 24 GB GDDR5, ~346 GB/s, no tensor
// cores, FP16 ALU throughput = 1/64 of FP32 -> we never do FP16 math, we use
// FP32 accumulate + INT8 DP4A, which sm_61 supports).
#pragma once

#include <cstdint>
#include <cmath>
#include <cstring>

#ifdef __CUDACC__
#define BONSAI_HD __host__ __device__
#define BONSAI_D  __device__
#define BONSAI_H  __host__
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#else
#define BONSAI_HD
#define BONSAI_D
#define BONSAI_H
#endif

namespace bonsai {

// ---------------------------------------------------------------------------
// Tesla P40 (GP102) constants.  All tuning decisions in this kernel are made
// against these numbers and nothing else -- the kernel deliberately does not
// carry generic code paths for other GPUs.
// ---------------------------------------------------------------------------
struct p40_spec {
    static constexpr int    cc                = 61;      // sm_61
    static constexpr int    sm_count          = 30;      // GP102: 30 SMs
    static constexpr int    cores_per_sm      = 128;
    static constexpr int    cores             = 3840;
    static constexpr int    clock_khz         = 1531000; // boost
    static constexpr int    threads_per_sm    = 2048;
    static constexpr int    regs_per_sm       = 65536;
    static constexpr int    smem_per_sm       = 98304;   // 96 KiB
    static constexpr int    smem_per_block    = 49152;   // 48 KiB (opt-in)
    static constexpr int    max_blocks_per_sm = 32;
    static constexpr int    l2_bytes          = 3u * 1024u * 1024u; // 3 MiB
    static constexpr double mem_bw_gbs        = 346.0;   // GDDR5 peak
    static constexpr double fp32_tflops       = 2.0 * cores * clock_khz * 1e3 / 1e12;
    static constexpr int    has_dp4a          = 1;       // cc >= 610
};

// ---------------------------------------------------------------------------
// fp16 <-> fp32.  On device we use the hardware intrinsics; on the host we use
// a bit-exact software conversion so tests reproduce device rounding.
// ---------------------------------------------------------------------------
BONSAI_HD inline float f16_to_f32(uint16_t h) {
#ifdef __CUDACC__
    return __half2float(__ushort_as_half(h));
#else
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant =  h & 0x3FFu;
    uint32_t out;
    if (exp == 0) {
        if (mant == 0) { out = sign; }
        else {                       // subnormal
            exp = 127 - 15 + 1;
            while ((mant & 0x400u) == 0) { mant <<= 1; exp--; }
            mant &= 0x3FFu;
            out = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1Fu) {       // inf / nan
        out = sign | 0x7F800000u | (mant << 13);
    } else {
        out = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float f; std::memcpy(&f, &out, sizeof(f)); return f;
#endif
}

BONSAI_HD inline uint16_t f32_to_f16(float f) {
#ifdef __CUDACC__
    return __half_as_ushort(__float2half(f));
#else
    uint32_t x; std::memcpy(&x, &f, sizeof(x));
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t  exp  = (int32_t)((x >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;
    if (((x >> 23) & 0xFFu) == 0xFFu) {                 // inf / nan
        return (uint16_t)(sign | 0x7C00u | (mant ? 0x200u : 0u));
    }
    if (exp >= 0x1F) { return (uint16_t)(sign | 0x7C00u); }        // overflow
    if (exp <= 0) {                                     // subnormal / zero
        if (exp < -10) { return (uint16_t)sign; }
        mant |= 0x800000u;
        const uint32_t shift = (uint32_t)(14 - exp);
        uint32_t m = mant >> shift;
        if (mant & (1u << (shift - 1))) { m += 1; }     // round half up
        return (uint16_t)(sign | m);
    }
    uint32_t m = mant >> 13;
    if (mant & 0x1000u) {                               // round half up
        m += 1;
        if (m == 0x400u) { m = 0; exp += 1; if (exp >= 0x1F) { return (uint16_t)(sign | 0x7C00u); } }
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | m);
#endif
}

// A "half" stored as raw bits, used to lay out the GGUF blocks on the host.
struct hbits { uint16_t v; };

BONSAI_HD inline int dp4a(int a, int b, int c) {
#ifdef __CUDACC__
#if __CUDA_ARCH__ >= 610
    return __dp4a(a, b, c);
#else
    int r = c;
    for (int i = 0; i < 4; ++i) {
        const int ai = (a >> (8 * i)) & 0xFF; const int bi = (b >> (8 * i)) & 0xFF;
        r += (int8_t)ai * (int8_t)bi;
    }
    return r;
#endif
#else
    // Host emulation of __dp4a: four signed byte products.
    int r = c;
    for (int i = 0; i < 4; ++i) {
        const int8_t ai = (int8_t)((a >> (8 * i)) & 0xFF);
        const int8_t bi = (int8_t)((b >> (8 * i)) & 0xFF);
        r += (int)ai * (int)bi;
    }
    return r;
#endif
}

// Sum of the four signed bytes of a packed int (used to fold the PQ2_0 +1 bias).
BONSAI_HD inline int dp4a_sum(int a) {
    return dp4a(a, 0x01010101, 0);
}

} // namespace bonsai
