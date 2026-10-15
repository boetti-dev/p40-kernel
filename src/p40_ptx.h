// p40_ptx.h -- the few places where hand-written PTX beats what nvcc emits for
// the P40, plus the reasoning for what is deliberately NOT done in PTX.
//
// WHY THIS FILE IS SMALL
// ----------------------
// The decode kernel is bandwidth-bound: 7.2 GB per token / 346 GB/s = 20.8 ms,
// while the entire token costs only ~0.45 G warp-instructions, i.e. ~1-2 ms of
// issue on 30 SMs x 4 schedulers.  Instruction count therefore does not set the
// tok/s number, so replacing arithmetic with PTX buys nothing.  What DOES
// matter is the memory path (cache policy, load width, request coalescing) and
// occupancy (register pressure -> warps in flight -> latency hiding).
//
// So only three things are hand-written:
//   1. read-only loads (ld.global.nc) for the weights and activations, which
//      are strictly read-only for the kernel's lifetime;
//   2. an explicit dp4a.u32.s32, because the weight operand is an *unsigned*
//      byte (0..3) while the activation operand is signed -- nvcc has to guess,
//      and stating it removes any sign-extension ambiguity;
//   3. an L2 prefetch to keep the weight stream ahead of the consumers.
//
// Everything else (cvt.f32.f16 for the scales, shfl.sync reductions, the
// dp4a-based code extraction) is already emitted optimally by nvcc, and adding
// PTX for it would only add risk.
#pragma once

#include "p40_common.h"

namespace bonsai {

// --- 1. read-only 16-bit load through the non-coherent (texture) path -------
// Used for the PQ2_0 code words and the fp16 scales.  The pointers are 2-byte
// aligned by construction (block_pq2_0 is a 34-byte struct), so we must never
// widen this to a 32-bit load.
BONSAI_HD inline uint16_t p40_ldg_u16(const uint16_t * p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 320
    uint16_t v;
    asm("ld.global.nc.u16 %0, [%1];" : "=h"(v) : "l"(p));
    return v;
#else
    return *p;
#endif
}

// Two adjacent uint16 (one 16-code word) as a uint32.
BONSAI_HD inline uint32_t p40_ldg_u16x2(const uint16_t * p) {
    return (uint32_t)p40_ldg_u16(p) | ((uint32_t)p40_ldg_u16(p + 1) << 16);
}

// --- 2. explicit mixed-signedness dp4a -------------------------------------
// `w` holds four unsigned 2-bit codes (0..3); `a` holds four signed int8
// activations.  dp4a.u32.s32 states exactly that.
BONSAI_HD inline int p40_dp4a_us(uint32_t w, int a, int c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
    int r;
    asm("dp4a.u32.s32 %0, %1, %2, %3;" : "=r"(r) : "r"(w), "r"(a), "r"(c));
    return r;
#else
    return dp4a((int)w, a, c);   // host reference / pre-sm61 fallback
#endif
}

// --- 3. L2 prefetch for the streaming weight reads -------------------------
BONSAI_HD inline void p40_prefetch_l2(const void * p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200
    asm volatile("prefetch.global.L2 [%0];" :: "l"(p));
#else
    (void)p;
#endif
}

} // namespace bonsai