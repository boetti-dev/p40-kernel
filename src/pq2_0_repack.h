// pq2_0_repack.h -- aligned repacked layout for the PQ2_0 weights, plus the
// numeric core that both the device kernel and the host test use.
//
// WHY
// ---
// Measured on a P40 with tools/bwtest.cu (docs/NEXT_STEPS.md):
//
//   aligned uint4 stream           251.8 GB/s
//   PQ2_0 addressing, pure read    183.6 GB/s   <- the 34-byte stride costs 37%
//   stock decode (achieves)        157.5 GB/s
//
// The 34-byte PQ2_0 block is only 2-byte aligned, so the decode kernel cannot
// use wide, coalesced loads.  This header defines a repacked layout that is
// 8-byte aligned and contiguous across a warp, at EXACTLY the same total size.
//
// LAYOUT OF ONE ROW  (nblocks = n_in / 128)
// ----------------------------------------
//   offset 0                    : d[nblocks]      fp16 scales, 16-byte aligned
//   offset 2*nblocks            : codes[32*nblocks]
//
//   row_stride = 34 * nblocks        <-- identical to the original 34-byte
//                                        blocks, so no extra memory at all
//
// The code bytes are stored as 4 planes of nblocks*8 bytes, plane `p` holding
// chunk p of every block.  Flattened, that is a single contiguous run of
// nblocks*4 units of 8 bytes, and unit `u` is
//
//   plane p = u / nblocks        block b = u % nblocks
//
// so the address of a unit is simply  codes + u*8  -- contiguous, 8-byte
// aligned, and one warp reads 32 units = 256 contiguous bytes per instruction.
// The bit order inside a unit is the original PQ2_0 order, so the existing
// dp4a code-word extraction (w >> 2k) & 0x03030303 still applies unchanged.
//
// Alignment holds for this model: every row width has nblocks a multiple of 8
// (5120/128=40, 17408/128=136, 6144/128=48, 10240/128=80, 12288/128=96,
// 1024/128=8), so 2*nblocks is 16-byte aligned and 32*nblocks keeps the 16-byte
// alignment of the code planes.
#pragma once

#include "pq2_0.h"

namespace bonsai {

// --- geometry --------------------------------------------------------------
inline int  repack_row_stride(int nblocks)   { return 34 * nblocks; }
inline int  repack_d_offset(int)             { return 0; }
inline int  repack_codes_offset(int nblocks) { return 2 * nblocks; }
inline int  repack_n_units(int nblocks)      { return 4 * nblocks; }

BONSAI_HD inline int repack_plane_of_unit(int u, int nblocks) { return u / nblocks; }
BONSAI_HD inline int repack_block_of_unit(int u, int nblocks) { return u % nblocks; }

// activation chunk index for a unit: block b contributes chunks b*4 + 0..3
BONSAI_HD inline int repack_act_chunk(int u, int nblocks) {
    return repack_block_of_unit(u, nblocks) * 4 + repack_plane_of_unit(u, nblocks);
}

// --- repack / unpack (host) ------------------------------------------------
// Requires nblocks % 8 == 0 so that both offsets stay 16-byte aligned.
inline bool repack_requires(int nblocks) { return nblocks > 0 && (nblocks % 8) == 0; }

inline void repack_row_pq2_0(const block_pq2_0 * src, int nblocks, uint8_t * dst) {
    for (int b = 0; b < nblocks; ++b) {
        std::memcpy(dst + repack_d_offset(nblocks) + 2 * b, &src[b].d, 2);
        for (int p = 0; p < 4; ++p) {
            // unit u = p*nblocks + b holds chunk p of block b: 8 bytes
            std::memcpy(dst + repack_codes_offset(nblocks) + (p * nblocks + b) * 8,
                        src[b].qs + p * 8, 8);
        }
    }
}

inline void unpack_row_pq2_0(const uint8_t * src, int nblocks, block_pq2_0 * dst) {
    for (int b = 0; b < nblocks; ++b) {
        std::memcpy(&dst[b].d, src + repack_d_offset(nblocks) + 2 * b, 2);
        for (int p = 0; p < 4; ++p) {
            std::memcpy(dst[b].qs + p * 8,
                        src + repack_codes_offset(nblocks) + (p * nblocks + b) * 8, 8);
        }
    }
}

// --- the numeric core: dot of one repacked unit against one activation chunk -
// A unit is 8 code bytes (32 elements) and is 8-byte aligned, so unlike the
// original layout it CAN be read as two aligned uint32 words.
BONSAI_HD inline uint32_t repack_load_u32(const uint8_t * p) {
#if defined(__CUDA_ARCH__)
    return *(const uint32_t *)p;          // 8-byte aligned unit: safe
#else
    uint32_t v; std::memcpy(&v, p, 4); return v;
#endif
}

// Integer contribution of one unit (before the scale), matching the reference
// arithmetic exactly:  sum((code-1) * a) = sum(code*a) - sum(a).
BONSAI_HD inline int repack_unit_sumi(const uint8_t * codes, const block_q8_1 * act,
                                      int nblocks, int u) {
    const uint8_t * q = codes + repack_codes_offset(nblocks) + (size_t)u * 8;
    const block_q8_1 * a = act + repack_act_chunk(u, nblocks);

    const uint32_t w0 = repack_load_u32(q);
    const uint32_t w1 = repack_load_u32(q + 4);

    const int * ap = (const int *)a->qs;      // 32 activations, 4-byte aligned
    const int a0 = ap[0], a1 = ap[1], a2 = ap[2], a3 = ap[3];
    const int a4 = ap[4], a5 = ap[5], a6 = ap[6], a7 = ap[7];

    int s = -q8_1_sum(a);
    s = dp4a((int)((w0 >> 0) & 0x03030303u), a0, s);
    s = dp4a((int)((w0 >> 2) & 0x03030303u), a1, s);
    s = dp4a((int)((w0 >> 4) & 0x03030303u), a2, s);
    s = dp4a((int)((w0 >> 6) & 0x03030303u), a3, s);
    s = dp4a((int)((w1 >> 0) & 0x03030303u), a4, s);
    s = dp4a((int)((w1 >> 2) & 0x03030303u), a5, s);
    s = dp4a((int)((w1 >> 4) & 0x03030303u), a6, s);
    s = dp4a((int)((w1 >> 6) & 0x03030303u), a7, s);
    return s;
}

// Float contribution of one unit: scale of its block x activation scale x sumi.
BONSAI_HD inline float repack_unit_dot(const uint8_t * row, const block_q8_1 * act,
                                       int nblocks, int u) {
    const uint16_t d = *(const uint16_t *)(row + repack_d_offset(nblocks) +
                                          2 * repack_block_of_unit(u, nblocks));
    const float dw = f16_to_f32(d);
    const float da = f16_to_f32(act[repack_act_chunk(u, nblocks)].d);
    return dw * da * (float)repack_unit_sumi(row, act, nblocks, u);
}

// Whole-row dot through the repacked layout.  This is the definition the device
// kernel must reproduce; the host test compares it against the original-layout
// reference (pq2_0_row_dot_perm) and requires an exact match.
inline float repack_row_dot(const uint8_t * row, const block_q8_1 * act, int nblocks) {
    float acc = 0.0f;
    for (int u = 0; u < repack_n_units(nblocks); ++u) acc += repack_unit_dot(row, act, nblocks, u);
    return acc;
}

} // namespace bonsai
