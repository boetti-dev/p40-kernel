#!/usr/bin/env python3
"""gpu_predict.py -- decode/prefill prediction for the Bonsai-27B engine on a
given GPU, from first principles (bytes on the wire / achievable bandwidth).

The model is deliberately simple, because the decode is bandwidth-bound:
    t_token = (weight_bytes + kv_bytes_per_token * context) / (bw * efficiency)
and everything else (dp4a/issue time, Hadamard, GDN state, attention) is
verified to be a small correction, not the driver.

It reproduces the P40 numbers in docs/P40_OPTIMIZATION.md and
docs/KV_CACHE.md, which is the sanity check that makes the V100 prediction
in docs/V100_PREDICTION.md comparable.

Usage:
    python3 tools/gpu_predict.py                 # P40 vs V100-32B table
    python3 tools/gpu_predict.py --bw 900 --sm 80 --fp32-per-sm 64 --name V100-32B
"""
import argparse

# --- model constants (Ternary-Bonsai-2-27B, PQ2_0) -------------------------
WEIGHTS_PQ2_0 = 7.21e9      # bytes (2.13 bit/weight)
WEIGHTS_PTQ1_0 = 5.95e9     # bytes (1.75 bit/weight), smaller/faster on BW-bound
KV_LAYERS, KV_HEADS, HEAD_DIM = 16, 4, 256
KV_ELEMS_PER_TOKEN = KV_LAYERS * KV_HEADS * HEAD_DIM      # 16384
GDN_STATE = 144 * 1024 * 1024                             # constant
CONTEXT = 262144

KV_BYTES = {
    "f16":   2.0,
    "q8_0":  34.0 / 32.0,      # 34 B per 32 elements
    "q4_0":  18.0 / 32.0,
}


def kv_per_token(kind_k, kind_v):
    return KV_ELEMS_PER_TOKEN * (KV_BYTES[kind_k] + KV_BYTES[kind_v])


def decode_ms(bw, eff, ctx, kv_k="q8_0", kv_v="q8_0", weights=WEIGHTS_PQ2_0):
    n = weights + kv_per_token(kv_k, kv_v) * ctx
    return n / (bw * eff) * 1e3


# --- card specs -----------------------------------------------------------
# `eff` is the FRACTION OF PEAK BANDWIDTH the decode actually reaches.  It is
# NOT a constant: measured on the shipped model card plus a real P40 run it
# ranges from 24% (H100, instruction/launch bound) to 71% (L4, memory bound).
# Using a blanket 0.88 was the original error in this model -- it made every
# prediction ~2x too optimistic.
CARDS = {
    # name          bw GB/s   SMs  FP32/SM  clock     L2 B      VRAM GiB  eff
    "P40":       dict(bw=346e9,  sms=30, fp32=128, clk=1.53e9, l2=3 << 20, vram=24,  eff=0.455),
    "V100-32B":  dict(bw=900e9,  sms=80, fp32=64,  clk=1.53e9, l2=6 << 20, vram=32,  eff=0.50),
}

# Measured reference points (tg128, PQ2_0, from the model card + a P40 run).
# Used by --calibrate to sanity-check the model.
MEASURED = [
    # name,        tok/s,  peak GB/s
    ("P40",          21.90, 346),     # this project, real Tesla P40, stock path
    ("P40 (bonsai)", 11.39, 346),     # this project, real Tesla P40, our kernel
    ("L4",           29.80, 300),
    ("RTX 4090",     81.20, 1008),
    ("RTX 5090",    129.90, 1792),
    ("H100 SXM",    113.90, 3350),
    ("A100 SXM",     73.90, 2039),
    ("L40S",         74.40, 864),
]


def measured_efficiencies():
    print("measured decode efficiency (tg128, PQ2_0, 6.70 GiB model):")
    for name, tps, bw in MEASURED:
        gbs = WEIGHTS_PQ2_0 * tps
        print(f"  {name:14s} {tps:6.1f} tok/s -> {gbs/1e9:7.1f} GB/s "
              f"= {100.0*gbs/(bw*1e9):5.1f}% of {bw} GB/s")
    print()


def describe(name, eff=None):
    c = CARDS[name]
    if eff is None:
        eff = c["eff"]
    print(f"=== {name}  ({c['bw']/1e9:.0f} GB/s, {c['sms']} SMs, "
          f"{c['vram']} GiB, L2 {c['l2']>>20} MiB, eff {eff:.0%}) ===")
    print(f"  FP32 TFLOP/s : {2*c['sms']*c['fp32']*c['clk']/1e12:5.1f}")
    print(f"  issue warp/cyc: {c['sms']*4*c['clk']/1e9:6.1f} e9/s")

    w = WEIGHTS_PQ2_0 / (c["bw"]) * 1e3
    print(f"  weight stream (PQ2_0) at peak: {w:5.1f} ms -> {1000/w:6.1f} tok/s ceiling")
    w4 = WEIGHTS_PTQ1_0 / (c["bw"]) * 1e3
    print(f"  weight stream (PTQ1_0) at peak: {w4:5.1f} ms -> {1000/w4:6.1f} tok/s ceiling")
    wr = decode_ms(c["bw"], eff, 0)
    print(f"  realistic ({eff:.0%} BW): {wr:5.1f} ms -> {1000/wr:6.1f} tok/s")

    print("  context scaling (Q8_0 KV, realistic) :")
    for ctx in (4096, 32768, 131072, 262144):
        ms = decode_ms(c["bw"], eff, ctx)
        gb = (WEIGHTS_PQ2_0 + kv_per_token("q8_0", "q8_0") * ctx) / 1e9
        print(f"    ctx {ctx:6d}: {gb:5.2f} GB moved, {ms:6.2f} ms -> {1000/ms:5.1f} tok/s")
        assert gb < c["vram"] * 1.073741824, "does not fit VRAM"
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bw", type=float, help="GB/s")
    ap.add_argument("--sm", type=int)
    ap.add_argument("--fp32-per-sm", type=int, default=64)
    ap.add_argument("--clk", type=float, default=1.53, help="GHz")
    ap.add_argument("--vram", type=float, default=32)
    ap.add_argument("--l2", type=float, default=6)
    ap.add_argument("--name", default="custom")
    ap.add_argument("--eff", type=float, default=None,
                    help="fraction of peak bandwidth reached (default: per-card, "
                         "calibrated against real measurements; 0.24..0.71 in practice)")
    a = ap.parse_args()

    if a.bw:
        CARDS[a.name] = dict(bw=a.bw * 1e9, sms=a.sm, fp32=a.fp32_per_sm,
                             clk=a.clk * 1e9, l2=int(a.l2 * (1 << 20)), vram=a.vram,
                             eff=a.eff if a.eff else 0.50)

    measured_efficiencies()

    for n in ("P40", "V100-32B", a.name):
        if n in CARDS:
            describe(n, a.eff)

    # KV bytes/token for the doc table
    print("KV bytes/token:")
    for k in ("f16", "q8_0", "q4_0"):
        for v in ("f16", "q8_0", "q4_0"):
            print(f"  {k:5s}/{v:5s} = {kv_per_token(k, v):8.0f} B/token "
                  f"({kv_per_token(k, v)*CONTEXT/2**30:5.2f} GiB @262k)")


if __name__ == "__main__":
    main()