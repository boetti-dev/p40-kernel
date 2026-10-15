#!/usr/bin/env python3
"""check_gguf.py -- verify that the geometry hardcoded in include/bonsai_model.h
and src/pq2_0.h matches the shipped Ternary-Bonsai-2-27B GGUF file.

This is the bridge between the kernel constants and the real artefact: it reads
the GGUF header of a local file (or the first bytes of a streamed download) and
asserts every number the P40 kernel relies on.

Usage:
    python3 tools/check_gguf.py /path/to/Ternary-Bonsai-2-27B-PQ2_0.gguf
    python3 tools/check_gguf.py --header /path/to/head.bin    # partial download
"""
import struct
import sys
import re

GGML_TYPE_PQ2_0 = 142
GGML_TYPE_F16 = 1
GGML_TYPE_F32 = 0
GGML_TYPE_BF16 = 30

# ---- expected geometry (must mirror include/bonsai_model.h) ----------------
EXPECT_KV = {
    "general.architecture": "qwen35",
    "qwen35.block_count": 64,
    "qwen35.embedding_length": 5120,
    "qwen35.feed_forward_length": 17408,
    "qwen35.attention.head_count": 24,
    "qwen35.attention.head_count_kv": 4,
    "qwen35.attention.key_length": 256,
    "qwen35.attention.value_length": 256,
    "qwen35.full_attention_interval": 4,
    "qwen35.ssm.conv_kernel": 4,
    "qwen35.ssm.state_size": 128,
    "qwen35.ssm.group_count": 16,
    "qwen35.ssm.time_step_rank": 48,
    "qwen35.ssm.inner_size": 6144,
    "qwen35.rope.dimension_count": 64,
    "prism.hadamard.block_size": 1024,
    "prism.hadamard.transform": "normalized-sylvester-walsh-hadamard",
    "prism.hadamard.axis": "input-last-dimension",
}

# tensor name -> (n_in, n_out, ggml_type)  (None = don't care)
EXPECT_TENSORS = {
    "output.weight": (5120, 248320, GGML_TYPE_PQ2_0),
    "token_embd.weight": (5120, 248320, GGML_TYPE_PQ2_0),
    "blk.X.attn_norm.weight": (5120, None, GGML_TYPE_F32),
    "blk.X.post_attention_norm.weight": (5120, None, GGML_TYPE_F32),
    "blk.X.ffn_down.weight": (17408, 5120, GGML_TYPE_PQ2_0),
    "blk.X.ffn_gate.weight": (5120, 17408, GGML_TYPE_PQ2_0),
    "blk.X.ffn_up.weight": (5120, 17408, GGML_TYPE_PQ2_0),
    # linear attention (48 layers)
    "blk.X.attn_qkv.weight": (5120, 10240, GGML_TYPE_PQ2_0),
    "blk.X.attn_gate.weight": (5120, 6144, GGML_TYPE_PQ2_0),
    "blk.X.ssm_out.weight": (6144, 5120, GGML_TYPE_PQ2_0),
    # full attention (16 layers)
    "blk.X.attn_q.weight": (5120, 12288, GGML_TYPE_PQ2_0),
    "blk.X.attn_k.weight": (5120, 1024, GGML_TYPE_PQ2_0),
    "blk.X.attn_v.weight": (5120, 1024, GGML_TYPE_PQ2_0),
    "blk.X.attn_output.weight": (6144, 5120, GGML_TYPE_PQ2_0),
}

TENSOR_COUNT = {"blk.X.attn_qkv.weight": 48, "blk.X.attn_q.weight": 16,
                "blk.X.attn_k.weight": 16, "blk.X.attn_v.weight": 16,
                "blk.X.attn_output.weight": 16, "blk.X.attn_gate.weight": 48,
                "blk.X.ssm_out.weight": 48, "blk.X.attn_norm.weight": 64}


class Reader:
    def __init__(self, buf):
        self.b = buf
        self.o = 0

    def u32(self):
        v = struct.unpack_from("<I", self.b, self.o)[0]; self.o += 4; return v

    def u64(self):
        v = struct.unpack_from("<Q", self.b, self.o)[0]; self.o += 8; return v

    def s(self):
        n = self.u64()
        v = self.b[self.o:self.o + n].decode("utf-8", "replace"); self.o += n; return v

    def skip(self, t):
        if t in (0, 1, 7): self.o += 1
        elif t in (2, 3): self.o += 2
        elif t in (4, 5, 6): self.o += 4
        elif t in (10, 11, 12): self.o += 8
        elif t == 8: self.s()
        elif t == 9:
            at = self.u32(); n = self.u64()
            for _ in range(n): self.skip(at)
        else: raise ValueError(f"unknown kv type {t}")


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__); return 2
    path = args[-1]
    with open(path, "rb") as f:
        buf = f.read()

    r = Reader(buf)
    magic = buf[0:4]; r.o = 4
    ver = r.u32(); n_tensors = r.u64(); n_kv = r.u64()
    fails = []
    if magic != b"GGUF":
        fails.append(f"bad magic {magic!r}")
    print(f"GGUF v{ver}  tensors={n_tensors}  kv={n_kv}")

    kv = {}
    for _ in range(n_kv):
        k = r.s(); t = r.u32()
        if k in EXPECT_KV:
            if t == 8: kv[k] = r.s()
            elif t == 4: kv[k] = struct.unpack_from("<I", buf, r.o)[0]; r.o += 4
            elif t == 5: kv[k] = struct.unpack_from("<i", buf, r.o)[0]; r.o += 4
            else: r.skip(t)
        else:
            r.skip(t)

    for k, want in EXPECT_KV.items():
        got = kv.get(k)
        ok = (got == want)
        print(f"  {'OK ' if ok else 'BAD'} {k} = {got}")
        if not ok:
            fails.append(f"{k}: got {got}, want {want}")

    shapes = {}
    mtp_tensors = []
    try:
        for _ in range(n_tensors):
            name = r.s(); nd = r.u32()
            dims = [r.u64() for _ in range(nd)]
            tt = r.u32(); r.u64()
            if re.search(r"nextn|mtp|draft|eagle", name, re.I):
                mtp_tensors.append(name)
            g = re.sub(r"blk\.\d+\.", "blk.X.", name)
            shapes.setdefault((g, tuple(dims), tt), 0)
            shapes[(g, tuple(dims), tt)] += 1
    except (struct.error, ValueError) as e:
        print(f"  (partial header, tensor list truncated: {e})")

    print("\ntensor type/geometry checks:")
    for (g, dims, tt), cnt in sorted(shapes.items()):
        exp = EXPECT_TENSORS.get(g)
        if exp is None:
            continue
        n_in, n_out, etype = exp
        ok = (dims[0] == n_in) and (n_out is None or dims[1] == n_out) and (tt == etype)
        want_count = TENSOR_COUNT.get(g)
        ok = ok and (want_count is None or cnt == want_count)
        print(f"  {'OK ' if ok else 'BAD'} {g:32s} dims={dims} type={tt} x{cnt}")
        if not ok:
            fails.append(f"{g}: dims={dims} type={tt} x{cnt}, want ({n_in},{n_out}) type={etype} x{want_count}")

    print()
    if mtp_tensors:
        print(f"MTP/nextn block detected: {len(mtp_tensors)} tensors")
        for n in mtp_tensors[:6]:
            print(f"  e.g. {n}")
        fails.append(
            "this file carries an MTP (nextn) block; the Bonsai-P40 kernel has no "
            "MTP support (no nextn layer, no small-batch verify path, no second "
            "cache stream) -- see docs/MTP.md. The trunk weights are still valid; "
            "a context that would use MTP cannot be built with this engine.")
    else:
        print("MTP/nextn block: none (trunk-only file, matches the engine)")

    print()
    if fails:
        print(f"FAILED ({len(fails)}):")
        for f in fails:
            print("  -", f)
        return 1
    print("All GGUF geometry checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
