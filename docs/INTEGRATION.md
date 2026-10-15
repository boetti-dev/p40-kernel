# Einbindung in prismml/llamacpp

Der Kernel-Stack ist so geschnitten, dass er den PQ2_0-Pfad im PrismML-Fork
(`PrismML-Eng/llama.cpp`, Branch `prism`) ersetzt bzw. daneben gebaut werden
kann.

## Variante A — als eigenständige Static Library

```bash
./tools/build_p40.sh                 # erzeugt build/libbonsai_p40.a
```

Linken gegen `build/libbonsai_p40.a`, Header `include/bonsai_p40.h` einbinden
und die Kernel direkt aufrufen. Beispiel für eine Zeile der Backbone
(`n_in = 5120`, `n_out = 17408`, `xq` = PQ2_0-natives Q8_1):

```c
#include "bonsai_p40.h"

// 1. Aktivierungen in das PQ2_0-native Q8_1-Layout quantisieren
bonsai_p40_quant_activations(x, xq, 5120, stream);

// 2. Decode-Mat-Vec:  y = W . x
bonsai_p40_gemv(W, xq, y, /*n_out=*/17408, /*n_in=*/5120, stream);

// 3. Rotation vor den ternären Projektionen
bonsai_p40_fwht(x, signs, 5120, stream);

// 4. Norm / Attention / GDN
bonsai_p40_rmsnorm(x, w, y, /*rows=*/1, /*n=*/5120, 1e-6f, stream);
bonsai_p40_gdn_step(state, q, k, v, g, beta, out, /*k_heads=*/16, /*v_heads=*/48, 128, 128, stream);
bonsai_p40_attn(q, k, v, out, /*n_tok=*/n, 24, 4, 256, 1.f/16.f, stream);
```

## Variante B — als Backend-Dateien im Fork

1. `src/pq2_0.h`, `src/p40_common.h`, `src/p40_ptx.h`, `src/fwht.h`, `src/ops.h`,
   `src/kv_cache.h` nach `ggml/src/ggml-cuda/bonsai-p40/` kopieren.
2. `src/kernels/gemv_p40.cu` → ersetzt für `sm_61` den Aufruf von
   `mul_mat_vec_pq2_0_mc(...)` in `mmvq.cu` (Decode, Batch 1).
3. `src/kernels/attn_p40.cu` → GDN-Pfad (`gated_delta_net.cu`) und
   Full-Attention für `sm_61`.
4. `src/kernels/kv_p40.cu` → KV-Cache-Streams.
5. In `ggml-cuda/CMakeLists.txt` die `.cu`-Dateien ergänzen.

### Wichtig: `gemm_p40.cu` NICHT einhängen — der Fork hat MMQ für Pascal

**Don't replace the fork's MMQ.** Der Fork hat bereits einen Pascal-spezifischen
PQ2_0-MMQ-Pfad:

```
ggml/src/ggml-cuda/mmq-config-pascal.cuh:
  CASE(GGML_TYPE_PQ2_0, 256, 2, 64,  8, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, ...)
  ... I=64, J in {8,16,24,32,40,48,64}, nthreads=256, occupancy=2
```

`ggml_cuda_mmq_get_config()` fällt für `__CUDA_ARCH__ < VOLTA` genau darauf
zurück, `ggml_cuda_should_use_mmq()` aktiviert es ab `GGML_CUDA_CC_DP4A` über
den DP4A-Pfad. Dieser Pfad ist Pascal-getunt und dem mitgelieferten
`gemm_p40.cu` (256×16-Kachel, eine Zeile pro Lane) **vorzuziehen**.

`src/kernels/gemm_p40.cu` ist deshalb als **Standalone-Fallback** gedacht: nur
verwenden, wenn ohne den Fork gebaut wird. Siehe `docs/MMQ.md`.

Da der Stack ausschließlich `sm_61` baut, sollte er über
`__CUDA_ARCH__ == 610` gegated werden, damit die anderen Architekturen des Forks
unberührt bleiben.

## Loader-Constraints

Der Kernel geht davon aus, dass **genau** diese Datei geladen wird. Der Loader
sollte hart ablehnen, wenn:

* `general.architecture != "qwen35"`
* ein großes Tensor nicht `GGML_TYPE_PQ2_0` (142) ist
* die Geometrie von `include/bonsai_model.h` abweicht
  (block_count 64, embd 5120, ff 17408, heads 24/4, head_dim 256,
   ssm state 128 / groups 16 / inner 6144, hadamard block 1024)

`tools/check_gguf.py` prüft genau diese Bedingungen und kann als
Pre-Flight-Check in den Loader eingebaut werden.

## Modell herunterladen

```bash
huggingface-cli download prism-ml/Ternary-Bonsai-2-27B-gguf \
    Ternary-Bonsai-2-27B-PQ2_0.gguf --local-dir models/
python3 tools/check_gguf.py models/Ternary-Bonsai-2-27B-PQ2_0.gguf
```

Die Datei passt vollständig in die 24 GB der P40 (7.21 GB Gewichte + KV-Cache),
also ist ein voller GPU-Offload ohne Auslagerung möglich.

## Generation-Parameter

Wie im Model-Card empfohlen (denkt vor dem Antworten, Reasoning zählt gegen das
Ausgabelimit):

```
-c 65536 -n 16384 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.05
```

`reasoning_effort: "medium"` verkürzt das Reasoning; `"high"` ist im
Chat-Template nicht gültig (dort `xhigh`).
