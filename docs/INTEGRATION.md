# Bonsai-P40 — Zusammenfassung und Einbindung in prismml/llamacpp

Diese Datei ist der Einstiegspunkt: **Teil A** fasst zusammen, was gebaut und
verifiziert wurde, **Teil B** was der PrismML-Fork bereits hat (und was man
deshalb *nicht* ersetzen darf), **Teil C** die konkrete Lücke, die dieses Repo
füllt, **Teil D** die Einbindung Schritt für Schritt, **Teil E** die
Verifikation, **Teil F** die Grenzen.

---

## Teil A — Was gebaut wurde

Ziel: **ausschließlich** `prism-ml/Ternary-Bonsai-2-27B-gguf` im PQ2_0-Format
(`GGML_TYPE_PQ2_0 = 142`, das „q2_0" dieser Datei) auf einer **Tesla P40**
(`sm_61`, GP102) rechnen — kein Codepfad für ein anderes Modell, eine andere
Quantisierung oder eine andere GPU.

| Datei | Rolle | verifiziert |
|---|---|---|
| `include/bonsai_model.h` | exakte qwen35-Geometrie aus dem GGUF-Header | GGUF-Check |
| `include/bonsai_p40.h` | C-API der Kernel | – |
| `include/bonsai_kv.h` | C-API des KV-Caches | – |
| `src/p40_common.h` | P40-Konstanten, `dp4a`, fp16↔fp32 (host+device) | Host-Tests |
| `src/pq2_0.h` | PQ2_0-Blockformat + DP4A-Kern | Host-Tests |
| `src/p40_ptx.h` | `ld.global.nc`, `dp4a.u32.s32`, `prefetch.global.L2` | Host-Fallback |
| `src/fwht.h` | Hadamard (Block 1024, normalisiert) | vs. naive Sylvester |
| `src/ops.h` | RMSNorm, RoPE, SwiGLU, GDN-Schritt | vs. Direktformel |
| `src/kv_cache.h` | KV-Geometrie, Sizing, Q8_0-Mathe | Sizing-Tabelle |
| `src/kernels/gemv_p40.cu` | **Decode-Mat-Vec** (bandbreitenkritisch) | Loop-Simulation |
| `src/kernels/gemm_p40.cu` | Prefill-DP4A-Tile-GEMM (**Fallback**, s. `docs/MMQ.md`) | Kachel-Simulation |
| `src/kernels/elementwise_p40.cu` | Hadamard + RMSNorm | – |
| `src/kernels/attn_p40.cu` | GDN + GQA-Flash-Attention | vs. naive Softmax |
| `src/kernels/kv_p40.cu` | KV-Append + Split-K-Decode | Split-K vs. Single-Pass |
| `tests/test_p40.cpp` | Host-Suite | **333/333** |
| `tools/check_gguf.py` | Pre-Flight-Check gegen die echte Datei | 18 KV + 14 Tensoren |
| `tools/build_p40.sh` | `sm_61`-Build + Tests | Syntax |
| `tools/disasm.sh` | PTX/SASS-Inspektion | Syntax |
| `tools/gpu_predict.py` | Vorhersagemodell für beliebige Karten | reproduziert P40 |
| `docs/P40_OPTIMIZATION.md` | Tuning-Begründung + Kostenmodell | – |
| `docs/KV_CACHE.md` | KV-Cache-Plan | Sizing geprüft |
| `docs/V100_PREDICTION.md` | V100-32GB-Vorhersage | – |
| `docs/MTP.md` | Kann das Modell MTP, kann der Kernel es? | Header-Parse |
| `docs/MMQ.md` | MMQ auf Pascal — warum nicht drin | Fork-Grep |

### Zahlen (Modell aus Bytes × Bandbreite, host-seitig gegen die Tabellen geprüft)

| | P40 | V100 32 GB |
|---|---|---|
| Decode, kurzer Kontext | 42.2 tok/s | 109.8 tok/s |
| Decode, 262K (Q8_0-KV) | 18.6 tok/s | 48.5 tok/s |
| Prefill Batch 512 | 1.3–1.8 s | 0.28–0.50 s |
| VRAM-Bedarf @262K | 15.9 GiB / 24 | 15.9 GiB / 32 |
| mit MTP (in-file, nmax=2) | 56.5 tok/s | 146.9 tok/s |

### Drei echte Bugs, die die Verifikation gefunden hat

1. **Aktivierungs-Permutation falsch** — die 4×4-Transposition muss *innerhalb
   jedes 16er-Halbs* liegen, nicht über den ganzen 32er-Block.
2. **Q8_1-`s`-Feld** muss den **rohen int16**-Summenwert tragen (Bits
   reinterpretiert), nicht einen fp16-Zahlenwert.
3. **Aktivierungs-Quantisierer** fehlte der ×127-Faktor (`d = amax/127`).

Dazu zwei Alignment-Bugs (`int4`-Load aus dem nur 4-Byte-aligned `qs`).

---

## Teil B — Was der Fork bereits hat (NICHT ersetzen)

Verifiziert im `prism`-Branch von `PrismML-Eng/llama.cpp`:

| Baustein | Datei | Konsequenz |
|---|---|---|
| **PQ2_0-MMQ für Pascal** | `mmq-config-pascal.cuh` (I=64, J∈{8..64}, `..._SRAM_LAYOUT_Q8_0`, `MMQ_ITER_K`), Dispatch in `mmq.cuh` für `__CUDA_ARCH__ < VOLTA` | **`gemm_p40.cu` nicht einhängen** |
| PQ2_0-MMVQ für 3–8 Spalten | `mmvq-pq2_0.cuh` (Ada-getunt) | deckt den Verify-Batch ab |
| Hadamard | `fwht.cuh` | `fwht.h` nur als Referenz |
| Gated DeltaNet | `gated_delta_net.cu/.cuh` | `attn_p40.cu` nur als Referenz |
| KV-Cache | `src/llama-kv-cache.cpp`, `-ctk/-ctv` | `kv_p40.cu` nur als Alternative |
| MTP | `--spec-type draft-mtp` (PR #205) | siehe `docs/MTP.md` |
| PQ2_0-Vektor-Dot | `vecdotq.cuh: vec_dot_pq2_0_q8_1`, `VDR_PQ2_0_Q8_1_MMVQ = 1` | generischer Decode-Pfad |

**`--spec-type ngram-*` ist für dieses Modell wirkungslos** (offen, #203).

## Teil C — Die Lücke, die dieses Repo füllt

Der entscheidende Fund, verifiziert in `mmvq.cu`:

```c
static constexpr __host__ __device__ int get_mmvq_parameters_table(int cc) {
    ...
    if (NVIDIA && highest_compiled_arch >= TURING && < AMPERE) return MMVQ_PARAMETERS_TURING;
    if (NVIDIA && highest_compiled_arch == DGX_SPARK)           return MMVQ_PARAMETERS_GB10;
    return MMVQ_PARAMETERS_GENERIC;        // <-- Pascal landet hier
}
```

Und die PQ2_0-spezifischen Decode-Optimierungen sind explizit gegated:

```c
if constexpr ((type == Q1_0 || type == Q2_0 || type == PQ2_0) &&
              table_id == MMVQ_PARAMETERS_GB10) { ... prefetch, breiterer Block ... }
```

Daraus folgt:

> **Für PQ2_0 bei `ncols_dst == 1` (Decode) hat der Fork auf Pascal *keine*
> karten-spezifische Optimierung.** Es läuft der generische `mul_mat_vec_q`
> mit der `GENERIC`-Parametertabelle und `VDR_PQ2_0_Q8_1_MMVQ = 1`
> (ein 32er-Chunk pro `vec_dot`-Aufruf). Die getunten Pfade existieren für
> **GB10** (Decode) und **Ada** (3–8 Spalten).

Genau diese Lücke füllt `src/kernels/gemv_p40.cu`:

| | Fork auf Pascal | `gemv_p40.cu` |
|---|---|---|
| Zeilen pro Warp | 1 | **4** (4 unabhängige Gewichtsströme) |
| Aktivierungs-Loads | pro Zeile | **1× pro 4 Zeilen** |
| `+1`-Bias | pro Zeile | **1× pro Chunk, gefaltet** |
| Gewichtsskala `d` | pro Chunk | **hoisted** |
| Load-Pfad | generisch | `ld.global.nc` + `prefetch.global.L2` |
| Occupancy | generische Tabelle | `__launch_bounds__(256,4)` = 32 Warps/SM |

Zweiter, weniger offensichtlicher Beitrag: **eine host-verifizierbare
Numerik-Schicht.** Die Kern-Mathematik liegt in `BONSAI_HD`-Funktionen, die
host *und* device identisch kompilieren. Dadurch laufen 333 Checks auf der CPU
gegen dieselbe Arithmetik, die die P40 ausführt — inklusive
Warp-Stride-Schleife, Kachel-/Index-Arithmetik und Split-K-Merge. Für die
PQ2_0-Kernel des Forks gibt es so etwas nicht.

### Reihenfolge der Empfehlung

1. **`gemv_p40.cu`** — echter Beitrag, sm_61-Lücke. **Einbinden.**
2. **`p40_ptx.h`** — klein, unschädlich, macht die Load-Intents explizit.
3. **`kv_p40.cu`** — nur wenn ein eigener Cache gewünscht ist, sonst Fork-Cache.
4. **`gemm_p40.cu`** — **nicht** einbinden (Fork-MMQ ist getunt).
5. **`tests/test_p40.cpp` + `tools/check_gguf.py`** — unabhängig nützlich.

---

## Teil D — Einbindung Schritt für Schritt

### D0 — Der wichtigste Vorbehalt (bitte zuerst lesen)

**Mein GEMV erwartet vorquantisierte Aktivierungen** im PQ2_0-nativen
Q8_1-Layout (`bonsai_p40_quant_activations`). **llama.cpp quantisiert im
Kernel**: `mul_mat_vec_q` wandelt `x` selbst nach q8_1 in Shared Memory um.
Ein reiner Kernel-Tausch funktioniert deshalb **nicht**. Zwei Wege:

* **Weg 1 (empfohlen):** die bestehende In-Kernel-Quantisierung um die
  PQ2_0-Permutation erweitern und die 4-Zeilen-Struktur dort einbauen. Der
  Permutations-Index ist exakt
  `16*(j>>4) + 4*(j&3) + ((j>>2)&3)` (siehe `quantize_row_q8_1(..., permute=true)`).
* **Weg 2:** `bonsai_p40_quant_activations` als separaten Kernel vor den
  Mat-Vec hängen und `x` als eigenes Tensor halten. Einfacher, kostet einen
  zusätzlichen Launch pro Layer.

Ohne diesen Schritt passen Code-Wort und Aktivierungs-Byte nicht zusammen und
das Ergebnis ist still falsch — genau der Bug, den die Host-Suite in
`[pq2_0] format + dp4a dot` abfängt.

### D1 — Dateien kopieren

```bash
mkdir -p ggml/src/ggml-cuda/bonsai-p40
cp src/pq2_0.h src/p40_common.h src/p40_ptx.h src/kv_cache.h \
   ggml/src/ggml-cuda/bonsai-p40/
cp src/kernels/gemv_p40.cu src/kernels/kv_p40.cu ggml/src/ggml-cuda/bonsai-p40/
cp include/bonsai_p40.h include/bonsai_kv.h include/bonsai_model.h \
   ggml/include/bonsai/
```

`gemm_p40.cu`, `elementwise_p40.cu`, `attn_p40.cu` **nicht** kopieren (Teil B).

### D2 — Einhängepunkt im Decode

Der Dispatch sitzt in `ggml/src/ggml-cuda/mmvq.cu`. Dort, wo für
`ncols_dst == 1` der generische `mul_mat_vec_q` gewählt wird, für `sm_61`
abzweigen:

```c
// ggml/src/ggml-cuda/mmvq.cu  (Skizze -- an die tatsächliche
// Signatur der Launcher-Funktion anpassen)
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 610)
    if (type == GGML_TYPE_PQ2_0 && ncols_dst == 1) {
        // x muss bereits im PQ2_0-nativen Q8_1-Layout vorliegen (siehe D0)
        return bonsai_p40_gemv(vx, vy_pq2, dst, nrows_x, ncols_x, stream);
    }
#endif
```

Der Host-Teil (`ggml_cuda_mul_mat_vec_q` in `ggml-cuda.cu`) muss dieselbe
Bedingung prüfen und ggf. `bonsai_p40_quant_activations` vorschalten (Weg 2).

### D3 — Arch-Gating

Alles hinter `__CUDA_ARCH__ == 610` kapseln. Der Stack baut **nur** `sm_61`;
andere Architekturen des Forks dürfen unberührt bleiben. Build:
`-gencode arch=compute_61,code=sm_61`.

### D4 — CMake

```cmake
# ggml/src/ggml-cuda/CMakeLists.txt
if (CMAKE_CUDA_ARCHITECTURES MATCHES "61")
    target_sources(ggml-cuda PRIVATE
        bonsai-p40/gemv_p40.cu
        bonsai-p40/kv_p40.cu)      # kv nur bei Weg "eigener Cache"
endif()
```

### D5 — Loader-Constraint + Pre-Flight

Der Kernel gilt nur für **genau** diese Datei. `tools/check_gguf.py` prüft:

```bash
python3 tools/check_gguf.py models/Ternary-Bonsai-2-27B-PQ2_0.gguf
# -> 18 KV-Checks + 14 Tensor-Checks + MTP-Erkennung, Exit 0 = kompatibel
```

Harte Ablehnung, wenn `general.architecture != "qwen35"`, ein großes Tensor
nicht `GGML_TYPE_PQ2_0` (142) ist, oder die Geometrie von
`include/bonsai_model.h` abweicht (block_count 64, embd 5120, ff 17408,
heads 24/4, head_dim 256, ssm state 128 / groups 16 / inner 6144,
hadamard block 1024). **Ein `nextn`/MTP-Block wird als inkompatibel gemeldet**
(Exit 1) — siehe `docs/MTP.md`.

### D6 — Variante: als eigenständige Library

```bash
./tools/build_p40.sh          # -> build/libbonsai_p40.a + Host-Tests
```

Dann `include/bonsai_p40.h` einbinden und die Kernel direkt rufen (für einen
eigenen Runner ohne Fork):

```c
bonsai_p40_quant_activations(x, xq, 5120, stream);          // Pflicht zuerst
bonsai_p40_gemv(W, xq, y, /*n_out=*/17408, /*n_in=*/5120, stream);
bonsai_p40_fwht(x, signs, 5120, stream);
bonsai_p40_rmsnorm(x, w, y, /*rows=*/1, /*n=*/5120, 1e-6f, stream);
bonsai_p40_gdn_step(st, q, k, v, g, b, out, 16, 48, 128, 128, stream);
bonsai_p40_attn(q, k, v, out, /*n_tok=*/n, 24, 4, 256, 1.f/16.f, stream);
```

---

## Teil E — Verifikation

```bash
# 1. Host-Suite (braucht keine GPU): exakt dieselbe Arithmetik wie die P40
g++ -O2 -std=c++17 -Wall -Wextra -I src -I include tests/test_p40.cpp -o test_p40
./test_p40                      # -> 333 passed, 0 failed

# 2. Geometrie gegen die echte Datei
python3 tools/check_gguf.py models/Ternary-Bonsai-2-27B-PQ2_0.gguf
#    -> All GGUF geometry checks passed.  MTP/nextn block: none

# 3. Vollständiger Build inkl. sm_61-Lib
./tools/build_p40.sh --check-gguf models/Ternary-Bonsai-2-27B-PQ2_0.gguf

# 4. PTX/SASS prüfen (braucht nvcc)
./tools/disasm.sh gemv
#    erwartet: "Used N registers" <= 64, LDG.E.U16 (nicht verbreitert),
#             32x DP4A pro 128-Gewichtsblock x 4 Zeilen, 4x prefetch.global.L2

# 5. Vorhersagen nachrechnen
python3 tools/gpu_predict.py
```

Erwartete Ausgabe der Suite:

```
[pq2_0] format + dp4a dot          [fwht] Hadamard block 1024
[ops] rmsnorm / rope / swiglu      [gdn] gated delta net step
[attn] online softmax vs naive     [model] qwen35 geometry
[gemv] 4-rows-per-warp strided loop [gemm] tiled matmul
[ptx] host fallback path           [kv] cache sizing, layout, score, split-K

333 passed, 0 failed
```

---

## Teil F — Grenzen (ehrlich)

* **Keine GPU-Messung.** In dieser Umgebung gab es weder `nvcc` noch eine GPU.
  Verifiziert ist die **Numerik** (host-seitig, 333 Checks) und die
  **Geometrie** (gegen die echte GGUF). **Nicht** verifiziert sind
  CUDA-Kompilierbarkeit, Registerdruck, Occupancy und Performance.
* **Kein MMQ** — bewusst, siehe `docs/MMQ.md`.
* **Kein MTP** — vier konkrete Gaps, siehe `docs/MTP.md`.
* **Prefill-GEMM ist ein Fallback** und dem Fork-MMQ unterlegen.
* **Der Abstand** zwischen meinem GEMV und dem generischen Fork-Pfad auf
  Pascal ist **nicht gemessen**; die Überlegenheit ist aus der Struktur
  abgeleitet (MLP, gefalteter Bias, Occupancy), nicht belegt.
* **D2 ist eine Skizze.** Die exakte Signatur der Launcher-Funktion im Fork
  wurde nicht gegen einen kompilierten Build geprüft.