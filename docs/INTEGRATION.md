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

## Teil D — Einbindung

> **Es gibt eine fertige, applybare Integration:** `integrations/prismml/`
> (Diff + `git am`-Patch, gegen `prism` @ `b858f3b` verifiziert).
> Die folgende Beschreibung erklärt, was sie tut.

### D0 — Der wichtigste Vorbehalt (bitte zuerst lesen)

**Mein GEMV erwartet vorquantisierte Aktivierungen** im PQ2_0-nativen
Q8_1-Layout (`bonsai_p40_quant_activations`). **llama.cpp quantisiert im
Kernel**: `mul_mat_vec_q` wandelt `x` selbst nach q8_1 in Shared Memory um.
Ein reiner Kernel-Tausch funktioniert deshalb **nicht**. Die fertige Integration
löst das, indem sie **beide** Kernel nacheinander ruft (Quantisieren in einen
32-KiB-Scratch, dann Mat-Vec) — das ist Weg 2 unten und kostet einen
zusätzlichen Launch pro Layer.

Würde man stattdessen nur den Mat-Vec tauschen, passten Code-Wort und
Aktivierungs-Byte nicht zusammen und das Ergebnis wäre **still falsch** — genau
der Bug, den `[pq2_0] format + dp4a dot` in der Host-Suite abfängt.

### D1 — Einhängepunkt: `ggml-cuda.cu`, nicht `mmvq.cu`

**Korrektur zu einer früheren Fassung dieses Dokuments:** dort stand, man solle
`mmvq.cu` patchen. Das ist der falsche Ort — `mmvq.cu` ist stark templatisiert
und gefused (Fusion, MoE, SoA-Layouts), ein Hook dort wäre kaum prüfbar. Der
saubere Punkt ist `ggml_cuda_mul_mat()` in `ggml/src/ggml-cuda/ggml-cuda.cu`,
unmittelbar vor dem Stock-`mmvq`-Zweig:

```c
    if (bonsai_p40_should_use(src0->type, cc, ne11) &&
            bonsai_p40_mul_mat(src0, src1, dst, ctx.stream())) {
        return;
    }
    if (ggml_cuda_should_use_mmvq(src0->type, cc, ne11)) { ... }   // unverändert
```

`bonsai_p40_mul_mat()` gibt `false` zurück, sobald irgendetwas nicht exakt
passt, und dann läuft alles unverändert weiter.

### D2 — Die Übernahme-Entscheidung ist eine getestete Funktion

Der einzige Gate, der den Kernel aktiviert, liegt in
`src/p40_dispatch_predicate.h` als **reine C++-Funktion ohne ggml- und
CUDA-Typen**:

```cpp
bool p40_decode_path_ok(const p40_mulmat_shape & s);
```

Damit ist sie host-testbar — `[dispatch] PrismML MUL_MAT takeover predicate`
prüft alle sechs echten Zeilenbreiten des Modells und **14 Ablehnungsgründe**
(falscher Typ, `GGML_TYPE_Q2_0` statt PQ2_0, Batch > 1, MoE/3D, nicht
zusammenhängender Zeilenstride, `n_in` nicht Vielfaches von 128, …). Ein
falsches „ja" würde still falsche Ergebnisse liefern; deshalb ist das getestet
und nicht inline.

### D3 — Makro-Kollisionen (der Stolperstein)

Der Fork definiert in `ggml-common.h` die **Makros** `QK_PQ2_0`, `QK8_1`,
`QK8_0`. Deklariert man dieselben Namen als Konstanten, ersetzt der
Präprozessor sie und aus `static constexpr int QK8_1 = 32;` wird
`static constexpr int 32 = 32;`:

```
<command-line>: error: expected unqualified-id before numeric constant
```

Alle eigenen Konstanten heißen deshalb `BONSAI_*` und **erben** die Werte des
Forks, wenn er vorhanden ist:

```cpp
#ifdef QK8_1
static constexpr int BONSAI_QK8_1 = QK8_1;   // inherit ggml's value
#else
static constexpr int BONSAI_QK8_1 = 32;
#endif
```

`tools/check_macro_collisions.py <fork>` prüft das gegen **1882** Fork-Makros
und muss `no macro collisions` liefern. Aktuell: 0 Kollisionen.

### D4 — CMake

`ggml/src/ggml-cuda/CMakeLists.txt` globt nur `*.cu` im Top-Level, also braucht
das Unterverzeichnis zwei Zeilen:

```cmake
file(GLOB   SRCS "bonsai-p40/*.cu")
list(APPEND GGML_SOURCES_CUDA ${SRCS})
file(GLOB   SRCS_BONSAI_HDRS "bonsai-p40/*.h" "bonsai-p40/*.cuh")
list(APPEND GGML_HEADERS_CUDA ${SRCS_BONSAI_HDRS})
```

### D5 — Arch-Gating

Der Kernel ist auf `cc == 610` und ein Shape-Prädikat begrenzt; auf jeder
anderen GPU liefert `bonsai_p40_should_use()` `false` und es ändert sich
nichts. `sm_61` ist in den Default-Arch-Listen des Forks enthalten
(`61-virtual`).

### D6 — Loader-Constraint + Pre-Flight

```bash
python3 tools/check_gguf.py models/Ternary-Bonsai-2-27B-PQ2_0.gguf
# -> 18 KV-Checks + 14 Tensor-Checks + MTP-Erkennung, Exit 0 = kompatibel
```

Harte Ablehnung, wenn `general.architecture != "qwen35"`, ein großes Tensor
nicht `GGML_TYPE_PQ2_0` (142) ist, oder die Geometrie von
`include/bonsai_model.h` abweicht. **Ein `nextn`/MTP-Block wird als
inkompatibel gemeldet** (Exit 1) — siehe `docs/MTP.md`.

### D7 — Variante: als eigenständige Library

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