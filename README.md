# Bonsai-P40

Ein CUDA-Kernel-Stack, der **ausschließlich** das Modell
[`prism-ml/Ternary-Bonsai-2-27B-gguf`](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf)
im **PQ2_0**-Format (`GGML_TYPE_PQ2_0 = 142`, das „q2_0" dieses Modells) auf
einer **NVIDIA Tesla P40** (GP102, `sm_61`) rechnet.

Es gibt bewusst **keinen** Codepfad für ein anderes Modell, eine andere
Quantisierung oder eine andere GPU: alle Kachelgrößen, Strides, Occupancy-Ziele
und Vektorisierungen sind auf genau diese Karte und genau diese Tensorformen
zugeschnitten.

---

## 1. Was hier drin ist

| Datei | Zweck |
|---|---|
| `include/bonsai_model.h` | Exakte Geometrie von qwen35 (aus dem GGUF-Header gelesen) |
| `include/bonsai_p40.h` | Öffentliche C-API der Kernel |
| `include/bonsai_kv.h` | Öffentliche C-API des KV-Caches |
| `src/p40_common.h` | Host/Device-Primitive, P40-Konstanten, `dp4a`, fp16↔fp32 |
| `src/kv_cache.h` | KV-Cache-Geometrie, Sizing, Q8_0-Mathe (host **und** device) |
| `src/p40_ptx.h` | Handgeschriebene PTX nur dort, wo sie den Speicherpfad/Occupancy betrifft |
| `src/pq2_0.h` | PQ2_0-Blockformat + `dp4a`-Kern (host **und** device) |
| `src/fwht.h` | Hadamard-Rotation (Block 1024, normalisiert, Sylvester) |
| `src/ops.h` | RMSNorm, RoPE, SwiGLU, Gated-DeltaNet-Schritt, Conv1d |
| `src/kernels/gemv_p40.cu` | Decode-Mat-Vec (bandbreitenkritisch) |
| `src/kernels/gemm_p40.cu` | Prefill-Mat-Mul (DP4A-Tile-GEMM) |
| `src/kernels/elementwise_p40.cu` | Hadamard + RMSNorm |
| `src/kernels/attn_p40.cu` | Gated DeltaNet + GQA-Flash-Attention |
| `src/kernels/kv_p40.cu` | KV-Cache: Append/Quantisierung + Split-K-Decode-Attention |
| `tests/test_p40.cpp` | Host-Verifikation der gesamten Kernel-Mathematik |
| `tools/check_gguf.py` | Prüft die Kernel-Konstanten gegen die echte GGUF-Datei |
| `tools/disasm.sh` | PTX/SASS-Inspektion der Kernel (braucht `nvcc`) |
| `tools/gpu_predict.py` | Decode-/Prefill-Vorhersage für eine beliebige Karte aus Bandbreite/MACs |
| `tools/build_p40.sh` | Build für `sm_61` + Host-Tests |

## 2. Die Karte: warum hier alles anders ist als bei Ampere/Ada

Die P40 ist **Pascal (GP102)**. Die wichtigsten Konsequenzen:

* **Kein FP16-ALU.** Die FP16-Rate auf GP102 ist **1/64** der FP32-Rate.
  FP16 wird deshalb **nur als Speicher-/Konvertierungsformat** benutzt (die
  PQ2_0-Skalen `d`), niemals als Rechenformat. Jede Akkumulation ist FP32.
* **DP4A ist vorhanden** (`sm_61 ≥ 610`): `__dp4a` liefert 4 Int8-MACs pro
  Instruktion. Das ist die einzige brauchbare „Tensor"-Einheit dieser Karte und
  trägt die gesamte ternäre Mat-Mul.
* **Keine Tensor Cores.** Es gibt keine `mma`-Pfade, also wird alles über
  `dp4a` + FP32-Skalierung gebaut.
* **Bandbreite ist das Limit, nicht FLOPS.** 24 GB GDDR5 @ ~346 GB/s,
  30 SMs, 3 MB L2.

## 3. Das Format: PQ2_0 („q2_0")

Aus der Referenzimplementierung im PrismML-Fork verifiziert:

```
block_pq2_0 : 128 Gewichte, 2-Bit-Codes, ein fp16-Skalenwert pro 128
   code c ∈ {0,1,2,3}  ->  Wert = (c-1) * d
   Element j liegt in Byte j/4, Bits (j%4)*2, little endian
   Blockgröße = 2 (fp16 d) + 32 (Codes) = 34 Byte, nur 2-Byte-aligned
```

Das Checkpoint ist ternär, es kommen also nur die Codes `{0,1,2}` vor; der
Kernel behandelt trotzdem `3` (→ +2), weil der Referenz-Quantisierer ihn
erzeugen kann.

Die Aktivierungen werden zu **Q8_1** quantisiert, aber in einem
**PQ2_0-nativen, transponierten Layout**: innerhalb jedes 16er-Halbs wird eine
4×4-Transposition gespeichert, sodass das extrahierte 2-Bit-Codewort `k` Byte
für Byte zu Aktivierungswort `k` passt. Dadurch fällt **jedes** Bit-Unpacken pro
Gewicht weg. Der `+1`-Bias der Codes wird durch eine einzige Integer-Subtraktion
pro 32er-Chunk erledigt (`Σ(c-1)a = Σca − Σa`, und `Σa` steht als roher
int16-Summenwert im `s`-Feld des Q8_1-Blocks).

## 4. Verifikation

Alles, was ohne GPU prüfbar ist, ist geprüft:

```
$ g++ -O2 -std=c++17 -I src -I include tests/test_p40.cpp -o test_p40 && ./test_p40
[pq2_0] format + dp4a dot
[fwht] Hadamard block 1024
[ops] rmsnorm / rope / swiglu
[gdn] gated delta net step
[attn] online softmax vs naive
[model] qwen35 geometry
[gemv] 4-rows-per-warp strided loop
[gemm] tiled matmul
[ptx] host fallback path
[kv] cache sizing, layout, score, split-K

333 passed, 0 failed
```

```
$ python3 tools/check_gguf.py Ternary-Bonsai-2-27B-PQ2_0.gguf
... alle 18 KV- und 14 Tensor-Checks OK ...
All GGUF geometry checks passed.
```

Die Kernel-Mathematik liegt in `BONSAI_HD`-Funktionen (`src/pq2_0.h`), die
**identisch** von Host und Device kompiliert werden. Der Host-Test führt also
exakt dieselbe Arithmetik aus, die die P40 ausführt — inklusive der
Warp-Stride-Schleife des GEMV und der Kachel-/Index-Arithmetik des GEMM.

## 5. Bauen

```bash
./tools/build_p40.sh                                  # Host-Tests + sm_61-Lib
./tools/build_p40.sh --check-gguf Ternary-Bonsai-2-27B-PQ2_0.gguf
# oder
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=61 && cmake --build build && ctest --test-dir build
```

Der CUDA-Build braucht `nvcc` (≥ 11.0 für `sm_61`). Ohne `nvcc` laufen
zumindest die Host-Tests.

## 6. Einbindung in prismml/llamacpp

Siehe [`docs/INTEGRATION.md`](docs/INTEGRATION.md) — die Kernel ersetzen im
PrismML-Fork die `PQ2_0`-Pfade in `ggml/src/ggml-cuda/` und der Loader akzeptiert
nur `GGML_TYPE_PQ2_0` mit der qwen35-Geometrie.

## 7. Grenzen (ehrlich)

* Die numerische Korrektheit ist auf dem Host bewiesen. Die **Performance** auf
  einer echten P40 (Bandbreiten-Auslastung, Occupancy) konnte in dieser
  Umgebung **nicht** gemessen werden — es stand kein `nvcc` und keine GPU zur
  Verfügung. Die Tuning-Entscheidungen sind aus den GP102-Datenblattswerten
  abgeleitet und in `docs/P40_OPTIMIZATION.md` begründet.
* Die **PTX- und SASS-Ausgabe** wurde ebenfalls nicht kompiliert/verifiziert
  (kein `nvcc`). `src/p40_ptx.h` benutzt ausschließlich etablierte
  Instruktionsformen (`ld.global.nc`, `dp4a.u32.s32`, `prefetch.global.L2`),
  jede hinter `__CUDA_ARCH__`-Guards mit Host-Fallback, der mitgetestet wird.
  `tools/disasm.sh` ist der vorgesehene Weg, die Behauptungen auf einer
  Maschine mit Toolkit nachzuprüfen (Registerzahl, `LDG.E.U16`, `DP4A`,
  `prefetch`).
* Die Gated-DeltaNet- und Attention-Kernel sind korrekt, aber bewusst einfach
  gehalten: sie sind neben dem 7.2-GB-Gewichtsstream pro Token praktisch
  kostenlos (Zustand ~3 MB), und die volle Attention ist nicht der
  bandbreitenkritische Pfad.
