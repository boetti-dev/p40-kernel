# PrismML-Fork-Integration (fertig, applybar)

Hier liegt die **fertige** Integration in `PrismML-Eng/llama.cpp` — kein Entwurf.

| Datei | Verwendung |
|---|---|
| `bonsai-p40.diff` | `git apply` (reiner Diff) |
| `0001-bonsai-p40.patch` | `git am` / `git format-patch`-Format |

> ⚠️ **Der Patch ist OPT-IN und standardmäßig AUS.** Gemessen auf einer echten
> P40 war der Pfad **1.92× langsamer** (11.39 vs 21.90 tok/s tg128).
> Aktivieren nur mit `BONSAI_P40=1`; Messung und Analyse in `docs/MEASURED.md`.
> Die 9 Zeilen Hook in Bestandsdateien sind trotzdem harmlos — sie rufen nur
> `bonsai_p40_should_use()`, das ohne `BONSAI_P40=1` `false` liefert.

Basis: Branch **`prism`**, Commit **`b858f3b`** (`tests: portable popcount in
arclab_init_hadamard (#362)`).

## Anwenden

```bash
git clone -b prism https://github.com/PrismML-Eng/llama.cpp
cd llama.cpp
git apply /pfad/zu/integrations/prismml/bonsai-p40.diff      # oder: git am 0001-bonsai-p40.patch
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61
cmake --build build -j
```

## Inhalt

**11 Dateien, +897 Zeilen.** Davon nur **9 Zeilen** in bestehenden Dateien:

* `ggml/src/ggml-cuda/ggml-cuda.cu` — 1 Include + 4 Zeilen Hook in
  `ggml_cuda_mul_mat()`, direkt vor dem Stock-`mmvq`-Zweig
* `ggml/src/ggml-cuda/CMakeLists.txt` — 4 Zeilen Glob für `bonsai-p40/`

Der Rest ist neu und liegt komplett unter `ggml/src/ggml-cuda/bonsai-p40/`
(`mmvq.cu` wird **nicht** angefasst — es ist stark templatisiert und gefused,
ein Hook dort wäre schwer zu prüfen).

## Was verifiziert wurde

| Prüfung | Ergebnis |
|---|---|
| `git apply --check` gegen frischen `prism`-Clone | **OK** |
| Ergebnis nach Apply vs. Entwicklungs-Clone | **byte-identisch** (`diff -r`) |
| Makro-Kollisionen gegen 1882 Fork-Makros | **keine** (`tools/check_macro_collisions.py`) |
| Host-Suite, standalone **und** mit den Fork-Makros (`-DQK8_0=32 -DQK8_1=32 -DQK_PQ2_0=128 -DQI_PQ2_0=4`) | **355/355** |
| Include-Graph der neuen Dateien | alle Auflösungen lokal bzw. `ggml/include/ggml.h` |

## Was *nicht* verifiziert wurde (bitte beachten)

**Es wurde nicht kompiliert und nicht gebenchmarkt.** In der Umgebung, in der
das entstand, gab es **kein `cmake`** (nur `make`) und **kein `nvcc`**, also ist
llama.cpp dort grundsätzlich nicht baubar. Konsequenzen:

* Die CUDA-Kompilierbarkeit der neuen Dateien ist **nicht** belegt.
* Der Geschwindigkeitsvorteil gegenüber dem Stock-Pfad ist **nicht** gemessen —
  er ist aus der Struktur argumentiert (Memory-Level-Parallelität, gefalteter
  Bias, Occupancy), nicht belegt.

Bitte zuerst genau das prüfen:

```bash
# 1. baut es?
cmake --build build -j 2>&1 | tail

# 2. greift der Pfad? (Logzeile beim Laden)
#    und: läuft es schneller?
llama-bench -m Ternary-Bonsai-2-27B-PQ2_0.gguf -ngl 99 -fa on -p 0 -n 128
#    A/B: einmal mit `return false` in bonsai_p40_should_use() -> Stock-Pfad
```

## Rücknahme

```bash
git apply -R /pfad/zu/integrations/prismml/bonsai-p40.diff
```

Da nur 9 Zeilen in Bestandsdateien geändert werden und der Hook hinter
`cc == 610` sowie einem Shape-Prädikat sitzt, ist der Rückweg trivial und der
Einfluss auf andere GPUs null.

## Verwandte Dokumente

* `docs/INTEGRATION.md` — Begründung, Lücke im Fork, Verifikationskommandos
* `ggml/src/ggml-cuda/bonsai-p40/README.md` (im Patch) — Kernel-Details
* `docs/MMQ.md`, `docs/MTP.md`, `docs/KV_CACHE.md`, `docs/P40_OPTIMIZATION.md`