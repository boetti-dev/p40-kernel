# Vorhersage: Ternary-Bonsai-2-27B auf einer Tesla V100 32 GB

> **KORRIGIERT — die Zahlen unten waren ~1.76× zu optimistisch.**
> Das ursprüngliche Modell rechnete mit 88 % der Peak-Bandbreite; gemessen
> auf einer echten P40 sind es **45.5 %** (21.90 tok/s statt 42.2).
> Korrigierte Werte: **P40 ~21.8 tok/s**, **V100-32GB ~62.4 tok/s**
> (statt 109.8). Siehe `docs/MEASURED.md` für Messung, Kalibrierung und
> Begründung. Die Zeilen unten sind historisch und NICHT korrigiert.


Diese Vorhersage ist mit **demselben Modell** gerechnet wie die P40-Zahlen, und
sie reproduziert sie (20.8 ms / 48 tok/s Ceiling, 8.50 GiB KV @262k, 42.2 tok/s
realistisch vs. dokumentierte 41–44). Damit sind beide Zahlen vergleichbar.
Nachrechnen: `python3 tools/gpu_predict.py`.

## 1. Hardwarevergleich

| | Tesla P40 (GP102) | Tesla V100 32 GB (GV100) | Faktor |
|---|---|---|---|
| Architektur | `sm_61` (Pascal) | `sm_70` (Volta) | – |
| Speicherbandbreite | 346 GB/s GDDR5 | **900 GB/s HBM2** | **2.60×** |
| SMs | 30 | 80 | 2.67× |
| FP32-Kerne/SM | 128 | 64 | 0.50× |
| FP32 gesamt | 11.8 TFLOP/s | 15.7 TFLOP/s | 1.33× |
| FP16-ALU (nicht-TC) | 1/64 Rate | **volle Rate (2× FP32)** | 128× |
| Tensor Cores | **keine** | FP16 + INT8 | – |
| INT8 über `dp4a` | ja | ja | – |
| INT8 Tensor Core | – | 125 TMAC/s | – |
| L2 | 3 MiB | 6 MiB | 2× |
| VRAM | 24 GiB | **32 GiB** | 1.33× |
| Register/SM, Threads/SM | 64 K, 2048 | 64 K, 2048 | 1× |

Zwei Punkte sind für den Entwurf entscheidend:

1. **Volta hat nur 64 FP32-Kerne/SM**, also trotz 2.67× mehr SMs nur 1.33× die
   FP32-Rate der P40 — aber **2.60× die Bandbreite**. Die V100 ist damit
   *noch stärker bandbreitengebunden* als die P40.
2. **FP16 läuft auf Volta mit voller Rate** (2× FP32). Die P40-Regel „FP16
   niemals als Rechentyp" fällt weg. Für den Decode ändert das nichts (der ist
   bandbreitengebunden), für den Prefill ist es zusammen mit den Tensor Cores
   der große Hebel.

## 2. Vorhersage Decode (Batch 1) — PQ2_0, Q8_0-KV

| | P40 | V100 32 GB |
|---|---|---|
| Gewichtsstrom, Peak | 20.8 ms → 48.0 tok/s | **8.0 ms → 124.8 tok/s** |
| + 88 % realer Bandbreite (**falsch, siehe Banner**) | 23.7 ms → 42.2 tok/s | **9.1 ms → 109.8 tok/s** |
| Ceiling mit **PTQ1_0** (5.95 GB) | 17.2 ms → 58.2 tok/s | **6.6 ms → 151.3 tok/s** |

Mit Kontext (Q8_0-KV, 34816 B/Token, 88 % Bandbreite):

| Kontext | bewegt | P40 | **V100 32 GB** |
|---|---|---|---|
| 0 | 7.21 GB | 42.2 tok/s | **109.8 tok/s** |
| 4 096 | 7.35 GB | 41.4 | **107.7** |
| 32 768 | 8.35 GB | 36.5 | **94.8** |
| 131 072 | 11.77 GB | 25.9 | **67.3** |
| 262 144 | 16.34 GB | 18.6 | **48.5** |

**Kernaussage: ~110 tok/s bei kurzem Kontext, ~95 bei 32K, ~48 bei vollem
262K — also 2.6× die P40 bei jedem Kontext.** Mit PTQ1_0 kommen ~+21 % dazu
(~133 tok/s kurz).

### Warum es weiterhin bandbreitengebunden bleibt

Gegenprobe wie in `docs/P40_OPTIMIZATION.md`:

| | P40 | V100 |
|---|---|---|
| Warp-Instruktionen/Token (~2e8) | 1.09 ms | **0.41 ms** |
| Bandbreitenzeit | 20.8 ms | 8.0 ms |
| Anteil Issue | 5 % | **5 %** |

Die Instruktionslast skaliert mit (2.67× mehr Scheduler), bleibt also der
gleiche 5 %-Beitrag. Tensor Cores sind für den Decode **nutzlos** (`mma` braucht
M ≥ 8). Der einzige Hebel bleibt Memory-Level-Parallelität:

* In Flight: 80 SM × 32 Warps × 4 Zeilen × 128 B = **1.31 MB**, benötigt
  (900 GB/s × 600 ns) = **0.54 MB** → ok mit Reserve.
* Also bleiben **4 Zeilen/Warp, `__launch_bounds__(256,4)`, `dp4a`,
  `ld.global.nc`, `prefetch.global.L2` alle unverändert.** Wichtigster Befund:
  die V100-Anpassung ist für den Decode eine **Neukompilierung, kein
## 3. Vorhersage Prefill — hier liegen die Tensor Cores

Prefill bei Batch 512 = 24.9 TFLOP = 12.47 TMAC:

| Pfad | Peak | realistisch | Zeit |
|---|---|---|---|
| P40, `dp4a` | 23.5 TMAC/s | 30–40 % | **1.3–1.8 s** |
| V100, FP16-TC | 62.5 TMAC/s | 15–30 % | 0.66–1.33 s |
| **V100, INT8-TC** | **125 TMAC/s** | 20–35 % | **0.28–0.50 s** |

**→ Prefill ~4–7× schneller als auf der P40.** Der Grund ist einzig der
INT8-Tensor-Core-Pfad (`mma` über `s8`-Fragmente), nicht die Bandbreite.

Wichtige Einschränkung, ehrlich: **Volta hat kein `cp.async` (kommt mit
`sm_80`) und kein `ldmatrix` (kommt mit `sm_75`).** Die Fragment-Ladung muss
per-Lane von Hand geschehen, es gibt kein doppelt gepuffertes Async-Staging.
Genau deshalb ist die Effizienzspanne (20–35 %) breiter als bei einem
Ampere-Kernel, und deshalb lautet die Vorhersage „0.3–0.5 s" statt „0.1 s".

## 4. Was konkret angepasst würde (die „ähnliche Anpassung")

Für den Decode: fast nichts.

| Datei | Änderung |
|---|---|
| Build | `-arch=sm_70` statt `sm_61`, `CMAKE_CUDA_ARCHITECTURES 70` |
| `p40_common.h` | CC 70, 80 SMs, 900 GB/s, 6 MiB L2 (sonst gleich) |
| `gemv_p40.cu` | **unverändert** — `dp4a`, 4 Zeilen/Warp, PTX-Layer gilt 1:1 |
| `kv_p40.cu` | **unverändert** — Split-K hilft genauso; `n_split` evtl. auf 64 |
| `p40_ptx.h` | `ld.global.nc`, `dp4a.u32.s32`, `prefetch.global.L2` alle `sm_70`-gültig |
| `gemm_p40.cu` | **ersetzt** durch einen `mma.sync`-Kernel (INT8, `s8`-Fragmente) |
| `elementwise` / `attn` | nur `-arch` anpassen |

Der Speicher-Header profitiert vom größeren VRAM: statt Q8_0/Q8_0 ist
**F16/F16 möglich** (16.00 GiB KV + 7.21 GiB Gewichte = 23.2 GiB von 32), was
Qualität statt Geschwindigkeit kauft:

| KV | @262k | KV-Array | Gesamt (Gew.+KV) | V100 tok/s @262k |
|---|---|---|---|---|
| Q4_0/Q4_0 | 4.50 GiB | 4.83 GB | 12.04 GB | **65.8** |
| Q8_0/Q4_0 | 6.50 GiB | 6.98 GB | 14.19 GB | **55.8** |
| **Q8_0/Q8_0** (empfohlen) | 8.50 GiB | 9.13 GB | 16.34 GB | **48.5** |
| F16/Q8_0 | 12.25 GiB | 13.15 GB | 20.36 GB | **38.9** |
| F16/F16 | 16.00 GiB | 17.18 GB | 24.39 GB | **32.5** |

Alle Varianten passen in 32 GiB (34.4 GB nutzbar) — inklusive F16/F16, das auf
der P40 unmöglich war. Empfehlung V100: **Q8_0/Q8_0 als Default**, **F16/F16
nur wenn Qualität über Durchsatz geht** und der Kontext unter ~200K bleibt.

## 5. Vorhergesagte Kennzahlen (Zusammenfassung)

| Metrik | P40 | **V100 32 GB** | Faktor |
|---|---|---|---|
| Decode, kurzer Kontext | 42.2 tok/s | **109.8 tok/s** | 2.60× |
| Decode, 32K | 36.5 | **94.8** | 2.60× |
| Decode, 262K (Q8_0-KV) | 18.6 | **48.5** | 2.61× |
| Decode, PTQ1_0, kurz | 51.2 | **133** | 2.60× |
| Prefill Batch 512 | 1.3–1.8 s | **0.28–0.50 s** | 4–7× |
| VRAM-Bedarf (Q8_0-KV, 262K) | 15.9 GiB / 24 | **15.9 GiB / 32** | – |
| längster Kontext | 262K (passt) | 262K (passt, mit Reserve) | – |

## 6. Unsicherheiten

* **±10 %** aus der erreichbaren Bandbreite (88 % angenommen; HBM2 streamt
  typisch 85–92 %, GDDR5 85–90 %).
* **Prefill ist die unsicherste Zahl** (±50 %): kein `cp.async`, kein
  `ldmatrix` auf `sm_70`, und der Tensor-Core-Wirkungsgrad hängt stark davon
  ab, wie gut das 2-Bit-Unpacking mit den `s8`-Fragmenten verschmilzt.
* **Nicht gemessen.** Weder P40 noch V100 standen zur Verfügung (kein `nvcc`,
  keine GPU). Alle Zahlen sind Byte × Bandbreite bzw. MAC × Rate, und im
  Host-Test (`tests/test_p40.cpp`) gegen die Sizing-Tabellen geprüft.
* **Kein Reasoning-Effekt** eingerechnet: das Modell denkt lange; die tok/s
  oben sind reine Decode-Rate, nicht Antwort-Tokens/s.

## 7. Was mit dem Cache passiert

Der KV-Cache-Plan aus `docs/KV_CACHE.md` gilt unverändert — 16 volle
Attention-Layer mit wachsendem Q8_0-Cache, 48 Gated-DeltaNet-Layer mit
konstantem 144-MiB-Zustand. Die V100 ändert nur die **Default-Wahl**:

* P40 erzwungen: **Q8_0/Q8_0**, weil F16 nicht passt (7.21 + 16.00 = 23.2 von
  24 GiB, kein Platz für Aktivierungen/Kontext).
* V100 frei: Q8_0/Q8_0 (48.5 tok/s @262k) bis F16/F16 (32.5 tok/s @262k).

Zweiter V100-Vorteil: `n_split` kann von 32 auf **64** erhöht werden, weil 80
SMs die Splits breiter verteilen. Das verschiebt den Erwartungswert nicht (der
Decode ist ohnehin bandbreitengebunden), macht den Split-K-Merge aber robuster
gegen Tail-Last bei sehr langem Kontext.

## 8. Ehrliche Zusammenfassung

* **Decode: 2.6× die P40** — reine Bandbreitenskala, kein Redesign.
* **Prefill: 4–7× die P40** — *das* ist die eigentliche V100-Anpassung
  (INT8-Tensor-Cores), und sie hat die größte Unsicherheit.
* **VRAM entspannt** — F16-KV wird erstmals möglich, was Qualität statt
  Durchsatz kauft.
* **Nicht gemessen.** Die Vorhersage ist ein belastbares Modell, kein Benchmark.
  `python3 tools/gpu_predict.py --bw 900 --sm 80 --name V100` rechnet sie nach.
  Redesign**.