# Gemessen: der P40-Kernel ist 2× LANGSAMER als der Stock-Pfad

**Dies ist die wichtigste Datei im Repo.** Sie korrigiert Vorhersagen, die sich
auf echter Hardware als falsch erwiesen haben.

## 1. Die Messung

Auf einer echten Tesla P40, `llama-bench -p 0 -n 128 -ngl 99 -fa on`, PQ2_0:

```
=== BONSAI_P40=1 ===
| qwen35 27B PQ2_0 - 2.13 bpw (group 128) | 6.70 GiB | 26.90 B | CUDA | 99 | 1 | tg128 | 11.39 ± 0.02 |

=== BONSAI_P40=0 ===
| qwen35 27B PQ2_0 - 2.13 bpw (group 128) | 6.70 GiB | 26.90 B | CUDA | 99 | 1 | tg128 | 21.90 ± 0.00 |
```

| | tok/s | effektive Bandbreite | % der Peak |
|---|---|---|---|
| **Stock-Pfad** (`BONSAI_P40=0`) | **21.90** | 157.5 GB/s | 45.5 % |
| **mein Kernel** (`BONSAI_P40=1`) | **11.39** | 81.9 GB/s | 23.7 % |

**Ergebnis: mein Kernel ist 1.92× langsamer.** Der Patch ist deshalb auf
**OFF by default** gestellt (`BONSAI_P40=1` muss explizit gesetzt werden).

## 2. Der Fehler in meinem Vorhersagemodell

Ich hatte mit **88 % der Peak-Bandbreite** gerechnet und daraus 42 tok/s für die
P40 abgeleitet. Diese Annahme war für **alle** Karten falsch. Mit den
Model-Card-Werten und der P40-Messung:

| Karte | gemessen (tg128) | effektiv | % von Peak |
|---|---|---|---|
| **P40 (Stock)** | **21.90** | 157.9 GB/s | **45.6 %** |
| P40 (mein Kernel) | 11.39 | 82.1 GB/s | 23.7 % |
| L4 | 29.80 | 214.9 GB/s | 71.6 % |
| L40S | 74.40 | 536.4 GB/s | 62.1 % |
| RTX 4090 | 81.20 | 585.5 GB/s | 58.1 % |
| RTX 5090 | 129.90 | 936.6 GB/s | 52.3 % |
| A100 SXM | 73.90 | 532.8 GB/s | 26.1 % |
| H100 SXM | 113.90 | 821.2 GB/s | 24.5 % |

Die Effizienz schwankt zwischen **24 % und 72 %**, und das Model-Card nennt den
Grund selbst: auf H100/A100/Blackwell ist der Decode „limited by instruction
throughput and launch overhead", auf L4/L40S/Ada ist er speichergebunden. Es
gibt **keine** universelle Effizienzzahl.

Mit der gemessenen P40-Effizienz (45.5 %) sagt das korrigierte Modell
**21.8 tok/s** voraus — gegen **21.90 gemessen**. Die Kalibrierung sitzt jetzt.

## 3. Warum meine Optimierung scheiterte

Meine Begründung war: *„4 Zeilen pro Warp ⇒ 4 unabhängige Gewichtsströme in
Flight ⇒ mehr Memory-Level-Parallelität pro Warp ⇒ mehr Bandbreite."*

Die Prämisse war falsch. Der Stock-Warp deckt schon von sich aus einen
zusammenhängenden Bereich ab:

```c
// ggml/src/ggml-cuda/mmvq.cu:784
for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
//  qi/vdr = 8   -> 8 Threads pro 128er-Block
//  blocks_per_iter = vdr * warp_size / qi = 1 * 32 / 8 = 4
```

Der Warp liest also **4 K-Blöcke pro Iteration = 136 zusammenhängende Bytes**,
mit 8 Lanes pro Block. Der Speicher-Fußabdruck pro Iteration ist winzig; die
Bandbreite kommt aus **vielen Warps**, nicht aus MLP pro Warp.

Mein Kernel liest stattdessen 4 Zeilen × 32 Blöcke = **36 Cache-Lines pro
Iteration**. Das erhöht nicht die Bandbreite, sondern die Last pro Warp über die
Scoreboard-Kapazität hinaus: nach wenigen offenen Loads stallt der Warp, der
Rest serialisiert. **MLP-pro-Warp war der falsche Hebel.**

### Hypothesen für die verbleibende 2×-Lücke

Nicht bestätigt — dafür braucht es `-Xptxas -v` und ein Profil:

1. **Occupancy.** `__launch_bounds__(256, 4)` ergibt 32 Warps/SM. Der
   Stock-Kernel hat niedrige Registerlast und damit mehr Warps — und bei einem
   latenzgebundenen Bandbreitenkernel schlagen mehr Warps mehr MLP-pro-Warp.
2. **Der zusätzliche, serialisierte Quantisierungs-Launch.** Ich habe die
   Aktivierungsquantisierung aus dem Mat-Vec herausgezogen (≈401 MUL_MATs pro
   Token ⇒ 401 zusätzliche, abhängige Launches). Der Stock-Pfad quantisiert
   gefused in Shared Memory.
3. **`p40_prefetch_l2`** — 4 Prefetch-Instruktionen pro Block-Iteration, die
   die Demand-Loads ohnehin abdecken. Reiner Issue-Overhead.
4. **Registerdruck/Spills** aus dem Vier-Zeilen-Body unter dem 64-Register-Cap.

## 4. Korrigierte Zahlen

`tools/gpu_predict.py` rechnet jetzt mit kalibrierten Effizienzen:

| | alte Vorhersage | **korrigiert** | gemessen |
|---|---|---|---|
| P40, kurzer Kontext | 42.2 tok/s | **21.8** | **21.90** ✓ |
| P40, 32K | 36.5 | **18.9** | – |
| P40, 262K (Q8_0-KV) | 18.6 | **9.6** | – |
| V100 32 GB, kurzer Kontext | 109.8 | **62.4** | – |
| V100 32 GB, 262K | 48.5 | **~28** | – |
| MTP-Faktor (aus PR #205) | 1.338× | 1.338× (gemessen) | – |

Die **V100-Vorhersage war um 1.76× zu optimistisch**; `docs/V100_PREDICTION.md`
ist entsprechend korrigiert.

## 5. Was daraus folgt

* **Der Stock-Pfad bleibt der Standard.** Der Fork hat für PQ2_0 auf Pascal
  keine kartenspezifische Optimierung (`MMVQ_PARAMETERS_GENERIC`), aber der
  generische Pfad ist für diese Karte bereits nahe am praktischen Limit von
  ~158 GB/s.
* **Der nächste Schritt ist Messen, nicht Raten.** Ohne `-Xptxas -v` und ein
  Profil (Registerzahl, `stall`-Gründe, erreichte Occupancy) ist jede weitere
  Änderung blind. Die Reihenfolge wäre: Registerzahl prüfen → Occupancy
  hochsetzen → Quantisierung wieder in den Kernel fusionieren.
* **Die 45.5 % der P40 sind selbst die interessante Zahl.** Wenn dort noch
  etwas zu holen ist, dann über das Zugriffsmuster des Stock-Pfads — nicht über
  eine Umverteilung der Arbeit pro Warp.
* **Was unabhängig davon trägt:** die host-verifizierte Numerik (355 Checks),
  der GGUF-Pre-Flight-Check, das kalibrierte Vorhersagemodell, die
  Makro-Kollisionsprüfung und die Analyse in `docs/MMQ.md`/`docs/MTP.md`.
  Diese Teile sind durch die Messung nicht entwertet.

## 6. Reproduktion

```bash
M=models/Ternary-Bonsai-2-27B-PQ2_0.gguf
for v in 1 0; do echo "=== BONSAI_P40=$v ==="; \
  BONSAI_P40=$v ./build/bin/llama-bench -m $M -ngl 99 -fa on -p 0 -n 128; done
python3 tools/gpu_predict.py     # Effizienz-Kalibrierung + korrigierte Vorhersagen
```

Build `5ae2e16 (1)`, 1 CUDA-Device, `Tesla P40, compute capability 6.1`,
22905 MiB sichtbar.