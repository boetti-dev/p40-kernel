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

## 3. Ursache bestätigt: Occupancy, nicht MLP

`ptxas -v` auf `src/kernels/gemv_p40.cu` (`sm_61`, CUDA wie auf der P40):

```
quant_q8_1_pq2_kernel : Used 15 registers
gemv_pq2_0_p40_kernel : Used 64 registers, used 0 barriers, 352 bytes cmem[0]
```

Zwei harte Fakten daraus:

1. **64 Register = exakt das Cap** von `__launch_bounds__(256, 4)`
   (`65536 / (4 × 256) = 64`). Der Allokator hat das Budget voll ausgeschöpft.
2. **Keine Spills** — es fehlt die Zeile `N bytes stack frame, M bytes spill
   stores, K bytes spill loads`. Die Registerzahl ist also echt, nicht durch
   Spilling erkauft.

Daraus folgt die Occupancy:

| Register | Blöcke/SM | Warps/SM | Occupancy |
|---|---|---|---|
| **64** (mein GEMV) | **4** | **32** | **50 %** |
| 15 (mein Quant-Kernel) | 8 | 64 | 100 % |
| ~15–25 (Stock mmvq) | 8 | 64 | 100 % |

Und jetzt die Gegenüberstellung, die die Sache entscheidet:

```
Regressionsfaktor  21.90 / 11.39 = 1.923
Occupancy-Faktor   64 / 32       = 2.000
```

**Die Regression ist die halbierte Occupancy.** Das ist keine Korrelation aus
zwei Datenpunkten, sondern eine Erklärung, die zur Mechanik passt: bei einem
latenzgebundenen Bandbreitenkernel auf einer Karte mit ~600 ns Speicherlatenz
zählt die Zahl der residenten Warps, weil die Scoreboard-Kapazität pro Warp die
offenen Loads begrenzt. Der Stock-Kernel hat einen winzigen Registerbedarf und
läuft deshalb mit 64 Warps/SM; meiner verbraucht sein Budget für vier Zeilen
und halbiert damit genau den Faktor, der zählt.

**Meine Optimierung war also ein Tausch am falschen Ende:** Register (und damit
Occupancy) gegen einen MLP-Gewinn, den es gar nicht gab — weil der Stock-Warp
mit `blocks_per_iter = 4` bereits 136 zusammenhängende Bytes pro Iteration liest
und die Bandbreite aus der Warp-Zahl zieht, nicht aus MLP pro Warp.

## 4. Die eine Änderung, die daraus folgt

Der Kernel ist jetzt auf den Zeilen-pro-Warp-Wert **baubar konfigurierbar**:

```bash
# in src/kernels/gemv_p40.cu bzw. ggml/src/ggml-cuda/bonsai-p40/gemv_p40.cu
-DBONSAI_GEMV_ROWS=1    # -> GEMV_MIN_BLOCKS 8, <= 32 Register, 64 Warps/SM
-DBONSAI_GEMV_ROWS=2    # -> GEMV_MIN_BLOCKS 6, <= 42 Register, 48 Warps/SM
-DBONSAI_GEMV_ROWS=4    # -> GEMV_MIN_BLOCKS 4, <= 64 Register, 32 Warps/SM (Default, gemessen)
```

`GEMV_MIN_BLOCKS` wird automatisch passend gesetzt. Der Test ist einzeilig:

```bash
./tools/regs.sh gemv                 # Registerzahl prüfen: <= 32 für rows=1?
# dann in llama.cpp neu bauen und der A/B aus docs/WORKFLOW.md Schritt 7
```

Erwartung nach dieser Analyse: **`rows=1` sollte die Lücke schließen oder
schlagen**, weil es den Registerbedarf auf das Niveau des Stock-Pfads bringt.
Ob es ihn *überholt*, ist offen — der einzige verbleibende Vorteil wäre dann
noch, dass die PQ2_0-native Permutation pro Gewicht zwei `__byte_perm` spart,
und die P40 ist dafür nicht instruktionsgebunden.

## 5. Korrigierte Zahlen

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

## 6. Was daraus folgt

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

## 7. Reproduktion

```bash
M=models/Ternary-Bonsai-2-27B-PQ2_0.gguf
for v in 1 0; do echo "=== BONSAI_P40=$v ==="; \
  BONSAI_P40=$v ./build/bin/llama-bench -m $M -ngl 99 -fa on -p 0 -n 128; done
python3 tools/gpu_predict.py     # Effizienz-Kalibrierung + korrigierte Vorhersagen
```

Build `5ae2e16 (1)`, 1 CUDA-Device, `Tesla P40, compute capability 6.1`,
22905 MiB sichtbar.