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

## 4. Ein Bug in meinem eigenen Patch (gefunden durch `regs.sh`)

Der erste `regs.sh`-Lauf auf einer P40 zeigte:

```
gemv_pq2_0_p40_kernel : 152 bytes stack frame, 184 bytes spill stores, 184 bytes spill loads
                        Used 32 registers
```

Das war **nicht** die gemessene Konfiguration — es war ein Fehler, den ich mit
dem `BONSAI_GEMV_ROWS`-Schalter eingebaut hatte:

```cpp
constexpr int GEMV_ROWS_PER_WARP = BONSAI_GEMV_ROWS;   // constexpr!
...
#if GEMV_ROWS_PER_WARP >= 4        // <-- Praeprozessor sieht KEIN constexpr
```

Der Präprozessor kann `constexpr int` nicht lesen; er sieht den Bezeichner als
`0`, also wurde `#if 0 >= 4` falsch und **jede** Konfiguration landete im
`#else`-Zweig → `GEMV_MIN_BLOCKS 8` → 32-Register-Cap → **Spills im
Vier-Zeilen-Body**. Korrigiert auf `#if BONSAI_GEMV_ROWS >= 4` (das Makro).

**Lehre für die Verifikation:** mein erster Test dieser Logik war wertlos, weil
ich im Testkern `GEMV_ROWS_PER_WARP` selbst als Makro definiert hatte — ich
habe meinen Test getestet, nicht den Code. Der Test extrahiert jetzt den Block
**wörtlich** aus `src/kernels/gemv_p40.cu` und präprozessiert ihn:

```
  none                 -> MINBLOCKS=4 ROWS=4
  -DBONSAI_GEMV_ROWS=1 -> MINBLOCKS=8 ROWS=1
  -DBONSAI_GEMV_ROWS=2 -> MINBLOCKS=6 ROWS=2
  -DBONSAI_GEMV_ROWS=4 -> MINBLOCKS=4 ROWS=4
```

Und `tools/regs.sh` prüft diese Zuordnung jetzt **bei jedem Lauf** gegen den
echten Präprozessor und bricht ab, wenn Skript und Quelle auseinanderlaufen.

## 5. Der Sweep auf der P40 (aus `./tools/regs.sh sweep`)

```
config-block check (script expectation vs. real preprocessor):
  OK: rows=1->8, rows=2->6, rows=4->4 min-blocks

  rows=1  min_blocks=8  regs=32   spill_stores=0  stack=0  -> 64 warps/SM
  rows=2  min_blocks=6  regs=40   spill_stores=0  stack=0  -> 48 warps/SM
  rows=4  min_blocks=4  regs=64   spill_stores=0  stack=0  -> 32 warps/SM
```

Damit ist alles bestätigt, was zu prüfen war:

* **Der Registerbedarf skaliert exakt mit der Zeilenzahl** (32 / 40 / 64) — die
  vier Zeilen sind der Grund für die halbierte Occupancy.
* **Keine Spills in irgendeiner Konfiguration.** Der Vier-Zeilen-Body passt in 64
  Register; er ist nur teuer, nicht kaputt.
* **`rows=1` passt in genau 32 Register ohne Spills → 64 Warps/SM → 100 %
  Occupancy.** Das ist die einzige Konfiguration, die den Registerbedarf auf das
  Niveau des Stock-Pfads bringt, und damit der Test, der die Frage entscheidet.

### Der A/B-Test für `rows=1`

```bash
cd llama.cpp
M=../p40-kernel/models/Ternary-Bonsai-2-27B-PQ2_0.gguf

# rows=1 -> 32 Register, 64 Warps/SM.  Ueber die CMake-Option, kein sed noetig.
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61 \
      -DCMAKE_BUILD_TYPE=Release -DBONSAI_GEMV_ROWS=1
cmake --build build --config Release -j"$(nproc)"

BONSAI_P40=1 ./build/bin/llama-bench -m $M -ngl 99 -fa on -p 0 -n 128   # mein Kernel
BONSAI_P40=0 ./build/bin/llama-bench -m $M -ngl 99 -fa on -p 0 -n 128   # Stock
```

Erwartung nach der Occupancy-Analyse: **`rows=1` schließt die Lücke**
(≈ 21.9 tok/s). Ob es den Stock-*überholt*, bleibt offen: der einzige dann
verbleibende Unterschied ist, dass die PQ2_0-native Permutation pro Gewicht zwei
`__byte_perm` spart — und die P40 ist dafür nicht instruktionsgebunden. Ich
rechne mit **Gleichstand, nicht mit Gewinn**.

## 6. Ergebnis des `rows=1`-Tests: Occupancy war die Haelfte, Architektur der Rest

```
=== BONSAI_P40=1 (rows=1) ===   tg128  17.34 +/- 0.00
=== BONSAI_P40=0 (stock) ===   tg128  21.91 +/- 0.01
```

| Konfiguration | tok/s | effektiv | % der Peak | Occupancy |
|---|---|---|---|---|
| rows=4 | 11.39 | 81.9 GB/s | 23.7 % | 50 % |
| **rows=1** | **17.34** | **124.7 GB/s** | **36.0 %** | **100 %** |
| Stock | **21.91** | 157.5 GB/s | 45.5 % | 100 % |

* **Die Occupancy-Hypothese ist bestaetigt.** rows=4 -> rows=1 bringt **1.52x**,
  bei gleicher Occupancy (64 Warps/SM). Die Occupancy erklaert damit **57 %**
  der urspruenglichen Luecke.
* **Es bleibt 1.26x zum Stock** (21.91 / 17.34), bei *gleicher* Occupancy. Der
  Rest ist also nicht Tuning, sondern Architektur.

### Was die Restluecke erklaert: Aktivierungs-Traffic

Mein Kernel hat **kein Shared-Memory-Staging** der Aktivierung. Jeder Warp
liest sie pro Zeile neu aus L2. Bei 4 Zeilen/Warp war das noch 1x pro 4 Zeilen;
bei rows=1 ist es **1x pro Zeile**:

| Matvec | Aktivierungen aus L2 | Gewichte aus DRAM | Faktor |
|---|---|---|---|
| 5120 -> 17408 | **100.27 MB** | 23.67 MB | **4.24x** |
| 5120 ->  6144 | 35.39 MB | 8.36 MB | 4.24x |
| 17408 -> 5120 | 100.27 MB | 23.67 MB | 4.24x |

Der Kernel erzeugt also **4.24x so viel L2-Leseverkehr wie DRAM-Gewichtsverkehr**.
L2 ist schneller als DRAM, aber diese Menge konkurriert mit dem Gewichtsstrom um
Bandbreite und MMU, und sie ist voellig vermeidbar: der Stock-Pfad quantisiert
die Aktivierung **einmal pro Block in Shared Memory**, und alle Zeilen des
Blocks lesen sie von dort.

Dazu kommt der **zusaetzliche, serialisierte Quantisierungs-Launch**: mein Pfad
macht zwei abhaengige Launches pro MUL_MAT (~400 pro Token), der Stock einen
gefusten. Die Mehrzeit von 12.03 ms/Token entspricht 30 us pro MUL_MAT — zu viel
fuer einen reinen Launch (3-8 us), aber konsistent mit Launch *plus*
Abhaengigkeitsstall *plus* dem L2-Verkehr.

### Verdikt

**Der Ansatz kann den Stock-Pfad auf dieser Karte nicht schlagen.** Der Grund ist
strukturell, nicht Feintuning:

* Was der Stock-Pfad richtig macht — Quantisierung fusioniert, Aktivierung in
  Shared Memory, damit alle Zeilen sie wiederverwenden — ist genau das, was ihn
  auf 45.5 % der Peak-Bandbreite bringt.
* Was mein Entwurf hinzufuegt — die vorpermutierte Aktivierung, die pro Gewicht
  zwei `__byte_perm` spart — ist ALU-Ersparnis auf einer Karte, die **nicht**
  instruktionsgebunden ist.
* Um die Luecke zu schliessen, muesste ich den Kernel auf die Stock-Struktur
  umbauen (fusionierte Quantisierung + Shared-Staging). Dann bleibt als einziger
  Unterschied die `__byte_perm`-Ersparnis — und die ist auf der P40 wertlos.

**Empfehlung: `BONSAI_P40` bleibt aus (Default).** Der Patch bleibt trotzdem im
Repo, weil die 9 Zeilen Hook harmlos sind und die Messung reproduzierbar ist;
und weil `-DBONSAI_GEMV_ROWS` ein sauberer Nachweis der Occupancy-Abhaengigkeit
ist.

Der Default wurde auf **rows=1** gesetzt: wer den Pfad einschaltet, bekommt die
beste Variante dieses Kernels (17.34 statt 11.39), auch wenn sie langsamer ist
als der Stock.

## 7. Die Aenderung, die daraus folgt

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

## 8. Korrigierte Zahlen

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

## 9. Was daraus folgt

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

## 10. Reproduktion

```bash
M=models/Ternary-Bonsai-2-27B-PQ2_0.gguf
for v in 1 0; do echo "=== BONSAI_P40=$v ==="; \
  BONSAI_P40=$v ./build/bin/llama-bench -m $M -ngl 99 -fa on -p 0 -n 128; done
python3 tools/gpu_predict.py     # Effizienz-Kalibrierung + korrigierte Vorhersagen
```

Build `5ae2e16 (1)`, 1 CUDA-Device, `Tesla P40, compute capability 6.1`,
22905 MiB sichtbar.