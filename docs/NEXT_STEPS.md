# Was noch geht — und was nicht

Kurzantwort auf „also keine weitere Optimierung mehr möglich?": **Mein Ansatz
ist erledigt, aber es sind noch drei Wege offen — einer davon kostet null
Codezeilen und bringt voraussichtlich ~10 %.**

## Erledigt: mein eigener Kernel

Belegt durch zwei gemessene Konfigurationen und eine quantifizierte Ursache:

| | tok/s | % der Peak | warum |
|---|---|---|---|
| rows=4 | 11.39 | 23.7 % | 64 Register → 32 Warps/SM (Occupancy) |
| rows=1 | 17.34 | 36.0 % | 32 Register → 64 Warps/SM |
| **Stock** | **21.91** | **45.5 %** | fusionierte Quantisierung + Shared-Staging |

Die Occupancy erklärte 57 % der Lücke; die restlichen 43 % sind architektonisch
(kein Shared-Memory-Staging → **4.24× so viel L2-Leseverkehr wie
Gewichtsverkehr**). Um das zu beheben, müsste der Kernel auf die Stock-Struktur
umbauen — und dann bleibt als einziger Unterschied die `__byte_perm`-Ersparnis,
die auf einer nicht instruktionsgebundenen Karte wertlos ist.

**Fazit: nicht weiterverfolgen.** Details in `docs/MEASURED.md`.

---

## Weg 1 — PTQ1_0 statt PQ2_0 (kostet nichts, ~10 %)

Die Model-Card-Tabelle zeigt ein klares Muster:

| Karte | PQ2_0 | PTQ1_0 | Faktor | Charakter der Karte |
|---|---|---|---|---|
| L4 | 29.8 | **32.1** | **1.077×** | memory-bound (71 % der Peak) |
| L40S | 74.4 | **81.8** | **1.099×** | memory-bound |
| RTX 6000 Ada | 82.8 | **90.4** | **1.092×** | memory-bound |
| RTX 4090 | 81.2 | **91.1** | **1.122×** | memory-bound |
| RTX 5090 | 129.9 | 120.5 | 0.928× | instruction-bound |
| RTX PRO 6000 | 124.8 | 117.9 | 0.945× | instruction-bound |
| A100 SXM | 73.9 | 54.7 | 0.740× | instruction-bound |
| H100 SXM | 113.9 | 86.9 | 0.763× | instruction-bound |

PTQ1_0 hat **17 % weniger Gewichtsbytes** (5.95 statt 7.21 GB) und gewinnt genau
auf den Karten, wo Speicher der Engpass ist. Die P40 erreicht 45.5 % der
Peak-Bandbreite und verbraucht nur **~4 % ihrer Issue-Kapazität** — sie ist also
**memory-bound**, wie L4/L40S/4090.

**Vorhersage: 21.91 × ~1.10 ≈ 24.1 tok/s** (Spanne 23.6–24.6), rein durch die
Wahl der anderen Datei. Null Codeänderung.

```bash
huggingface-cli download prism-ml/Ternary-Bonsai-2-27B-gguf \
    Ternary-Bonsai-2-27B-PTQ1_0.gguf --local-dir models

M=models/Ternary-Bonsai-2-27B-PTQ1_0.gguf
./build/bin/llama-bench -m $M -ngl 99 -fa on -p 0 -n 128    # Decode
./build/bin/llama-bench -m $M -ngl 99 -fa on -p 512 -n 0    # Prefill
```

**Einschränkung, geprüft:** `mmq-config-pascal.cuh` enthält **keinen
`GGML_TYPE_PTQ1_0`-Eintrag** (nur PQ2_0, Q1_0, Q2_0, Q4_0, Q4_1, Q5, Q6, Q8,
IQ*, MXFP4, NVFP4). Für den **Decode** ist PTQ1_0 dagegen vorhanden und sogar
speziell optimiert (`vec_dot_ptq1_0_q8_1`, `VDR_PTQ1_0_Q8_1_MMVQ = 4`,
`ptq1_0_pt_block_dot`, SoA-Layout). Erwartung: **Decode gewinnt, Prefill fällt
möglicherweise auf einen langsameren Pfad zurück** — beide Tests oben laufen
lassen und getrennt bewerten.

### Kostet Weg 1 Genauigkeit? **Nein.**

Das ist die naheliegende Sorge und die Antwort ist belegt: **PTQ1_0 ist
verlustfrei gegenüber PQ2_0.** Beide speichern *dieselben* ternären g128-Werte
{−1, 0, +1} mit *derselben* einen FP16-Skala pro 128 Gewichte. PTQ1_0 packt sie
nur dichter.

### Die Rechnung

| | Werte pro Block | Skalen pro Block | Bytes pro Block | bpw |
|---|---|---|---|---|
| `block_pq2_0` | 128 | 1 × fp16 | 2 + 32 = **34** | 2.125 |
| `block_ptq1_0` | 128 | 1 × fp16 | 24 + 2 + 2 = **28** | 1.750 |

`block_ptq1_0` = `qs[24]` (5 Trits pro Byte → 120 Werte) + `qh[2]` (4 Trits pro
Byte → 8 Werte) + `d`. **3⁵ = 243 ≤ 256**, deshalb passen 5 Trits verlustfrei in
ein Byte. Die 17 % weniger Bytes sind reine Packdichte, keine Präzision.

Der Fork sagt es in seinem eigenen Quelltext (`ggml-common.h`):

```
// PTQ1_0: ... 1.75 bpw vs PQ2_0's 2.125, and lossless for checkpoints that
// are already ternary at group 128
```

Und der Model-Card beschreibt den Unterschied **nur** als Durchsatz-Trade:
„PQ2_0 stores each trit in a 2-bit slot, **trading footprint for cheaper
unpacking**" — von Genauigkeit ist nirgends die Rede.

### Empirisch geprüft: PQ2_0 nutzt seinen vierten Zustand nicht

PQ2_0 hat 4 Codes (0,1,2,3 → Werte −1, 0, +1, **+2**). PTQ1_0 kann den vierten
nicht darstellen. Wenn Code 3 nie vorkommt, trägt der viere Zustand **keine
Information** — dann ist der Mehrverbrauch von PQ2_0 reine Verschwendung.

Ich habe die ausgelieferte `Ternary-Bonsai-2-27B-PQ2_0.gguf` gescannt (nur die
Code-Bytes, Skalen übersprungen; Block-Offsets aus dem Header, die **relativ**
zum Datenanfang sind):

| Tensor | Codes | Code == 3 | Datei-Offset |
|---|---|---|---|
| `output.weight` | 1 271 398 400 | **0** | 11 120 992 |
| `blk.21.attn_qkv.weight` | 52 428 800 | **0** | 2 835 173 728 |
| `blk.42.ffn_up.weight` | 89 128 960 | **0** | 5 036 675 424 |
| `blk.63.ffn_up.weight` | 89 128 960 | **0** | 7 182 473 568 |
| **Summe** | **1 502 085 120** | **0** | (über die ganze Datei verteilt) |

**1,50 Milliarden ternäre Codes, null Mal Code 3.** Wäre auch nur 0,01 % der
Werte +2, sähe man ~150 000 Treffer. Der vierte Zustand ist unbenutzt, also ist
PTQ1_0 für dieses Checkpoint verlustfrei.

Nachvollziehbar: Header parsen, Tensor-Offsets relativ zum Datenanfang
(`data_start = align32(header_ende)`) auflösen, dann in jedem 34-Byte-Block die
Bytes `2..33` auf ein 2-Bit-Feld gleich `0b11` prüfen.

### Wo die Genauigkeit tatsächlich liegt

Der verlustbehaftete Schritt ist die **Ternarisierung + Hadamard-Faltung** — und
der ist in beiden Packs identisch. Zur Einordnung (Model-Card, 14-Benchmark-
Durchschnitt):

| Variante | Größe | Durchschnitt |
|---|---|---|
| Qwen3.8-27B FP16 | 54.66 GB | 86.32 |
| Qwen3.8-27B UD-Q4_K_XL | 17.56 GB | 85.18 |
| **Bonsai 2 27B (ternär, g128)** | **5.95 / 7.21 GB** | **84.78** |
| Qwen3.8-27B IQ2_XXS | 7.27 GB | 72.59 |

Also: die ternäre Darstellung kostet **1,5 Punkte** gegenüber FP16 (86.32 →
84.78) und ist **12 Punkte besser** als die konventionelle IQ2_XXS. Für mehr
Genauigkeit müsste man die Darstellung wechseln (z. B. UD-Q4_K_XL: +0.4 Punkte
für **3× die Größe**), nicht die Packung.

Auch die `F16`-Datei des Repos hilft nicht: sie ist dieselben ternären Gewichte,
nur in FP16 entpackt — also numerisch dasselbe Modell, nur 54 GB groß.

**Fazit: PTQ1_0 ist gratis.** Die 17 % sind Packung, nicht Präzision.

Die Zahl „45.5 % der Peak" ist mehrdeutig: sie kann heißen, dass der Kernel
schlecht ist, oder dass die P40 schlicht langsamer ist als ihr Datenblatt.

```bash
nvcc -O3 -std=c++17 -arch=sm_61 tools/bwtest.cu -o build/bwtest
./build/bwtest            # 1 GiB Puffer, reiner Read + Copy
```

Das Skript zieht die Schlussfolgerung selbst:

* **≤ 1.25× der Stock-Rate** → die Karte ist die Grenze, der Stock-Kernel ist
  fertig, **Weg 1 ist der einzige Hebel**.
* **> 1.25×** → es gibt Kernel-Headroom → **Weg 3**.

## Weg 2 — GEMESSEN: das Zugriffsmuster ist der Flaschenhals

`./build/bwtest` auf einer echten P40 (das Kontrollkernel läuft mit 2 Registern
bei voller Occupancy, liest **nur Gewichte**, keine Aktivierungen, keine
Compute):

| Test | GB/s | % der 346-GB/s-Peak |
|---|---|---|
| alignierter `uint4`-Stream | **251.8** | 72.8 % |
| **PQ2_0-Adressierung, reiner Read** | **183.6** | **53.1 %** |
| Stock-Decode (erreicht) | 157.5 | 45.5 % |

Damit zerfällt die Lücke in **zwei getrennte, quantifizierte Verluste**:

```
Pattern-Strafe   251.8 / 183.6 = 1.371x  (+37.1 %)   allein durch die Adressierung
Stock vs Muster  183.6 / 157.5 = 1.166x  (+16.6 %)   der Stock holt 85.8 % seines Musters
Gesamtpotenzial  251.8 / 157.5 = 1.599x
```

Die 34-Byte-Blöcke (2-Byte-aligniert, 4-Byte-Worte) kosten **37 % Bandbreite,
ohne jede Rechenarbeit**. Das ist kein Occupancy- und kein Issue-Effekt — es ist
reine Adressierung.

### Was das für die erwartete Ausbeute heißt

| Szenario | effektiv | PQ2_0 | PTQ1_0 |
|---|---|---|---|
| Stock heute | 157.5 GB/s | 21.9 tok/s | 26.5 tok/s |
| nur Muster-Limit (ohne Repack) | 183.6 | 25.5 | 30.9 |
| Repack, konservativ (85.8 % des neuen Limits) | 216.0 | **30.0** (1.37×) | **36.3** (1.66×) |
| Repack, ideal | 251.8 | **35.0** (1.60×) | **42.3** (1.93×) |

**Realistische Spanne: 27–33 tok/s mit PQ2_0, 32–37 tok/s mit PTQ1_0** — also
**1.2×–1.7×** gegenüber heute. Die Spanne ist bewusst breit: die *Speicherseite*
ist gemessen, ob ein Kernel das neue Limit auch erreicht, ist es nicht.

### Der Repack — vier Entwurfs-Constraints, alle gemessen

Das ist jetzt kein Raten mehr, sondern das Ergebnis der Messungen:

1. **Gewichtsladungen müssen ≥ 8 Byte aligniert und über den Warp
   zusammenhängend sein.** Die 37 % oben sind der Preis für 34-Byte-Blöcke mit
   4-Byte-Worten. Also: `d` getrennt ablegen, Code-Bytes in ein aligniertes
   Layout packen (z. B. SoA nach Chunk, damit ein Warp 256 zusammenhängende
   Bytes pro Instruktion liest).
2. **Aktivierungen müssen aus Shared Memory kommen.** Mein Kernel las sie pro
   Zeile aus L2 — **4.24× das Gewichtsvolumen** (gemessen, `docs/MEASURED.md`).
   Der Stock staged sie einmal pro Block.
3. **Register ≤ 32 für volle Occupancy.** Gemessen: rows=4 (64 Register, 50 %
   Occupancy) → 11.39 tok/s; rows=1 (32 Register, 100 %) → 17.34 tok/s. Faktor
   1.52× allein daraus.
4. **Quantisierung fusioniert.** Mein separater Launch kostete ~400 abhängige
   Launches pro Token.

Und der entscheidende Punkt: **ein Repack kann nur ein eigener Loader machen.**
Der Fork muss die GGUF-Tensoren *in place* konsumieren; mein Engine liest die
Datei selbst und kann die 7.21 GB (bzw. 5.95 GB mit PTQ1_0) **einmal beim Laden**
in ein aligniertes Layout umpacken. Kosten: eine Vollkopie beim Start. Nutzen:
bis zu 1.6× pro Token.

Das ist genau die Position, die mein erster Ansatz hatte — und die ich mit der
(durch `bwtest` widerlegten) Begründung „breitere Loads sparen nur
Instruktionen" weggeworfen habe. Siehe `docs/P40_OPTIMIZATION.md` Abschnitt 2b.

## Weg 3 — ergänzend: eine Pascal-MMVQ-Tabelle

Der Fork hat `MMVQ_PARAMETERS_TURING` und `MMVQ_PARAMETERS_GB10`, und
`get_mmvq_parameters_table()` fällt für Pascal auf **`MMVQ_PARAMETERS_GENERIC`**
zurück. Für PQ2_0 bei `ncols_dst == 1` gibt es auf Pascal also **keine
kartenspezifische Parametrisierung** — genau die Lücke, die ich früher
identifiziert habe.

Das ist aber eine **Ergänzung am Stock-Kernel**, kein Ersatz: eine
`MMVQ_PARAMETERS_PASCAL`-Tabelle (nwarps / rows_per_cuda_block / blocks_per_iter
für `ncols_dst == 1`) plus `__ldg` auf den Gewichtsladungen. Dafür braucht es
aber **ein Profil** (welche Stall-Gründe dominieren), sonst ist es Raten — und
genau das war der Fehler bei meinem Kernel.

## Weg 4 — Prefill ist eine eigene Achse

Prefill ist **rechengebunden** (12.5 TMAC bei Batch 512) und läuft über den
Fork-MMQ, den ich nicht angefasst habe. Die P40 hat `dp4a` mit ~23.5 TMAC/s
Peak; der Wirkungsgrad dort ist **nie gemessen** worden. Das ist eine
unabhängige Frage mit eigenem Potenzial — und `pp512` fehlt in der Model-Card
komplett für Pascal-Karten.

## Was sich *nicht* lohnt

* **Meinen Kernel umbauen** — siehe oben, strukturell aussichtslos.
* **`-DBONSAI_GEMV_ROWS=2`** — 40 Register, 75 % Occupancy; liegt zwischen den
  beiden gemessenen Punkten, also zwischen 17.34 und 21.91, nicht darüber.
* **`--spec-type ngram-*`** — für dieses Modell wirkungslos (offen, #203).
* **MTP** — vier konkrete Gaps, und der MTP-Layer ist Q8_0 (`docs/MTP.md`).
  Der Faktor 1.338× ist gemessen, aber nicht auf `sm_61`.
* **MMQ durch meinen GEMM ersetzen** — der Fork-MMQ ist Pascal-getunt
  (`docs/MMQ.md`).

## Reihenfolge

1. **Repack auf 8/16-Byte-Alignment** (Weg 2) — gemessen: das Muster kostet 37 %, Potenzial 1.37x bis 1.60x. Der einzige Hebel mit dieser Größe.
   Muster bewiesen und Weg 2 wird konkret. (read-only ist bereits gemessen:
   251.8 GB/s = 1.60x.)
2. **PTQ1_0 benchmarken** (Weg 1) — null Aufwand, ~10 % erwartet, und
   verlustfrei (siehe oben).
3. **Repack auf 16-Byte-Alignment** (Weg 2) — der einzige Hebel mit bis zu
   1.6x, und nur mit eigenem Loader möglich.
4. **Profil** ziehen, daraus ggf. eine Pascal-MMVQ-Tabelle (Weg 3).
5. Parallel: **`pp512` messen** (Weg 4) — unabhängige Achse.
