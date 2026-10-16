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

## Weg 2 — die Decke der Karte messen (entscheidet über Weg 3)

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

## Weg 3 — nur falls Weg 2 Headroom zeigt: eine Pascal-MMVQ-Tabelle

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

1. **PTQ1_0 benchmarken** (Weg 1) — null Aufwand, ~10 % erwartet.
2. **`bwtest` laufen lassen** (Weg 2) — beantwortet, ob überhaupt noch
   Kernel-Headroom existiert.
3. Nur bei Headroom: **Profil** ziehen und daraus eine Pascal-MMVQ-Tabelle
   (Weg 3).
4. Parallel: **`pp512` messen** (Weg 4) — völlig unabhängige Achse.
