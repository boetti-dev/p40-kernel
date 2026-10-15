# P40-Optimierung: warum die Kernel so aussehen

Diese Datei begründet jede Tuning-Entscheidung mit den GP102-Daten und dem
konkreten Modell. Es gibt keine „generische" Optimierung — alle Zahlen unten
gelten nur für die Tesla P40 und Ternary-Bonsai-2-27B in PQ2_0.

## 1. Das Bandbreiten-Budget

PQ2_0 wiegt **7.21 GB** (2.13 bit/Gewicht über das gesamte Sprachmodell). Bei
Batch 1 muss für **jedes** Token der komplette Gewichtssatz einmal durch den
Speicher:

```
7.21e9 Byte / 346e9 Byte/s = 20.8 ms   ->   48.0 tok/s  (Hardware-Obergrenze)
```

Realistisch (85–90 % der Peak-Bandbreite) landet man bei **~41–44 tok/s**. Das
ist das Ziel; alles andere ist diesem Ziel untergeordnet.

Zum Vergleich die Rechenlast pro Token: ~24.35 G MAC im Backbone ≈ 48.7 GFLOP.
Bei ~12 TFLOPS FP32 sind das ~4 ms — **deutlich unter** den 20.8 ms. Der Decode
ist also eindeutig **bandbreiten-, nicht rechengebunden**. Genau deshalb ist der
Kern ein `dp4a`-Mat-Vec, der den Gewichtsstrom maximal effizient liest.

Aktivierungen: pro Token ~5 KB (5120 × 1 Byte). Das passt zusammen mit den
Zwischenpuffern dauerhaft in die **3 MB L2** — Aktivierungs-Traffic ist
vernachlässigbar.

## 2. GEMV (`gemv_pq2_0_p40_kernel`)

* **Vier Ausgabezeilen pro Warp.** 32 Lanes strizen über die `n_in/128`
  PQ2_0-Blöcke (`for b = lane; b < nblocks; b += 32`), und jede Lane hält
  **vier** Zeilen gleichzeitig. Die Aktivierungsbytes eines 128er-Blocks werden
  einmal geladen und von allen vier Zeilen benutzt; gleichzeitig hat der Warp
  vier unabhängige Gewichtsströme in Flight statt einem. Bei ~600 ns
  GDDR5-Latenz braucht ein Warp mehrere offene Cache-Lines: 30 SM × 24 Warps ×
  4 Lines × 128 B ≈ 3.7 MB in Flight, gegenüber den ~208 KB, die für 346 GB/s
  tatsächlich nötig sind. Ohne Shared Memory, ohne `__syncthreads`.
* **Der `+1`-Code-Bias wird gefaltet.** `value = code - 1`, also
  `Σ(code-1)·a = Σcode·a − Σa`. Das `−Σa` (ein int16-Load + Negate) passiert
  **einmal pro 32er-Chunk** und wird von allen vier Zeilen wiederverwendet,
  statt viermal pro Zeile.
* **Die Gewichtsskala `d` wird einmal pro Zeile und Block geladen**, außerhalb
  der Chunk-Schleife.
* **`dp4a` statt FP16.** Der Code-Wort-Extrakt `(w >> 2k) & 0x03030303` liefert
  vier 2-Bit-Codes als Bytes; die passenden Aktivierungs-Bytes liegen dank des
  transponierten Q8_1-Layouts an exakt derselben Stelle. Ein `dp4a` pro
  Code-Wort, 8 pro 128er-Block.
* **2-Byte-Zugriffe.** `block_pq2_0` ist nur 2-Byte-aligned (34-Byte-Stride).
  Ein `uint32`-Load wäre misaligned → deshalb zwei `uint16`-Loads pro Codewort,
  genau wie im PrismML-Fork.
* **`__launch_bounds__(256, 4)`.** 256 Threads, ≤ 64 Register → 32 Warps/SM.
  Das ist der Kompromiss, der die Vier-Zeilen-Register (4 Pointer, 4 Skalen,
  4 Akkumulatoren, 8 Aktivierungswörter, 8 Codewörter) spillfrei hält.
* **Aktivierungs-Quantisierung als eigener Kernel** (ein Warp pro 32er-Chunk):
  `amax` per Shuffle, `d = amax/127`, roher int16-Summenwert in `s`, Store in
  der transponierten Position `16*(lane>>4) + 4*(lane&3) + ((lane>>2)&3)`.
* **`dp4a` statt FP16.** Der Code-Wort-Extrakt `(w >> 2k) & 0x03030303` liefert
  vier 2-Bit-Codes als Bytes; die passenden Aktivierungs-Bytes liegen dank des
  transponierten Q8_1-Layouts an exakt derselben Stelle. Ein `dp4a` pro
  Code-Wort, 8 pro 128er-Block.
* **2-Byte-Zugriffe.** `block_pq2_0` ist nur 2-Byte-aligned (34-Byte-Stride).
  Ein `uint32`-Load wäre misaligned → deshalb zwei `uint16`-Loads pro Codewort,
  genau wie im PrismML-Fork.
* **`__launch_bounds__(256, 8)`.** 8 Blöcke × 256 Threads = 2048 Threads =
  **64 Warps/SM** (GP102-Maximum). Das erzwingt ≤ 32 Register/Thread und ist
  nötig, um die GDDR5-Latenz zu verstecken. Die GEMV-Innenschleife bleibt
  deutlich darunter.
* **Aktivierungs-Quantisierung als eigener Kernel** (ein Warp pro 32er-Chunk):
  `amax` per Shuffle, `d = amax/127`, roher int16-Summenwert in `s`, Store in
  der transponierten Position `16*(lane>>4) + 4*(lane&3) + ((lane>>2)&3)`.

Hinweis zur Lastverteilung: für `n_in = 5120` ist `nblocks = 40`, bei
Stride 32 machen 8 Lanes zwei Iterationen. Das ist bewusst einfach gehalten;
eine Aufteilung in zusammenhängende Blöcke pro Lane wäre der nächste
Tuning-Schritt.

### 2a. Warum die PTX/SASS-Ebene den tok/s-Wert *nicht* heben kann

Diese Rechnung ist der Grund, warum `src/p40_ptx.h` so klein ist. Pro Token:

| Größe | Wert |
|---|---|
| Gewichte, die bewegt werden müssen | 7.21 GB |
| Bandbreiten-Zeit (Peak) | 20.8 ms |
| MACs | 24.35 G |
| `dp4a`-Instruktionen (128 Gewichte je Warp-`dp4a`) | 5.2e7 |
| gesamte Warp-Instruktionen inkl. Index/Extrakt (~4×) | ~2e8 |
| Issue-Kapazität (30 SM × 4 Sched × 1.53 GHz) | 184e9 /s |
| **Issue-Zeit** | **~1.1 ms** |

Die Instruktionslast ist also **~5 %** des Bandbreiten-Budgets. Eine
SASS-Optimierung, die die Instruktionszahl halbiert, spart 0.5 ms von 20.8 ms.
Deshalb wird PTX nur dort eingesetzt, wo es den **Speicherpfad** oder die
**Occupancy** betrifft — nicht die Arithmetik:

| PTX | Wirkung | Datei |
|---|---|---|
| `ld.global.nc.u16` | Gewichte/Aktivierungen laufen über den Read-only-Pfad, statt dass nvcc die Aliasing-Freiheit erst beweisen muss | `p40_ldg_u16` |
| `dp4a.u32.s32` | macht die gemischte Vorzeichenlage explizit: Gewicht ist **unsigned** (Codes 0..3), Aktivierung **signed** | `p40_dp4a_us` |
| `prefetch.global.L2` | hält den Gewichtsstrom einen Block voraus | `p40_prefetch_l2` |

**Bewusst nicht** in PTX gemacht: `cvt.f32.f16` (nvcc emittiert es bereits für
`__half2float`), die `shfl.sync`-Reduktionen, und der Code-Wort-Extrakt.
Letzterer ist genau der Grund, warum das Aktivierungs-Layout überhaupt
transponiert wird: mit `dp4a` braucht man **kein** `prmt`/Byte-Shuffle pro
Gewicht. Nachprüfbar mit `tools/disasm.sh`.

## 2b. Warum es *keinen* Weight-Repack gibt

Naheliegend wäre, die 7.21 GB einmal beim Laden in ein 4-Byte-alignetes Layout
umzupacken (`d` getrennt, `qs` gepackt), damit man `ld.global.v4` (128 Bit)
benutzen kann. Das ist bewusst **nicht** gemacht:

* Weniger/w breitere Loads reduzieren nur die Instruktionszahl → laut Tabelle
  oben irrelevant.
* Die Ladelast ist nur *scheinbar* misaligned: eine Lane liest 8 Byte an
  `34·b + 2 + 8·iqs`; über die 4 `iqs`-Iterationen werden alle 32 Codebytes
  jedes Blocks gelesen, und die 128-Byte-Linien werden mit `-dlcm=ca` im L1
  zwischen den Iterationen wiederverwendet. Die DRAM-Effizienz ist damit
  bereits ~100 %.
* Ein Repack kostet 7.21 GB zusätzlichen Speicher und eine Vollkopie, ohne
  belegbaren Nutzen.

## 3. GEMM / Prefill (`gemm_pq2_0_p40_kernel`)

Prefill ist rechengebunden (bei Batch 512: 25 TFLOP → ~2 s auf der P40, das ist
die Physik der Karte ohne Tensor Cores). Der Kernel ist ein klassisches
DP4A-Tile-GEMM:

* Blockkachel **256 Zeilen × 16 Token**, Warp = 32 Zeilen (eine pro Lane).
* Der 16-Spalten-Aktivierungs-Slab für den aktuellen k-Block wird **einmal** in
  2 KB Shared Memory gestaged und von allen Lanes gelesen → die Gewichte
  dominieren den DRAM-Verkehr und werden pro Kachel genau einmal gelesen.
* `k`-Schritt = ein PQ2_0-Block = 4 Q8_1-Chunks; pro Lane 8 Code-Wörter, 16
  Akkumulatoren, `dp4a` mit den gestagten Aktivierungen.

## 4. Elementwise / Hadamard / RMSNorm

Reine Streaming-Kernel, mit genug Warps um die GDDR5 zu sättigen und
128-Bit-Vektorzugriffen. Die Hadamard-Rotation (Block 1024, normalisiert) läuft
im Shared Memory; die Signum-Vektoren aus den `prism.hadamard.*`-Metadaten
werden vorangestellt.

## 5. Gated DeltaNet und Attention

Diese Pfade sind **neben** dem Gewichtsstrom praktisch kostenlos:

* GDN-Zustand: 48 Value-Heads × 128 × 128 × 4 B = **3 MB** pro Token lesen +
  schreiben → ~18 µs bei 346 GB/s. Der Kernel tiled die Zustands**spalten**
  (unabhängig!), ein Block pro (Head, 32 Spalten) mit 16 KB Shared Memory.
* Attention: nur **16 von 64** Layern sind voll; 24 Q-Heads / 4 KV-Heads,
  Head-Dim 256. Online-Softmax (Flash-Stil), ein Warp pro (Head, Query), jede
  Lane besitzt 8 Head-Dimensionen. Alles FP32.

## 6. Was Pascal erzwingt (Zusammenfassung)

| Eigenschaft | Konsequenz im Code |
|---|---|
| FP16-ALU 1/64 Rate | FP16 nur als Skalen-Speicherformat, nie als Rechentyp |
| `dp4a` ab `sm_61` | gesamte ternäre Mat-Mul über Int8-DP4A |
| keine Tensor Cores | kein `mma`/`wgmma`-Pfad |
| 64 KB Register/SM | `__launch_bounds__` + ≤ 32 Reg für 64 Warps/SM |
| 3 MB L2 | Aktivierungen dauerhaft L2-resident, kein Prefetch nötig |
| 2-Byte-Alignment der PQ2_0-Blöcke | zwei `uint16`-Loads statt `uint32` |
