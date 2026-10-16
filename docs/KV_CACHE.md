# Bonsai-P40 KV-Cache

Dies beschreibt den KV-Cache-Plan für Ternary-Bonsai-2-27B auf der P40.
Der Ist-Zustand ist in `src/kv_cache.h` (Geometrie + Mathe), `include/bonsai_kv.h`
(API) und `src/kernels/kv_p40.cu` (Kernel) implementiert.

## 1. Die entscheidende Struktur-Eigenschaft: Hybrid Attention

qwen35 ist **kein** reiner Transformer. Von den 64 Blöcken sind nur
**16 Full-Attention** (`(i+1) % 4 == 0`); die anderen **48 sind Gated DeltaNet**
mit einem *konstant großen* rekurrenten Zustand pro Head. Pro Token:

| | Full-Attention | Gated DeltaNet |
|---|---|---|
| Layer | 16 | 48 |
| pro Token | 4 kv-heads × 256 dim = 16384 K (+ 16384 V) | 48 heads × 128 × 128 (fix) |
| wächst mit ctx | ja | **nein** |
| Speicher @262k | siehe Tabelle | 3 MiB/Layer = **144 MiB konstant** |

Das ist der Grund, warum 262K auf 24 GB überhaupt geht: 48 der 64 Layer
wachsen überhaupt nicht, und der wachsende Teil ist nur 16 × 4 × 256 Elemente
pro Token.

## 2. Sizing auf der P40 (16 Layer, 4 kv-heads, head_dim 256, 262144 Positionen)

| Typ | Bytes/Token | Cache @262k | KV-Leserate | + 20.8 ms | tok/s |
|---|---|---|---|---|---|
| F16 / F16 | 65536 | **16.00 GiB** | 49.7 ms | 70.5 ms | ~7 |
| **Q8_0 / Q8_0** | 34816 | **8.50 GiB** | 26.4 ms | 47.2 ms | **~9.6** |
| Q8_0 / Q4_0 | 26624 | 6.50 GiB | 20.2 ms | 41.0 ms | ~11 |
| Q4_0 / Q4_0 | 18432 | 4.50 GiB | 14.0 ms | 34.8 ms | ~13 |

**F16 ist keine Option:** 7.21 GiB Gewichte + 16.00 GiB Cache = **23.21 GiB**
von 24 GiB — kein Platz für Aktivierungen, CUDA-Kontext, den 144-MiB-GDN-State
oder den Graph-Workspace. Q8_0/Q8_0 ist der Default: halbiert den KV-Traffic
gegenüber F16, und 8-bit K hält die Attention-Scores unverzerrt.

Wichtig zur Einordnung der KV-Leserate: sie **wächst mit dem Kontext** und
überholt den Gewichtsstrom (20.8 ms) erst irgendwo bei ~200K:

| Kontext | KV-Read | tok/s (Q8_0) |
|---|---|---|
| 4 096 | 0.41 ms | ~21 |
| 32 768 | 3.30 ms | ~19 |
| 262 144 | 26.4 ms | ~10 |

Bei kurzen Kontexten ist der Cache also irrelevant, bei 262K ist er der
teuerste Posten nach den Gewichten.

## 3. Layout

Head-major, K und V in **getrennten** Arrays, damit sie unterschiedlich
quantisiert werden können:

```
K[ ((layer * n_kv_head + kv_head) * capacity + pos) * (head_dim/32) + blk ]
```

Ein `(layer, kv_head)`-Paar ist über `pos` **ein zusammenhängender Lauf** von
`capacity * 8 * 34` Byte. Genau das macht den Decode-Read zu einem sauberen
DRAM-Stream auf der P40. Ein token-major Layout würde pro Position um
`n_kv_head * head_dim` Byte springen und die Requests verschwenden.

`block_q8_0` = 34 Byte (1 fp16-Skala + 32 int8). Damit reduziert sich der Score
auf **DP4A**: Byte `i` eines K-Blocks paart mit Byte `i` des Query-Blocks, ohne
jeden Shuffle — genau die Ausrichtung, die `dp4a` will.

### Lane-Mapping

4 Lanes pro Q8_0-Block, 8 Blöcke pro 256-weitem Head:

* Lane `l` besitzt Block `l >> 2` und die vier Dimensionen `4*(l & 3) .. +3`
* Score pro Lane = **ein** `dp4a`; Reduktion über 32 Lanes = 5 Shuffles
* jede 32-Byte-Block-Lesung ist ein vollständig coalesced Warp-Zugriff

## 4. Decode-Aufteilung (Split-K)

Bei 262K muss ein Query gegen 262144 Positionen laufen. Ein einzelner Block
schafft das nicht mit voller Bandbreite, deshalb **Split-K**: die 32 Splits
werden auf `(n_split, n_head)` Blöcke verteilt, jeder Split läuft als
Online-Softmax über seinen Positions-Chunk, und ein kleiner Combiner
verschmilzt die Teil-Ergebnisse mit Reskalierung:

```
acc = acc_a * exp(m_a - m) + acc_b * exp(m_b - m)
l   = l_a   * exp(m_a - m) + l_b   * exp(m_b - m)
m   = max(m_a, m_b)
```

`n_split == 1` degeneriert zum einfachen Online-Softmax. Leere Splits sind exakt
neutral, weil `m = -inf` und `exp(-inf - m) == 0`.

## 5. Append (Prefill und Decode)

`bonsai_kv_append`-Kernel: ein Warp pro `(token, kv-head)`-Zeile.

1. Lane `l` liest ihre 4 Floats `blk*32 + sub*4 .. +3`
2. `amax` über die 4er-Segmente per `shfl_xor(1,4)` + `shfl_xor(2,4)`
3. `d = amax/127`, Quantisierung, dann **ein 4-Byte-Store** pro Lane

Damit ist das Schreiben eines kompletten K/V-Tokens 16 Layer × 4 Heads = 64
Zeilen, also ein einzelner kleiner Grid-Launch pro Layer.

## 6. Gated DeltaNet State

Der Zustand ist `[48 layers][48 value heads][128][128]` **FP32** = 144 MiB. Er
wird zusammen mit dem Cache allokiert (`bonsai_kv_gdn_state()`), ist konstant
groß und wird bei `bonsai_kv_reset()` genullt. Bewusst FP32: Lesen+Schreiben
sind 0.87 ms/Token (~4% des Gewichtsstroms) — die Präzision nicht wert, halbiert
zu werden.

## 7. Speicherbudget (Q8_0/Q8_0, 262K)

| Posten | Größe |
|---|---|
| Gewichte (PQ2_0) | 7.21 GiB |
| KV-Cache K | 4.25 GiB |
| KV-Cache V | 4.25 GiB |
| GDN-State (48 Layer) | 0.14 GiB |
| Split-K-Workspace + Q8-Query | ~1 MiB |
| **Summe** | **~15.85 GiB** von 24 GiB |

Bleiben ~8 GiB für Aktivierungen, CUDA-Kontext und Compute-Buffer — komfortabel.

## 8. Was offen ist

* **Kein Paging/Eviction.** Gewollt: das Modell hat ein festes 262144-Fenster
  und kein Sliding Window. Füllt der Kontext, ist er voll.
* **Prefill-Attention** läuft noch über `bonsai_p40_attn` (dichter Puffer); der
  Cache-Prefill-Pfad schreibt nur. Für sehr lange Prompts würde man die
  Prefill-Attention direkt gegen den Cache rechnen und den dichten Puffer
  sparen — als nächster Schritt notiert.
* **Q4_0 V** (24.4 tok/s @262k) ist als Option vorgesehen, aber derzeit nicht
  implementiert; das Layout trennt K und V genau dafür.
* **Keine Messung auf echter P40** möglich (kein `nvcc`, keine GPU in dieser
  Umgebung). Alle Zahlen sind aus den Bytes × 346 GB/s abgeleitet und im
  Host-Test gegen die Sizing-Tabelle geprüft.