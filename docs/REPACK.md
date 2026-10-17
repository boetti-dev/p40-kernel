# Repack: das alignierte PQ2_0-Layout

Der Fix für den in `docs/NEXT_STEPS.md` gemessenen Flaschenhals: **die
34-Byte-Blockadressierung kostet 37 % Bandbreite** (251.8 GB/s aligniert vs.
183.6 GB/s im PQ2_0-Muster, bei reiner Leseoperation ohne Compute).

## Layout (verifiziert)

`src/pq2_0_repack.h`, geprüft durch `tests/test_p40.cpp` (`[repack] ...`).

```
eine Zeile (nblocks = n_in/128):
  offset 0            : d[nblocks]        fp16-Skalen, 16-Byte-aligniert
  offset 2*nblocks    : codes[32*nblocks]

  row_stride = 34 * nblocks      <-- exakt die Originalgröße, kein Zusatzspeicher
```

Die Code-Bytes liegen als **4 Planes** à `nblocks*8` Byte: Plane `p` enthält
Chunk `p` jedes Blocks. Flachgelegt ist das ein **zusammenhängender Lauf von
`4*nblocks` Units à 8 Byte**, und Unit `u` ist

```
Plane p = u / nblocks        Block b = u % nblocks
```

Die Adresse einer Unit ist damit schlicht `codes + u*8` — **zusammenhängend und
8-Byte-aligniert**. Ein Warp liest 32 Units = **256 zusammenhängende Byte pro
Instruktion**. Die Bit-Reihenfolge *innerhalb* einer Unit bleibt die originale
PQ2_0-Ordnung, deshalb gilt die bestehende `(w >> 2k) & 0x03030303`-Extraktion
unverändert.

Die Ausrichtung hält für **jede** Zeilenbreite dieses Modells, weil `nblocks`
immer durch 8 teilbar ist (5120/128=40, 17408/128=136, 6144/128=48,
10240/128=80, 12288/128=96, 1024/128=8). Damit ist `2*nblocks` 16-Byte-aligniert
und `32*nblocks` erhält diese Ausrichtung.

## Was verifiziert ist (2065 Checks, host-seitig)

| Prüfung | Ergebnis |
|---|---|
| `row_stride == 34*nblocks` | **größenneutral** |
| `d`- und Codes-Offset 16-Byte-aligniert | ✓ für alle 6 Breiten |
| alle `4*nblocks` Unit-Adressen 8-Byte-aligniert | ✓ |
| `unit → (plane, block)` Bijektion | ✓ |
| Repack → Unpack | **bitgenau** |
| **`repack_row_dot` == `pq2_0_row_dot_perm`** | **bitgenau** (Integer-Pfad) |
| jede Aktivierungs-Chunk genau 1× konsumiert | ✓ |
| Kernel-Schleife (R=1/2/4): Unit-Abdeckung, Zeilen-Clamping, Dot | ✓ |

Der letzte Punkt ist eine Host-Simulation der Kernel-Schleifenstruktur
(`[repack] R-rows-per-warp loop simulation`) — sie hat beim Schreiben bereits
einen echten Fehler in meiner eigenen Simulation gefunden (R Zeilen in **einen**
Akkumulator summiert statt in `acc[r]`), was genau der Zweck dieses Vorgehens ist.

## Der Kernel

`src/kernels/gemv_repack_p40.cu` — erfüllt alle vier gemessenen Constraints:

1. **8-Byte-alignierte, warp-zusammenhängende Gewichtsladungen** — zwei
   alignierte `uint32`-Loads pro Unit statt vier `uint16`.
2. **Aktivierungen aus Shared Memory** — kooperativ einmal pro Block
   quantisiert (8 Lanes pro Chunk, bankkonfliktfrei).
3. **Registerbudget als Schalter** — `RP_ROWS` bestimmt die Occupancy.
4. **Quantisierung fusioniert** — kein zusätzlicher Launch.

## Die offene Spannung — und warum `RP_ROWS` ein Build-Schalter ist

Eine Lane, die **eine** 8-Byte-Unit verarbeitet, braucht dafür **32 Byte
Aktivierung**: das Verhältnis Aktivierung:Gewicht ist **4:1** bei `RP_ROWS=1`.
R Zeilen mit derselben Unit amortisieren das auf `4:R` — kostet aber R
Zeilenzeiger und R Akkumulatoren. Und eine frühere Messung sagt, dass Register
hier viel kosten (64 Register → 50 % Occupancy → **−35 %**).

Nur eine Messung entscheidet:

```bash
./tools/regs.sh repack          # Register/Spills für RP_ROWS = 1, 2, 4
```

| `-DBONSAI_REPACK_ROWS=` | Register | Warps/SM | Aktivierung:Gewicht |
|---|---|---|---|
| 1 | ≤ 32 | 64 | 4:1 |
| 2 | ≤ 42 | 48 | 2:1 |
| **4** | ≤ 64 | 32 | **1:1** |

## Messen

```bash
# 1. Registerbild
./tools/regs.sh repack

# 2. Host-Verifikation des Layouts (kein nvcc noetig)
g++ -O2 -std=c++17 -I src -I include tests/test_p40.cpp -o t && ./t   # 2065/2065

# 3. Auf der Karte: A/B gegen den Stock-Pfad
#    Der Repack braucht einen eigenen Loader (der Fork konsumiert GGUF in place),
#    deshalb laeuft die Messung ueber einen eigenen Runner oder ueber
#    bonsai_p40_repack + bonsai_p40_gemv_repack aus include/bonsai_p40.h.
```

## Erwartete Ausbeute

| Szenario | effektiv | PQ2_0 | PTQ1_0 |
|---|---|---|---|
| Stock heute | 157.5 GB/s | 21.9 tok/s | 26.5 tok/s |
| Repack, konservativ | 216.0 GB/s | 30.0 tok/s | 36.3 tok/s |
| Repack, ideal | 251.8 GB/s | 35.0 tok/s | 42.3 tok/s |

**Spanne: 27–33 tok/s (PQ2_0) bzw. 32–37 tok/s (PTQ1_0)**, also 1.2×–1.7×. Die
Speicherseite ist gemessen; ob ein Kernel das neue Limit erreicht, ist es
**nicht** — deshalb eine Spanne und keine Zahl.

## Grenzen

* Der Kernel ist **nicht kompiliert und nicht gemessen** — hier gibt es kein
  `nvcc` und keine GPU. Verifiziert sind Layout, Round-Trip und die komplette
  Dot-Mathematik auf dem Host.
* `bonsai_p40_repack` / `bonsai_p40_gemv_repack` sind neu in der C-API; der
  Repack selbst ist eine reine Byte-Permutation (Kernel im selben File).
* Der Repack kostet **eine Vollkopie beim Laden** (7.21 GB bzw. 5.95 GB) — bei
  ~30 s Ladezeit für 7 GB von Disk vernachlässigbar gegenüber 1.2–1.7× pro Token.
* Nur sinnvoll mit **eigenem Loader**; in den Fork lässt es sich nicht einhängen,
  weil dort die GGUF-Tensoren in place konsumiert werden.
