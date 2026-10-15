# MMQ auf Pascal — ist es drin?

**Nein. In diesem Repo ist keine MMQ enthalten.** Belegt durch ein `grep` über
den gesamten Quellbaum:

```
$ grep -rniE 'mmq|mmq-config|mul_mat_q|SRAM_LAYOUT|load_tiles|cp_async' src include tests tools
(nur zwei Doku-Treffer, kein Code)
```

Das ist keine Lücke aus Versehen, sondern eine bewusste Entscheidung — und die
Begründung ist wichtiger als die Antwort.

## 1. Was stattdessen drin ist

`src/kernels/gemm_p40.cu` ist ein **eigener, einfacher DP4A-Tile-GEMM** für
Prefill, der die *Rolle* von MMQ spielt, aber nicht MMQ ist:

| | `gemm_p40.cu` (hier) | Fork-Pascal-MMQ |
|---|---|---|
| Kachel | 256 Zeilen × **16** Spalten | **64** Zeilen × J, J ∈ {8,16,24,32,40,48,64} |
| Threads / Occupancy | 256 / 2 | 256 / 2 |
| Lane-Mapping | **eine ganze Zeile pro Lane** (16 Akkumulatoren) | Sub-Tile über die Warp verteilt |
| SRAM-Layout | schlicht: Aktivierungen als int8-Slab | `GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0` |
| K-Pipeline | keine (`MMQ_ITER_K` fehlt) | `MMQ_ITER_K` |
| Dequant-Staging | entfällt (native PQ2_0-Codes) | opakes Tile-Staging |

## 2. Warum der Fork-MMQ auf Pascal existiert — und funktioniert

Verifiziert im `prism`-Branch:

```
ggml/src/ggml-cuda/mmq.cuh:
  ggml_cuda_mmq_get_config(...):
    #elif __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
        return ggml_cuda_mmq_get_config_ampere(type, J, fallback);
    #else
        return ggml_cuda_mmq_get_config_pascal(type, J, fallback);   <-- sm_61

  ggml/src/ggml-cuda/mmq-config-pascal.cuh:
    CASE(GGML_TYPE_PQ2_0, 256, 2, 64,  8, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, true);
    CASE(GGML_TYPE_PQ2_0, 256, 2, 64, 16, ...);   ... bis J=64
```

`ggml_cuda_should_use_mmq()` aktiviert den Pfad ab `GGML_CUDA_CC_DP4A` (also
`sm_61`) über den DP4A-Zweig. **PQ2_0 hat auf Pascal explizite Kacheln** —
MMQ ist für dieses Format auf dieser Karte also nicht nur vorhanden, sondern
getunt.

## 3. Die Konsequenz: `gemm_p40.cu` sollte man *nicht* einhängen

Meine ursprüngliche `INTEGRATION.md` empfahl, den Fork-MMQ-Pfad durch
`gemm_p40.cu` zu ersetzen. **Das war falsch** und ist korrigiert: der
Fork-MMQ ist Pascal-getunt und meiner einfachen Kachel vorzuziehen. Richtig ist:

* **Mit dem Fork bauen** → Fork-MMQ für Prefill, meine Datei bleibt draußen.
  Mein Beitrag liegt dann bei Decode (`gemv_p40.cu`, `kv_p40.cu`), wo der
  Hebel tatsächlich anders aussieht.
* **Ohne Fork bauen / standalone testen** → `gemm_p40.cu` ist der Fallback.

## 4. Warum MMQ auf Pascal *nicht* der große Hebel ist

Das ist der Punkt, der die Frage einordnet. Auf Ampere/Hopper bringt MMQ
Tensor-Core-`mma` über `s8`-Fragmente — Hardware, die Pascal **nicht hat**.
Auf `sm_61` ist MMQ ein *weiter verbesserter `dp4a`-Tile-GEMM*; der Gewinn
kommt aus Kachelgröße, SRAM-Layout und K-Pipelining, nicht aus neuer Hardware.

Und Prefill ist für dieses Modell gemessen **nicht** der Engpass:

| Phase | P40 | warum |
|---|---|---|
| Decode (tg128) | 42 tok/s | 7.21 GB/Token — **reines Bandbreitenlimit** |
| Prefill (pp512) | ~1.3–1.8 s | 12.5 TMAC, `dp4a`-Limit |

Der Decode ist das, was der Nutzer bei jedem Token spürt, und der ist
bandbreitengebunden — dort hilft MMQ **überhaupt nicht**. MMQ ist eine
Prefill-Optimierung, und Prefill ist der einmalige Prompt-Verarbeitungsschritt.

## 5. Was ich stattdessen optimiert habe

Weil MMQ den Decode nicht berührt, liegt der Aufwand dort, wo er zählt:

* 4 Zeilen/Warp + Aktivierungs-Reuse (Decode)
* `+1`-Bias gefaltet, Skalen außerhalb der Chunk-Schleife
* `ld.global.nc` + `prefetch.global.L2`
* Split-K-Attention für den KV-Cache bei langem Kontext

## 6. Falls MMQ doch gewünscht ist

Zwei Wege, in dieser Reihenfolge:

1. **Fork-MMQ nutzen** (empfohlen, 0 Zeilen Code): gegen den Fork bauen, nichts
   einhängen. Der PQ2_0-Pascal-Pfad ist fertig.
2. **Eigene Kachel nachziehen**: `GEMM_BN` 16 → 64 und Lane-Mapping auf
   Sub-Tiles umstellen (statt „eine Zeile pro Lane"), plus doppelt gepufferter
   Aktivierungs-Slab. Das ist ein Rewrite von `gemm_p40.cu`, nicht ein Parameter.

Wichtig bei Weg 2: **`cp.async` gibt es auf Pascal nicht** (ab `sm_80`), das
Pipelining müsste wie im Fork über `MMQ_ITER_K` und Register-Double-Buffering
laufen.

## 7. Nicht gemessen

Es gibt hier weder `nvcc` noch GPU, also ist der Abstand zwischen meiner Kachel
und dem Fork-MMQ **nicht quantifiziert**. Die Aussage „MMQ ist vorzuziehen" ist
aus der Struktur abgeleitet (Kachelgröße, SRAM-Layout, K-Pipelining), nicht
gemessen. Der Host-Test prüft nur die Korrektheit meines GEMM (Kachel-/Index-
Arithmetik gegen eine Referenz-Matmul), nicht seine Geschwindigkeit.