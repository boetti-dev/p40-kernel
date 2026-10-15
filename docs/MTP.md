# MTP (Multi-Token Prediction) — kann das Modell es, kann der Kernel es?

Kurzantwort: **Das ausgelieferte offizielle GGUF hat keinen MTP-Head. MTP gibt
es für dieses Modell trotzdem — aber nur über ein separates Community-Paket.
Der Bonsai-P40-Kernel kann es nicht, und zwar aus vier konkreten Gründen.**

## 1. Faktische Prüfung (nicht geraten)

Ich habe den Header der ausgelieferten Datei geparst:

| Prüfung | Ergebnis |
|---|---|
| Tensoren in `Ternary-Bonsai-2-27B-PQ2_0.gguf` | **851** |
| Tensoren mit `nextn` / `mtp` / `draft` / `eagle` | **0** |
| KV-Keys mit MTP-Bezug | **0** |
| Layerzählung | exakt 64 (48 linear + 16 voll) + 3 global |

Die Tensoren gehen exakt auf: 3 global + 64×(ffn_down/gate/up, attn_norm,
post_attention_norm) + 48×(ssm_*, attn_qkv, attn_gate) + 16×(attn_q/k/v/output,
q/k_norm) = 851. **Kein Platz für einen MTP-Block.** Dasselbe gilt für F16 und
PTQ1_0 (gleiche Architektur).

`KNOWN_ISSUES.md` sagt zwar „Multi-token prediction (MTP) is refused for these
files" — gemeint ist damit der Runtime-Guard, nicht ein Head in der Datei.

## 2. Wo MTP dann herkommt

Es existiert ein Community-Paket
[`ProCreations/Ternary-Bonsai-2-27B-MTP`](https://huggingface.co/ProCreations/Ternary-Bonsai-2-27B-MTP)
mit `Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf`: **Trunk PQ2_0 + ein MTP-Layer,
in dieselbe Datei gepackt.**

Verifiziert über die HF-API (ohne Download):

| | Trunk | MTP-Bundle | Δ |
|---|---|---|---|
| Parameter | 26 895 998 464 | 27 320 697 856 | **+424 699 392** |
| Dateigröße | 7.21 GB | 7.66 GB | **+0.451 GB** |
| Architektur | qwen35, 262144 ctx | identisch | – |

Die Rechnung geht exakt auf: 424 699 392 × 34/32 B = **0.451 GB** → der
MTP-Layer ist **Q8_0**, nicht PQ2_0. (Da das Paket `token_embd`/`output` nicht
dupliziert, teilt der MTP-Head das Vokabular des Trunks — genau das ist der
Punkt, der ihn profitabel macht.)

Die Runtime-Seite ist im PrismML-Fork vorhanden: `--spec-type draft-mtp`,
Merged PR [#205](https://github.com/PrismML-Eng/llama.cpp/pull/205)
(2026-09-21) behebt die fehlende Hadamard-Inverse beim
`layer.nextn.embed_tokens`-Lookup. `--spec-type ngram-*` hingegen ist für
dieses Modell wirkungslos ([#203](https://github.com/PrismML-Eng/llama.cpp/issues/203), offen).

## 3. Kann der Bonsai-P40-Kernel es? Nein — vier Gaps

| # | Gap | Konsequenz |
|---|---|---|
| 1 | **Kein `nextn`-Layer** | `include/bonsai_model.h` hat hart 64 Layer; der Trunk-Graph kennt keinen MTP-Block. |
| 2 | **MTP-Layer ist Q8_0, mein GEMV nur PQ2_0** | `bonsai_p40_gemv` akzeptiert ausschließlich `block_pq2_0` (Typ 142). Für den MTP-Layer bräuchte es einen zweiten Mat-Vec-Pfad (`block_q8_0`) — trivial: dieselbe Struktur, `d*sumi` ohne Bias. |
| 3 | **Kein Small-Batch-Verify-Pfad** | Die Verifikation rechnet 1..`nmax+1` Tokens in **einem** Forward. Mein GEMV ist strikt Batch-1 (eine Zeile pro Warp), mein GEMM zielt auf Batch ≥ 16. Genau das Band **2–8 Spalten fehlt** — im Fork ist es `mul_mat_vec_pq2_0_mc` („3-8 columns"). **Das ist der eigentliche Implementierungsaufwand.** |
| 4 | **Kein Accept/Rollback + kein zweiter KV-Stream** | MTP braucht einen Cache-Stream für den MTP-Layer und einen Rollback des GDN-Zustands bei Nicht-Annahme (der Fork: „GDN rollback by replay"). |

## 4. Vorhersage (kalibriert an der Messung, nicht extrapoliert)

Bytes pro Verifikations-Pass:

```
rho = (2 x Vokabular + 1 Layer) / Trunk
    = (2 x 5120*248320*34/128 + 7.21e9/64) / 7.21e9
    = (0.675 + 0.113) / 7.21 = 0.109
Verifier-Pass zieht also 1 + 2*rho = 1.219x die Bytes eines Trunk-Tokens.
```

Kalibriert an PR #205 (gemessen 1.338×, Acceptance 68.1 %, nmax=2) ergeben sich
**1.630 effektiv akzeptierte Tokens pro Pass** — nicht 2.14, wie ein naives
geometrisches Modell liefert (Acceptance ist bedingt, nicht unabhängig). Damit
gibt das Modell exakt 1.338× und reproduziert die Messung.

| Karte | heute | mit MTP (nmax=2) | Faktor |
|---|---|---|---|
| P40 | 42.2 tok/s | **56.5 tok/s** | 1.34× |
| V100 32 GB | 109.8 tok/s | **146.9 tok/s** | 1.34× |

Warum der Faktor übertragbar ist: der Draft ist auf **beiden** Karten
bandbreitengebunden, nicht rechengebunden —

| Karte | Draft/Pass: Compute | Draft/Pass: Bandbreite |
|---|---|---|
| P40 | 0.25 ms | **5.18 ms** |
| V100 | 0.05 ms | **1.99 ms** |
### Die Sidecar-Falle

`-md <separater Drafter>` dupliziert das Vokabular (248320×5120 ≈ 92 % der
Drafter-Bytes) → rho ≈ 0.43 → für dieselben 1.63 Tokens werden 1.86× Bytes
gezogen → **0.88×, also Nettoverlust**. PR #205 misst real 5.16 vs 16.6 t/s.
Deshalb ist nur das **in-file**-MTP sinnvoll, und nur dafür lohnt der
Implementierungsaufwand.

## 5. Was zu bauen wäre (Reihenfolge)

1. `bonsai_p40_gemv_q8_0` — Q8_0-Mat-Vec, ~40 Zeilen, identische 4-Zeilen-Struktur.
2. `bonsai_p40_gemv_n` — Batch-2..8-Pfad (das 2–8-Spalten-Band), der eigentliche Kern.
3. `nextn`-Layer in `bonsai_model.h` (1 Layer, Aufteilung wie ein voller Layer).
4. KV-Stream für den MTP-Layer (1 Layer statt 16 → 1/16 der Bytes: 0.53 GiB Q8_0 @262k).
5. Accept/Rollback im Decoder-Loop (GDN-Zustand per Replay zurücksetzen).

## 6. Der GGUF-Checker meldet es jetzt

`tools/check_gguf.py` erkennt einen `nextn`/`mtp`-Block und **scheitert mit
Begründung**, statt still durchzuwinken:

```
MTP/nextn block detected: 2 tensors
  e.g. blk.64.nextn.attn_q.weight
FAILED: this file carries an MTP (nextn) block; the Bonsai-P40 kernel has no MTP
support (no nextn layer, no small-batch verify path, no second cache stream)
```

Verifiziert mit einer synthetischen GGUF-Datei mit `nextn`-Tensoren (Exit 1,
klare Meldung) sowie mit der echten Trunk-Datei (`MTP/nextn block: none`,
Exit 0).

## 7. Nicht gemessen

Die 1.338× stammen aus PR #205 auf einer RTX 4080 SUPER (Ada, `sm_89`). Auf
`sm_61`/`sm_70` ist MTP nicht gemessen; das Modell oben überträgt sie über den
Byte-Verbrauch, was zulässig ist, weil der Draft nachweislich
bandbreitengebunden ist. `nmax=3` ist **nicht** extrapoliert — dafür fehlen
Messdaten für die bedingte Acceptance.