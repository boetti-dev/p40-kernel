# Kompletter Workflow auf deinem lokalen System

Von Null bis laufendem Modell auf der P40, inklusive **Beweis**, dass der
Kernel greift. Alles ist copy-paste-fähig; jeder Schritt hat eine Prüfung.

> Getestet wurde hier nur, was **ohne GPU und ohne `cmake`** prüfbar ist
> (Host-Suite 355/355, GGUF-Checks, Patch applyt sauber). Schritte 5–8 laufen
> erstmals auf **deinem** Rechner. Rechne mit Überraschungen und melde sie.

---

## 0. Was am Ende läuft

| Variante | Ergebnis |
|---|---|
| **A** | `llama-cli` / `llama-server` aus dem PrismML-Fork, mit dem P40-Kernel als Decode-Pfad |
| **B** | `libbonsai_p40.a` + die Host-Suite, ohne Fork (eigener Runner) |

Für A brauchst du nur den Fork + den Patch aus diesem Repo. Für B nur dieses Repo.

---

## 1. Voraussetzungen prüfen

```bash
# Compiler / Build
gcc --version | head -1
cmake --version | head -1
make --version | head -1
python3 --version

# CUDA: sm_61 MUSS verfügbar sein
nvcc --version | tail -2
nvcc --list-gpu-arch | grep -w compute_61
```

**Wichtig:** Zeile `compute_61` **muss** erscheinen. Wenn nicht, ist dein
CUDA-Toolkit zu neu — **Pascal wurde in CUDA 13 aus den unterstützten
Architekturen entfernt**. Dann brauchst du ein **CUDA 12.x**-Toolkit
(11.8 geht auch).

```bash
# GPU sichtbar?
nvidia-smi
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
```

Erwartet: `Tesla P40, 24576 MiB, 6.1`.

**P40-spezifische Pflichten** (sonst throttelt oder stirbt die Karte):

```bash
sudo nvidia-smi -pm 1                 # Persistence Mode: weniger Latenz
sudo nvidia-smi -pl 200               # optional: Power-Limit senken
```

* Die P40 ist **passiv gekühlt**. Sie braucht Server-Airflow. In einem normalen
  Desktop-Gehäuse ohne Kanal über der Karte taktet sie runter oder schaltet ab.
* Die P40 hat **keinen Videoausgang** — für die Konsole brauchst du eine zweite
  GPU oder SSH.
* Bei mehreren GPUs: `export CUDA_VISIBLE_DEVICES=0` (Index der P40).

---

## 2. Dieses Repo klonen

```bash
git clone https://github.com/boetti-dev/p40-kernel.git
cd p40-kernel
git checkout cline/yzkepqca          # Branch mit dem aktuellen Stand
git log --oneline -1
```

---

## 3. Ohne GPU verifizieren (2 Minuten)

Das läuft überall und prüft die Numerik, die die P40 später ausführt.

```bash
# 0. Statische Checks (kein Compiler nötig): Makro-Kollisionen, zerschnittene
#    Funktionen, Klammerbilanz. Fängt genau die drei Fehlerklassen ab, die hier
#    schon aufgetreten sind.
python3 tools/check_sources.py

# 1. Host-Suite
g++ -O2 -std=c++17 -Wall -Wextra -I src -I include tests/test_p40.cpp -o test_p40
./test_p40
```

Erwartet:

```
[pq2_0] format + dp4a dot          [fwht] Hadamard block 1024
[ops] rmsnorm / rope / swiglu      [gdn] gated delta net step
[attn] online softmax vs naive     [model] qwen35 geometry
[gemv] 4-rows-per-warp strided loop [gemm] tiled matmul
[ptx] host fallback path           [kv] cache sizing, layout, score, split-K
[dispatch] PrismML MUL_MAT takeover predicate

355 passed, 0 failed
```

Zusätzlich: derselbe Test mit den ggml-Makros des Forks — das prüft die
Integrations-Härtung (Makro-Kollisionen):

```bash
g++ -O2 -std=c++17 -DQK8_0=32 -DQK8_1=32 -DQK_PQ2_0=128 -DQI_PQ2_0=4 \
    -I src -I include tests/test_p40.cpp -o test_p40_fork && ./test_p40_fork | tail -2
```

---

## 4. Modell laden und prüfen

```bash
pip install -U "huggingface_hub[cli]"
# neuere Versionen: hf download ...

huggingface-cli download prism-ml/Ternary-Bonsai-2-27B-gguf \
    Ternary-Bonsai-2-27B-PQ2_0.gguf --local-dir models
# optional, nur für Bild-Eingabe:
huggingface-cli download prism-ml/Ternary-Bonsai-2-27B-gguf \
    Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf --local-dir models

ls -lh models/            # ~7.2 GB für PQ2_0
```

Geometrie gegen die Kernel-Konstanten prüfen:

```bash
python3 tools/check_gguf.py models/Ternary-Bonsai-2-27B-PQ2_0.gguf
```

Erwartet: `18 KV-Checks` + `14 Tensor-Checks` OK,
`MTP/nextn block: none`, Exit-Code 0.

> Lädt jemand die MTP-Variante (`...-PQ2_0-MTP-Q8_0.gguf`), meldet das Skript
> `MTP/nextn block detected` und **Exit 1** — das ist korrekt, der Kernel
> unterstützt MTP nicht (`docs/MTP.md`).

---

## 5. Variante A — PrismML-Fork bauen (empfohlen)

### 5.1 Fork klonen und Patch anwenden

```bash
cd ..
git clone --depth 1 -b prism https://github.com/PrismML-Eng/llama.cpp
cd llama.cpp

git apply --check  ../p40-kernel/integrations/prismml/bonsai-p40.diff   # muss still durchlaufen
git apply          ../p40-kernel/integrations/prismml/bonsai-p40.diff
git status --short | head
```

Erwartet: 11 Dateien — 9 neu unter `ggml/src/ggml-cuda/bonsai-p40/`, 2 geändert
(`ggml-cuda.cu`, `CMakeLists.txt`), insgesamt +928 Zeilen.

Alternative im `git am`-Format (mit Commit-Message):

```bash
git am < ../p40-kernel/integrations/prismml/0001-bonsai-p40.patch
```

### 5.2 Bauen — **nur für sm_61**

```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61 -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j"$(nproc)"
```

`CMAKE_CUDA_ARCHITECTURES=61` ist Absicht: nur `sm_61`, kürzere Buildzeit,
kleinere Binaries — und der P40-Pfad ist exakt dafür gegated. Binaries landen in
`build/bin/` (`llama-cli`, `llama-server`, `llama-bench`).

### 5.3 Prüfen, dass der Pfad wirklich greift

```bash
BONSAI_P40_VERBOSE=1 ./build/bin/llama-cli \
    -m ../p40-kernel/models/Ternary-Bonsai-2-27B-PQ2_0.gguf \
    -ngl 99 -fa on -c 4096 -n 32 -p "Hallo" 2>&1 | grep -i bonsai-p40
```

Erwartet:

```
bonsai-p40: sm_61 PQ2_0 decode path active (n_in=..., n_out=...)
```

**Kommt diese Zeile nicht**, greift der Kernel nicht — dann ist entweder
`compute_cap != 6.1` oder ein Shape passt nicht (siehe Schritt 9).

---

## 6. Variante A — Starten

```bash
./build/bin/llama-cli \
    -m ../p40-kernel/models/Ternary-Bonsai-2-27B-PQ2_0.gguf \
    -ngl 99 -fa on \
    -c 65536 -n 16384 \
    --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.05 \
    -p "Erkläre in drei Sätzen, was ein Tensor Core ist."
```

Wichtig zu den Parametern (aus dem Model-Card, sonst wirkt das Modell kaputt):

* Das Modell **denkt vor der Antwort**, und das Reasoning zählt gegen `-n`.
  Mit `-n 256` bricht es mitten im Gedanken ab → **leere Antwort**. `-n 16384`
  mit `-c 65536` ist der empfohlene Bereich.
* `--min-p 0.05` ist llama.cpp-Default, hier nur explizit.
* `reasoning_effort: "medium"` verkürzt das Denken; `"high"` ist im
  Chat-Template **ungültig** (gültig sind `low`, `medium`, `xhigh`).

Als Server:

```bash
./build/bin/llama-server \
    -m ../p40-kernel/models/Ternary-Bonsai-2-27B-PQ2_0.gguf \
    -ngl 99 -fa on -c 65536 -n 16384 --host 0.0.0.0 --port 8080
# dann: http://<host>:8080
```


---

## 7. Der entscheidende Test: A/B-Benchmark

> **Bereits gemessen:** auf einer echten P40 war `BONSAI_P40=1` **1.92×
> langsamer** (11.39 vs 21.90 tok/s). Der Pfad ist deshalb **standardmäßig
> AUS**. Siehe `docs/MEASURED.md` — inklusive Analyse, warum die
> 4-Zeilen-pro-Warp-Prämisse nicht hielt. Die folgenden Kommandos dienen dem
> Nachvollziehen und dem Testen von Änderungen.

Der Schalter wirkt zur Laufzeit, **kein Neubauen nötig**:

```bash
cd llama.cpp
M=../p40-kernel/models/Ternary-Bonsai-2-27B-PQ2_0.gguf

for v in 0 1; do
  echo "=== BONSAI_P40=$v ==="
  BONSAI_P40=$v ./build/bin/llama-bench -m $M -ngl 99 -fa on -p 0 -n 128
done
```

**Erwartung auf einer P40: `BONSAI_P40=0` (Stock) gewinnt deutlich.** Das ist
der gemessene Stand, kein Fehler in der Installation. Die erwartete
Größenordnung auf einer P40 ist **~22 tok/s** — nicht die 42, die eine frühere
Version dieser Doku vorhergesagt hatte (falsche Effizienzannahme, siehe
`docs/MEASURED.md`).

Wenn du Änderungen am Kernel testest, ist das die Messlatte:
`BONSAI_P40=1` muss `BONSAI_P40=0` **übertreffen**, sonst ist die Änderung
wertlos.

* Beide Werte **exakt gleich** → der Pfad greift nicht (`BONSAI_P40_VERBOSE=1`).
* `BONSAI_P40=1` langsamer → erwartet beim aktuellen Stand.

Prefill getrennt ansehen (`-p 512 -n 0`): davon ist der Patch **nicht**
betroffen, Prefill läuft weiter über den Fork-MMQ.

---

## 8. Variante B — nur dieses Repo, ohne Fork

```bash
cd p40-kernel
./tools/build_p40.sh
# -> build/libbonsai_p40.a  + Host-Tests laufen mit
```

Register-/Spill-Report der Kernel (**erster Schritt bei Performance-Fragen**):

```bash
./tools/regs.sh              # alle Kernel
./tools/regs.sh gemv         # nur gemv_p40.cu
```

Zu prüfen: `Used N registers` für `gemv_pq2_0_p40_kernel` muss **<= 64** sein
(wegen `__launch_bounds__(256,4)`). Steht dort mehr oder `spill stores/loads > 0`,
sind Spills die Ursache der gemessenen Regression — dann Launch-Bounds
lockern (`__launch_bounds__(256, 2)` → 128 Register) und den A/B aus
Schritt 7 wiederholen. Siehe `docs/MEASURED.md`.

SASS/PTX der Kernel ansehen (prüft die Behauptungen aus
`docs/P40_OPTIMIZATION.md`):

```bash
./tools/disasm.sh gemv
# erwartet: "Used N registers" <= 64, LDG.E.U16 (nicht verbreitert),
#           32x DP4A pro 128-Gewichtsblock x 4 Zeilen, 4x prefetch.global.L2
```

Vorhersagen für deine Karte nachrechnen:

```bash
python3 tools/gpu_predict.py
python3 tools/gpu_predict.py --bw 346 --sm 30 --fp32-per-sm 128 --name "mein P40"
```

---

## 9. Fehlersuche

| Symptom | Ursache / Lösung |
|---|---|
| `nvcc --list-gpu-arch` ohne `compute_61` | Toolkit zu neu (CUDA 13 hat Pascal entfernt) → CUDA 12.x installieren |
| Ausgabe eines Tools passt nicht zum Quelltext / Zeile fehlt | **Stale Binary**: wenn der `nvcc`-Lauf fehlschlaegt, laeuft das alte Binary weiter und man sieht die alte Ausgabe. Immer erst `rm -f build/bwtest build/libbonsai_p40.a` und dann neu bauen — oder `./tools/build_p40.sh` benutzen, das mit `set -e` beim ersten Fehler abbricht. |
| `Built target test_p40`, aber `./build/bin/llama-bench` fehlt | Du baust in **p40-kernel** statt in **llama.cpp**. `llama-bench` existiert nur im Fork; p40-kernel baut nur `test_p40` + `libbonsai_p40`. Also `cd llama.cpp` und dort konfigurieren/bauen. `-DGGML_CUDA=ON`/`-DBONSAI_GEMV_ROWS` werden in p40-kernel still ignoriert (nur eine Warnung über unbenutzte Variablen) — deshalb läuft cmake dort ohne Fehler durch. |
| `cmake: command not found` | `sudo apt install cmake` |
| `fatal error: bonsai_p40.h: Datei nicht gefunden` | Include-Pfad fehlt: `bonsai_p40.h` liegt in `include/`, also `-I src -I include` angeben (oder einfach `./tools/regs.sh` benutzen) |
| Build bricht in `bonsai-p40/*.cu` ab | Fehlermeldung komplett aufbewahren (Schritt 10) |
| `bonsai-p40:`-Zeile fehlt | `compute_cap` prüfen; ist `BONSAI_P40` auf `0` gesetzt? |
| Leere Antworten | `-n` zu klein (Reasoning frisst das Budget) → `-n 16384 -c 65536` |
| HTTP 500 bei `reasoning_effort: "high"` | `xhigh` oder `medium` verwenden |
| CUDA out of memory | `-c` senken; bei 262K braucht der Q8_0-KV-Cache ~8.5 GiB |
| Karte taktet runter / schaltet ab | Airflow fehlt (P40 ist passiv) → `nvidia-smi -q -d PERFORMANCE` |
| Patch lässt sich nicht anwenden | Fork-Stand weicht ab → `git log -1` muss `b858f3b` zeigen |

Patch zurückrollen:

```bash
git apply -R ../p40-kernel/integrations/prismml/bonsai-p40.diff
```

---

## 10. Wenn etwas nicht baut: das brauche ich

```bash
nvcc --version
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv
cmake --build build --config Release -j1 2>&1 | tail -60   # -j1 = klare Reihenfolge
git -C . log --oneline -1
git -C ../p40-kernel log --oneline -1
```

Mit der ersten echten Compiler-Meldung ist das in einem Durchgang zu fixen —
genau das war hier **nicht** prüfbar (kein `cmake`, kein `nvcc`).

---

## Kurzfassung

```bash
# 1. Dieses Repo
git clone https://github.com/boetti-dev/p40-kernel.git && cd p40-kernel
git checkout cline/yzkepqca
g++ -O2 -std=c++17 -I src -I include tests/test_p40.cpp -o t && ./t    # 355/355

# 2. Modell
huggingface-cli download prism-ml/Ternary-Bonsai-2-27B-gguf \
    Ternary-Bonsai-2-27B-PQ2_0.gguf --local-dir models
python3 tools/check_gguf.py models/Ternary-Bonsai-2-27B-PQ2_0.gguf

# 3. Fork + Patch + Bau
cd .. && git clone --depth 1 -b prism https://github.com/PrismML-Eng/llama.cpp
cd llama.cpp
git apply ../p40-kernel/integrations/prismml/bonsai-p40.diff
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61 -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j"$(nproc)"

# 4. Beweis, dass der Kernel greift, und A/B
BONSAI_P40_VERBOSE=1 ./build/bin/llama-cli \
    -m ../p40-kernel/models/Ternary-Bonsai-2-27B-PQ2_0.gguf \
    -ngl 99 -fa on -c 4096 -n 32 -p "Hallo" 2>&1 | grep bonsai-p40
for v in 1 0; do BONSAI_P40=$v ./build/bin/llama-bench \
    -m ../p40-kernel/models/Ternary-Bonsai-2-27B-PQ2_0.gguf -ngl 99 -fa on -p 0 -n 128; done
```

---

## Was hier verifiziert ist und was nicht

**Verifiziert (ohne GPU):** Host-Suite 355/355, davon 19 Checks für die
Fork-Übernahme-Entscheidung; GGUF-Geometrie gegen die echte Datei; der Patch
applyt sauber auf einen frischen `prism`-Clone und ist byte-identisch; keine
Makro-Kollisionen gegen 1882 Fork-Makros.

**Nicht verifiziert:** CUDA-Kompilierbarkeit, Registerdruck, Occupancy,
Performance. Diese Umgebung hatte **kein `cmake`** und **kein `nvcc`**, also
war llama.cpp dort grundsätzlich nicht baubar. Schritte 5–8 laufen erstmals bei
dir — die A/B-Zahl aus Schritt 7 ist das, was am Ende zählt.