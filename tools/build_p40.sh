#!/usr/bin/env bash
# build_p40.sh -- build the Bonsai-P40 kernels for the Tesla P40 (sm_61).
#
#   ./tools/build_p40.sh                 # build libbonsai_p40.a + run host tests
#   ./tools/build_p40.sh --check-gguf F  # also validate a GGUF against the kernel constants
#
# The P40 is GP102 / sm_61.  We compile for sm_61 ONLY: there is no generic
# fatbin, no PTX JIT fallback and no code path for any other architecture.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${ROOT}/build"
mkdir -p "${BUILD}"

ARCH="-gencode arch=compute_61,code=sm_61"
NVCC="${NVCC:-nvcc}"
CXX="${CXX:-g++}"

# --- 0. static checks (catch the bug classes that need no compiler) --------
echo "== static checks =="
python3 "${ROOT}/tools/check_sources.py"

# --- 1. host test suite (always runs, needs no GPU) -------------------------
echo "== host verification =="
"${CXX}" -O2 -std=c++17 -I "${ROOT}/src" -I "${ROOT}/include" \
        "${ROOT}/tests/test_p40.cpp" -o "${BUILD}/test_p40"
"${BUILD}/test_p40"

# --- 2. CUDA kernels for sm_61 ---------------------------------------------
if ! command -v "${NVCC}" >/dev/null 2>&1; then
    echo
    echo "nvcc not found -- skipping the CUDA build."
    echo "Install the CUDA toolkit (>= 11.0 for sm_61) and re-run, or set NVCC=/path/to/nvcc."
    exit 0
fi

echo
echo "== CUDA build (sm_61 only) =="
NVCCFLAGS=(-O3 -std=c++17 ${ARCH}
           -Xptxas -dlcm=ca           # default L1 caching helps the GEMV weight stream
           --use_fast_math
           -I "${ROOT}/src" -I "${ROOT}/include"
           -Xcompiler -fPIC)

OBJS=()
for f in "${ROOT}"/src/kernels/*.cu; do
    o="${BUILD}/$(basename "${f}" .cu).o"
    echo "  nvcc ${f##*/}"
    "${NVCC}" "${NVCCFLAGS[@]}" -c "${f}" -o "${o}"
    OBJS+=("${o}")
done

ar rcs "${BUILD}/libbonsai_p40.a" "${OBJS[@]}"
echo "  -> ${BUILD}/libbonsai_p40.a"

# --- 2b. bandwidth ceiling probe (independent of the kernels) --------------
if [ -f "${ROOT}/tools/bwtest.cu" ]; then
    echo "  nvcc tools/bwtest.cu"
    "${NVCC}" -O3 -std=c++17 ${ARCH} "${ROOT}/tools/bwtest.cu" -o "${BUILD}/bwtest"
    echo "  -> ${BUILD}/bwtest   (run it to measure the card's ceiling)"
fi

# --- 3. optional GGUF check -------------------------------------------------
if [[ "${1:-}" == "--check-gguf" && -n "${2:-}" ]]; then
    echo
    echo "== GGUF geometry check =="
    python3 "${ROOT}/tools/check_gguf.py" "${2}"
fi

echo
echo "Done."
