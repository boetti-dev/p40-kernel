#!/usr/bin/env bash
# regs.sh -- ptxas resource report for the P40 kernels.
#
# Prints registers / shared memory / spill counts per kernel.  This is the FIRST
# thing to check when a kernel underperforms: `__launch_bounds__(256,4)` caps the
# kernel at 64 registers, and if the compiler has to spill, occupancy and
# throughput collapse.  See docs/MEASURED.md.
#
#   ./tools/regs.sh              # all kernels
#   ./tools/regs.sh gemv         # only gemv_p40.cu
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ARCH="${ARCH:-sm_61}"
NVCC="${NVCC:-nvcc}"

# src/kernels/*.cu include "../pq2_0.h" (resolved relative to the file) and
# "bonsai_p40.h" (lives in include/).  Both -I paths are required.
INCS=(-I "${ROOT}/src" -I "${ROOT}/include")

case "${1:-all}" in
    gemv) FILES=("src/kernels/gemv_p40.cu") ;;
    gemm) FILES=("src/kernels/gemm_p40.cu") ;;
    attn) FILES=("src/kernels/attn_p40.cu") ;;
    elem) FILES=("src/kernels/elementwise_p40.cu") ;;
    kv)   FILES=("src/kernels/kv_p40.cu") ;;
    all)  FILES=(src/kernels/*.cu) ;;
    *)    echo "usage: $0 [all|gemv|gemm|attn|elem|kv]" >&2; exit 2 ;;
esac

OUT="$(mktemp -d)"
trap 'rm -rf "${OUT}"' EXIT

if ! command -v "${NVCC}" >/dev/null 2>&1; then
    echo "nvcc not found (set NVCC=/path/to/nvcc)" >&2
    exit 1
fi

echo "ptxas resource report for ${ARCH}"
echo

fail=0
for f in "${FILES[@]}"; do
    base="$(basename "${f}" .cu)"
    echo "=== ${base} ==="
    # shellcheck disable=SC2086
    if ! "${NVCC}" -O3 -std=c++17 -arch="${ARCH}" --use_fast_math \
            "${INCS[@]}" --ptxas-options=-v \
            -c "${ROOT}/${f}" -o "${OUT}/${base}.o" 2> "${OUT}/${base}.log"; then
        echo "  COMPILE FAILED:"
        sed 's/^/    /' "${OUT}/${base}.log" | head -30
        fail=1
        continue
    fi
    # ptxas prints one "Function properties" block and one "Used N registers"
    # line per instantiated kernel.
    grep -E 'Function properties|Used [0-9]+ registers|spill|bytes smem|cmem' \
        "${OUT}/${base}.log" | sed 's/^/  /' || echo "  (no ptxas report)"
    echo
done

cat <<'EOF'
WHAT TO LOOK FOR

  gemv_pq2_0_p40_kernel  (declared __launch_bounds__(256, 4))
    "Used N registers"  -> MUST be <= 64.  More than 64 is impossible under the
                           launch bounds, so a higher number means the compiler
                           spilled instead: look for "spill stores/loads" > 0.
                           Spills in the four-row body are the prime suspect for
                           the measured 1.92x slowdown (docs/MEASURED.md).

  quant_q8_1_pq2_kernel  (declared __launch_bounds__(256, 8))
    "Used N registers"  -> should be well under 32.

  If gemv_pq2_0_p40_kernel shows spills, the fix is to relax the launch bounds
  (e.g. __launch_bounds__(256, 2) -> 128 registers) and re-run the A/B in
  docs/WORKFLOW.md step 7.
EOF

exit "${fail}"
