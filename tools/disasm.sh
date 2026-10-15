#!/usr/bin/env bash
# disasm.sh -- inspect the generated PTX and SASS of the P40 kernels.
#
# Run this on a machine with the CUDA toolkit.  It is the tool to use when
# checking the claims in docs/P40_OPTIMIZATION.md: register counts (which set
# occupancy), the ld.global.nc / dp4a / prefetch instructions, and whether the
# decode inner loop really keeps four independent weight streams in flight.
#
#   ./tools/disasm.sh              # PTX + SASS for all four kernels
#   ./tools/disasm.sh gemv         # only gemv_p40.cu
#   ./tools/disasm.sh gemv sass    # only SASS, only gemv
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${ROOT}/build/disasm"
mkdir -p "${OUT}"

WHICH="${1:-all}"
WHAT="${2:-both}"

NVCC="${NVCC:-nvcc}"
CUOBJDUMP="${CUOBJDUMP:-cuobjdump}"
ARCH="sm_61"

FILES=()
case "${WHICH}" in
    gemv)   FILES=("src/kernels/gemv_p40.cu") ;;
    gemm)   FILES=("src/kernels/gemm_p40.cu") ;;
    attn)   FILES=("src/kernels/attn_p40.cu") ;;
    elem)   FILES=("src/kernels/elementwise_p40.cu") ;;
    all)    FILES=("src/kernels/gemv_p40.cu" "src/kernels/gemm_p40.cu"
                   "src/kernels/attn_p40.cu" "src/kernels/elementwise_p40.cu") ;;
    *)      echo "unknown target '${WHICH}' (gemv|gemm|attn|elem|all)" >&2; exit 2 ;;
esac

for f in "${FILES[@]}"; do
    base="$(basename "${f}" .cu)"
    echo "=== ${base} ==="

    if [[ "${WHAT}" == "both" || "${WHAT}" == "ptx" ]]; then
        echo "--- PTX ---"
        "${NVCC}" -O3 -std=c++17 -arch=${ARCH} --use_fast_math \
            -I "${ROOT}/src" -I "${ROOT}/include" \
            -ptx "${ROOT}/${f}" -o "${OUT}/${base}.ptx" \
            -Xptxas -v 2>&1 | grep -E 'ptxas info|registers|smem|cmem' || true
        cp "${OUT}/${base}.ptx" "${OUT}/${base}.ptx"
        echo "  written: ${OUT}/${base}.ptx"
    fi

    if [[ "${WHAT}" == "both" || "${WHAT}" == "sass" ]]; then
        echo "--- SASS (sm_61) ---"
        "${NVCC}" -O3 -std=c++17 -arch=${ARCH} --use_fast_math \
            -I "${ROOT}/src" -I "${ROOT}/include" \
            -cubin "${ROOT}/${f}" -o "${OUT}/${base}.cubin" -Xptxas -v
        "${CUOBJDUMP}" -sass "${OUT}/${base}.cubin" > "${OUT}/${base}.sass"
        echo "  written: ${OUT}/${base}.sass"
        echo "  kernel resource usage:"
        "${CUOBJDUMP}" -res-usage "${OUT}/${base}.cubin" || true
    fi
    echo
done

cat <<'EOF'
WHAT TO LOOK FOR (see docs/P40_OPTIMIZATION.md)

  decode GEMV (gemv_pq2_0_p40_kernel)
    - "Used N registers" must be <= 64 so __launch_bounds__(256,4) is met.
    - the weight loads must be LDG.E.U16 (via ld.global.nc), never a 2-byte
      load widened to 32-bit (that would be a misaligned access: the PQ2_0
      block is a 34-byte struct, so qs is only 2-byte aligned).
    - the inner loop must contain 32 DP4A per 128-weight block x 4 rows
      (i.e. 8 DP4A per row per 32-element chunk).
    - prefetch.global.L2 should appear four times per block iteration (one per
      row's weight stream).

  prefill GEMM (gemm_pq2_0_p40_kernel)
    - DP4A must dominate the inner loop; the per-element scale must be a
      single FFMA (acc += (dw*D)*sumi), not two FMUL + FADD.
    - shared memory usage should be ~2.5 KB (X_s 2 KB + D_s/S_s 256 B each).
EOF