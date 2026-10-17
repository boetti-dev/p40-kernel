#!/usr/bin/env bash
# regs.sh -- ptxas resource report for the P40 GEMV, plus a sweep over the
# rows-per-warp configurations.
#
# This is the FIRST thing to check when the kernel underperforms: the register
# count fixes the occupancy, and on a latency-bound bandwidth kernel the number
# of resident warps is what matters (docs/MEASURED.md).
#
#   ./tools/regs.sh              # default configuration
#   ./tools/regs.sh sweep        # rows = 1, 2, 4 with registers/spills/occupancy
#   BONSAI_GEMV_ROWS=1 ./tools/regs.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ARCH="${ARCH:-sm_61}"
NVCC="${NVCC:-nvcc}"
SRC="${ROOT}/src/kernels/gemv_p40.cu"

# src/kernels/*.cu include "../pq2_0.h" (relative to the file) and
# "bonsai_p40.h" (lives in include/).  BOTH -I paths are required.
INCS=(-I "${ROOT}/src" -I "${ROOT}/include")

command -v "${NVCC}" >/dev/null 2>&1 || { echo "nvcc not found (set NVCC=...)" >&2; exit 1; }

OUT="$(mktemp -d)"
trap 'rm -rf "${OUT}"' EXIT

# --- single source of truth for the mapping, asserted below ---------------
min_blocks_for_rows() {
    case "$1" in
        1) echo 8 ;;   # <= 32 regs -> 64 warps/SM
        2) echo 6 ;;   # <= 42 regs -> 48 warps/SM
        *) echo 4 ;;   # <= 64 regs -> 32 warps/SM
    esac
}
warps_for_rows() {
    case "$1" in 1) echo 64 ;; 2) echo 48 ;; *) echo 32 ;; esac
}

# Assert the script's expectation against the REAL preprocessor behaviour of the
# real file.  This exists because a `#if` cannot see a `constexpr int`: writing
# `#if GEMV_ROWS_PER_WARP >= 4` silently made every configuration take the #else
# branch (32-register cap, spills).  That shipped once.
assert_mapping() {
    local blk="${OUT}/blk.inc" bad=0 got want r
    awk '/^#ifndef BONSAI_GEMV_ROWS/{f=1} f{print} f&&/^#endif/{n++} n==2{exit}' \
        "${SRC}" > "${blk}"
    [ -s "${blk}" ] || { echo "could not extract the config block" >&2; return 1; }
    for r in 1 2 4; do
        got=$( (cat "${blk}"; echo 'MB=GEMV_MIN_BLOCKS') \
               | gcc -E -P -DBONSAI_GEMV_ROWS="$r" -x c - 2>/dev/null | tail -1 )
        want="MB=$(min_blocks_for_rows "$r")"
        if [ "${got}" != "${want}" ]; then
            echo "  MISMATCH rows=${r}: preprocessor='${got}' script expects '${want}'" >&2
            bad=1
        fi
    done
    return "${bad}"
}

# Extract the two lines that matter for gemv_pq2_0_p40_kernel.
report_one() {
    local rows="$1" log="${OUT}/gemv_r${1}.log" obj="${OUT}/gemv_r${1}.o"
    if ! "${NVCC}" -O3 -std=c++17 -arch="${ARCH}" --use_fast_math \
            "${INCS[@]}" --ptxas-options=-v -DBONSAI_GEMV_ROWS="${rows}" \
            -c "${SRC}" -o "${obj}" 2> "${log}"; then
        echo "  COMPILE FAILED (rows=${rows}):"
        sed 's/^/    /' "${log}" | head -20
        return 1
    fi
    local regs spill stack
    regs=$(grep -A3 'Function properties for.*gemv_pq2_0_p40_kernel' "${log}" \
           | grep -oE 'Used [0-9]+ registers' | head -1 | grep -oE '[0-9]+' || echo '?')
    stack=$(grep -A3 'Function properties for.*gemv_pq2_0_p40_kernel' "${log}" \
            | grep -oE '[0-9]+ bytes spill stores' | head -1 | grep -oE '^[0-9]+' || echo 0)
    spill=$(grep -A3 'Function properties for.*gemv_pq2_0_p40_kernel' "${log}" \
            | grep -oE '[0-9]+ bytes stack frame' | head -1 | grep -oE '^[0-9]+' || echo 0)
    printf '  rows=%-2s min_blocks=%-2s regs=%-4s spill_stores=%-5s stack=%-5s -> %s warps/SM%s\n' \
        "${rows}" "$(min_blocks_for_rows "${rows}")" "${regs}" "${spill:-0}" "${stack:-0}" \
        "$(warps_for_rows "${rows}")" \
        "$( [ "${spill:-0}" != 0 ] && echo '  <-- SPILLS: cap too tight' )"
}

echo "ptxas resource report for ${ARCH}"
echo
echo "config-block check (script expectation vs. real preprocessor):"
if assert_mapping; then echo "  OK: rows=1->8, rows=2->6, rows=4->4 min-blocks"; else
    echo "  FAILED -- the mapping in regs.sh and in gemv_p40.cu disagree" >&2; exit 1; fi
echo

# Register report for the repacked-layout kernel: it has its own row knob
# (BONSAI_REPACK_ROWS) and its own min-blocks mapping in gemv_repack_p40.cu.
report_repack_one() {
    local rows="$1"
    local f="${ROOT}/src/kernels/gemv_repack_p40.cu"
    local log="${OUT}/rp_r${1}.log" obj="${OUT}/rp_r${1}.o"
    if ! "${NVCC}" -O3 -std=c++17 -arch="${ARCH}" --use_fast_math \
            "${INCS[@]}" --ptxas-options=-v -DBONSAI_REPACK_ROWS="${rows}" \
            -c "${f}" -o "${obj}" 2> "${log}"; then
        echo "  COMPILE FAILED (rows=${rows}):"
        sed 's/^/    /' "${log}" | head -20
        return 1
    fi
    local regs spill
    regs=$(grep -A3 'Function properties for.*gemv_repack_p40_kernel' "${log}" \
           | grep -oE 'Used [0-9]+ registers' | head -1 | grep -oE '[0-9]+' || echo '?')
    spill=$(grep -A3 'Function properties for.*gemv_repack_p40_kernel' "${log}" \
            | grep -oE '[0-9]+ bytes spill stores' | head -1 | grep -oE '^[0-9]+' || echo 0)
    printf '  rows=%-2s regs=%-4s spill_stores=%-5s -> %s warps/SM%s\n' \
        "${rows}" "${regs}" "${spill:-0}" "$(warps_for_rows "${rows}")" \
        "$( [ "${spill:-0}" != 0 ] && echo '  <-- SPILLS' )"
}

if [ "${1:-}" = "repack" ]; then
    echo "aligned repacked GEMV (gemv_repack_p40_kernel):"
    for r in 1 2 4; do report_repack_one "${r}"; done
    echo
    echo "  activation:weight byte ratio is 4/R; the target is <=32 regs (64 warps/SM)."
elif [ "${1:-}" = "sweep" ]; then
    echo "sweep over rows-per-warp:"
    for r in 1 2 4; do report_one "${r}"; done
else
    ROWS="${BONSAI_GEMV_ROWS:-1}"
    echo "default build (BONSAI_GEMV_ROWS=${ROWS}):"
    report_one "${ROWS}"
    echo
    echo "  './tools/regs.sh sweep'  -> all three gemv rows configurations"
    echo "  './tools/regs.sh repack' -> the aligned repacked kernel"
fi

cat <<'EOF'

WHAT TO LOOK FOR

  P40 register file: 65536 per SM, max 2048 threads (64 warps) per SM.
  min_blocks=8 -> <= 32 regs -> 8 blocks/SM -> 64 warps/SM -> 100% occupancy
  min_blocks=6 -> <= 42 regs -> 6 blocks/SM -> 48 warps/SM ->  75%
  min_blocks=4 -> <= 64 regs -> 4 blocks/SM -> 32 warps/SM ->  50%

  gemv_pq2_0_p40_kernel  (all three MEASURED on a real P40, docs/MEASURED.md)
    rows=1 : regs=32, 64 warps/SM -> 17.34 tok/s   <- default, best variant
    rows=2 : regs=40, 48 warps/SM -> untested
    rows=4 : regs=64, 32 warps/SM -> 11.39 tok/s   <- the original regression
    stock  : ~15-25 regs, 64 warps/SM -> 21.91 tok/s  <- still faster than all

  Occupancy explained 57% of the rows=4 gap; the rest is architectural (no
  shared-memory activation staging -> 4.24x more L2 read traffic than weight
  traffic).  See docs/MEASURED.md section 6.

  Any non-zero "spill stores" is a red flag: spills live in local memory, which
  is DRAM traffic, on a kernel that is already bandwidth-bound.
EOF
