#!/usr/bin/env bash
# Run DDU-SDDiP vs SDDiP for every combination of
# instance type (A B C D) and N_SAA (25 50).
#
# Usage:
#   bash run_all.sh [TYPE [N_SAA [SCRIPT]]]
#
#   TYPE   : A | B | C | D          (default: all four)
#   N_SAA  : number of SAA scenarios (default: 25 and 50)
#   SCRIPT : G | G2 | both          (default: both)
#
# Examples:
#   bash run_all.sh                  # all 8 combinations, both scripts
#   bash run_all.sh A                # type A, both N_SAA values, both scripts
#   bash run_all.sh A 25             # type A, N_SAA=25, both scripts
#   bash run_all.sh A 25 G           # type A, N_SAA=25, only G
#   bash run_all.sh A 25 G2          # type A, N_SAA=25, only G2
#   bash run_all.sh "" "" G2         # all combinations, only G2 
#
# Logs are written to facility_location/results/<TYPE>/run_G_<TYPE>_N<N_SAA>.log
#                                                  and run_G2_<TYPE>_N<N_SAA>.log
# Failed runs are recorded in facility_location/results/failed_runs.txt

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUNNER_G="$SCRIPT_DIR/problem_src/run_comparison_G.jl"
RUNNER_G2="$SCRIPT_DIR/problem_src/run_comparison_G2.jl"
FAILED_LOG="$SCRIPT_DIR/results/failed_runs.txt"

TYPES=("A" "B" "C" "D")
NSAA_VALUES=(25 50)
WHICH="both"

if [ $# -ge 1 ] && [ -n "$1" ]; then TYPES=("$1"); fi
if [ $# -ge 2 ] && [ -n "$2" ]; then NSAA_VALUES=("$2"); fi
if [ $# -ge 3 ]; then WHICH="$3"; fi

if [[ "$WHICH" != "G" && "$WHICH" != "G2" && "$WHICH" != "both" ]]; then
    echo "ERROR: SCRIPT must be G, G2, or both (got '$WHICH')"
    exit 1
fi

mkdir -p "$SCRIPT_DIR/results"
> "$FAILED_LOG"   # clear/create the failed runs file

FAILED=0
SUCCEEDED=0

run_script() {
    local SCRIPT="$1"
    local TAG="$2"
    local LOG="$3"
    local ITYPE="$4"
    local NSAA="$5"

    echo "----------------------------------------"
    echo "  Script: $TAG  |  type=$ITYPE  N_SAA=$NSAA"
    echo "  Log: $LOG"
    echo "----------------------------------------"

    if julia "$SCRIPT" "$ITYPE" "$NSAA" 2>&1 | tee "$LOG"; then
        echo "  Done: $TAG  type=$ITYPE  N_SAA=$NSAA"
        SUCCEEDED=$((SUCCEEDED + 1))
    else
        echo "  FAILED: $TAG  type=$ITYPE  N_SAA=$NSAA  (see $LOG)"
        echo "FAILED: $TAG  type=$ITYPE  N_SAA=$NSAA  log=$LOG" >> "$FAILED_LOG"
        FAILED=$((FAILED + 1))
    fi
    echo ""
}

for ITYPE in "${TYPES[@]}"; do
    for NSAA in "${NSAA_VALUES[@]}"; do
        LOG_DIR="$SCRIPT_DIR/results/$ITYPE"
        mkdir -p "$LOG_DIR"

        echo "========================================"
        echo "  Combination: type=$ITYPE  N_SAA=$NSAA"
        echo "========================================"

        if [[ "$WHICH" == "G" || "$WHICH" == "both" ]]; then
            run_script "$RUNNER_G"  "G"  "$LOG_DIR/run_G_${ITYPE}_N${NSAA}.log"  "$ITYPE" "$NSAA"
        fi
        if [[ "$WHICH" == "G2" || "$WHICH" == "both" ]]; then
            run_script "$RUNNER_G2" "G2" "$LOG_DIR/run_G2_${ITYPE}_N${NSAA}.log" "$ITYPE" "$NSAA"
        fi
    done
done

echo "========================================"
echo "  All runs attempted."
echo "  Succeeded: $SUCCEEDED"
echo "  Failed:    $FAILED"
if [ $FAILED -gt 0 ]; then
    echo "  Failed runs recorded in: $FAILED_LOG"
    cat "$FAILED_LOG"
fi
echo "========================================"
