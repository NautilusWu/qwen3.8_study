#!/usr/bin/env bash
# Recompute the tables from the published per-task results (no GPU, no task files needed).
#   scripts/recheck.sh            the three locked comparisons, Flash Next vs 27B for each usage
#   scripts/recheck.sh all        all six configurations, every pair (reported only; Holm over all 15 pairs)
#   scripts/recheck.sh maxlimit   Flash Next low thinking, 65,536 vs 260,096-token limit (no time limit)
set -euo pipefail
cd "$(dirname "$0")/.."
D=data/per_task_400
case "${1:-locked}" in
  locked)
    for m in deep_A65 low_D2 vote4_low_CL4; do
      echo "== $m"
      python3 code/stats.py flashnext_$m=$D/flashnext_$m.csv qwen27b_$m=$D/qwen27b_$m.csv
    done
    echo "Holm over the three pairs: see data/tests_400_preregistered.json (holm_over_3)" ;;
  all)
    args=()
    for f in $D/flashnext_deep_A65.csv $D/flashnext_low_D2.csv $D/flashnext_vote4_low_CL4.csv \
             $D/qwen27b_deep_A65.csv $D/qwen27b_low_D2.csv $D/qwen27b_vote4_low_CL4.csv; do
      args+=("$(basename "$f" .csv)=$f")
    done
    python3 code/stats.py "${args[@]}" ;;
  maxlimit)
    python3 code/stats.py --limit-min 100000 low_65536=$D/flashnext_low_D2.csv low_260096=$D/flashnext_low_D2_maxlimit.csv ;;
  *) sed -n '2,5p' "$0"; exit 2 ;;
esac
