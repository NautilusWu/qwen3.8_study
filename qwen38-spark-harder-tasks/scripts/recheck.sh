#!/usr/bin/env bash
# Recompute the tables and tests from the published per-task results (no GPU, no task files needed).
#   scripts/recheck.sh            all: tables, then the tests
#   scripts/recheck.sh tables     per-group tables, checked against data/summary.csv
#   scripts/recheck.sh tests      paired tests (exact McNemar, bootstrap 95% intervals, 10,000 resamples, seed 2026)
set -euo pipefail
cd "$(dirname "$0")/.."
S="python3 code/stats.py --limit-min 40 --boot 10000 --seed 2026"
tables() { python3 code/tables.py; }
tests() {
  for set in h250 arxivmath; do
    echo "== PLAN_10, $set: Flash Next deep + 32K budget vs 27B vote over 4 low-thinking branches"
    $S flashnext_B=data/per_task_$set/flashnext_deep_budget.csv qwen27b_CL4=data/per_task_$set/qwen27b_vote4_low_CL4.csv
  done
  D=data/per_task_72
  for m in low_D2 vote4_low_CL4 deep_A65; do
    echo "== PLAN_09, 72 tasks, $m: Flash Next vs 27B"
    $S flashnext=$D/flashnext_$m.csv qwen27b=$D/qwen27b_$m.csv
  done
  echo "== PLAN_09, 72 tasks, Flash Next only (reported only): deep + budget, deep, vote, low"
  $S budget=$D/flashnext_deep_budget.csv deep=$D/flashnext_deep_A65.csv vote=$D/flashnext_vote4_low_CL4.csv low=$D/flashnext_low_D2.csv
}
case "${1:-all}" in
  all) tables; echo; tests ;;
  tables) tables ;;
  tests) tests ;;
  *) sed -n '2,5p' "$0"; exit 2 ;;
esac
