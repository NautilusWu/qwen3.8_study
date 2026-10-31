#!/usr/bin/env bash
# Recompute the comparison tables from the published per-task results (no GPU, no task files needed).
#   scripts/recheck.sh 72     the 72-task model comparison (dev + confirm sets of Part 4)
#   scripts/recheck.sh 400    the 400 new tasks, all six configurations
set -euo pipefail
cd "$(dirname "$0")/.."
set="${1:?usage: recheck.sh 72|400}"
args=()
for f in data/per_task_${set}/*.csv; do args+=("$(basename "$f" .csv)=$f"); done
python3 code/stats.py --limit-min 40 "${args[@]}"
