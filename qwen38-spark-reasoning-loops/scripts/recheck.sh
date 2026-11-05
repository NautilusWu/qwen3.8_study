#!/usr/bin/env bash
# Recompute the tables from the published per-task results (no GPU, no task files needed).
#   scripts/recheck.sh            400 tasks: Flash Next greedy + budget, sampling + budget, greedy; 27B greedy + budget, greedy
#   scripts/recheck.sh pilot      stage 1, 79 tasks: G vs S, B, D
set -euo pipefail
cd "$(dirname "$0")/.."
D=data/per_task_400
case "${1:-400}" in
  400)
    echo "== Flash Next, deep thinking"
    python3 code/stats.py --limit-min 40 greedy_budget=$D/flashnext_deep_greedy_budget.csv \
        sampling_budget=$D/flashnext_deep_sampling_budget.csv greedy=$D/flashnext_deep_greedy.csv
    echo "== Qwen3.8-27B, deep thinking"
    python3 code/stats.py --limit-min 40 greedy_budget=$D/qwen27b_deep_greedy_budget.csv greedy=$D/qwen27b_deep_greedy.csv ;;
  pilot)
    python3 - <<'PY'
import csv, collections, subprocess, sys, tempfile, os
rows = list(csv.DictReader(open('data/stage1_pilot79.csv')))
tmp = tempfile.mkdtemp()
args = []
for arm in ('G', 'S', 'B', 'D'):
    p = os.path.join(tmp, arm + '.csv')
    with open(p, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['task_id', 'passed', 'wall_s'])
        for r in rows:
            if r['arm'] == arm:
                w.writerow([r['task_id'], r['passed'], r['wall_s']])
    args.append(f'{arm}={p}')
print(subprocess.run([sys.executable, 'code/stats.py', '--limit-min', '40', *args], capture_output=True, text=True, check=True).stdout)
PY
    ;;
  *) sed -n '2,4p' "$0"; exit 2 ;;
esac
