#!/usr/bin/env bash
# Acceptance check after setup.sh: runs the pipeline end to end on the example tasks in examples/ and prints a table.
#   scripts/smoke_test.sh                     # strategies D2, CL4, H on 5 tasks + 4 coding agents on 1 task (~15 min)
#   SMOKE_AGENTS=opencode scripts/smoke_test.sh   # fewer agents; SMOKE_AGENTS= skips the agents
# Passes when every step runs and every answer is graded. How many answers are correct is printed, not checked:
# it is the model's result, not the pipeline's (ours: D2 4/5, CL4 5/5, H 5/5, every agent 1/1).
# Needs the GPU free (stop any other model server first). Results: code/strategies/results/smoke-*,
# code/coding_agents/results/smoke-agents.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export TASKS_DIR="$ROOT/examples/tasks" GRADER_DIR="$ROOT/examples/grader" AGENT_TASKS_DIR="$ROOT/examples/agent_tasks"
AGENTS="${SMOKE_AGENTS-qwen,opencode,codex,claude}"

if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
    echo "the GPU is in use; stop other model servers first:" >&2
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv >&2
    exit 1
fi

fail=0
cd "$ROOT/code/strategies"
for spec in "D2 smoke-D2" "CL smoke-CL4 --k 4" "H smoke-H"; do
    set -- $spec
    rm -rf "results/$2" "results/$2.log"
    mkdir -p results
    echo "== $spec"
    python3 run_method.py "$@" > "results/$2.log" 2>&1 || { echo "FAILED: $spec (log: code/strategies/results/$2.log)"; tail -5 "results/$2.log"; fail=1; }
done
if [ -n "$AGENTS" ]; then
    cd "$ROOT/code/coding_agents"
    rm -rf results/smoke-agents results/smoke-agents.log
    mkdir -p results
    echo "== agents $AGENTS"
    python3 run_all.py smoke-agents --agents "$AGENTS" > results/smoke-agents.log 2>&1 \
        || { echo "FAILED: agents (log: code/coding_agents/results/smoke-agents.log)"; tail -5 results/smoke-agents.log; fail=1; }
fi

python3 - "$ROOT" "$AGENTS" <<'PY' || fail=1
import json, sys
from pathlib import Path
root, agents = Path(sys.argv[1]), [a for a in sys.argv[2].split(',') if a]
ok = True
print(f"\n{'step':<22}{'graded':>8}{'correct':>9}{'min/correct':>13}")
for run in ('smoke-D2', 'smoke-CL4', 'smoke-H'):
    d = root / 'code/strategies/results' / run
    graded = len((d / 'graded.jsonl').read_text().splitlines()) if (d / 'graded.jsonl').exists() else 0
    s = json.loads((d / 'summary.json').read_text()) if (d / 'summary.json').exists() else {}
    ok &= graded == 5
    print(f"{run:<22}{graded:>6}/5{s.get('correct', '-'):>9}{s.get('min_per_correct') or '-':>13}")
f = root / 'code/coding_agents/results/smoke-agents/results.jsonl'
rows = [json.loads(l) for l in f.read_text().splitlines()] if agents and f.exists() else []
for a in agents:
    r = next((x for x in rows if x['agent'] == a), None)
    ok &= r is not None
    print(f"{'agent ' + a:<22}{(1 if r else 0):>6}/1{(int(r['passed']) if r else '-'):>9}"
          f"{(round(r['wall_s'] / 60, 2) if r and r['passed'] else '-'):>13}")
sys.exit(0 if ok else 1)
PY
[ $fail -eq 0 ] && echo "SMOKE TEST PASSED: the pipeline runs and grades end to end" || echo "SMOKE TEST FAILED"
exit $fail
