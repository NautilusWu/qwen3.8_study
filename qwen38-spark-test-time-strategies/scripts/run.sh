#!/usr/bin/env bash
# The study's runs on your own tasks (see REPRODUCE.md for the task and grader format).
#   TASKS_DIR=... GRADER_DIR=... scripts/run.sh strategies <prefix> [split]
#       A65, D2, H and CL4 with the confirmation-phase settings on TASKS_DIR/<split>.jsonl (default confirm),
#       then the five pre-registered tests C1-C5 (code/strategies/results/<prefix>-tests.json).
#   AGENT_TASKS_DIR=... scripts/run.sh agents <run_id> [qwen,opencode,codex,claude]
#       every coding agent on every task, once each (code/coding_agents/results/<run_id>/).
#   scripts/run.sh recheck
#       recompute C1-C5 from the published per-task results (data/per_task.csv); no GPU needed.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
case "${1:-}" in
  strategies)
    : "${TASKS_DIR:?set TASKS_DIR}" "${GRADER_DIR:?set GRADER_DIR}" "${2:?usage: run.sh strategies <prefix> [split]}"
    p="$2"; split="${3:-confirm}"
    cd "$ROOT/code/strategies"
    python3 run_method.py A65 "$p-A65" --split "$split" --limit-min 40 --context 81920
    python3 run_method.py D2 "$p-D2" --split "$split" --limit-min 40 --context 81920 --max-tokens 65536
    python3 run_method.py H "$p-H" --split "$split" --limit-min 40 --context 81920
    python3 run_method.py CL "$p-CL4" --k 4 --split "$split" --limit-min 40 --context 81920 --max-tokens 65536
    python3 analyze_confirm.py "$p" ;;
  agents)
    : "${AGENT_TASKS_DIR:?set AGENT_TASKS_DIR}" "${2:?usage: run.sh agents <run_id> [agents]}"
    cd "$ROOT/code/coding_agents"
    python3 run_all.py "$2" --agents "${3:-qwen,opencode,codex,claude}" ;;
  recheck)
    cd "$ROOT/code/strategies"
    python3 analyze_confirm.py --csv "$ROOT/data/per_task.csv" ;;
  *) sed -n '2,9p' "$0"; exit 2 ;;
esac
