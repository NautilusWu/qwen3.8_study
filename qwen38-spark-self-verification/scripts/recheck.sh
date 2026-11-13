#!/usr/bin/env bash
# Recompute everything in the README from the published per-task files (no GPU, no task files needed).
#   scripts/recheck.sh            all: rules, arms, tables, tests
#   scripts/recheck.sh rules      the selection rules' own checks on hand-made records, and the prompts' hash
#   scripts/recheck.sh arms       A, B, B2 and C recomputed from the recorded times, checked against data/*/arm_*.csv
#   scripts/recheck.sh tables     every table, checked against data/summary.csv and data/tester_summary.csv
#   scripts/recheck.sh tests      the paired tests, checked against the locked analyses (data/tests.json)
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
rules() { (cd code/rules && python3 check_p11_rules.py && python3 check_p11c_rules.py); python3 code/prompts.py; }
arms() { python3 code/arms.py; }
tables() { python3 code/tables.py; }
tests() { python3 code/tests_check.py; }
case "${1:-all}" in
  all) rules; echo; arms; echo; tables; echo; tests ;;
  rules) rules ;;
  arms) arms ;;
  tables) tables ;;
  tests) tests ;;
  *) sed -n '2,7p' "$0"; exit 2 ;;
esac
