#!/usr/bin/env bash
# Recompute every table from the published per-task results and compare with data/summary.csv and
# data/tests_preregistered.json (no GPU, no task files needed). Expected last lines:
#   summary.csv: all 6 rows match
#   tests: all match
set -euo pipefail
cd "$(dirname "$0")/.."
python3 code/tables.py --check
