#!/usr/bin/env bash
# Recompute every table and test from the published data and compare with data/summary.csv,
# data/tests_preregistered.json and data/reported_only.json (no GPU, no task files needed). The cross-plan rows also read
# Part 10's and Part 11's per-task files from the sibling folders of this repository. Expected last lines:
#   summary.csv: all 7 rows match
#   tests: all match
#   reported only: all match
set -euo pipefail
cd "$(dirname "$0")/.."
python3 code/tables.py --check
