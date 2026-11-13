#!/usr/bin/env bash
# Recompute every table of the README from the published files (no GPU, no task files needed): code/tables.py prints
# them, recomputes data/summary.csv and runs code/stats.py on the per-task files (the locked test, and 0.6.0 in this
# study against Part 5's run), then compares both with the published files.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -B code/tables.py --check
