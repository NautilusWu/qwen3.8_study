# Reproduce

## What can and cannot be reproduced

| | How |
|---|---|
| The tables in the README from the per-task results | `scripts/recheck.sh`, `scripts/recheck.sh all`, `scripts/recheck.sh maxlimit` (no GPU) |
| The figures | `figures/make_figures.py` (needs matplotlib) |
| Serving Flash Next with this study's settings, and a check on example tasks | steps 1-2 below |
| The MTP sweep, on the example tasks or your own | step 4 |
| Running the comparison with your own tasks | step 3 (uses Part 4's runner) |
| Re-running our scores | **not possible from this repository**: the 400 tasks are private (kept unseen for future model comparisons) |

## 0. Prerequisites

Part 4's setup: a DGX Spark, Docker with the NVIDIA Container Toolkit, and [Part 4's `scripts/setup.sh`](https://github.com/NautilusWu/qwen3.8_study/blob/main/qwen38-spark-test-time-strategies/REPRODUCE.md), which builds the TensorFold 0.6.0 images and the sandbox image (`SKIP_MODELS=1` skips the 27B checkpoints if you only want Flash Next). Check out Part 4's folder next to this one. Machine and versions used here are in the README's Conditions. Nothing else may run on the GPU during a run: Flash Next takes about 90 GiB of the 128 GB.

## 1. Download Flash Next (105.9 GB, 50 files)

Into the same Hugging Face cache Part 4 uses (`HF_DIR`, mounted as `/hf`), at the commit used:

```bash
HF_HOME="$HF_DIR" hf download local-inference-lab/Qwen3.8-Flash-Next-NVFP4 --revision 7c4f1bc1a2d6847e0cbc01ac6b823f00251de8dd
```

`code/flash_next.py` expects it at `/hf/hub/models--local-inference-lab--Qwen3.8-Flash-Next-NVFP4/snapshots/7c4f1bc1a2d6847e0cbc01ac6b823f00251de8dd`; set `FN_TARGET` to another path inside `/hf` if yours differs.

## 2. Check on Part 4's example tasks

```bash
cd code
export HF_DIR=... TASKS_DIR=../../qwen38-spark-test-time-strategies/examples/tasks GRADER_DIR=../../qwen38-spark-test-time-strategies/examples/grader
python3 run_flash_next.py ../../qwen38-spark-test-time-strategies/code/strategies D2 fn-check --limit-min 40 --context 81920 --max-tokens 65536
```

Expected: the server starts (about 4 minutes the first time, which compiles kernels; about 90 s later), its log in `<Part 4>/code/strategies/results/fn-check/server_start.log` (Part 4's runner writes its results there) says `a chain stops before a later draft under 60%`, and all 5 tasks are graded. Ours: 5/5 correct, 2,541 output tokens (greedy, so the same tokens should come back on the same software).

## 3. Run with your own tasks

In `code/`, with `HF_DIR`, `TASKS_DIR` and `GRADER_DIR` set for your tasks. `code/run_flash_next.py` runs Part 4's `run_method.py` unchanged, with Flash Next as the server (`code/flash_next.py` lists every difference from Part 4's 27B server). The study's three configurations, on your `TASKS_DIR/confirm.jsonl` (task and grader format as in Part 4's REPRODUCE.md):

```bash
S=../../qwen38-spark-test-time-strategies/code/strategies
O="--split confirm --limit-min 40 --context 81920 --max-tokens 65536"
python3 run_flash_next.py $S D2 fn-D2 $O
python3 run_flash_next.py $S CL fn-CL4 --k 4 $O
python3 run_flash_next.py $S A65 fn-A65 $O
```

`A65` (deep thinking) always uses a 65,536-token limit; for a different limit use method `A` (the same request otherwise), which takes `--max-tokens`. Draft settings: `FN_EXTRA` (default `--mtp-confidence 0.6 --mtp-drafts 6`). For the 27B side, run the same methods with Part 4's runner directly, then compare with `python3 stats.py NAME=<results>/graded.jsonl ...` (paired by task id).

To record memory during a run: `python3 memlog.py <checkpoint folder on the host> mem.jsonl` in a second terminal (available memory every 2 s, n-gram tables in the page cache every 20 s).

## 4. MTP sweep

In `code/`, with the environment of step 2:

```bash
python3 mtp_sweep.py ../../qwen38-spark-test-time-strategies/code/strategies sweep1 --settings 0.7:6,0.6:6,0.5:6,0:3,0:4,serial
```

Restarts the server for every setting (`THRESHOLD:DRAFTS`, or `serial` for no drafts), sends the same replies (by default deep thinking on `ex_primes` and `ex_gcd_cpp`, low thinking on all 5 examples), checks that every setting returns the same tokens, and prints the time per setting. Keep the display state fixed for the whole sweep; switching it changed our timings by 10-17%.

## 5. Recompute the tables

From this folder's root:

```bash
scripts/recheck.sh            # the three locked comparisons
scripts/recheck.sh all        # all six configurations, every pair (reported only)
scripts/recheck.sh maxlimit   # low thinking, 65,536 vs 260,096-token limit
```

`code/stats.py` (as in Part 5) prints totals and, per pair, the tasks only one side solved, the exact McNemar p, Holm-adjusted p over the pairs it prints, and paired bootstrap 95% intervals (10,000 resamples, seed 2026). The locked Holm adjustment over the three pairs is in `data/tests_400_preregistered.json` (`holm_over_3`).
