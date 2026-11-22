# Reproduce

## What can and cannot be reproduced

| | How |
|---|---|
| Every table in the README, from the per-task results | `scripts/recheck.sh` (no GPU) |
| The figures | `figures/make_figures.py` (needs matplotlib) |
| A check on Part 4's public example tasks: both models at one and four streams, the 27B at two, four and eight | step 2 |
| Both studies' runs on your own tasks | step 3 |
| Re-running our scores | **not possible from this repository**: both task sets are private (kept unseen for future model comparisons) |

## 1. Prerequisites

The runner, the 27B server and the grader are Part 4's; the Flash Next server is Part 6's.
- Check out [Part 4's folder](https://github.com/NautilusWu/qwen3.8_study/blob/main/qwen38-spark-test-time-strategies/REPRODUCE.md) and [Part 6's folder](https://github.com/NautilusWu/qwen3.8_study/blob/main/qwen38-spark-flash-next/REPRODUCE.md) next to this one, as in the repository.
- Follow Part 4's setup: the TensorFold 0.6.0 image, the 27B checkpoints (NVFP4 target and DFlash2 drafter) and the sandbox image.
- Follow Part 6's step 1 for the Flash Next checkpoint.

Nothing else may run on the GPU during a run. Each block starts its own server, and only one model is loaded at a time:
- the 27B left 84-90 GiB available (up to 8 streams);
- Flash Next left 33-36 GiB.

## 2. Check on Part 4's example tasks

All commands in this step run in `code/`, with this environment (add `HF_DIR`, `TF_CACHE_DIR`, and `FN_TARGET` if your Flash Next checkpoint is elsewhere, as in Parts 4 and 6):

```bash
cd code
export TASKS_DIR=../../qwen38-spark-test-time-strategies/examples/tasks GRADER_DIR=../../qwen38-spark-test-time-strategies/examples/grader
P4=../../qwen38-spark-test-time-strategies/code/strategies
P6=../../qwen38-spark-flash-next/code
```

**Four streams** (one block per model, 5 tasks, four at a time):

```bash
python3 run_streams.py $P4 check-4streams --model 27b=27b --model flashnext=flashnext:$P6 --streams 4 --orders "27b,flashnext" --split dev
```

**One stream:**

```bash
python3 run_streams.py $P4 check-1stream --model 27b=27b --model flashnext=flashnext:$P6 --streams 1 --orders "flashnext,27b" --split dev
```

Expected: greedy replies do not depend on the number of streams. The output tokens per task should be the same in both runs:

| Model | Correct | Output tokens (`ex_five_numbers`, `ex_gcd_cpp`, `ex_palindrome`, `ex_primes`, `ex_trains`) |
|---|---|---|
| 27B | 4 of 5 (`ex_five_numbers` wrong) | 1,780, 577, 116, 741, 523 = 3,737 |
| Flash Next | 5 of 5 | 607, 399, 120, 900, 515 = 2,541 |

**Ours.** The two runs took about 9 minutes together. They used a fresh copy of the Part 4, 6 and 12 folders and a new, empty kernel cache.
- **Exit codes:** both runs 0; all 4 blocks finished.
- **Replies:** correct answers and tokens as in the table, for both models in both runs. All 20 replies were character for character the same as in this study's two compatibility checks.
- **Concurrency:** the server reported at most 4 running requests at four streams, and 1 at one stream.

| | 27B, 4 streams | 27B, 1 stream | Flash Next, 4 streams | Flash Next, 1 stream |
|---|---|---|---|---|
| Server start | 87.5 s (first start: kernels built) | 15.1 s | 142.1 s (first start) | 87.9 s |
| Block time (5 tasks) | 25.4 s | 38.0 s | 26.3 s | 38.7 s |
| Lowest available memory | 85.4 GiB | 90.1 GiB | 33.5 GiB | 34.1 GiB |

On 5 tasks, the time comparison in `stats.json` means nothing.

**The 27B at 2, 4 and 8 streams** (PLAN_21b's three arms in one run; `@N` gives an arm its own stream count):

```bash
python3 run_streams.py $P4 check-2-4-8 --model 27b_p2=27b@2 --model 27b_p4=27b@4 --model 27b_p8=27b@8 \
  --orders "27b_p2,27b_p4,27b_p8" --split dev
```

Expected: the same replies as above in all three arms (4 of 5 correct, 3,737 tokens), and `same_text.csv` all 1.

**Ours.** About 4 minutes (2026-11-19, 23:55-23:59), in a fresh copy of the Part 4, 6 and 12 folders with a new, empty kernel cache.
- **Exit code** 0; all 3 blocks finished.
- **Replies:** 4 of 5 correct in every arm, tokens as in the table, and all 15 replies character for character the same as in PLAN_20's compatibility check.

| | 2 streams | 4 streams | 8 streams |
|---|---|---|---|
| Server start | 84.5 s (first start: kernels built) | 21.2 s | 21.2 s |
| Block time (5 tasks) | 29.2 s | 24.5 s | 21.9 s |
| Lowest available memory | 84.3 GiB | 87.9 GiB | 88.0 GiB |
| Largest `requests_running` seen | 2 | 4 | 5 (only 5 tasks) |

## 3. The studies on your own tasks

All commands run in `code/`, with `TASKS_DIR` and `GRADER_DIR` pointing at your tasks. The task and grader format is in Part 4's REPRODUCE.md.

[code/run_streams.py](code/run_streams.py) runs both models on every group of tasks, in alternating blocks:
- a fresh server with `--parallel N` for every block, each block in its own process, after waiting for the GPU to be released;
- with one stream, one task at a time, waiting for the server to be idle before each;
- with N streams, N workers taking tasks from one queue in the group's order;
- two failed blocks in a row stop the run;
- grading at the end in the network-isolated sandbox, pinned to the Cortex-A725 cores (`SANDBOX_CPUS`).

It writes:
- per-task CSVs with the columns of `data/per_task_400/`, plus the shared time `shared_s`;
- `stats_<model>.csv`, ready for `stats.py`;
- `same_text.csv` and `blocks.csv`;
- with exactly two arms, `stats.py`'s comparison of them.

Task text and model output stay in `results/` and are never printed.

**PLAN_19, one stream, this study's 8 shuffled groups and ABBA order:**

```bash
python3 run_streams.py $P4 one-stream --model 27b=27b --model flashnext=flashnext:$P6 --streams 1 \
  --shuffle-seed 2026 --split confirm --limit-min 40 --max-tokens 65536
```

**PLAN_20, four streams, the same groups and order:**

```bash
python3 run_streams.py $P4 four-streams --model 27b=27b --model flashnext=flashnext:$P6 --streams 4 \
  --shuffle-seed 2026 --split confirm --limit-min 40 --max-tokens 65536
```

With two models, the default order is AB BA BA AB AB BA BA AB (8 groups). Note that `--shuffle-seed` shuffles your task file's order: our groups were cut from set O's private order, so the same seed gives different groups on other tasks.

**PLAN_21b, the 27B at 2, 4 and 8 streams, the same groups, each group's three stream counts in a Williams order:**

```bash
python3 run_streams.py $P4 streams-2-4-8 --model 27b_p2=27b@2 --model 27b_p4=27b@4 --model 27b_p8=27b@8 \
  --orders "27b_p2,27b_p4,27b_p8;27b_p4,27b_p8,27b_p2;27b_p8,27b_p2,27b_p4;27b_p8,27b_p4,27b_p2;27b_p2,27b_p8,27b_p4;27b_p4,27b_p2,27b_p8;27b_p2,27b_p4,27b_p8;27b_p8,27b_p4,27b_p2" \
  --shuffle-seed 2026 --split confirm --limit-min 40 --max-tokens 65536
python3 stats.py --limit-min 40 27b_p8=../results/streams-2-4-8/stats_27b_p8.csv 27b_p4=../results/streams-2-4-8/stats_27b_p4.csv
```

The second command is PLAN_21b's Q1 (8 against 4 streams). With more than two arms, the runner does not run `stats.py` itself. Our PLAN_21b runner polled `/health` every second, this one every 2 s; that only changes the largest `requests_running` it can see.

**Four streams against one stream, for one model** (PLAN_20's Q2):

```bash
python3 stats.py --limit-min 40 27b_4=../results/four-streams/stats_27b.csv 27b_1=../results/one-stream/stats_27b.csv
```

`--method` takes any of Part 4's single-reply methods (default `D2`, low thinking). `--model` can be given once, to run one model alone.

## 4. Recompute the tables

```bash
scripts/recheck.sh
```

Expected last lines:
- `summary.csv: all 7 rows match`
- `tests: all match`
- `reported only: all match`

It recomputes:
- every locked test (PLAN_19, PLAN_20, PLAN_21b), exactly as the locked analyses did: `stats.py` for the paired tests, and for the set comparison each set's tasks resampled independently in the locked analysis's order;
- everything looked at after the results.

```bash
python3 figures/make_figures.py
```
Redraws the figures (needs matplotlib).
