# Reproduce

## What can and cannot be reproduced

| | How |
|---|---|
| Every table in the README, from the published files | `scripts/recheck.sh` (step 5; no GPU) |
| The figures | `figures/make_figures.py` (needs matplotlib) |
| Serving the 27B with TensorFold 1.2.5 at this study's settings, with a check on example tasks | steps 1-2 |
| The comparison with your own tasks: both engines, alternating blocks | step 3 |
| The speed microbench | step 4 |
| Re-running our scores | **not possible from this repository**: the 400 tasks are private (kept unseen for future model comparisons) |

## 0. Prerequisites

- **Part 4's setup:** a DGX Spark, Docker with the NVIDIA Container Toolkit, and [Part 4's `scripts/setup.sh`](https://github.com/NautilusWu/qwen3.8_study/blob/main/qwen38-spark-test-time-strategies/REPRODUCE.md). It builds the TensorFold 0.6.0 images (whose base image, `nvcr.io/nvidia/pytorch@sha256:2140e699…`, 1.2.5 also runs in) and the sandbox image, and downloads the two checkpoints.
- **Part 4's folder** checked out next to this one. Its `code/strategies` is used unchanged.
- **Driver R580 or newer** for 1.2.5.
- **Nothing else on the GPU.** One server runs at a time (the lowest available memory in our runs was 75 GiB on 1.2.5 and 81 GiB on 0.6.0), and another GPU job can make the memory guard stop the run.

## 1. Download and check TensorFold 1.2.5 (46.6 MiB)

Into a new, empty folder outside this repository:

```bash
mkdir tf-1.2.5 && cd tf-1.2.5
curl -LO https://github.com/ashhart/TensorFold/releases/download/v1.2.5/tensorfold-1.2.5-linux-aarch64.tar.gz
curl -LO https://github.com/ashhart/TensorFold/releases/download/v1.2.5/tensorfold-1.2.5-linux-aarch64.tar.gz.sha256
echo "1a0192b28d6ab6ec2749f3ff25515379dba4895171e781b30f810afc971ae4f8  tensorfold-1.2.5-linux-aarch64.tar.gz" | sha256sum -c
sha256sum -c tensorfold-1.2.5-linux-aarch64.tar.gz.sha256
tar xzf tensorfold-1.2.5-linux-aarch64.tar.gz
(cd tensorfold-1.2.5-linux-aarch64 && sha256sum -c --quiet SHA256SUMS && cat VERSION)
chmod -R a-w tensorfold-1.2.5-linux-aarch64
export TF_NATIVE_DIR=$PWD/tensorfold-1.2.5-linux-aarch64 TF_NATIVE_ARCHIVE=$PWD/tensorfold-1.2.5-linux-aarch64.tar.gz
```

Expected:
- the two `OK` lines;
- no output from `SHA256SUMS`, then `1.2.5`.

We found no public key for the `.sig` file, so it was not checked. The archive holds 28 entries, with no absolute paths and no `..`. `code/native_engine.py` mounts the folder read-only and runs `bin/tensorfold-native` in the base image; nothing is installed.

## 2. Check on Part 4's example tasks

```bash
cd code
export HF_DIR=... TASKS_DIR=../../qwen38-spark-test-time-strategies/examples/tasks GRADER_DIR=../../qwen38-spark-test-time-strategies/examples/grader
python3 run_engines.py ../../qwen38-spark-test-time-strategies/code/strategies check --split dev --first new
```

This runs the 5 example tasks with CL4: first on 1.2.5, then on 0.6.0, with a server start before each. It first prints the release check, which should show `"files_ok": true, "version": "tensorfold-native 1.2.5", "version_ok": true` and, with `TF_NATIVE_ARCHIVE` set, `"archive_ok": true`.

Expected:
- every task gets a vote result on both engines;
- `results/check/` holds the two runs, `per_task_tf125.csv`, `per_task_tf060.csv`, `branches.csv` and `blocks.csv`;
- the last output is `stats.py`'s comparison.

1.2.5's server log (`results/check/tf125/server_start_b0.log`) should mention the FP8 caches, `DFlash2` and 4 streams at once. A start takes about a minute on 1.2.5 and about 20 s on 0.6.0.

**Ours** (2026-11-11, on a fresh copy of this folder next to Part 4's, as in the repository): about 4 minutes in all.
- The release check passed.
- Both engines answered 5/5 tasks correctly, with the same vote rule per task: `majority` on the three reasoning tasks, `first_with_code` on the two coding tasks. 11 of 11 finished branches were correct on each engine.
- Server starts took 66.4 s on 1.2.5, of which 46.3 s was the verification window measurement, and 21.2 s on 0.6.0.
- The lowest available memory was 76.8 and 85.2 GiB.
- The 1.2.5 server had only local connections.

The study's locked code gave the same outcome before the lock (`data/compat_check.json`). Sampling differs between the engines and between software versions, so your replies will not match ours token for token.

To use 1.2.5 in Part 4's runner for any of its methods instead (no comparison):

```bash
python3 run_native.py ../../qwen38-spark-test-time-strategies/code/strategies CL tf125-check --k 4 --split dev --limit-min 40 --context 81920 --max-tokens 65536
```

## 3. The comparison on your own tasks

In `code/`, with `HF_DIR`, `TASKS_DIR` and `GRADER_DIR` set for your tasks (task and grader format as in Part 4's REPRODUCE.md):

```bash
python3 run_engines.py ../../qwen38-spark-test-time-strategies/code/strategies cmp1 --split confirm
```

- **Groups.** The tasks, in file order, form as many groups as `--first` has entries, of nearly equal size. The default is the study's order: `new,old,old,new,old,new,new,old`, 8 groups. Each group is run by both engines, the named one first.
- **Before each block and each task.** The server restarts and gets the warm-up request before every block. Before each task, the runner waits until the server is idle; a branch the vote cancelled may still be finishing, and that wait is not timed.
- **Failures.** A block that fails prints `FAILED` and the next one runs. Its tasks count as wrong at the time limit. The memory guard (12 GiB) stops the run.
- **Results.** Everything goes to `results/cmp1/` in this folder:
  - both runs, graded;
  - the per-task, per-branch and per-block CSVs, with the columns of `data/` except those that compare texts with other runs;
  - `stats.json`: `stats.py` comparing `tf125` and `tf060`.
- **Grading.** Grading containers are pinned to the cores in `SANDBOX_CPUS` (default `0-4,10-14`, the GB10's Cortex-A725 cores, away from the server's). Graders with time limits depend on the core type; set it empty to leave them unpinned. Grading runs 50 items per container.

`code/native_engine.py` lists every difference between the two servers. The 0.6.0 server is Part 4's own `start_server`, unchanged.

## 4. Speed microbench

```bash
python3 microbench.py ../../qwen38-spark-test-time-strategies/code/strategies bench1
```

Six rounds, alternating the engines (`--order old,new,new,old,old,new`), each with a fresh server start.
- **Prompts:** fixed public prompts, all in the script.
- **One stream:** short code and short chat (thinking off) and a long thinking answer.
- **4 streams at once.**
- **Cold prompts** of about 2K, 8.6K and 36K tokens, with a different start in every round so no prompt cache can help.

It writes `results/bench1/microbench.csv` (the columns of `data/microbench.csv`) and the medians in `summary.json`.

Keep the display state fixed for the whole run: in Part 2 the display changed decoding speed by 9-18%.

## 5. Recompute the tables

From this folder's root:

```bash
scripts/recheck.sh
```

It runs `code/tables.py --check`, which prints every table in the README from `data/`. Then:
- it recomputes `data/summary.csv`;
- it runs `code/stats.py` on the per-task files: the locked test, and 0.6.0 in this study against Part 5's run;
- it compares both with the published files.

Expected last lines:
- `summary.csv: all 3 rows match`
- `tests: all match`

`code/stats.py` (as in Parts 5-9) prints totals and, per pair:
- the tasks only one side solved, and the exact McNemar p;
- paired bootstrap 95% intervals (10,000 resamples, seed 2026) for the accuracy difference and the time ratio.

The per-task files are in the same task order as the locked analysis, so the intervals come out identical.

**Columns.**
- **Per-task files:**
  - `wall_s`: seconds until the vote's choice, at most 2,400;
  - `output_tokens`: the tokens of all 4 branches, counting 0 for a cancelled branch (a cancelled stream reports no count);
  - `vote_rule`: how the answer was chosen: `first_with_code`, `majority`, `plurality`, `tie_earliest`, or `no_answer` when no branch gave an answer the vote could read;
  - `chosen_same_text_as_*`: 1 when the chosen answer's text equals that of the run named.
- **`data/branches.csv`:**
  - `end_s`: seconds from the task's start;
  - `first_token_s`: the time from sending to the first token;
  - `tok_s_from_wall`: tokens per second between the first token and the end;
  - `tok_s_engine`: 1.2.5's own figure;
  - `same_text_*`: 1 when the branch with the same seed finished with the same text in the run named.
