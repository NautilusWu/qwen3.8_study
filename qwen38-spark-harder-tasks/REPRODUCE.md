# Reproducing Part 8

| What | How |
|---|---|
| The tables and tests in the README, from the per-task results | `scripts/recheck.sh` (no GPU); it gives exactly the numbers in [data/tests.json](data/tests.json) and [data/summary.csv](data/summary.csv) |
| The figures | `figures/make_figures.py` (needs matplotlib) |
| The two configurations of step 2, on your own tasks | step 2 below (Part 4's and Part 7's runners, unchanged) |
| The ArXivMath problems | step 3 |

What cannot be reproduced from this folder:
- our scores on h250 and on Part 4's 72 tasks, since those tasks and their graders are private;
- our ArXivMath scores exactly, since the local answer checker is not included.

## 0. Prerequisites

- **27B:** Part 4's setup ([Part 4 REPRODUCE.md](https://github.com/NautilusWu/qwen3.8_study/blob/main/qwen38-spark-test-time-strategies/REPRODUCE.md) steps 0-2: a DGX Spark, Docker with the NVIDIA Container Toolkit, the TensorFold 0.6.0 images, the sandbox image and the 27B checkpoints, all from Part 4's `scripts/setup.sh`, and its example check).
- **Flash Next:** in addition, the checkpoint as in [Part 6 REPRODUCE.md](https://github.com/NautilusWu/qwen3.8_study/blob/main/qwen38-spark-flash-next/REPRODUCE.md) step 1 (105.9 GB, commit `7c4f1bc1`), and Part 7's runner ([Part 7 REPRODUCE.md](https://github.com/NautilusWu/qwen3.8_study/blob/main/qwen38-spark-reasoning-loops/REPRODUCE.md) steps 0-2, including its check on the example tasks).
- **Folders:** check out Parts 4, 6 and 7 next to this folder (the repository has them side by side).
- **GPU:** nothing else may run on it during a run.

This part adds no new runner. The only code change to the runs is the vote key for `\boxed{}` answers ([code/boxed_vote.py](code/boxed_vote.py)). Check it with:

```bash
python3 code/boxed_vote.py
```

It should print `boxed_vote self-check passed`.

## 1. Recompute the tables and tests

From this folder's root:

```bash
scripts/recheck.sh
```

- **`code/tables.py`** recomputes every group of the README's tables from `data/per_task_*` and checks them against `data/summary.csv`. It ends with `summary.csv: all 45 rows match`. It also prints, per group:
  - tasks solved by either configuration and by neither;
  - for the 27B vote, how often 3 of 4 branches agree and how often at least one branch is right;
  - for Flash Next, how many replies ran past 32,768 tokens.
- **`code/stats.py`** (as in Parts 5-7) prints the paired tests:
  - totals;
  - per pair, the tasks only one side solved and the exact McNemar p;
  - paired bootstrap 95% intervals of the accuracy difference and of the time ratio (10,000 resamples, seed 2026, each task's time capped at 40 minutes).

  Its `p_holm` is over the pairs of one call. The README's Holm corrections are over the plans' families:
  - step 2, two tests: the smaller p times 2, the larger times 1, then made monotone;
  - step 1, three tests.

  Both are also in `data/tests.json`.

The figures:

```bash
cd figures && python3 make_figures.py
```

## 2. The two configurations on your own tasks

Task and grader format as in Part 4's REPRODUCE.md step 3 (`TASKS_DIR/<split>.jsonl`, `GRADER_DIR/grading.py`).

**27B, vote over 4 low-thinking branches (CL4)**, with Part 4's runner, at the settings of this part:

```bash
cd ../qwen38-spark-test-time-strategies/code/strategies
TASKS_DIR=/path/to/tasks GRADER_DIR=/path/to/grader \
  python3 run_method.py CL my-cl4 --k 4 --split confirm --limit-min 40 --context 81920 --max-tokens 65536
```

- For problems answered in `\boxed{}`, add the case from [code/boxed_vote.py](code/boxed_vote.py) to `answer_key` in that folder's `common.py`, and give those tasks `"subcategory": "arxivmath"`.
- Without it, a task whose grader type is not one of Part 4's types votes on the whole final text.

**Flash Next, deep thinking + 32K thinking budget (B)**, with Part 7's runner:

```bash
cd ../qwen38-spark-reasoning-loops/code
S4=../../qwen38-spark-test-time-strategies/code/strategies
TASKS_DIR=/path/to/tasks GRADER_DIR=/path/to/grader \
  python3 loop_arms.py $S4 my-fn-b --model flashnext --part6-code ../../qwen38-spark-flash-next/code --arms B
```

Both runners start and stop their own server, grade with Part 4's grader in the network-isolated sandbox, and write `graded.jsonl` under Part 4's `code/strategies/results/<run_id>/`. To compare the two per task, write one CSV per run (`task_id,passed,wall_s`) and run `code/stats.py --limit-min 40 a=a.csv b=b.csv`.

**Grading cores (optional).** The GB10 has two kinds of CPU cores: 0-4 and 10-14 are Cortex-A725, 5-9 and 15-19 are the faster Cortex-X925. Tests with time limits can pass on one kind and fail on the other. In step 2 the grading container was pinned to the A725 cores, which needs `--cpuset-cpus 0-4,10-14` in the `docker run` arguments of the sandbox (`SANDBOX` in Part 4's `common.py`).

## 3. The ArXivMath problems

| Set | Source | Commit | File | Bytes | sha256 |
|---|---|---|---|---|---|
| 08/2026 (57) | huggingface.co/datasets/MathArena/arxivmath-0826 | `977a3a0083198e11ac4f20fe79288351cebd6281` | `data/train-00000-of-00001.parquet` | 55,180 | `a635931866406bbe7eb5cee8a4e47e188634b3c25b0b0b83dc06d36435e75609` |
| 09/2026 (53) | huggingface.co/datasets/MathArena/arxivmath-0926 | `4c38882866fb19388cbb4a229c160040b191d4fe` | `data/train-00000-of-00001.parquet` | 239,973 | `3f9f6f1575f5da9dbe0f2a8124a033b0b6587e64d3a789a4ad30309fdd925b49` |

License CC BY-SA 4.0, for evaluation (see the datasets' README). Citation: Dekoninck et al., "Beyond Benchmarks: MathArena as an Evaluation Platform for Mathematics with LLMs", arXiv:2605.00674.

- **Task id:** `am0826_NN` and `am0926_NN`, where NN is `problem_idx`.
- **Prompt:** the user message is the `problem` field followed by `"\n\nPut your final answer within \\boxed{}."`.
- **Grading:** the last `\boxed{...}` of the answer is compared with the `answer` field. Our checker decides mathematical equivalence locally (sets, intervals, tuples, and expressions compared at sample points with sympy) and is not part of this folder. MathArena's own grading uses model judges.
- **Newer papers:** the published per-task files give each problem's source paper (`source_arxiv`) and the date of its first arXiv version. `after_release` = 1 for the 58 problems first posted after 2026-08-25.
