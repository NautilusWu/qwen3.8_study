# Reproducing Part 9

| What | How |
|---|---|
| The selection rules, on hand-made cases | `scripts/recheck.sh rules` (no GPU) |
| Every rule's choice and time on every task, from the recorded times | `scripts/recheck.sh arms` |
| The tables in the README | `scripts/recheck.sh tables`; it gives exactly [data/summary.csv](data/summary.csv) and [data/tester_summary.csv](data/tester_summary.csv) |
| The tests of the four plans | `scripts/recheck.sh tests`; it gives exactly the locked analyses' numbers in [data/tests.json](data/tests.json) |
| The figures | `figures/make_figures.py` (needs matplotlib) |
| The method, checked on Part 4's example tasks | step 2 (GPU, a few minutes) |
| The method on your own tasks | step 3 |

What cannot be reproduced from this folder:
- **Our scores on h250**, since its tasks, hidden tests and references are private.
- **The test-quality columns of `tasks.csv`** on your own tasks: they need each task's reference solution and known wrong answers. The runner records everything else.

## 0. Prerequisites

- **Recomputing:** Python 3.10 or later, standard library only. No GPU and no task files.
- **Figures:** matplotlib.
- **Running the method:**
  - Part 4's setup ([Part 4 REPRODUCE.md](https://github.com/NautilusWu/qwen3.8_study/blob/main/qwen38-spark-test-time-strategies/REPRODUCE.md) steps 0-2): a DGX Spark, Docker with the NVIDIA Container Toolkit, the TensorFold 0.6.0 images, the sandbox image and the 27B checkpoints, all from Part 4's `scripts/setup.sh`.
  - Part 4's folder next to this one (the repository has them side by side). The runner uses Part 4's `code/strategies` unchanged: its server, requests, sandbox and grading.
  - Nothing else may run on the GPU during a run.

## 1. Recompute everything

From this folder's root:

```bash
scripts/recheck.sh
```

It runs four steps and stops at the first failure:

1. **rules:** the checks of [code/rules/](code/rules/) on hand-made records, then the prompts' hash. They print `p11_rules checks passed` (twice: the second check file imports the first), `p11c_rules checks passed` and `prompts check passed`.
2. **arms:** [code/arms.py](code/arms.py) rebuilds each task's record from `data/<run>/branches.csv` and `tasks.csv`, applies the rules, and compares every choice, time, result and label with `data/<run>/arm_*.csv`. It ends with `arms: all match`.
3. **tables:** [code/tables.py](code/tables.py) recomputes every group of the README's tables and ends with `summary.csv: all 184 rows match; tester_summary.csv: all 5 rows match`. It also prints, per run:
   - the plan's gates;
   - where "verified" comes from;
   - the composed escalation estimate;
   - the two acceptance comparisons of the development runs (exact McNemar);
   - M2's exact interval.
4. **tests:** [code/tests_check.py](code/tests_check.py) runs [code/stats.py](code/stats.py) (as in Parts 5-8) with the same configurations as each locked analysis. It compares the totals, the tasks only one side solved, the McNemar p values and the bootstrap intervals, and ends with `tests: all match`. Add `--print` to see the full output.

`python3 code/prompts.py` prints both prompt hashes; the one for runs 3-4 must equal the hash recorded in those runs.

The figures:

```bash
cd figures && python3 make_figures.py
```

**Why the intervals match exactly.** `stats.py` sorts task ids before resampling. The anonymous ids H001-H250 are numbered in the sorted order of the private ids (as in Part 8), so the published files draw the same resamples as the locked analyses did.

## 2. Check on Part 4's example tasks

```bash
cd code
export HF_DIR=... TASKS_DIR=../../qwen38-spark-test-time-strategies/examples/tasks GRADER_DIR=../../qwen38-spark-test-time-strategies/examples/grader
python3 selfverify.py ../../qwen38-spark-test-time-strategies/code/strategies check --split dev --check
```

`--check` runs two things on the two coding tasks among Part 4's examples (a Python and a C++ task):
1. **The test path, without the model:** hand-written tests in the test writer's format are run in the sandbox on a right answer, a wrong answer and an empty code block, plus the C++ forms the runner must handle (tests written as a whole `int main()`, and a `return 0;` after a failed check). It prints a table and stops if any result is not the expected one.
2. **The method:** the server starts with `--parallel 5`, and each task gets its five requests, the test runs and the rules, as in step 3. Then every branch is graded and the rules are recomputed from the CSV files.

It ends with `CHECK PASSED` when the test path gave the expected results, no request failed, every branch and every test run was recorded, and the CSV recheck matched. How many answers are right or verified is printed, not checked: it is the model's result, not the pipeline's.

Ours, about 3 minutes with the kernel cache already built: `CHECK PASSED`, the test path 9/9 as expected, every branch right on both tasks.
- **Python task:** the test writer's tests passed all 4 branches and failed on empty code, so the answer was verified.
- **C++ task:** the tests did not compile, because they used `std::LLONG_MAX`, which is a macro and not in `std`. No branch passed, so B2 fell back to the first branch with code (right, unverified). This is the failure the README counts as "test code fails by itself".

The model runs with sampling, so your texts and these outcomes can differ.

## 3. The method on your own tasks

```bash
cd code
export HF_DIR=... TASKS_DIR=<your tasks> GRADER_DIR=<your grader>
python3 selfverify.py ../../qwen38-spark-test-time-strategies/code/strategies myrun --split confirm
```

- **Tasks and grader** as in Part 4's REPRODUCE.md step 3. Only the coding tasks are run, and the grader needs the `python_tests` and `cpp_tests` types of Part 4's example grader.
- **Each task must end with its sentence about the answer format** ("Return only the complete function ..." or "Do not write a main function. Return only the code ..."). `RETURN_TAIL` in [code/prompts.py](code/prompts.py) removes it for the test writer. The runner checks this before the server starts.
- **Two settings** besides Part 4's:
  - `SANDBOX_CPUS` (default `0-4,10-14`): the cores for the sandbox containers. These are the DGX Spark's Cortex-A725 cores, where our runs ran the tests; the model server runs on the others. Set it to empty to leave the containers unpinned.
  - `CPP_FAIL_COUNTER` (default `fails`): the failure counter of your grader's C++ harness. The C++ wrapper increments it when the tests return early with a non-zero value. Part 4's example grader calls it `fails`; ours called it `failures`.

What [code/selfverify.py](code/selfverify.py) does for each task, as our runs did:
1. **It sends five requests together** and lets all five run to the end:
   - **4 code branches,** exactly Part 4's CL4: the task as the prompt, `reasoning_effort: low`, temperature 0.6, top_p 0.95, top_k 20, seeds 1000-1003.
   - **The test writer:** `T_PROMPT` with the task's format sentence removed, the same sampling, `reasoning_effort: low`, seed 2000.
2. **It takes the tests:** the longest code block of the test writer's answer, as the grader takes code. A C++ block that is a whole `int main() {...}` is replaced by its body, and the statements go into `CPP_WRAP`.
3. **It checks the tests on empty code** (for B2), in their own container, timed.
4. **It runs the tests on every branch with code,** each in its own container. The limits are 30 seconds per run and 60 seconds for compiling C++ (the example grader's). It records each pass or fail and the container's time.
5. **It chooses** with the rules A, B and B2 of [code/rules/](code/rules/). Times are computed from the recorded end times: each test run starts as soon as both the branch and the tests exist.

After the last task it stops the server and grades every branch with your grader. Results go to `results/<run_id>/` in this folder:
- `records.jsonl`: every request's text, the tests, every test run and every grade. It holds the model's code, so keep it private if your tasks are.
- `branches.csv`, `tasks.csv`, `arm_A.csv`, `arm_B.csv`, `arm_B2.csv`: the columns of `data/`, without text.
- `summary.json`: per rule, the correct answers, minutes per correct answer, verified answers and verified answers that are wrong.

`python3 code/arms.py --dir results/<run_id>` recomputes the rules from the CSV files; the runner does this at the end.

A live runner would cancel the other branches as soon as the rule has chosen. Our runs let every request finish, so that every rule could be computed from the same run, and this runner does the same.

This runner was written for publication from our run code, on top of Part 4's public code. It was checked on the example tasks (step 2); it was not rerun on h250.
