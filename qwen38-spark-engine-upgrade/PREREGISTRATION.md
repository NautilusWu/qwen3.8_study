# Pre-registration

One plan, written and sha256-locked before the run. Below is an English summary of the Chinese original; the hash is of the original file as locked. The run script checked the lock at the start and again at the end, and both checks passed. Results were appended only after the run, below the locked text.

| Plan | Locked | sha256 (locked text) | Run |
|---|---|---|---|
| PLAN_13: TensorFold 1.2.5 vs 0.6.0, CL4, 400 tasks | 2026-11-10, after the download check, the compatibility check and a smoke test on public examples, before any of the 400 tasks | `b3700c6ca21c1b2f996e66247b8d69dc621e099f3105ca2ed914522ea8b99c91` | speed microbench, then 16 alternating blocks |

The lock covered 22 files:
- the plan;
- the run code;
- the code it imports (Part 4's runner and server code, the grader, the loop detector, `stats.py`);
- the 400-task prompt file, sha256 `591b703379981311905ec9ed7a2beef5c34f880beafd92f34e759fe18e09a85b` (not published; the same file as Parts 5-7);
- the release archive and its binary;
- Part 5's 0.6.0 run used for the reproduction check;
- the compatibility check's results.

## Before the plan was locked: download and compatibility check

**Download.** The official release `tensorfold-1.2.5-linux-aarch64.tar.gz` (48,839,854 bytes) from the GitHub release `v1.2.5` (commit `8be9c408c2825102b475dc6049c6aa3cf5d9e910`).
- Its sha256, `1a0192b28d6ab6ec2749f3ff25515379dba4895171e781b30f810afc971ae4f8`, matched the release's `.sha256` file.
- All 23 files in the archive's own `SHA256SUMS` matched.
- The `.sig` signature could not be checked, because no public key was found. This was recorded in the plan.

**Version choice.** TensorFold has no separate stable channel: every tag is a full release, and 1.2.5 was the newest (2026-11-08). The README's `--profile stable` is a configuration profile, not a release channel. The plan fixed 1.2.5 and was not to change it if a newer version appeared during the study.

**Compatibility check** on Part 4's 5 public example tasks, both engines. All 6 criteria were met ([data/compat_check.json](data/compat_check.json)). The rule was to stop and report on any failure, not to change settings to make it run.

| Criterion | Result |
|---|---|
| 1. Release | sha256 and files verified; `--version` prints `tensorfold-native 1.2.5` |
| 2. Start | `--parallel 1` and `--parallel 4` start; `--precision full` and `--kv-cache fp8` are accepted. The startup lines show the FP8 caches, DFlash2, 4 streams at once, and the memory estimate within the budget left after the 40 GiB reserve. A start takes about 60-63 s (0.6.0: 15-18 s). |
| 3. Example tasks, CL4 | Every task got a vote result, with the same vote rule per task as 0.6.0. Chosen answers 5/5 correct (0.6.0: 5/5); finished branches 11/11 correct. The engine's statistics show effort `low`. Median tokens of finished branches: 671 against 730 (0.92). The server was idle within 1 s after each decision. |
| 4. Determinism | greedy twice: same text; same seed twice: same text, another seed: different; the 4 branches of a task not all identical |
| 5. Speed | a public long reply at 67.5 against 70.0 tok/s (0.96); the criterion was at least half of 0.6.0 |
| 6. Memory | lowest available 76.0 GiB, guard not fired, containers removed |

Reported only:
- greedy replies under concurrency were identical to replies run alone, on both engines;
- the 1.2.5 server process had only local connections.

A smoke test of the runner followed: 4 public examples in 4 alternating blocks, 4/4 correct on both engines.

## PLAN_13: the engine, CL4, 400 tasks

- **Question:** on one DGX Spark, with the same checkpoint, drafter, tasks and settings, does TensorFold 1.2.5 give a shorter time per correct answer than 0.6.0 with CL4, and does its accuracy drop? CL4 had the most correct answers on these tasks in Part 5 (373/400). Other usages were out of scope.
- **Variable: the engine.**
  - 0.6.0: v0.6.0 `c4646171` with the recipe's five patches, Part 4's image.
  - 1.2.5: the official release, run in the same base image, not built locally.
  - The plan also lists everything that comes with the version and cannot be fixed:
    - prompt processing with bf16 activations, because 1.2.5 refuses `--prefill-fp8`, the only change in math;
    - the kernels, the scheduler and the sampler (the same settings and seeds draw different tokens);
    - the KV pool;
    - no per-token trace;
    - the verification window cost measured at every start.
  - Conclusions are about "switching to 1.2.5" as a whole.
- **Unchanged:**
  - checkpoint `nvidia/Qwen3.8-27B-NVFP4` @`482ca0f3` and drafter `z-lab/Qwen3.8-27B-DFlash2` @`50307d4c`;
  - decode math (bf16 activations; NVFP4/FP8 weights);
  - FP8 KV cache, kept the same at the user's choice;
  - the 40 GiB memory reserve, the container limits, CPU pinning and host network;
  - the tasks, prompts, grader, context 81,920, output limit 65,536 and 40 minutes per task;
  - thinking on, `reasoning_effort: low`, temperature 0.6 / top_p 0.95 / top_k 20, seeds 1000-1003;
  - the vote and cancellation rules (Part 4's runner, imported unchanged), and the memory guard (12 GiB).
- **Design.** Both engines were run fresh in this study, because Part 5's 0.6.0 timings are from October with the display state not recorded. The 400 tasks, in their fixed order, form 8 groups of 50. Each group is run by both engines, the first engine per group being 1.2.5, 0.6.0, 0.6.0, 1.2.5, 0.6.0, 1.2.5, 1.2.5, 0.6.0, so drift in the machine's state affects both sides alike.
  - Before every block: stop the server, wait for a free GPU, start the server, and send the same warm-up request.
  - Before each task: wait until the server is idle.
  - A failed block prints `FAILED` and the next block runs; its tasks count as wrong at 40 minutes. A task running 10 minutes past its limit stops the block. The memory guard stops the whole run.
  - Grading comes after all blocks.
- **Step 1, speed microbench** (report only, about 40 minutes): 6 rounds, old new new old old new, on fixed public prompts.
- **The only test, M1:**
  - CL4 with 1.2.5 against 0.6.0.
  - Accuracy: exact two-sided McNemar at α = 0.05. There is one test, so no correction.
  - Time: the ratio of minutes per correct answer (1.2.5 / 0.6.0) with a paired bootstrap 95% interval (`stats.py`, 10,000 resamples, seed 2026), with at most 40 minutes per task.
- **Decision rules, written before the run:**

| Result | Reading | Next |
|---|---|---|
| accuracy significantly lower (p < 0.05, 1.2.5 worse) | "1.2.5 loses accuracy" | do not upgrade, whatever the speed |
| not lower, time interval's upper end < 1 | "1.2.5 faster, no accuracy loss detected" | switch multi-branch use to 1.2.5; the daily single-stream service is the user's decision |
| not lower, interval's lower end > 1 | "1.2.5 slower" | stay on 0.6.0 |
| otherwise | "no speed difference detected" | stay on 0.6.0 (no reason to switch for speed) |

  - A significant accuracy gain would be reported as such.
  - Conclusions apply to CL4 only.
- **Reported only:**
  - whether 0.6.0 reproduces Part 5's run (same correct tasks, same texts, and its time against Part 5's);
  - M1 on the tasks both engines ran;
  - the two engines' outputs: identical texts, tokens, limit hits, timeouts, loops by the offline loop detector, vote outcomes;
  - speed (tokens/s, first token, and accepted tokens per step if an engine reports them);
  - groups (Python, C++, reasoning), memory, errors, and the microbench tables.
- **Power stated in advance.** With McNemar on 400 paired tasks, accuracy differences under about 3-4 percentage points cannot be detected. From earlier 400-task runs, the time interval was expected to be about ±6%.

## Results (appended after the run; lock verified before and after)

**Run.** 2026-11-10, 01:42-14:45.
- All 16 blocks completed, 800 task runs.
- No failed block, no timeout, no request error, no branch at the output limit, and the memory guard did not fire.
- The display was on throughout (778 of 778 checks).
- There were no deviations from the plan.

**M1:**

| | 1.2.5 | 0.6.0 |
|---|---|---|
| Correct | 377/400 | 373/400 |
| Total time | 378.8 min | 357.3 min |
| Minutes per correct answer | 1.005 | 0.958 |

- Accuracy: 13 tasks only 1.2.5 solved, 9 only 0.6.0, p = 0.5235. Difference +1.0 percentage points (95% interval -1.25 to +3.25).
- Time ratio 1.049 (95% interval 0.912-1.22).
- **Reading under the rules: "no speed difference detected", stay on 0.6.0.** No accuracy difference was detected either.
- The time interval (-9% to +22%) is much wider than the ±6% expected, so only a large speed difference could have been detected.

**Reported only:**

- **Reproduction of Part 5's run by 0.6.0.**
  - 373/400, the same right and wrong on all 400 tasks.
  - 385/400 chosen answers with the same text, and 828 same-seed finished branches with the same text.
  - Time ratio against Part 5's run 1.036 (1.032-1.039): 3.6% slower, cause not investigated. Both engines alternated within this run, so M1 is unaffected.
- **M1 on the tasks both engines ran:** all 400, identical to M1.
- **Outputs.**
  - 0/400 chosen answers and 0 branches with the same text on both engines.
  - Finished-branch tokens per task: median ratio 1.009. Total output tokens 2,937,369 against 2,797,868 (+5.0%).
  - Cancelled branches 767 against 760.
  - Vote outcomes (1.2.5 / 0.6.0): first branch with code 200 / 200, majority 183 / 183, plurality 7 / 10, tie (earliest) 7 / 7, no answer 3 / 0.
  - The offline loop detector flagged 2 branches on 1.2.5 and 1 on 0.6.0, all on one reasoning task, none at the output limit.
- **Speed in the run.**
  - Finished-branch decoding, median: 66.8 against 62.2 tok/s. 1.2.5's own figure is 66.7; 0.6.0's request statistics have no speed.
  - First token, median: 0.453 against 1.065 s.
  - Neither engine reports accepted draft tokens per step (`drafts` is only a flag), so that item could not be reported.
- **Groups, memory, starts.** See the README. Lowest available memory 75.0 against 81.0 GiB. Starts took 63.4 s on 1.2.5 and 18.2-21.2 s on 0.6.0, measured in 3-second polling steps. No crashes.

**Looked at after the results** (not a pre-registered test):
- 1.2.5 was faster on 271/400 tasks (median per-task ratio 0.925).
- Its higher total comes from a few long tasks: T116 18.7 against 0.7 minutes, T114 14.3 against 1.5, together 30.7 minutes, more than the whole 21.5-minute gap.
- With different tokens on each engine, every task is one draw, and one run cannot tell a tendency to long tails from chance.
