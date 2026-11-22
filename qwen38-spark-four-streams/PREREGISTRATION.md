# Pre-registration

This part has four plans, locked with sha256:
- PLAN_19 and PLAN_20 twice each: the first lock before the compatibility check (P0), the final lock after P0 had passed, before any of the 400 tasks;
- PLAN_21 once, before its P0. P0 failed, so the plan stopped there;
- PLAN_21b once, before its first block. It used PLAN_21's P0 results for the 27B.

How to read this file:
- It is an English summary of the Chinese originals. Each hash is of the original file as locked.
- Every run script checked its locks at the start and again at the end; all checks passed.
- Results were appended only after the runs, below the locked text. A check confirmed that each plan file still begins with its locked copy (`verify-prefix`).

| Plan | Lock | Time (2026-11-19) | sha256 (locked text) | Files in the lock |
|---|---|---|---|---|
| PLAN_19: 27B and Flash Next on set O, one stream | before P0 | 00:36 | `b0cb973e8dfe6566b4cc798b95bec38ef084d21929272f23e31d82eff47833a4` | 22 |
| | final, after P0 | 00:45 | `7828f2285194c44c80dd18c99e92a4f548056b188c506d9753c609a97d9c9ebf` | 11 |
| PLAN_20: the same at four streams | before P0 | 08:24 | `1b67912abc9811e2ba1582921eecdfbb70fe0dd24f7c9d90bb5e267be1043e19` | 21 |
| | final, after P0 | 08:32 | `ccf4de4301025037bedfe721c4d585fe17030804af9a0f2b04e7df07af02513d` | 9 |
| PLAN_21: both models at 2, 4 and 8 streams | before P0 (stopped at P0) | 20:23 | `ddbd0d88e8141451a9d8ad0cef7c8322f091f9e89e4eaa793ee643eef2e17be4` | 27 |
| PLAN_21b: the 27B alone at 2, 4 and 8 streams | before the first block | 21:28 | `225088c01cac0fc346567d615325502bb649f76c58b59081366413ea701634ed` | 8 |

**What the locks covered:**
- the plans;
- all run, grading and analysis code;
- the code it imports (Part 4's runner and server code, Part 6's Flash Next setup, `stats.py`);
- Part 4's example tasks;
- the compatibility check results.
- **Set O's files** (not published): prompts sha256 `d5b5c5aae963143e6112fb705124f7e1aaf2504b626a7b23496580b642a052a9`, hidden test cases `0e6e6ce2c2da83e8734091b25651b0aad73cba3eeb65616afd50bd590fb1a6ee`, grader `40002303295a541b003c39ab9111473e4a9e74c7bea81e5b49fe2e3c7699e7bd`.
- **For PLAN_19 only:**
  - set T's prompt file, sha256 `591b703379981311905ec9ed7a2beef5c34f880beafd92f34e759fe18e09a85b` (the same file as Parts 5-11);
  - the two set T runs it compares with;
  - the list of topic-similar tasks.
- **For PLAN_20 only:** PLAN_19's code and its one-stream results, which PLAN_20 compares with.
- **For PLAN_21:** PLAN_19's code, PLAN_19's and PLAN_20's results, and PLAN_20's example replies (P0 compared with them).
- **For PLAN_21b:** its own run and analysis code and PLAN_21's 27B check results. Its run script also checked, at the start and the end, every file of PLAN_21's lock, and that PLAN_21's plan had only been appended to.

## PLAN_19: one stream

**Why.** A second private set (set O) had just been added and checked. The questions: do the 27B and Flash Next rank the same on it, and is it easier or harder than set T?

**Questions:**
1. **Q1 (primary):** on set O, with D2, Flash Next against the 27B. Which answers more correctly, and which needs fewer minutes per correct answer? Paired by task.
2. **Q2 (secondary, difficulty):** for each model, accuracy on set O minus set T, overall and in three matching groups:
   - Python code;
   - C++ code;
   - set T's numeric reasoning against set O's JSON logic tasks.
3. **Q3 (reported only):** set O by kind and language, grader statuses, tokens, limit hits, and the results without the 24 topic-similar tasks.

**Fixed for both models:**
- **Servers and checkpoints:** as in the README (Part 4's 27B server, Part 6's Flash Next setup, TensorFold 0.6.0).
- **Settings:** `--parallel 1`, context 81,920, Part 4's `run_single(task, 'D2')`, output limit 65,536, 40 minutes per task (as in the set T runs).
- **Prompt:** the task's participant prompt, unchanged, as the user message.
- **Grading:** set O's own grader, imported unchanged, in single-reply mode, on the answer after the thinking.
  - The one change: every grading container also got `--cpuset-cpus 0-4,10-14` (the Cortex-A725 cores, as for the other sets).
  - A grader `infrastructure_error` was to be retried once.

**Compatibility check (P0).** All criteria had to pass; any failure was to stop the run, not to change settings.

| # | Criterion | 27B | Flash Next |
|---|---|---|---|
| 1 | the server starts and the warm-up reply is right | 15.1 s | 82.0 s |
| 2 | Part 4's 5 example tasks with D2: at least 4/5 correct | 4/5 | 5/5 |
| 3 | the first task of each kind of set O (O001, O161, O321, O381) gets a normal grader status, with no request error | 4 pass | 4 pass |
| 4 | through the pinned grading path, reference answers pass and the first known wrong answer is rejected | ok (one check for both) | |
| 5 | the memory guard does not fire (lowest available memory) | 89.9 GiB | 36.2 GiB |

**The main run (P2):**
- **Groups:** 400 tasks in a fixed shuffled order (seed 2026), cut into 8 groups of 50.
- **Order:** the two models in ABBA order:
  - 27B first in groups 0, 3, 4 and 7;
  - Flash Next first in groups 1, 2, 5 and 6.
- **Servers:** a fresh server for every block. Before each start, the runner waits until `nvidia-smi` lists no process (at most 300 s).
- **Failures:** two failed blocks in a row stop the run.
- **Grading:** after all generation.

**Rules, written in advance:**

| Test | Result | Verdict |
|---|---|---|
| Q1 accuracy | exact McNemar p < 0.05 | the model with more correct answers is more accurate (detectable) |
| | otherwise | no detectable accuracy difference |
| Q1 time | upper end of the 95% interval of the time ratio Flash Next / 27B < 1 | Flash Next faster (detectable) |
| | lower end > 1 | Flash Next slower (detectable) |
| | otherwise | no detectable time difference |
| Q2 (overall only) | both models' intervals of O − T below 0 | set O harder for both (detectable) |
| | both above 0 | set T harder for both (detectable) |
| | otherwise | no consistent detectable difference |

- **Q1 intervals:** paired bootstrap (10,000 resamples, seed 2026), 40-minute limit.
- **Q2 intervals:** each set's tasks resampled independently (10,000 resamples, seed 2026).
- **The plan's own caveats on Q2:**
  - it applies only to these two sets and these two models, and not to "tasks written by" either assistant;
  - the sets differ in their kinds of tasks and in how strictly they test, and the strata are there to show where a difference comes from.
- **Reported only:** the model gap on each set and its change; time and tokens per set.

**Power (from the plan):**
- Q1: about 10% of tasks were solved by only one model on set T, so the accuracy interval would be about ±3 points.
- Q2: with 400 tasks per set at about 90% correct, differences of about 6 points or more can be detected.

**Estimated time:** 17-33 hours. Actual: 3 hours 39 minutes. Set O's replies were about half as long as set T's, and only one reply ran to the token limit (13.6 minutes).

**Results:** see the README. All 16 blocks finished; 400/400 tasks per model; no request errors, no grader retries, the memory guard never fired.

**Deviation, found while preparing this report.** The plan's text says the reference check (criterion 4) would grade 6 tasks. The locked code graded 4: the four compatibility tasks O001, O161, O321 and O381. All four passed and all four wrong answers were rejected.
- The set's own self-check before the study had already run every reference answer and every known wrong answer through the same grader: 400 passed and 800 were rejected. It ran without the core pinning.
- This changes no result. It is also noted below the plan's locked text.

## PLAN_20: four streams

**Why.** PLAN_19 ran one request at a time. The question: which model is faster when the server runs four requests at once, and how much does that gain over one stream?

The plan quoted an earlier estimate as its reason, not as a test. In earlier runs where four branches of the same task ran at once, each 27B branch kept 81% of its single-stream speed, each Flash Next branch 54%.

**Questions:**
1. **Q1 (primary):** at four streams, Flash Next against the 27B, minutes per correct answer, on shared time.
2. **Q2 (primary, for each model separately):** four streams (PLAN_20) against one stream (PLAN_19), minutes per correct answer.
3. **Q3 (reported only):**
   - how much slower a single request gets;
   - the speed of one stream, and the total speed;
   - whether accuracy or the replies change;
   - timeouts and limit hits;
   - the share of busy time with fewer than four tasks running;
   - the server's reported number of running requests.

**Only one variable changes: concurrency.**

| | PLAN_19 (one stream) | PLAN_20 |
|---|---|---|
| Server | `--parallel 1` | `--parallel 4` |
| Client | one task at a time, waiting for the server to be idle before each | 4 workers take tasks from one queue in group order; the server is waited for once per block, after the warm-up |

Everything else was PLAN_19's: models, servers, context, request, limits, prompts, grading code (unchanged), the 8 groups and the ABBA order. The 40-minute limit counts from each task's own request.

The plan noted one built-in difference: TensorFold runs the 27B on a separate single-stream path at `--parallel 1`, and that difference is part of the comparison.

**Shared time (locked).**
- Each stretch of time is split evenly between the tasks running in it.
- Minutes per correct answer = all shared time / correct answers.
- Time at the end of a block, when fewer than four tasks run, counts too, and its share is reported.
- With one stream, a task's shared time is its own time.
- Intervals: paired bootstrap on shared time (`stats.py`, 10,000 resamples, seed 2026).
- The plan notes that a task's shared time depends on the other tasks running with it, so resampling tasks is an approximation.

**Compatibility check (P0):**

| # | Criterion | 27B | Flash Next |
|---|---|---|---|
| 1 | starts with `--parallel 4`, warm-up right | 21.2 s | 72.8 s |
| 2 | Part 4's 5 example tasks with D2, four at a time: at least 4/5 correct | 4/5 | 5/5 |
| 3 | set O's four compatibility tasks at once: normal grader statuses, no request errors | 4 pass (10.0 s) | 4 pass (20.1 s) |
| 4 | the server's `/health` reports 4 requests running at least once | 4 | 4 |
| 5 | the memory guard does not fire (lowest available memory) | 87.8 GiB | 35.6 GiB |

**The main run (P2):** 16 blocks in PLAN_19's order. Two failed blocks in a row, or the memory guard, stop the run.

**Rules, written in advance:**

| Test | Interval of the time ratio | Verdict |
|---|---|---|
| Q1: Flash Next / 27B at 4 streams | upper end < 1 | Flash Next faster at 4 streams (detectable) |
| | lower end > 1 | Flash Next slower at 4 streams (detectable) |
| | otherwise | no detectable time difference |
| Q2: 4 streams / 1 stream, each model | upper end < 1 | concurrency raises throughput (detectable); speed-up = 1 / ratio |
| | lower end > 1 | concurrency is slower (detectable) |
| | otherwise | no detectable difference |

- Accuracy (McNemar) is reported only in PLAN_20: PLAN_19 already compares it.
- **Three intervals, each at 95%, with no multiplicity correction.** The expected differences were large (about 2-3 times throughput, 1.5-2.5 times Flash Next / 27B), and this choice is written in the plan.
- **Q3's steady speed:** only the time when all four tasks run, with each task's tokens spread evenly over its own time. The plan calls this an approximation.
- **Q3 also:** the time ratio without the 5 tasks with the largest shared-time difference.

**Estimated time:** 2-3 hours. Actual: 2 hours 14 minutes, including P0 and grading.

**Results:** see the README.
- All 16 blocks finished; 400/400 per model; no request errors, timeouts, grader retries or memory-guard stops.
- The server reported 4 running requests in every block.
- Replies were identical to PLAN_19's for both models, 400/400.
- **Deviations: none.**

## PLAN_21: 2, 4 and 8 streams, both models (stopped at its check)

**Why.** PLAN_20 found that four streams raise throughput, so the next question was 2 and 8 streams. Eight streams against PLAN_20's four would compare two sessions, and sessions differ (Part 11: 14-15%). So the plan ran 2, 4 and 8 streams in one session, the four-stream arm doubling as a repeat of PLAN_20.

**Questions:**
1. **Q1 (primary, each model):** 8 streams against 4 streams, both this run, minutes per correct answer on shared time.
2. **Q2 (primary):** at 8 streams, Flash Next against the 27B.
3. **Reported only:** 2 against 4 streams; Flash Next against the 27B at 2 and 4; each stream count against PLAN_19's one stream; this run's four streams against PLAN_20's; for each model and count, identical replies, speeds, task times, idle share and the never-dry estimate.

**Only concurrency changes.** Everything else was PLAN_20's: models, servers, context 81,920 per stream, D2, limits, grading, the 8 groups. With N streams, the server ran `--parallel N` and N workers took tasks from one queue.

**Order:** 48 blocks. Each group ran its three stream counts back to back, in a Williams order (in the README); within a stream count the two models ran in PLAN_19's order for the first and third count, reversed for the second.

**Rules:** as PLAN_20's, for 8 against 4 (each model) and for Flash Next against the 27B at 8. Three intervals at 95%, no multiplicity correction.

**Compatibility check (P0)**, at 2 and 8 streams (4 had been checked in PLAN_20). The queue: set O's four compatibility tasks first, then Part 4's 5 examples, N at a time.

| # | Criterion | 27B, 2 | 27B, 8 | Flash Next, 2 | Flash Next, 8 |
|---|---|---|---|---|---|
| 1 | starts, warm-up right | 18.2 s | 21.2 s | 72.8 s | 72.9 s |
| 2 | Part 4's 5 examples: at least 4/5 correct | 4/5 | 4/5 | 5/5 | 5/5 |
| 3 | set O's four tasks: normal grader statuses, no request errors | 4 pass | 4 pass | **fail**: O001 and O161 server errors | **fail**: O161 server error |
| 4 | the server's `/health` reports N running at least once | 2 | 8 | 2 | 8 |
| 5 | the memory guard does not fire (lowest available memory) | 88.9 GiB | 86.6 GiB | 36.7 GiB | 33.2 GiB |

- The examples' replies were identical to PLAN_20's check in all four runs (5/5 each), and the set O tasks that were answered were identical to PLAN_19.
- **Decision: stop.** By the plan's rule, a failed criterion stops the run and no setting is changed. None of the 48 blocks ran.

## PLAN_21b: the 27B alone at 2, 4 and 8 streams

**Why.** The 27B passed every criterion of PLAN_21's check. Running Flash Next as planned would have counted rejected tasks as wrong, and any way around the error (staggered first requests, retries) would change the client. So PLAN_21b drops Flash Next's 24 blocks and changes nothing else.

**Questions:**
1. **Q1 (primary):** the 27B, 8 streams against 4 streams, both this run, minutes per correct answer on shared time.
2. PLAN_21's Q2 is cancelled.
3. **Reported only:** 2 against 4 streams; each stream count against PLAN_19's one stream (another session); this run's four streams against PLAN_20's (time ratio, busy time per block, identical replies); the numbers for each stream count.

**Rule:**

| Interval of the time ratio 8 streams / 4 streams | Verdict |
|---|---|
| upper end < 1 | 8 streams raise throughput over 4 (detectable); speed-up = 1 / ratio |
| lower end > 1 | 8 streams are slower than 4 (detectable) |
| otherwise | no detectable difference |

**Power (from the plan):** an expected speed-up of 1.2-1.4 times; PLAN_20's 27B interval was about ±9% of the ratio, so this was detectable.

**Order and blocks:** PLAN_21's, without the Flash Next blocks: 24 blocks, each group's three stream counts in the Williams order.

**Estimated time:** 2-2.5 hours. Actual: 2 hours 1 minute, including grading.

**Results:** see the README.
- All 24 blocks finished; 400/400 tasks at each stream count; no request errors, timeouts, limit hits, grader retries or memory-guard stops.
- The server reported N running requests in every block.
- Replies identical to PLAN_19's at every stream count, and to PLAN_20's at four streams, 400/400.
- The run script's lock checks passed at the start and the end.
- **Deviations: none.**

**After both plans: tracing Flash Next's error** (diagnostics, not a test, nothing in the results depends on it). The README describes the cause and the reproduction. The diagnostic servers used the same image. One run also had a copy of TensorFold's `mtp.py` mounted read-only, with only the error message changed so that it printed the cache numbers.

## Looked at after the results (reported only)

Everything in [data/reported_only.json](data/reported_only.json) was computed after the results. Only PLAN_21b's comparisons listed in its plan were planned (as reported only); nothing else was:
- **PLAN_19:**
  - the decomposition of the time ratio into token ratio / speed ratio / correct ratio;
  - the share of the time difference in the largest tasks;
  - the ratio without them.
- **PLAN_20 per block:** one stream's summed task times against four streams' busy time, with no time sharing.
- **A second estimate of the steady speed:** only tasks that had four tasks running for their whole time, with no spreading.
- **Minutes by number of tasks running**, and how long the last task of each block ran alone.
- **The "never runs dry" estimate:** all output tokens / steady speed / correct answers.
- **Correct answers per hour**, and the 90th percentile and maximum of task times.
- **The mixture-of-experts explanation in the README** is a hypothesis. It was not measured.
- **PLAN_21b, beyond its plan's list:** busy time per block at 8 against 4 streams, the speed of one stream and a task's own time against one stream, and minutes by number of tasks running.
