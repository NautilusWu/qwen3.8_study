# Pre-registration

Two plans, each written and sha256-locked before its runs. These are English summaries of the Chinese originals; the hashes are of the original files as locked. Each run script checked its lock at the start, and the lock was checked again after the run. Results were appended below the locked text only after the runs; the locked text itself is unchanged (checked: each plan file still starts with its locked copy, byte for byte).

| Plan | Locked | sha256 (locked text) | Runs |
|---|---|---|---|
| PLAN_09: Flash Next on Part 4's 72 tasks (step 1) | 2026-11-01 | `2fce285b69893d48657940b9f61720c3fde77861bf4c97ec4667a12a0a373fe8` | Flash Next: low, vote, deep, deep + budget; 27B: low and vote on the 24 hard tasks |
| PLAN_10: the two new sets (step 2) | 2026-11-03 | `01d2e0e6c838f89138eb0d6019ec8e645a05f4087877092ffce24710a9840a9c` | 27B vote and Flash Next deep + budget on h250 and ArXivMath |

Task files (not published):

| Set | sha256 of the prompt file |
|---|---|
| Part 4 hard tasks (24) | `fc636efd1be59edc5ffa3d675c76e2815dbe60274e3187e5e3aea08602f00852` |
| Part 4 confirmation tasks (48) | `5b2cd4bcf64cabc3700ed94f5cb65ebd6a27fe9b9709970900ed0cb53cd1c7d5` |
| h250 (250) | `136be07a5fbbf4b5377e26db9bbd72ca443da3c948127ea322b6014b7ee06c46` |
| ArXivMath (110, built from the public problems) | `4a487a23f9ea562fc83cfe73a417dae6c3d368c3122aeaa881449657f6d01f47` |

## PLAN_09: Flash Next on Part 4's 72 tasks

- **Question:**
  1. With the same method, does Flash Next answer more or fewer of Part 4's 24 hard and 48 confirmation tasks than the 27B?
  2. How much headroom do these tasks leave for comparing accuracy methods on Flash Next?
- **Methods:** low thinking (D2, greedy), a vote over 4 low-thinking branches (CL4), deep thinking (A65, greedy), and, for Flash Next only, deep thinking + 32,768-token thinking budget (B, greedy).
- **Settings:**
  - Flash Next as in Part 6; the 27B as in Part 5.
  - Context 81,920, output limit 65,536, 40 minutes per task.
- **27B baselines:**
  - Part 4's runs on the confirmation tasks (D2, CL4, A65) and its deep-thinking run on the hard tasks.
  - D2 and CL4 on the hard tasks were run again, because Part 4's hard runs used older settings (24,576-token limit, 25 minutes, 32,768 context).
- **B, composed** as in Part 7. Greedy decoding is deterministic, so:
  - B is run only on the tasks where Flash Next's deep reply ran past 32,768 output tokens; the other tasks take the deep result.
  - 5 tasks that finished within the budget are run again with the budget as a check: the text must be identical, or B is not composed and this is reported as a deviation.
- **Locked tests:** Flash Next against the 27B, paired over the 72 tasks, separately for D2, CL4 and A65, with Holm correction over the three.
  1. Accuracy: exact two-sided McNemar test. An adjusted p below 0.05 is written as "a detectable accuracy difference".
  2. Time: ratio of minutes per correct answer, Flash Next / 27B, paired bootstrap 95% interval (10,000 resamples, seed 2026). An interval that excludes 1 is written as "a detectable time difference".
  3. The accuracy difference, also with a 95% interval.
- **Capacity:** with about 10-15 tasks only one side solves, only differences of about 10-12 points are detectable.
- **Reported only:**
  - the full table;
  - Flash Next's internal comparisons (B vs A65, B vs CL4, D2 vs CL4);
  - headroom (tasks solved by any Flash Next configuration, by any 27B run, by none);
  - an estimated cascade (CL4, then B on reasoning tasks without 3 of 4 agreeing).
- **Rule written before the run (a reading, not a test):** if Flash Next's best single configuration answers at least 68/72, these tasks are near the ceiling for it too, and a harder set is needed.

**Results** (appended after the run):
- **Run:** 28.9 hours, nothing failed, display on throughout (1,727 one-minute records). B composed: 5/5 check tasks identical, 27 tasks run with the budget, 45 taken from A65.
- **Locked tests:** no detectable accuracy difference with any method (Holm-adjusted p 0.68, 1.0, 1.0). Time: Flash Next slower with D2 (1.37x, interval 1.06-1.79) and CL4 (1.87x, 1.43-2.52); no detectable time difference with A65 (1.03x, 0.74-1.42).
- **Reported only:**
  - B vs A65: 14 / 0 tasks only one side solved, p = 0.0001, time ratio 0.55.
  - B vs CL4: 4 / 0, p = 0.125.
  - Flash Next together 68/72; the 27B's 14 runs together 70/72; 2 tasks never solved.
  - Cascade estimate: 66/72 at 7.25 minutes per correct answer, worse than B alone (68/72, 5.90).
- **Reading:** B answered 68/72, so a harder set was needed.

**Deviation.** The plan said the 27B headroom count would include "all existing runs, including Part 4's agents". No agent ever ran on these 72 tasks. The count used all 14 of the 27B's runs on them.

**Robustness check (added after the run, reported only).** The GB10's CPU has two core types. Step 1's grader was not pinned to either, so a time-limited test could depend on the core a grading container got. All 464 graded answers of the reported configurations were graded again pinned to each core type: 0 of 464 differed from the recorded result.

## PLAN_10: the two new sets

- **Question:**
  1. How many tasks of the two new sets does each model's best configuration from step 1 answer: Flash Next B, and the 27B's CL4?
  2. Does h250 leave enough headroom for work on accuracy methods?
  3. Is there a detectable difference between the two configurations?
- **Tasks:** h250 (250) and ArXivMath (110), both frozen and read-only before the plan. Before the run, the references were graded on the runtime grading path: 250/250 and 110/110 passed.
- **Configurations:** Flash Next B and 27B CL4, settings as in step 1. Context 81,920, output limit 65,536, 40 minutes per task.
- **Not a one-variable comparison.** The plan states this: model and method both differ, so a difference cannot be attributed to either alone.
- **Code changes since step 1** (all before the lock):
  - task loading for the new sets;
  - the vote key for ArXivMath: the last `\boxed{}`, with spacing and layout-only commands removed (`code/boxed_vote.py`); this affects only 27B CL4 on ArXivMath;
  - the grading container pinned to the A725 cores, where h250's time limits were calibrated. The step-1 answers were graded again on both core types beforehand: none changed.
- **Run:** wait until the GPU has been idle for 5 minutes, check the lock, then 27B CL4 (h250, ArXivMath), then Flash Next B (h250, ArXivMath). Display on, logged every minute. Daily service stopped.
- **Locked tests:** Flash Next B against 27B CL4, paired, on h250 and on ArXivMath, Holm over the two.
  1. Accuracy: exact two-sided McNemar test, adjusted p below 0.05.
  2. Time: ratio of minutes per correct answer, bootstrap 95% interval excluding 1 (10,000, seed 2026).
  3. The accuracy difference with a 95% interval.
- **Capacity:** h250 can detect differences of about 6-8 points; ArXivMath only about 10 points or more.
- **Reading of h250's headroom (written before the run, not a test),** based on the better configuration's score:
  - at least 225/250: still near the ceiling, so accuracy work needs harder tasks;
  - 150-224: suitable for method work (tune on h250-dev, final check on h250-test);
  - below 150: very hard; first count how many tasks either configuration solves.
- **Reported only:**
  - the table by group: h250 all, dev and test; Python, C++ and process simulation; tiers A, B and C; ArXivMath all, the 58 problems from papers after both releases, and the other 52;
  - tasks solved by either configuration, and by neither;
  - 27B CL4: how often 3 of 4 branches agree, and how often at least one branch is right;
  - Flash Next B: how often it reaches the thinking budget, and how many of those it gets right;
  - the difference between the 58 newer and the 52 older problems, described without conclusions.
- **h250-test:** after this run its two baselines are known. Later method work tunes on h250-dev only; h250-test is reserved for final locked checks against these baselines.

**Results** (appended after the run):
- **Run:** 85.1 hours (estimate: 50-70), nothing failed, memory guard never fired, display on in all 5,105 one-minute records. Lock checked after the run.
- **h250:** Flash Next B 227/250 (6.52 minutes per correct answer), 27B CL4 208/250 (2.38).
  - 27 / 8 tasks only one side solved; Holm-adjusted p = 0.0038; +7.6 pp (3.2, 12.0); time ratio 2.74 (2.27, 3.34).
  - Both differences are detectable.
- **ArXivMath:** 10/110 (145.49) against 2/110 (828.01).
  - 10 / 2 tasks only one side solved; Holm-adjusted p = 0.0386; +7.3 pp (1.8, 13.6); time ratio 0.18 (0.06, 0.64).
  - Both differences are detectable.
- **Reading:** the better configuration answered 227/250, which is at least 225, so h250 is still near the ceiling for Flash Next. The 27B's 208/250 is in the 150-224 band.
- **Reported only:** see the README tables. The main points:
  - h250 together 235/250, neither 15/250;
  - ArXivMath together 12/110;
  - 27B CL4 agreement (3 of 4) on process simulation 27/50 (25 correct), on ArXivMath 23/110 (0 correct);
  - at least one branch right on h250 216/250, on ArXivMath 4/110;
  - Flash Next B over 32,768 tokens on h250 79/250 (61 correct), on ArXivMath 109/110 (10 correct);
  - newer vs older ArXivMath problems: Flash Next 5/58 vs 5/52, 27B 2/58 vs 0/52.

**Deviations and notes:**
1. **The run took 85.1 hours instead of the estimated 50-70.** Most of the excess was 27B CL4 on ArXivMath: 27.6 hours against an estimated 9.
2. **The ArXivMath time interval leaves out resamples.** The statistics script leaves out bootstrap resamples in which one side has no correct answer. On ArXivMath that removed 1,346 of 10,000 resamples, all of them without a correct 27B answer, so the reported interval is conditional on both sides having at least one. Counting the removed resamples as a ratio of 0 gives (0.00, 0.61); the reading is the same.
3. **The budget count uses output length.** "Reached the thinking budget" is counted as more than 32,768 output tokens (Part 7's definition), which includes the answer tokens.
4. **The grading was checked by hand** (added after the run, reported only). Because 27B CL4 answered only 2/110 ArXivMath problems, the graded answers were inspected. The wrong answers are wrong (for example 0 against a reference of 1, 30 against 31, infinity against 8), and 106 of the 110 chosen answers had a `\boxed{}`.
5. **27B branches that stopped inside the think block** (added after the run, reported only). 127 of the 424 finished 27B branches on ArXivMath stopped inside the think block without an answer (29 of 378 on h250). Step 1's 27B vote runs show the same.
6. **The h250 ids are anonymized** in the published per-task files. They are numbered in the sorted order of the private ids, so the published files reproduce the locked analysis's bootstrap intervals exactly.
