# Pre-registration

Four plans, each written and sha256-locked before its run. These are English summaries of the Chinese originals; the hashes are of the original files as locked. Each run script checked its lock at the start, and the lock was checked again after the run. Results were appended below the locked text only after the runs; the locked text itself is unchanged (checked: each plan file still starts with its locked copy, byte for byte).

| Plan | Locked | sha256 (locked text) | Run |
|---|---|---|---|
| PLAN_11: can the 27B verify its own code? | 2026-11-08 | `8f96aaef938f31270d4e1d9fb07cb2848cd44583b60b18562338b33a0499f343` | run 1: h250-dev, 50 coding tasks |
| PLAN_11b: the test writer thinks more (low → medium) | 2026-11-08 | `6a5cbfe05fd69c2a2a9d7297f1c005ea5b279a1dc50acf66bbb1616a6bea8558` | run 2: the same 50 tasks |
| PLAN_11c: expected values only from a brute force, a property or an example | 2026-11-09 | `dc9e4bfab73fa1107a7a3581d3fb9248378f38df74cd6bca01e854223270af77` | run 3: the same 50 tasks |
| PLAN_12: the locked check on tasks not used for development | 2026-11-10 | `fcedc69043244e5cb9007cde55976db15464346d542df3c9e765f5016b7ad762` | run 4: h250-test, 150 coding tasks |

- **Task file** (not published): h250, sha256 `136be07a5fbbf4b5377e26db9bbd72ca443da3c948127ea322b6014b7ee06c46` (the same file as Part 8).
- **Locked with each plan:** the run code, the selection rules and their checks, the statistics script, the grader, the tasks, their references and known wrong answers, and the earlier plans and results each plan compares with. That is 13, 22, 31 and 31 files.
- **Selection rules:** [code/rules/](code/rules/) holds `p11_rules.py` (sha256 `61b0c4fc…bc8738`) and `p11c_rules.py` (`78023514…f811e`), byte-identical to the locked files.

## Common to all four

- **Model:** the 27B as in Parts 5-8 (NVFP4 + DFlash2, FP8 KV cache), on TensorFold 0.6.0, with the server at `--parallel 5`.
- **Limits:** context 81,920, output limit 65,536, 40 minutes per task.
- **Requests:** five per task, all run to the end. Four are code branches identical to CL4 (low thinking, temperature 0.6, top_p 0.95, top_k 20, seeds 1000-1003). The fifth is the test writer T (seed 2000), which sees only the task.
- **Tests:** T's tests run on every branch with code, in the grader's sandbox, one container per branch. Selection uses only the model's tests; the hidden tests only grade the chosen answer.
- **Rules:** A, B, (C, B2) and O as in the README. Times are computed from the recorded times by the locked rule code.
- **Measure:** minutes per correct answer, each task capped at 40 minutes. Paired statistics with `code/stats.py` (exact McNemar; bootstrap 10,000 resamples, seed 2026).
- **Run state:** display on, daily service stopped, the GPU idle for 5 minutes before the start.

## PLAN_11 (run 1)

- **Question:** can the same local model reliably judge its own code?
  1. Does choosing among the 4 branches with the model's own tests give more correct answers?
  2. How many "verified" answers are wrong?
  3. How good are the tests: do they accept correct code, and how many known wrong answers do they reject?
  4. Does a second round help, in which a critic K lists the requirements the tests miss and adds tests?
  5. What does it all cost in minutes per correct answer?
- **Type:** exploration on h250-dev, reported only.
- **Gates for a locked check on h250-test (a rule, not a test).** B or C must meet all three:
  1. correct ≥ A;
  2. at most 2 verified answers wrong;
  3. minutes per correct ≤ 2 × A.
- **Changes before the lock**, after the first compatibility check on Part 4's two public example tasks:
  - In that check, T wrote a whole C++ `main()` (a compile error), and a stray `return 0;` would have hidden failed checks.
  - So C++ tests became the body of a function returning `int`, a whole `main()` is replaced by its body, and each language's format paragraph gained one sentence.
  - An empty-code check was added to the evaluation.

**Results** (appended after the run):
- **Run:** 3.5 hours, nothing failed, display on (212 records).
- **Main table:** A 43/50 (1.23 min), B 44/50 (2.99; verified 14, 0 wrong), C 44/50 (3.74; verified 6, 0 wrong), O 49/50.
- **Tests:** T accepted the reference on 13/50 tasks. Of the 37 rejections, 26 failed first at a hand-written expected value. T rejected 116/116 wrong-output and 34/37 too-slow known wrong answers, accepted 0/16 wrong branches, and rejected 129/182 correct branches.
- **Gates:** 1 and 2 met, 3 failed for B and C, so no locked check.

**Deviations and notes:**
1. **Compatibility check not met.** In the second check, T's tests rejected both reference solutions because of wrong expected values. The method was not changed (decided by the user before the lock), and this was recorded as a deviation with the expectation that many correct branches would be rejected.
2. **A gap in the gate rule.** It did not cover "gates 1 and 2 met, gate 3 failed". The closest case was used (no locked check). PLAN_11b's plan lists every case.
3. **A corrected median.** The per-task median times in the first write-up took the 26th of 50 values; they were corrected below the results (T 1.16 minutes, not 1.22).

## PLAN_11b (run 2)

- **One variable:** T's thinking, low → medium. The prompt, seeds and everything else as in run 1. The critic was dropped: it added nothing in run 1 and cost about 35 minutes.
- **Main comparison:** T accepting the reference, run 2 against run 1, paired by task, exact McNemar. It reads "more thinking makes the tests more correct" only if acceptance rises with p < 0.05.
- **Gates:** as in run 1, now with every case written out. If the main comparison is not confirmed, the next variable is T's prompt, not more thinking.

**Results:**
- **Run:** 2.9 hours, nothing failed, display on (174 records). 2 tasks got no usable tests: an unclosed code block, and no code block.
- **Main comparison:** 15/50 against 13/50 (6 gained, 4 lost), p = 0.754: **not confirmed**.
- **A test that checked nothing:** in one Python test file, all checks were in functions that were never called. It passed on empty code and on every known wrong answer.
- **Main table:** A 43/50 (1.28), B 45/50 (3.07; verified 16, 1 wrong).
- **Gates:** 1 and 2 met, 3 failed. By the plan, the next step changes T's prompt.

**Notes:**
1. **The branches were byte-identical to run 1** (texts, tokens, hidden results), although the plan expected small run-to-run differences. Runs 1 and 2 therefore compare T on the same code.
2. **One compatibility-check reference was wrong.** On one public example, T's test was right and the reference written for that example was wrong (an overflow). Rejecting a reference does not always mean T is wrong; on h250 the references are program-checked (Part 8).

## PLAN_11c (run 3)

- **One variable:** T's prompt, compared with run 1 (low thinking again). Expected values may come only from:
  1. a brute force written inside the tests, compared on about 200 random small inputs and on the edge cases;
  2. a property checkable without knowing the answer;
  3. an example stated in the task.

  Hand-computed values are forbidden, the largest-size test checks only properties, and every check must actually run. The prompt is in [code/prompts.py](code/prompts.py) (sha256 of the prompt pair `0dab2a0b6ef7…1b0993`, recorded in runs 3 and 4).
- **New rule B2 (reported only):** T's tests run once on an empty code block while the task runs. If they pass, they count as no tests.
- **Main comparison:** T accepting the reference, against run 1, exact McNemar, with the same reading as in run 2. The gates are judged on B; B2 is reported.
- **If not confirmed:** stop tuning T on the 27B and wait for a newer model.

**Results:**
- **Run:** 2.7 hours, nothing failed, display on (164 records). The branches were byte-identical to runs 1-2 again.
- **Main comparison:** 31/50 against 13/50 (22 gained, 4 lost), p = 0.0005: **confirmed**. Failures at a hand-written value fell from 26 to 7.
- **Main table:** A 43/50 (1.24), B 46/50 (2.29; verified 33, 1 wrong, which hidden tests judged too slow). B2 equals B on every task. O 49/50.
- **Gates:** all three met for B and B2, so the next plan was the locked check (PLAN_12).
- **Cost:** test code that fails by itself rose from 2 to 6 tasks, and 6/37 too-slow wrong answers passed T's tests.

**Deviations and notes:**
1. **The compatibility check was not met as written.** Its condition 3 is "the tests compile". On the C++ example, T wrote `std::LLONG_MIN`, a macro outside `std`, so the tests did not compile; the pipeline itself worked (checked offline on a copy with that name fixed). The method was not changed (decided by the user before the lock), and this was recorded as a deviation.
2. **Correction, 2026-11-10.** The C++ test runs labelled "crash" on the references were compile errors. The sandbox keeps only the last 500 characters of output, so long compiler messages lost their "compile error" prefix. The count of test code errors (6) and every test and gate are unaffected. The published data gives both labels.

## PLAN_12 (run 4, the locked check)

- **Question:** does the method of run 3, unchanged, still work on tasks that were not used to develop it?
- **Tasks:** the 150 coding tasks of h250-test, which Part 8 reserved for final locked checks. This is their first use for checking a method.
- **The only change from run 3 is the task set.** Two flow rules were added: a task on which all five requests fail, or on which the memory guard fires, stops the run; a stopped run may resume from the next task. Neither happened.
- **Decisions before the lock:** B2 is the main rule; M3 compares with Part 8's real CL4 (no new CL4 run); M2's threshold is CL4's error rate on these tasks.
- **Tests:**
  - **M1 (main):** B2 against A from the same run, exact two-sided McNemar, p < 0.05. "B2 more" is read as "self-verification makes the 27B answer more coding tasks"; "A more" as "worse, not recommended"; anything else as "not confirmed".
  - **M2:** the share of verified answers that are wrong. "Reliable" if the upper end of the exact (Clopper-Pearson) 95% interval is below 17/150 = 11.3%, CL4's error rate on these tasks in Part 8.
  - **M3:** the ratio of minutes per correct answer, B2 / Part 8's CL4, with a bootstrap 95% interval. It is read in bands: "under 2x" if the upper end is below 2, "over 2x" if the lower end is above 2, otherwise "about 2x". B2 / A from the same run is also reported.
  - No multiplicity correction: only M1 decides "more accurate".
- **Conclusion table, written before the run:**

| M1 | M2 | Conclusion |
|---|---|---|
| more accurate | reliable | self-verification works for the 27B on coding tasks; recommend B2 where accuracy matters |
| more accurate | not confirmed | more correct answers, but "verified" is not a guarantee |
| not confirmed | reliable | **no accuracy gain confirmed: B2 not recommended instead of CL4; "verified" can decide which tasks to escalate** |
| not confirmed | not confirmed | not recommended; stop this line until a newer model |
| A more accurate | — | not recommended; analyse why |

- **Capacity, stated before the run:** A was expected to miss about 17 tasks, and B2 can at best win those back. With no task lost, M1 needs 6 tasks that only B2 solves. So only a large gain is detectable.
- **Reported only:**
  - the main table and groups (Python and C++, tiers, specification-style or not);
  - where "verified" comes from;
  - Part 8's CL4 and Flash Next as external references;
  - the composed escalation estimate;
  - test quality with both failure labels;
  - the time breakdown;
  - the development and test numbers side by side.
- **Check before the lock:** with the run's C++ wrapper and without the hidden prelude, all 150 references and 444 known wrong answers pass an empty test. The compatibility check on Part 4's two public examples met all four conditions.

**Results:**
- **Run:** 10.7 hours (estimate 10-14), nothing failed, no stop or resume, memory guard never fired, display on (642 records). The lock was checked after the run.
- **M1:** 5 / 0, p = 0.0625: **not confirmed**.
- **M2:** 1/85, interval 0.03%-6.38%: **reliable**.
- **M3:** 2.12x (1.77, 2.61): **about 2x**. B2 / A from the same run: 1.96x (1.65, 2.37).
- **Conclusion (third row):** no accuracy gain confirmed; B2 is not recommended instead of CL4; "verified" can decide which tasks to escalate.
- **Reported only:** see the README. The composed escalation to Flash Next B (140/150, 7.06 minutes per correct answer) was slower than Flash Next alone (6.58), so Flash Next is not a useful escalation target here.

**Notes:**
1. **The title of the locked text still says "draft, not locked".** It was left over from the draft; the status line below it says locked. Nothing else is affected.
2. **The lock was created twice, both times before the run.** The plan text first stated a wrong number of locked files, so the lock was deleted, the text fixed, and the lock created again.
