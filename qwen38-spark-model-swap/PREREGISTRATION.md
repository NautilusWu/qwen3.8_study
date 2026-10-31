# Pre-registration

Four plans, each written and sha256-locked before its runs (English summaries of the Chinese originals; the hashes are of the original files as locked). Every run script checked the lock at start; results were appended only after the runs, below the locked text.

| Plan | Locked | sha256 (locked text) | Runs |
|---|---|---|---|
| PLAN_05 model swap | 2026-10-21 | `fce850a248a1e7b447aceb1f4d50a876f74efeb8b867e2b55cbd380152d66594` | 72 tasks, phase 1 (deep, low) and phase 2 (plan+low, vote) |
| PLAN_05 addendum 01: NVFP4 lm_head | 2026-10-21, before any measured run | `123dd3d5b08e8f0145cf7c1e27518d8f5124b3d77eaae0d1bd1aa879957f89d8` | (conversion and checks) |
| PLAN_06 400 new tasks | 2026-10-22 | `1b6458b8a3872338e043b3ccbfba4bd47f059b00a66f0fcd63345cff985876e2` | original vote, fine-tune vote, fine-tune deep |
| PLAN_06 addendum 01: reference configurations | 2026-10-23, after PLAN_06's results | `8cc04678ba71e297428075001ddf3b83848c0ab0528b9a3e65c4e304c79d046c` | original deep, original low, fine-tune low (reported only) |

400-task prompt file sha256: `591b703379981311905ec9ed7a2beef5c34f880beafd92f34e759fe18e09a85b` (not published).

## PLAN_05: fine-tune vs original on Part 4's 72 tasks

- One variable: the target checkpoint. Engine, image, drafter (z-lab DFlash2), FP8 KV/prefill, settings, seeds and run settings as in the original runs being compared (Part 4's dev and confirmation runs).
- A compatibility check (load, 5 example tasks, output sanity, draft acceptance) had to pass before any measured run.
- Locked tests on the 72 tasks, paired by task:
  - **M1** deep thinking accuracy: exact McNemar, two-sided p < 0.05 → "more" or "less accurate". **Result: 11 vs 1, p = 0.006 → more accurate.**
  - **M2** deep thinking minutes per correct answer: paired bootstrap 95% CI of the ratio (10,000 resamples, seed 2026); upper bound < 1 → faster. **Result: 0.41 (0.30, 0.54) → faster.**
  - **M3** low thinking accuracy, as M1. **Result: 5 vs 11, p = 0.21 → no detectable difference.**
  - **M4** low thinking time, as M2. **Result: 0.63 (0.44, 0.92) → faster**, accuracy not significantly lower.
  - Phase 2 (deep plan + low execution; 4-branch vote): reported only.

## PLAN_05 addendum 01: converting the fine-tune's lm_head to NVFP4

- Why: the compatibility check showed the published folder's only format difference from NVIDIA's checkpoint is a BF16 `lm_head`, which makes TensorFold decode about 3x slower (about 350 vs 110 ms per verification step) with similar draft acceptance. The time comparison would otherwise measure the file format, not the fine-tune.
- Change: quantize only `lm_head` with ModelOpt (block 16, weight-only) into NVIDIA's three-tensor layout; all other files symlinked unchanged.
- Locked checks, all passed: (1) identical tensor names/dtypes/shapes apart from the unused `lm_head.input_scale`; (2) relative error <= 0.15 (got 0.095), packing check (ModelOpt vs TensorFold dequant: 1.5e-8); (3) greedy outputs, BF16 head vs NVFP4 head, on 6 general prompts: at least 4 of 6 without divergence in the first 32 tokens (got 4: divergence at 12, 21, 80, 136, none, 41) **and** the diverging token within the other's top 5 — **deviation: this part could not be run, TensorFold returns no log-probabilities for this model family**; (4) per-step time <= 1.3x the original (got 114-118 ms vs 98-118 ms).

## PLAN_06: three configurations on 400 new tasks

- Tasks: 200 coding (30 C++) + 200 reasoning, each from a new task family not overlapping earlier sets; answers computed by programs and checked against brute force on small inputs; every reference answer accepted and at least two typical wrong answers rejected by the grader in a network-off sandbox; all references re-graded through the exact run-time grading path. Written before any model saw them; difficulty not tuned with any model.
- Configurations: original + 4-branch vote, fine-tune + 4-branch vote, fine-tune + deep thinking (Part 4 confirmation settings).
- Output: tables only, no "equally accurate" verdicts — totals, and for each pair: tasks only one side solved, exact McNemar p and Holm-adjusted p, paired bootstrap 95% CIs of the accuracy difference and of the time ratio.
- Code changes (locked): the task split, and grading in chunks of 50 items per sandbox container (same grader and limits).

## PLAN_06 addendum 01: three reference configurations (reported only)

Decided after seeing PLAN_06's results, to show readers where the three best configurations stand: original deep, original low, fine-tune low, on the same 400 tasks with the same settings. Reported only; they do not change PLAN_06's locked results.
