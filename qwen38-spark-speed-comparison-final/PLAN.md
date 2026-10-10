# Plan of the final test (written 2026-10-09, before any of the 48 sealed prompts was run)

This is the plan as written before the run, with internal file paths removed. The rules and the decision on `c_segtree` are unchanged; the decision was made before any sealed output existed.

## Question

On 72 prompts (24 dev + 48 sealed), is TensorFold + NVIDIA NVFP4 + DFlash2 at least as correct as the best vLLM configuration (NVIDIA NVFP4 + MTP5), and does its ~2x total-time advantage hold on new prompts? Secondary: does TensorFold + MLX 4-bit pass fewer prompts than TensorFold + NVIDIA NVFP4?

## Prompts

The 72 prompts were written and split once, before Part 1: 24 dev prompts (used in Parts 1 and 2) and 48 sealed prompts (24 coding, 24 reasoning, never sent to any model before this test). The split was fixed by a seed and stratified by category and subcategory.

## Runs (one launch each; outputs are token-deterministic per configuration)

1. vLLM 0.27.1 + NVIDIA NVFP4 + MTP5, launch command identical to the Part 2 MTP5 run; only the prompt file differs.
2. TensorFold 0.6.0 + NVIDIA NVFP4 + DFlash2, same image, flags and warm-up as Part 2.
3. TensorFold 0.6.0 + MLX 4-bit + DFlash2, same as 2 with the MLX 4-bit target.

TensorFold uses the vLLM run's prompt order and must match its prompt token hashes for every prompt.

## Conditions

Identical to Part 2: thinking on (template default), temperature 0, max_tokens 24,576, context 32,768, one request at a time, display off, no automatic thermal stop, TensorFold memory/driver watchdog on. The run starts only after the monitor is off.

## Checks before the run (done)

- The sealed prompt file, the dev prompt file and the reference answers match the sha256 recorded when the prompts were written.
- Grader self-test: 48/48 sealed graders accept their reference answer and reject known-wrong answers.
- Real grading path (the graders stored in the sealed file, applied to the text after `</think>`): 47/48 clean. Exception: `c_segtree` (performance-bound, 60 s time limit): the reference runs in 1.0 s, the known-wrong brute force in about 61 s, so it was rejected in 2 of 3 trials and accepted in 1.

## Analysis (fixed before the run)

- Both engines are graded by the same grader function in a network-less container.
- Primary comparison, TensorFold NVFP4 vs vLLM MTP5 on 71 prompts (24 dev from Part 2 + 47 sealed): count prompts passed only by TensorFold (b) and only by vLLM (c); exact two-sided sign test on b vs c.
  - "Quality loss detected" if c > b and p < 0.05.
  - Otherwise "no quality loss observed" (with the counts and p-value reported).
- Secondary, TensorFold MLX 4-bit vs TensorFold NVFP4 on 71 prompts: same test; "MLX 4-bit passes fewer" if p < 0.05.
- Speed: per-prompt paired geometric mean of total time (vLLM MTP5 / TensorFold), median output tok/s, total hours, on the 48 sealed prompts and on all 72.
- Also reported: pass counts by category, budget-exhausted prompts, temperature peaks, display state.
- No setting is changed because of sealed results. If anything is changed later, the sealed results become development evidence.

## c_segtree

Decision (before any sealed prompt was run): exclude `c_segtree` from the pass/fail decision. It is still run and graded with its original grader and reported separately. The primary and secondary tests use 71 prompts (24 dev + 47 sealed). Graders are not modified.

## Added after the run

The time-to-correct-answer measures in the README (minutes per correct answer, time on the prompts all configurations got right) were added after the run. They are descriptive and are not part of the pre-registered tests above.
