# Plans and when they were locked

Each step was planned in writing and the plan locked (hash recorded) before its run; analyses added after a run are marked as such in the results. The original plans are in Chinese and refer to the private task files; this is a faithful English summary. Times are US Eastern.

| Step | Locked | Plan hash (sha256) | What it decided |
|---|---|---|---|
| Exploration 1 (dev, 24 tasks) | 2026-10-17 22:30 | `56980bdb…62d03c982` | Methods A, D1, D2, D3, C4, C8, B; screening rule below |
| Addendum 1 (dev) | 2026-10-18 12:58 | `2a4dbd9c…a53a9784` | CL4 (parallel low thinking) vs D2 |
| Plan 2, step 1 (dev) | 2026-10-18 14:41 | `0ff91940…52bd2` (+ code `bc794861…3c3e2`) | Cooperative parallel methods P1, P2, with all prompts attached |
| Plan 2, step 2 (hard, 24 new tasks) | 2026-10-18 16:04 | `900c6c09…6db33a` (tasks `fc636efd…00852`) | A, D2, B, CL4, P1, P2 on the hard set; no task removed after results |
| Plan 3 (dev + hard) | 2026-10-19 00:22 | `392ae118…b0e8b7` | A65 (output limit 65,536) and H (deep direction + low execution) |
| Plan 4, confirmation (48 new tasks) | 2026-10-19 14:22 | `9ada93a2…636698e5` (tasks `5b2cd4bc…3cd1c7d5`) | C1–C5 below |

## Method definitions

| Code | Definition |
|---|---|
| A / A65 | Deep thinking (chat template default effort), greedy, output limit 24,576 / 65,536 |
| D1, D2 | `reasoning_effort` medium / low, greedy |
| D3 | Default effort with `thinking_budget` 8,192 (the server closes the thinking block at 8,192 tokens) |
| C4, C8 | 4 / 8 parallel branches with default effort, sampled (temperature 0.6, top_p 0.95, top_k 20, seed 1000+k); tasks with an exact answer: majority vote, stopping as soon as more than half agree, otherwise plurality (ties to the earliest); coding tasks: the first branch to finish with a code block |
| CL4 | As C4, every branch with `reasoning_effort: low` |
| B | Agent with one tool, `run_code(language, code)` (Python or C++ in a network-isolated sandbox, 20 s, output cut at 4,000 characters), up to 12 calls, 16,384 tokens per turn, earlier turns' thinking not sent back, 64K context |
| P1 | Low effort throughout: the model writes a JSON split into 2–4 independent subtasks (or declines, then answers directly); subtasks solved in parallel; a merge call writes the final answer |
| P2 | Low effort, 4 fixed parallel roles. Coding: implementation, tests from the specification, edge-case tests, a second implementation; both implementations run against both test sets in the sandbox; if neither passes all and both fail the same tests, the tests are treated as suspect; otherwise up to 2 repair rounds. Reasoning: direct answer, a program that computes the answer, an answer by another method, a constraint checker; majority, then the checker, then a judging call |
| H | Deep thinking writes a direction (≤300 words, no code or final answer, limit 16,384), then low effort answers the task given that direction (limit 65,536) |

## Rules

**Exploration screening** (relative to A on the same set): worth validating if minutes per correct ≤ 85% of A's, or ≥ 2 more correct with total time ≤ 125%. Addendum 1 and Plan 2 used analogous rules against D2, B and CL4 (85% time, or +2 correct within 150% time; P vs B/CL4 additionally required correct ≥ baseline − 1). Plan 3: "capacity under-estimated" if A65 has ≥ 2 more correct than A; H better than D2 if ≤ 85% time per correct, or ≥ 2 more correct within 200% total time; H "close to deep thinking at low cost" if correct ≥ A65 − 2 and total time ≤ 50%.

**Confirmation (Plan 4)**, 48 tasks, α = 0.05, exact sign test on discordant tasks, paired bootstrap intervals (10,000 resamples, seed 2026):

| Claim | Confirmed if |
|---|---|
| C1 | D2 minutes per correct ≤ 50% of A65's and the 95% interval of the ratio is below 1 |
| C2 | D2 vs A65 two-sided p ≥ 0.05 and D2 correct ≥ A65 − 3 |
| C3 | H correct ≥ A65 − 2, and H minutes per correct ≤ 50% of A65's with the interval below 1 |
| C4 | CL4 vs D2 one-sided p < 0.05 |
| C5 | CL4 minutes per correct ≤ 1.5x D2's |

Outcome: C1, C2, C3, C5 confirmed; C4 not confirmed (p = 0.055). See [README.md](README.md).

## Changes made before locking, after smoke tests (two dev tasks, not used for any decision)

- B: sending every turn's thinking back made the 12th request exceed the context; changed to not sending earlier thinking, and 64K context.
- P2: model-written tests imported the function as a module; import lines for the implementation are removed before running (implementation and tests share one file). The "suspect tests" rule was added after a smoke task showed both implementations failing the same wrong test.
