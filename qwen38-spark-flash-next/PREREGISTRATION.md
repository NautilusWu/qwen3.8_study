# Pre-registration

One plan and one addendum, each written and sha256-locked before its runs (English summaries of the Chinese originals; the hashes are of the original files as locked). Every run script checked the locks at start; results were appended only after the runs, below the locked text.

| Plan | Locked | sha256 (locked text) | Runs |
|---|---|---|---|
| PLAN_07 Flash Next vs 27B on the 400 tasks | 2026-10-25, after the compatibility check and MTP sweeps, before any of the 400 tasks | `4ea3ed0025f8359bab77aa4f6c8971e8cbf88ae7abc2b6d0b151a0bd3437750c` | Flash Next low, vote, deep |
| PLAN_07 addendum 01: the largest output limit | 2026-10-27, after seeing PLAN_07's low-thinking and vote results | `b86bf021b78855db886b999a064764aea2c21d91d049a20c22cbda613775d2ee` | Flash Next low with a 260,096-token limit (reported only) |

400-task prompt file sha256: `591b703379981311905ec9ed7a2beef5c34f880beafd92f34e759fe18e09a85b` (not published; the same file as Part 5).

## Before the plan: compatibility check and MTP sweeps (not locked)

- Compatibility check on Part 4's 5 public example tasks, as in Part 5: the server starts, every usage runs and is graded (5/5 each), a 60,653-token prompt with one hidden fact is answered correctly, and a reply runs to the 65,536-token limit; memory stays above 25 GiB available.
- MTP sweeps on the same 5 example tasks (data/mtp/): fixed draft counts 0-8 and thresholds 0-0.9. Every setting gave the same tokens as the default. After the display-on sessions, the user chose the threshold 0.6 with up to 6 drafts for the study.

## PLAN_07: Flash Next vs 27B, three usages, 400 tasks

- Variable: the model. Flash Next: local-inference-lab/Qwen3.8-Flash-Next-NVFP4 at commit `7c4f1bc1a2d6847e0cbc01ac6b823f00251de8dd`. 27B: Part 5's runs of nvidia/Qwen3.8-27B-NVFP4 at `482ca0f3` (reused, not rerun).
- Changes that come with the model, stated in advance: own MTP drafting (threshold 0.6, up to 6) instead of DFlash2; bf16 instead of FP8 KV cache; TensorFold's default memory reserve, no container memory limit and a 3 GiB memory guard instead of 40 GiB / 80 GB / 12 GiB (the 27B settings do not fit).
- Unchanged: engine and image (`qwen38-tensorfold-repro:0.6.0-trace`, sha256:75d748c5…), runner code (Part 4's, locked by hash), request settings, seeds, output limit 65,536, context 81,920, 40 minutes per task, grader and sandbox.
- Display on throughout the runs (the 27B runs did not record the display state; stated in advance as a limitation of the time comparison).
- Order: low thinking, vote, deep thinking (slowest last).
- Locked tests, three pairs (Flash Next vs 27B, same usage):
  1. Accuracy: exact two-sided McNemar, Holm-adjusted over the three pairs; adjusted p < 0.05 → "a detectable accuracy difference".
  2. Time: ratio of minutes per correct answer, paired bootstrap 95% CI (10,000 resamples, seed 2026); CI excluding 1 → "a detectable time difference".
  3. The accuracy difference's 95% CI is reported too. Reporting as tables; differences not detected are written as such, not as "equally accurate".
- Reported only: the full six-configuration table (correct, time, limit hits, no answer), Flash Next's internal comparisons, memory.

**Results** (appended after the runs; locks verified):

| Usage | Only Flash Next / only 27B | p | Holm p (3 pairs) | Accuracy difference (pp) | Time ratio | Reading under the rules |
|---|---|---|---|---|---|---|
| Deep thinking | 15 / 23 | 0.26 | 0.26 | -2.0 (-5.0, +1.0) | 1.01 (0.89, 1.15) | no detectable difference in accuracy or time |
| Low thinking | 30 / 9 | 0.0011 | 0.0022 | +5.25 (+2.25, +8.25) | 2.01 (1.60, 2.50) | Flash Next more accurate; slower |
| 4 branches + vote | 21 / 3 | 0.0003 | 0.0009 | +4.5 (+2.25, +7.0) | 2.43 (2.03, 2.90) | Flash Next more accurate; slower |

**Deviation:** the first deep-thinking run stopped after 4 tasks (the 5th was interrupted at 875 s) when another job started on the machine (a text-to-speech container using about 35 GB) and the memory guard fired. It was rerun from the start with the same command and locked code once the GPU was free; the interrupted run is kept locally and not counted. Low thinking and the vote were complete before this.

## PLAN_07 addendum 01: the largest output limit (reported only)

- Why: PLAN_07's low thinking had 5 replies at the 65,536-token limit and 7 without an answer; deep thinking was showing more. Decided after seeing these, so it is reported only and changes nothing in PLAN_07.
- Change: context 262,144 (the model's native window), output limit 260,096, no time limit per task; everything else as in PLAN_07.
- Planned runs: low thinking, then deep thinking; compare each with PLAN_07's run of the same usage.

**Results:**

- Low thinking: 384 correct, the same as with 65,536 (no task changed from wrong to right or back); 395 of 400 replies identical token for token; total time 758.4 vs 661.6 minutes. The 5 previously capped replies: 2 repeated themselves until 260,096 tokens (58 and 60 minutes), 3 stopped at 75,400, 91,027 and 95,326 tokens without an answer.
- **Deviation, deep thinking not completed:** the runner's deep-thinking method (`A65`) sets its own 65,536-token limit and ignores `--max-tokens`, so the run with the larger limit produced replies identical to PLAN_07's (91 of 91, including the 9 capped ones). We stopped it after 91 tasks; it is not reported. The method that reads `--max-tokens` (`A`, otherwise identical) was not run.
