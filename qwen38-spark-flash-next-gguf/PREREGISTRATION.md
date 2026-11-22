# Pre-registration

This part has three plans. Each was written and sha256-locked before the runs it governs.

- Below is an English summary of the Chinese originals. Each hash is of the original file as locked.
- Every run script checked its locks at the start and again at the end; all checks passed.
- Results were appended only after the runs, below the locked text. A check confirmed that each plan file still begins with its locked copy (`verify-prefix`).

| Plan | Locked | sha256 (locked text) | Files in the lock | Runs |
|---|---|---|---|---|
| PLAN_18: three GGUF files on llama.cpp | 2026-11-15, after the download, the compatibility check and the user's decision below, before any of the 400 tasks | `d863176d48ac37121110bca5d99a5c4d6daa416b1ff310318e5984be0123271c` | 44 | P1 tool calls, P2 400 tasks |
| PLAN_18b: fill after the interrupted run | 2026-11-16, before the fill; P2 not yet graded | `9f7fa1f50cc4fe2ca77113005a0de1aee3f49c8a2e2370a6f0084342fcf3a488` | 6 | the missing blocks |
| PLAN_18c: UD-IQ3_XXS + MTP against TensorFold, first lock | 2026-11-17, before its compatibility check | `75f29f27024ed35bbcd3c699307a93e1771ef46aff90056e2bdf075b2287c1c3` | 11 | compatibility check (P0) |
| PLAN_18c, final lock | 2026-11-18 00:03, after P0 had chosen N by its rule, before any of the 400 tasks | `7fc6934387130dd219911b9828d28f27c58981bf70be25f6a3a3ee20e09146da` | 18 | P2 400 tasks, P3 probe |

**What the locks covered:**
- the plans;
- all run and analysis code;
- the code it imports (Part 4's runner and server code, the grader, `stats.py`);
- the llama.cpp binary and its libraries;
- the chat template and the MTP head;
- the compatibility check results;
- the reference runs used for comparison (Part 6's TensorFold run);
- the 400-task prompt file, sha256 `591b703379981311905ec9ed7a2beef5c34f880beafd92f34e759fe18e09a85b` (not published; the same file as Parts 5-7 and 10).

The GGUF files themselves were checked by sha256 against Hugging Face at download ([data/downloads.json](data/downloads.json)).

## PLAN_18: three GGUF files, one engine

**Why.** A post on r/LocalLLM said ISTA-DASLab's GSQ-RCO IQ3_S is better than Unsloth's new UD-IQ4_XS. The evidence was personal use on Strata, and comments disagreed; one noted that Strata treats the two vendors' files differently, so the engine is a confound. The study changes only the weight file. TensorFold cannot read GGUF (1.2.7: "GGUF serving is coming"), so TensorFold + NVFP4 is a reference, reported only.

**Questions:**
1. **Accuracy (primary):** IQ3_S against UD-IQ4_XS (the post's claim), and IQ3_S against UD-IQ3_XXS (the same file size, 83.6 GB).
2. **Output length and loops (secondary).**
3. **Tool calls (secondary):** format errors on 32 recorded agent turns.
4. **Speed (reported only).**

**Fixed for all three files:**
- **Engine:** llama.cpp v1.0.0, commit `b1b770d9`, built locally.
- **Server flags:** the same for every file (see the README).
- **Chat template:** the official template from the NVFP4 checkpoint at `7c4f1bc1`.
- **Requests:** greedy, output limit 65,536, Part 4's request otherwise, with `"cache_prompt": false`.
  - The smoke test found that llama.cpp's prompt cache makes greedy replies depend on the previous request.
  - The TensorFold reference sent its requests unchanged.
- **No MTP:** this was the default setting. The plan also said, wrongly, that v1.0.0 had no MTP for this model (see the correction below).

**Compatibility check (P0), 9 criteria per file.** Any failure was to stop the run, not to change settings.

| # | Criterion | GSQ-RCO IQ3_S | UD-IQ3_XXS | UD-IQ4_XS |
|---|---|---|---|---|
| 1 | loads within 10 minutes, no errors; lowest available memory at least 3 GiB | 68.1 s, 55.1 GiB | 82.7 s, 57.0 GiB | 206.2 s, 47.1 GiB |
| 2 | 3 fixed texts tokenize to the same ids on all files | yes | yes | yes |
| 3 | templates render the same on all files; low and default effort differ; tools appear | yes | yes | yes |
| 4 | a greedy request twice gives the same reply | yes | yes | yes |
| 5 | Part 4's 5 examples with D2: at least 4/5 correct, no garbled replies | 5/5 | 5/5 | 4/5 (see below) |
| 6 | `--reasoning-budget 1024` holds | 1,023 thinking tokens | 1,023 | 1,023 |
| 7 | a recorded agent turn is parsed into `tool_calls` | yes | yes | yes |
| 8 | decoding at least 5 tokens/s (long reply) | 29.4 | 30.1 | 25.1 |
| 9 | listens on 127.0.0.1 only, no outside connections | yes | yes | yes |

Reported only: llama.cpp's rendering matched Python jinja2's for all three prompt kinds. The TensorFold reference started (82.1 s). The display was off throughout the check. Details: [data/compat_check.json](data/compat_check.json).

**Deviation: criterion 5 for UD-IQ4_XS.**
- One example (`ex_five_numbers`) looped in its thinking for 26,078 tokens. The model then stopped by itself without closing its thinking, so the reply had no answer and was graded wrong.
- That still meets the plan's text: at least 4 of 5 correct, no garbled replies.
- But the check's code also required every example to have an answer, so the pipeline stopped before the lock. Plan text and code had both been written before the results.
- **The user decided, after seeing this, to follow the plan's text.** That is a decision after the fact, not a pre-written rule. Nothing else changed: files, flags, requests, tasks and rules.
- The loop happened on a public example before the lock, so it is not counted in question 2.

**Choice of task set (rule written in advance).** The rule estimated GPU time from the long-reply speed and the reference's token counts:
- the harder 250-task set with deep thinking, if it fit in 100 hours;
- otherwise the 400 tasks with D2.

The estimates were 133.8 and 61.1 hours, so the run used **the 400 tasks, D2**.

**Design.**
- **P1, tool calls:** 32 agent turns recorded in an earlier study (OpenCode on our own private agent tasks) were replayed as recorded, output limit 8,192, three replies each: greedy, and sampled with temperature 1.0, top_p 0.95, top_k 20 and seeds 1 and 2. All 4 configurations ran them, 96 replies each.
  - Labels, assigned automatically: valid call, malformed, text only, cut. The rule is in [code/tool_call_labels.py](code/tool_call_labels.py).
- **P2, the 400 tasks:** 8 groups of 50 in file order. Every file ran every group, with a server restart before each block. The orders rotate so that, in the first 6 groups, each file holds each position twice:

  | Group | Order |
  |---|---|
  | 0 | IQ3_S, UD-IQ3_XXS, UD-IQ4_XS |
  | 1 | UD-IQ3_XXS, UD-IQ4_XS, IQ3_S |
  | 2 | UD-IQ4_XS, IQ3_S, UD-IQ3_XXS |
  | 3 | IQ3_S, UD-IQ4_XS, UD-IQ3_XXS |
  | 4 | UD-IQ3_XXS, IQ3_S, UD-IQ4_XS |
  | 5 | UD-IQ4_XS, UD-IQ3_XXS, IQ3_S |
  | 6 | IQ3_S, UD-IQ3_XXS, UD-IQ4_XS |
  | 7 | UD-IQ4_XS, IQ3_S, UD-IQ3_XXS |

  The limit was 120 minutes per task, a safety limit only.

**Tests and readings, written in advance:**
- **Q1 (primary):** IQ3_S vs UD-IQ4_XS and IQ3_S vs UD-IQ3_XXS, paired by task.
  - Exact McNemar p, Holm-adjusted over the two pairs.
  - Bootstrap 95% intervals for the accuracy difference and the time ratio: 10,000 resamples, seed 2026.
  - A Holm p below 0.05 means a detectable accuracy difference; otherwise "no detectable difference", with the intervals.
  - UD-IQ4_XS vs UD-IQ3_XXS is reported only.
- **The verdict on the post's claim** (IQ3_S vs UD-IQ4_XS only):

  | Result | Verdict |
  |---|---|
  | IQ3_S detectably more accurate | supported |
  | no detectable accuracy difference, but IQ3_S detectably better on loops or tool-call errors, and not worse on the other | partly supported |
  | UD-IQ4_XS detectably more accurate | not supported |
  | anything else | no detectable difference |
- **Q2 (secondary, no adjustment):**
  - total-token ratio with a bootstrap interval, and the median per-task ratio;
  - limit hits;
  - loops: a reply that hit the limit and whose last 8,000 characters compress (zlib) to under 10% of their size; thresholds of 5% and 15% as sensitivity;
  - McNemar on loops.
- **Q3 (secondary):** label counts, and McNemar on malformed calls.
- **Q4 (reported only):** decoding speed, median first token, minutes per correct answer.

**Power, as stated in the plan:** the tasks are near the ceiling (the reference missed 16 of 400). A clearly worse file can be seen; a small improvement cannot. With 96 replies per configuration, tool calls detect only very large differences.

**Not done:** MTP or other speculative decoding (see the correction below), tuning llama.cpp, Strata, rerunning the reference on the 400 tasks, and KL divergence (neither BF16 nor Q8_0 fits on one Spark).

## The interrupted run and PLAN_18b (the fill)

**What happened:**
- The run started on 2026-11-15 13:19, with the lock verified.
- A stopped llama-server stays in `nvidia-smi` for a few seconds. When the next block started at once, the runner's own check ("GPU is in use") failed it, and the whole block was skipped.
- 8 of 24 blocks were skipped this way, and so were UD-IQ3_XXS's tool-call replies.
- The user stopped the run on 2026-11-16 12:47. At that moment UD-IQ3_XXS had finished 19 of the 50 tasks in group 6.

Before the fill, the 400-task results had not been graded: only per-task times and token counts had been seen. The tool-call labels of the three other configurations had been seen.

**PLAN_18b** changed no test, task, file, flag or rule. It added two files:
- a wrapper that waits until `nvidia-smi` lists no process (at most 300 s) and then calls the locked start function;
- a runner that runs only the missing blocks, in the locked order, appending to the existing results. Two failed blocks in a row stop it.

It was checked in three ways before and during the fill:
- four switches between real servers worked (waits 0.0-0.1 s; this did not reproduce the original delay, so it only shows that the wait does no harm);
- the line counts and sha256 of all existing result files were recorded, and later shown to be unchanged prefixes;
- both locks were checked before and after.

The fill ran 2026-11-16 13:06 to 2026-11-17 21:31:
- 12 starts, all successful; 3 waits above zero (1.1, 0.1 and 1.1 s);
- each configuration ended with exactly 400 distinct tasks;
- UD-IQ3_XXS got all 96 tool-call replies.

Which tasks ran in the fill:

| | Tasks run in the fill |
|---|---|
| IQ3_S | 200 of 400 |
| UD-IQ3_XXS | 150 of 400, plus all tool-call replies |
| UD-IQ4_XS | 300 of 400 |

The `run` column of the per-task files and [data/blocks.csv](data/blocks.csv) show which blocks ran in the fill.

## PLAN_18 results (appended after the run)

| | GSQ-RCO IQ3_S | UD-IQ4_XS | UD-IQ3_XXS | Reference: TensorFold + NVFP4 + MTP (Part 6's run, reported only) |
|---|---|---|---|---|
| Correct | 383/400 | 379/400 | 380/400 | 384/400 |
| Minutes per correct answer | 3.21 | 2.77 | 2.15 | 1.72 |
| Output tokens | 2,086,062 | 1,598,688 | 1,432,914 | 2,052,826 |
| Hit the limit | 1 | 4 | 5 | 5 |
| Loops (thresholds 0.05 / 0.10 / 0.15) | 1 / 1 / 1 | 1 / 1 / 1 | 1 / 1 / 1 | |
| Over 40 minutes | 0 | 4 | 2 | 0 |
| Tool calls: valid / text / cut / malformed (of 96) | 72 / 22 / 2 / 0 | 74 / 22 / 0 / 0 | 73 / 22 / 1 / 0 | 71 / 22 / 3 / 0 |

**Q1:**

| Pair | Only first / only second | p | Holm p | Accuracy difference (pp) | Time ratio |
|---|---|---|---|---|---|
| IQ3_S vs UD-IQ4_XS | 15 / 11 | 0.557 | 1.0 | +1.0 (-1.5, +3.5) | 1.157 (0.941, 1.432) |
| IQ3_S vs UD-IQ3_XXS | 12 / 9 | 0.664 | 1.0 | +0.75 (-1.5, +3.0) | 1.489 (1.264, 1.794) |
| UD-IQ4_XS vs UD-IQ3_XXS (reported only) | 13 / 14 | 1.0 | | -0.25 (-2.75, +2.25) | 1.287 (1.013, 1.648) |

**Q2 and Q3:** loops 1 against 1 (p = 1.0) and malformed calls 0 against 0 in both primary pairs. Output tokens, IQ3_S against UD-IQ3_XXS: 1.456 times (1.255, 1.735), median per task 1.041. IQ3_S against UD-IQ4_XS: 1.305 times (1.079, 1.583), median per task 1.036.

**Verdict on the post's claim: no detectable difference.**

**Deviations and limits, as appended:**
1. Criterion 5 for UD-IQ4_XS and the user's decision (above).
2. The interrupted run and the fill (above). The fill broke the planned balance of order and time. Decoding speeds of original and fill blocks differ by about 5-6%, in different directions per file; the blocks hold different tasks, so this cannot be put down to timing.
3. UD-IQ3_XXS's group 6 ran in two parts (19 + 31 tasks). The first 19 have no llama.cpp timings, so its decoding speed uses 381 tasks.
4. The display was on for 6 of 1,945 records, on 2026-11-16, during 2 tasks of UD-IQ4_XS (one of them for about 7.5 minutes, about 5% slower than similar tasks).
   - **Sensitivity check** (computed after the results; the rule to drop the overlapping tasks was set before grading, but not in a locked file): without those 2 tasks (398 tasks), IQ3_S vs UD-IQ4_XS gives 14 / 11, p = 0.69, time 1.164 (0.946, 1.44). IQ3_S vs UD-IQ3_XXS gives 11 / 9, p = 0.824, time 1.48 (1.25, 1.79). Conclusions unchanged.
5. One run per configuration, greedy.

**Correction, appended 2026-11-17.** The plan's statement that llama.cpp v1.0.0 lacks Flash Next's MTP was wrong. The check had looked at an unmerged community pull request (#27836); the feature was merged through another one (commit `c061df198`, "Qwen4Exp: add MTP (#29761)", 2026-10-01).
- None of the three GGUF files contains MTP layers.
- The speed comparison with the reference (which drafts with MTP) was therefore even less fair than the plan said. This led to PLAN_18c.

**Second correction, appended 2026-11-18 (from PLAN_18c).** The first correction added that MTP "in principle does not change greedy output". For llama.cpp v1.0.0 that is not true: with and without MTP, only 9 of 400 replies were identical. Accuracy (381 against 380) and total tokens (1,431,344 against 1,432,914) did not visibly change.

## PLAN_18c: UD-IQ3_XXS + MTP against TensorFold + NVFP4 + MTP

**Why:** to replace PLAN_18's unfair speed comparison with a fair one. The user approved it, and the download of the MTP head, on 2026-11-17.

**Questions:**
1. **Q1 (primary):** on the 400 tasks with D2, with both run in alternating blocks in one session, which has the shorter time per correct answer: UD-IQ3_XXS + MTP (llama.cpp v1.0.0) or TensorFold + NVFP4 + MTP?
2. **Q2 (secondary):** in llama.cpp, with only MTP on or off changed, how do UD-IQ3_XXS's replies, accuracy and time change? PLAN_18's MTP-off run is from another session, so its time is reported only.
3. **Q3 (reported only):** the MTP speed-up of each file in one session, and from it estimated times per correct answer for IQ3_S and UD-IQ4_XS with MTP.
4. **Q4 (reported only):** does TensorFold reproduce Part 6's run?

**Configurations:**
- **UD-IQ3_XXS + MTP:** PLAN_18's UD-IQ3_XXS server unchanged, plus `-md <head> --spec-type draft-mtp --spec-draft-n-max N`, other draft settings at their defaults.
  - The head is Unsloth's self-contained Q8_0 file. The `shared` heads need code we did not find in v1.0.0.
  - N is chosen by P0's rule.
- **TensorFold + NVFP4:** exactly Part 6's run, apart from the time limit per task (120 minutes instead of 40, as for the GGUF runs; no task in Part 6 reached 40).
- **Both:** the 400 tasks, D2, greedy, output limit 65,536, no thinking budget.

**P0 (compatibility check and the choice of N, before the final lock).**
- **Probe tasks:** the first 6 coding and first 6 reasoning tasks of Part 4's private dev set (not the 400 tasks, not the sealed tasks), D2, capped at 8,192 tokens. They measure speed only and are not graded.
- **Runs:** UD-IQ3_XXS without MTP, then with N = 2, 3 and 4. Each run did the 12 probe tasks plus Part 4's 5 examples (65,536 tokens, 40 minutes), with a fresh server each time. Then TensorFold had to start and answer a warm-up correctly.
- **Rule for N:** decoding speed = sum of `predicted_n` / sum of `predicted_ms` over the 12 probe tasks. Take the fastest N; if a smaller N is within 97% of it, take the smallest such N.
- **Criteria** (all needed, otherwise stop and report):
  1. the draft model loads in every MTP start, and every request runs without error and with drafts;
  2. at the chosen N, at least 4 of the 5 examples are correct;
  3. TensorFold starts and answers the warm-up correctly;
  4. the memory guard does not fire, and the lowest available memory stays at least 3 GiB.
- Reported only: probe replies identical to MTP off.

**P0 result (appended by the lock step from the check's own output):**

| | Start (s) | Probe decoding (tokens/s) | Drafts accepted | Probe tokens | Identical to MTP off (of 12) | Examples | Example tokens | Lowest available memory (GiB) |
|---|---|---|---|---|---|---|---|---|
| MTP off | 10.0 | 30.57 | | 47,376 | | 5/5 | 3,118 | 53.0 |
| N = 2 | 24.5 | 51.35 | 0.911 | 53,367 | 1 | 5/5 | 3,154 | 49.3 |
| N = 3 | 30.1 | 55.45 | 0.863 | 49,587 | 1 | 5/5 | 14,012 | 49.5 |
| N = 4 | 20.1 | 57.90 | 0.792 | 44,382 | 0 | 4/5 | 67,480 | 49.5 |

TensorFold started in 82.0 s, warm-up correct, lowest available memory 37.6 GiB. All four criteria were met and no request failed. **N = 4; go; final lock.**

**Design:**
- **P2:** 8 groups of 50 tasks, with both configurations running every group in alternating order and a server restart before every block:

  | Group | First | Second |
  |---|---|---|
  | 0 | UD-IQ3_XXS + MTP | TensorFold |
  | 1 | TensorFold | UD-IQ3_XXS + MTP |
  | 2 | TensorFold | UD-IQ3_XXS + MTP |
  | 3 | UD-IQ3_XXS + MTP | TensorFold |
  | 4 | UD-IQ3_XXS + MTP | TensorFold |
  | 5 | TensorFold | UD-IQ3_XXS + MTP |
  | 6 | TensorFold | UD-IQ3_XXS + MTP |
  | 7 | UD-IQ3_XXS + MTP | TensorFold |

  Before each start the runner waited for the GPU to be released (at most 300 s). Two failed blocks in a row would stop the run.
- **P3, after P2:** IQ3_S, UD-IQ3_XXS and UD-IQ4_XS each ran the 12 probe tasks twice, without MTP and with N = 4, with a fresh server each time.

**Tests and readings, written in advance:**
- **Q1:** paired by task.
  - Time ratio (UD-IQ3_XXS + MTP / TensorFold) with a paired bootstrap 95% interval (10,000 resamples, seed 2026).
  - Secondary: exact McNemar and the accuracy difference with its interval.

  | Result | Reading |
  |---|---|
  | upper end below 1 | GGUF + MTP faster (detectable) |
  | lower end above 1 | TensorFold faster (detectable) |
  | otherwise | no detectable time difference, with the interval |

  If the accuracy p is below 0.05, say which is more accurate.
- **Q2:** identical replies (of 400), exact McNemar p, time ratio and interval (reported only, two sessions), decoding speeds, and draft acceptance (accepted / drafted).
- **Q3:** per file, decoding without and with MTP, speed-up, acceptance and identical replies (of 12). The estimate of minutes per correct answer with MTP is PLAN_18's value divided by the speed-up; for UD-IQ3_XXS it is placed next to Q1's measured value.
- **Q4:** identical replies, McNemar p, time ratio and interval.

**Power, as stated in the plan:** in PLAN_18, time-ratio intervals for similar comparisons were about ±20-30% wide, so time differences under about 20% will likely not be detected. Accuracy is near the ceiling, so only clear drops can be seen.

**Not done:** other llama.cpp draft settings (such as `--spec-draft-p-min`), the `shared` heads, MTP on the tool-call replay, the full 400 tasks for IQ3_S and UD-IQ4_XS with MTP (estimated from P3 instead), and any other engine version or setting.

## PLAN_18c results (appended after the run)

**The run:**
- P2 ran 2026-11-18 00:03-17:03 and P3 until 19:27.
- All 16 blocks finished 50 of 50 tasks. No block failed, no task timed out, no start had to wait for the GPU, and the memory guard never fired.
- Lowest available memory: 48.4 GiB (llama.cpp) and 35.8 GiB (TensorFold).
- The locks passed before P0, after the final lock and after the run.
- The monitor was off for all 1,265 display records.

**Q1:**

| | UD-IQ3_XXS + MTP (N = 4) | TensorFold + NVFP4 + MTP |
|---|---|---|
| Correct | 381/400 | 384/400 |
| Total minutes | 430.4 | 570.8 |
| Minutes per correct answer | 1.130 | 1.487 |
| Output tokens | 1,431,344 | 2,052,826 |
| Hit the limit | 4 | 5 |

The time ratio is 0.760, with an interval of 0.599-0.952. Its upper end is below 1, so the reading is **GGUF + MTP faster (detectable)**. On accuracy, 7 tasks were solved only by GGUF and 10 only by TensorFold (p = 0.6291); the difference is -0.75 pp (-2.75, +1.25), not detectable. Identical replies: 0 of 400.

**Where the difference comes from.** This was looked at after the results and is reported only; the README has the details:
- Over all tasks, TensorFold produced more tokens per second (59.9 against 55.4), but wrote 1.43 times as many tokens.
- 10 tasks account for 457,837 of the 621,482-token difference, and 131.5 of the 140.5 minutes.
- llama.cpp was faster on only 135 of 400 tasks; the median per-task time ratio is 1.186.
- The same UD-IQ3_XXS file without MTP (PLAN_18; text different on 391 of 400 tasks) was also short on 8 of those 10 tasks. So this is partly a property of the file, but which tasks become long is noisy: only 5 of 11 and 12 long replies coincide.

**Q2:**

| | With MTP | Without (PLAN_18) |
|---|---|---|
| Correct | 381/400 | 380/400 |
| Minutes per correct answer | 1.130 | 2.155 |
| Output tokens | 1,431,344 | 1,432,914 |
| Hit the limit | 4 | 5 |
| Decoding (tokens/s) | 55.93 | 29.27 (381 tasks) |

- Identical replies: 9 of 400.
- Accuracy: 9 tasks solved only with MTP, 8 only without (p = 1.0); the difference is +0.25 pp (-1.75, +2.25).
- Draft acceptance: 1,099,008 of 1,328,004 drafts (0.828).
- Time ratio: 0.524 (0.425, 0.644), reported only because the runs are from two sessions.

**Q3:**

| File | Off (tokens/s) | N = 4 | Speed-up | Accepted | Identical (of 12) | PLAN_18 minutes per correct answer | Estimate with MTP |
|---|---|---|---|---|---|---|---|
| IQ3_S | 29.77 | 53.98 | 1.813 | 0.767 | 0 | 3.209 | 1.770 |
| UD-IQ3_XXS | 26.04 | 57.42 | 2.205 | 0.792 | 0 | 2.155 | 0.977 (measured 1.130) |
| UD-IQ4_XS | 25.08 | 50.77 | 2.024 | 0.730 | 0 | 2.774 | 1.371 |

Notes on Q3 (looked at after the results):
- The UD-IQ3_XXS estimate was 13.5% too optimistic.
- P3's MTP-off run of UD-IQ3_XXS was 14-15% slower than P0's on every one of the 12 tasks (per-task ratios 0.846-0.861), with identical text, the same cores and the same binary. The cause is unknown. The MTP run matched P0 within 2%.
- With the 400-task speed-up (1.911) the estimate is 1.128, the measured value. So the other two estimates are probably about 15% optimistic as well.

**Q4:** 400 of 400 replies identical to Part 6's run, 384/400 both, the same 2,052,826 tokens. Time 570.8 against 661.6 minutes, ratio 0.863 (0.857, 0.869). Part 6 ran with the display on and this run with it off; earlier measurements put the display's effect at 10-17%, which fits, but it was not tested separately.

**Repeatability:** P0 and P3 ran the same 12 probe tasks. Without MTP, 12 of 12 replies were identical, and with N = 4 also 12 of 12. llama.cpp with MTP differs from llama.cpp without it, but each setting repeats itself.

**Deviations:**
1. Q3's estimates rest on P3, where one MTP-off run was 15% slower for unknown reasons. The check with the 400-task speed-up was done after the results.
2. UD-IQ3_XXS's decoding speed without MTP uses 381 tasks (PLAN_18, deviation 3).
3. Nothing else deviated from the plan, and no setting was changed.
