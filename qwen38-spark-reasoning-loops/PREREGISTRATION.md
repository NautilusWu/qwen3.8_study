# Pre-registration

Five plans, each written and sha256-locked before its runs (English summaries of the Chinese originals; the hashes are of the original files as locked). Every run script checked its lock at start; results were appended only after the runs, below the locked text.

| Plan | Locked | sha256 (locked text) | Runs |
|---|---|---|---|
| PLAN_08 stage 1: three fixes on 79 tasks (exploration, reported only) | 2026-10-28 | `8165594a512bedfcd720f7df812a8c52c18e9c7ec916b1d135bf95d447cf265d` | Flash Next deep thinking: S, B, D |
| Stage 1 addendum 00: fix of the compatibility check | 2026-10-29, before any task ran | `54a00fc69643b038cf53a24208bf65861afe8f37aeebd6bcafed4a736e194f6e` | (same runs) |
| PLAN_08 stage 2: sampling + budget on 400 tasks (locked tests) | 2026-10-30, after stage 1 | `760ac9a4361fdfac5c99dd573b256875c54f591d8c4c8546045e0cf2745aaa34` | Flash Next deep thinking: SB |
| Stage 2 addendum 01: greedy + budget on 400 tasks, Flash Next (reported only) | 2026-10-31, after stage 2 | `e1ebb4ab14c26bb1b7dc773dfa9ca4504df6de91172adfb7a437bcf42305335d` | Flash Next: 9 tasks |
| Addendum 02: greedy + budget on 400 tasks, 27B (reported only) | 2026-10-31, after addendum 01 (dated 2026-11-01 in its header, the day the run ended) | `96c84a612d65d456d412beab2c28f8015f83f2c51973458169aff18a0138b211` | 27B: 10 + 35 tasks |

400-task prompt file sha256: `591b703379981311905ec9ed7a2beef5c34f880beafd92f34e759fe18e09a85b` (not published; the same file as Parts 5 and 6).

## Before the plans: offline analysis (not locked)

The loop detector (`code/loopdetect.py`) was written and tuned on the stored replies of Parts 5-6 on the 400 tasks (3,263 replies: deep and low thinking of both models, and the vote branches that were not cancelled) before stage 1. Replayed as a stream, it fired on 20 of Flash Next's 29 deep-thinking replies that hit the limit, and on exactly 1 reply that answered (Flash Next low thinking, T355, a correct answer: a false alarm). The detector was then locked unchanged for stage 1. T355 happened to be among stage 1's 50 randomly drawn controls; in stage 1 (deep thinking) the detector did not fire on it.

## PLAN_08 stage 1: three fixes, 79 tasks

- Question: on Flash Next's deep thinking (greedy, Part 6's run), which way of stopping the loops rescues the looping tasks without breaking others, and at what time per correct answer? Exploration: results reported only, no "more accurate / faster" conclusions.
- Arms (everything else as Part 6's deep-thinking run: checkpoint @7c4f1bc1, image, MTP threshold 0.6 up to 6, bf16 KV, output limit 65,536, context 81,920, 40 minutes per task, grader):
  - G: greedy, Part 6's run (not rerun; greedy is deterministic, a rerun of 91 tasks had been identical);
  - S: the recommended sampling, temperature 1.0, top_p 0.95, top_k 20, seed 1234;
  - B: greedy + `thinking_budget` 32,768;
  - D: greedy + the locked detector on the stream; when it fires, stop, cut the thinking where the loop began, append "I keep repeating myself, so I will stop thinking here and give my final answer now." and `</think>`, continue as a raw completion (greedy, at most 8,192 tokens), all within the 40 minutes.
  - Derived from existing data, reported only: D-restart (G's time up to the detection + S's result) and D-low (G's time up to the detection + Part 6's low-thinking result).
- Tasks: the 29 tasks where Part 6's deep thinking hit the limit, plus 50 controls drawn from the other 371 with `random.Random(2026).sample`. List locked.
- Compatibility check first, on Part 4's 5 public example tasks: the locally rendered prompt equals the server's; S, B (budget 256 so it fires) and D (detector forced to fire after 1,500 characters) each answer at least 4 of 5; no request errors.
- Reported: per arm, correct, answered, limit hits, time, on all 79, the 29 and the 50; pairs with G (exact McNemar, 95% bootstrap intervals, 10,000 resamples, seed 2026); detector fires; budget hits and whether the model then answers.
- Rule for stage 2, written in advance: the arm with the most correct answers on the 79 goes on, provided it loses at most 1 control task against G; ties go to the shorter total time. Two arms that fix different kinds of loops may be combined, stated separately in the stage 2 plan; the user confirms.

### Stage 1 addendum 00: the compatibility check (before any task ran)

The first compatibility check stopped the queue: on one example the check compared the first 400 characters of the chat reply's thinking with the raw completion, but that thinking was only about 380 characters long, so the raw text (which goes on past `</think>`) necessarily differed. A diagnosis showed both paths generating the same 298 token ids, twice. The check was changed to compare generated token ids (stricter) and re-locked together with the plan and all code; the runner only gained a helper for the check. Nothing else changed.

**Stage 1 results** (appended after the run; the amended lock verified; 237 replies, about 20.4 hours, memory guard never fired, lowest available memory 32.8 GiB, display on):

| Arm | Correct of 79 | 29 looping | 50 controls | Answered | Hit the limit | Minutes per correct answer |
|---|---|---|---|---|---|---|
| G | 49 | 0 | 49 | 50 | 29 | 13.13 |
| S | **74** | **25** | 49 | 75 | 4 | 5.77 |
| B | 72 | 23 | 49 | **79** | **0** | **5.43** |
| D | 65 | 16 | 49 | 70 | 9 | 6.25 |
| D-restart (derived) | 67 | 18 | 49 | 68 | 9 | 8.93 |
| D-low (derived) | 66 | 17 | 49 | 69 | 9 | 7.19 |

Pairs with G: S 26 / 1 (p < 0.0001), +31.6 pp (+21.5, +43.0), time ratio 0.44 (0.32, 0.58); B 23 / 0, +29.1 (+19.0, +39.2), 0.41 (0.34, 0.50); D 16 / 0, +20.2 (+11.4, +29.1), 0.48 (0.36, 0.61). Detector: 20 fires, all on looping tasks, none on controls; all 20 then answered, 16 correctly. Budget: reached 32 times (29 looping, 3 controls), all 32 then answered, 25 correctly; on the controls B equals G.

Under the rule, S went on (74, no control lost on balance). Since B rescued tasks S did not, the user chose the combination S + B for stage 2.

## PLAN_08 stage 2: sampling + budget, 400 tasks (locked tests)

- Variable: decoding. G: Part 6's greedy deep-thinking run (existing). SB: temperature 1.0, top_p 0.95, top_k 20, seed 1234, plus `thinking_budget` 32,768. Everything else unchanged; request code: stage 1's runner, unchanged. Stated in advance: SB itself was not tested in stage 1; it was chosen after seeing stage 1.
- Tasks: all 400 (79 of them used in stage 1).
- Compatibility check: 5 example tasks with budget 256; at least 4 of 5 answered, no errors.
- Conditions: no other GPU job; display on; daily service stopped; memory guard 3 GiB.
- Locked tests, SB vs G, paired over the 400 tasks:
  1. Accuracy: exact two-sided McNemar, p < 0.05 → "a detectable accuracy difference".
  2. Time: ratio of minutes per correct answer SB / G, paired bootstrap 95% CI (10,000, seed 2026); CI excluding 1 → "a detectable time difference".
  3. The accuracy difference's 95% CI is reported as well. The two tests concern different quantities; no multiplicity adjustment.
- Reported only: the same comparison on the 321 tasks not used in stage 1; limit hits, no-answer, budget hits; SB beside Part 6's other configurations and Part 5's 27B.

**Stage 2 results** (appended after the run; lock verified; 400 tasks, about 24.4 hours, memory guard never fired, lowest available memory 30.7 GiB):

| | SB | G |
|---|---|---|
| Correct | **393** | 364 |
| Total time | 1459.8 min | 1457.0 min |
| Minutes per correct answer | 3.71 | 4.00 |

- Accuracy: SB only 34, G only 5, p < 0.0001, +7.25 pp (+4.25, +10.25) → **a detectable accuracy improvement**.
- Time: ratio 0.93 (0.81, 1.06) → **no detectable time difference**.
- Reported only: of the 29 tasks G looped on, SB solved 27; no SB reply hit the limit, all 400 answered; the budget was reached 34 times, always followed by an answer, 28 correct. 4 of the 5 tasks G solved and SB did not were budget-forced answers. On the 321 tasks outside stage 1: 317 vs 315 (p = 0.75), time ratio 1.36 (1.22, 1.53): sampling wrote 31% more tokens there. Coding 197 vs 176, reasoning 196 vs 188.

**Deviation: the display was not on throughout.** The user switched the display off for a while during the run; from per-task speeds against G, roughly tasks 100-215. The display changes only speed. At the user's request a worst-case check was added after the run: every SB time multiplied by 1.18 (the largest display effect measured), capped at 40 minutes: time ratio 1.10 (0.96, 1.26), still no detectable difference. On the 321 tasks: 1.61 (1.44, 1.81).

## Stage 2 addendum 01: greedy + budget, Flash Next, 400 tasks (reported only)

- Why: SB lengthened thinking on tasks that do not loop. Budget without sampling (B) is identical to greedy on those tasks and might be faster per correct answer. Decided after seeing stage 2: reported only, changes nothing in stage 2's conclusions.
- Composition, using greedy determinism (below the budget B = G, verified on 47 / 47 stage 1 controls): the 79 stage 1 tasks take stage 1's B; the 312 other tasks where G stayed within 32,768 tokens take G's result and time; the remaining 9 tasks are run new with stage 1's code, unchanged (list locked).
- Reported: B on 400 tasks; pairs with G and SB.

**Results** (lock verified; 9 tasks run, memory guard never fired, lowest available memory 32.2 GiB, display on): B 387 correct, 1157.0 minutes, 2.99 per correct answer, no limit hits. B vs G: 23 / 0, p < 0.0001, +5.75 pp (+3.50, +8.25), time ratio 0.75 (0.70, 0.80). B vs SB: 5 / 11, p = 0.21, -1.50 pp (-3.50, +0.50), time ratio 0.81 (0.73, 0.88). The 9 new tasks: same results as G (7 correct), 110.9 vs 159 minutes.

## Addendum 02: greedy + budget, 27B, 400 tasks (reported only)

- Why: addendum 01 showed the budget only helping on Flash Next; the daily service runs the 27B. Same method, reported only.
- Baseline: Part 5's 27B deep-thinking run (greedy; generated token ids stored). New runs with exactly its server and request settings (Part 4/5 runner, nvidia/Qwen3.8-27B-NVFP4 + DFlash2, FP8 KV, FP8 prefill, context 81,920, output limit 65,536, 40 minutes, same image), plus only `thinking_budget: 32768`.
- Composition: the 35 tasks where G went past 32,768 tokens are run new; the 365 others take G's result and time.
- Precondition (the 27B drafts with DFlash2, unlike Flash Next, so checked separately): 10 of the 365, drawn with `random.Random(2026)`, are run again and must give **exactly** G's token ids (10 / 10); otherwise no composition, and only the 35 tasks are reported. List locked.
- Compatibility check on 5 example tasks with budget 256. Display on; no other GPU job. Stated in advance: Part 5's run did not record the display state, so the time comparison on the 35 tasks carries that uncertainty.

**Results** (lock verified; 45 tasks, about 6.6 hours, memory guard never fired, lowest available memory 80.0 GiB):

- Precondition: **10 / 10 identical token ids**, same results. On those 10 tasks the new run took 1.09x Part 5's time.
- The 35 tasks: B 28 correct, G 17; B only 11, G only 0. G hit the limit 17 times, B never; B answered all 35. 370.9 vs 604.7 minutes.
- 400 tasks composed: B 383, 3.25 minutes per correct answer; G 372, 3.97. 11 / 0, p = 0.001, +2.75 pp (+1.25, +4.50), time ratio 0.82 (0.76, 0.87). Added after the run: with the 35 new times divided by 1.09 (as if run in Part 5's conditions), 0.80 (0.74, 0.86).
- B vs Part 5's low thinking: +5.0 pp (30 / 10, p = 0.002), time 3.8x (3.1, 4.6); vs the vote: +2.5 pp (20 / 10, p = 0.10), time 3.5x.
