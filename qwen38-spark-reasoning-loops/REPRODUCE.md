# Reproducing Part 7

| What | How |
|---|---|
| The tables in the README from the per-task results | `scripts/recheck.sh`, `scripts/recheck.sh pilot` (no GPU) |
| The figures | `figures/make_figures.py` (needs matplotlib) |
| Every arm (greedy, budget, sampling + budget, sampling, loop detector) on both models, checked on example tasks | steps 1-2 |
| The same on your own tasks | step 3 |
| The loop detector on replies you already have | step 4 (no GPU) |
| The budget in daily use | step 5 |

What cannot be reproduced from this folder: our scores themselves, since the 400 tasks and their grader are private.

## 0. Prerequisites

- **27B:** Part 4's setup ([Part 4 REPRODUCE.md](https://github.com/NautilusWu/qwen3.8_study/blob/main/qwen38-spark-test-time-strategies/REPRODUCE.md)): a DGX Spark, Docker with the NVIDIA Container Toolkit, the TensorFold 0.6.0 images, the sandbox image and the 27B checkpoints, all from Part 4's `scripts/setup.sh`.
- **Flash Next:** in addition, the checkpoint as in [Part 6 REPRODUCE.md](https://github.com/NautilusWu/qwen3.8_study/blob/main/qwen38-spark-flash-next/REPRODUCE.md) step 1 (105.9 GB, commit `7c4f1bc1`).
- Check out Part 4's and Part 6's folders next to this one (the repository has them side by side). The runner uses Part 4's `code/strategies` and, for Flash Next, Part 6's `code/flash_next.py`, both unchanged.
- On the host: Python 3 with `jinja2` (the detector arm renders the chat template locally to continue the thinking as a raw completion).
- Nothing else may run on the GPU during a run.

## 1. The runner

`code/loop_arms.py` runs deep thinking (the template's default effort) with output limit 65,536, context 81,920 and 40 minutes per task, as in Parts 4-6, with any of these arms:

| Arm | Request |
|---|---|
| `G` | greedy (temperature 0), as in Parts 4-6 |
| `B` | greedy + `thinking_budget` (default 32,768) |
| `SB` | temperature 1.0, top_p 0.95, top_k 20, seed 1234, + `thinking_budget` |
| `S` | temperature 1.0, top_p 0.95, top_k 20, seed 1234 |
| `D` | greedy; `code/loopdetect.py` watches the streamed thinking; when it fires, the request is stopped and the thinking, cut where the loop began, is continued as a raw completion (`/v1/completions`) after "I keep repeating myself, so I will stop thinking here and give my final answer now." and `</think>` (greedy, at most 8,192 tokens, within the same 40 minutes) |

It starts and stops the server itself (port 8114 for the 27B, 8115 for Flash Next), runs the arms one after another for each task, grades with Part 4's grader and sandbox, and prints a summary per arm. Results go to `<Part 4>/code/strategies/results/<run_id>/graded.jsonl` (Part 4's runner writes there).

## 2. Check on Part 4's example tasks

```bash
cd code
export HF_DIR=... TASKS_DIR=../../qwen38-spark-test-time-strategies/examples/tasks GRADER_DIR=../../qwen38-spark-test-time-strategies/examples/grader
S4=../../qwen38-spark-test-time-strategies/code/strategies
python3 loop_arms.py $S4 check-27b --model 27b --arms G,B,SB,S,D --split dev --check
python3 loop_arms.py $S4 check-fn --model flashnext --part6-code ../../qwen38-spark-flash-next/code --arms G,B,SB,S,D --split dev --check
```

`--check` first checks that the locally rendered prompt is the server's (two tasks: the same prompt token count, and the same generated token ids from the chat endpoint and the raw-completion endpoint). Then it runs the 5 example tasks with a 256-token budget, so that the budget is reached, and a detector that fires after 1,500 characters of thinking, so that the continuation path runs on tasks that do not loop. It ends with `CHECK PASSED` when the rendering matched, no request failed and every arm answered at least 4 of 5 tasks. The number of correct answers is printed, not checked.

Ours, each about 10 minutes plus the server start: `CHECK PASSED` for both models; the rendering matched (92 and 138 prompt tokens, identical token ids); the forced detector fired on the 2 tasks whose thinking was longer than 1,500 characters and both then answered. Correct answers (G / B / SB / S / D, of 5): 27B 5 / 3 / 4 / 5 / 5, Flash Next 5 / 5 / 4 / 5 / 5; with only 256 thinking tokens some budget answers are wrong, as expected. Greedy (G) wrote 13,032 tokens on the 27B and 11,823 on Flash Next in total; greedy is deterministic, so the same software should give the same tokens.

Flash Next: set `FN_TARGET` if your checkpoint is not at `/hf/hub/models--local-inference-lab--Qwen3.8-Flash-Next-NVFP4/snapshots/7c4f1bc1a2d6847e0cbc01ac6b823f00251de8dd` inside `HF_DIR`, as in Part 6.

## 3. Runs on your own tasks

With `TASKS_DIR` and `GRADER_DIR` set for your tasks (format as in Part 4's REPRODUCE.md, split `confirm` by default):

```bash
python3 loop_arms.py $S4 my-27b --model 27b --arms G,B
python3 loop_arms.py $S4 my-fn --model flashnext --part6-code ../../qwen38-spark-flash-next/code --arms G,B,SB
```

Other options: `--budget N` (thinking budget for `B` and `SB`), `--split`, `--tasks id,id` (a subset). Compare two arms per task with `code/stats.py` after writing one CSV per arm (`task_id,passed,wall_s`), as `scripts/recheck.sh pilot` does.

To save GPU time the way this study did for `B`: with greedy decoding, `B` is identical to `G` until the budget is reached, so only the tasks where your greedy reply ran past the budget need a `B` run. Check that on a few tasks first (identical token ids), as in [PREREGISTRATION.md](PREREGISTRATION.md) addendum 02.

## 4. The loop detector on replies you already have

```bash
python3 code/replay_detector.py <run>/graded.jsonl [...]
```

Replays each stored reply's thinking through `LoopDetector`, 256 characters at a time as a stream would arrive, and prints, per file, how many replies hit the output limit, on how many the detector fires, and on which replies that did answer (false alarms). It reads Part 4's `graded.jsonl` format (fields `task_id`, `text`, `completion_tokens` or `tokens`, `finish`, `passed`; vote runs with `branches`). In your own client: create a `LoopDetector()` per reply and call `feed(thinking_so_far)` as text arrives; it returns `None`, or a dict with `rule` (`exact` or `phrase`) and `cut` (the character position where the loop began).

## 5. The budget in daily use

Per request, add `"thinking_budget": 32768` to the chat completion body. For every request, start the server with `--thinking-budget 32768` (TensorFold 0.6.0; default 0, no limit), for example in Part 4's `common.start_server` arguments or your own `tensorfold serve` command. Keep the output limit above the budget.

## 6. Recompute the tables

From this folder's root:

```bash
scripts/recheck.sh            # 400 tasks: Flash Next greedy + budget, sampling + budget, greedy; 27B greedy + budget, greedy
scripts/recheck.sh pilot      # stage 1, 79 tasks: G, S, B, D
```

`code/stats.py` (as in Parts 5-6) prints totals and, per pair, the tasks only one side solved, the exact McNemar p, a Holm-adjusted p over the pairs it prints, and paired bootstrap 95% intervals (10,000 resamples, seed 2026). A time ratio is the first configuration's minutes per correct answer over the second's, with each task's time capped at 40 minutes.

The figures:

```bash
cd figures && python3 make_figures.py
```
