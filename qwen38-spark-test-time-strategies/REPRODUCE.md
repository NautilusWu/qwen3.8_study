# Reproduce

## What can and cannot be reproduced

| | How |
|---|---|
| The setup: engine, checkpoints, settings, every method, the agent harness | `scripts/setup.sh`, then `scripts/smoke_test.sh` (example tasks) |
| The statistics of the published results (C1–C5 from the per-task results) | `scripts/run.sh recheck`, no GPU needed; gives exactly [data/confirmation_tests.json](data/confirmation_tests.json) |
| The study's design on **your own** tasks | `scripts/run.sh strategies …` and `scripts/run.sh agents …` |
| Rerunning the published scores themselves | **Not possible from this repository.** The task sets, their graders and the agent task repositories are not published (they stay unseen for future tests, as in Parts 1–3). |

The examples in `examples/` are new, small tasks written only to check the pipeline; they are not from the study's task sets and say nothing about the published results.

## 0. Machine and what you install yourself

The scripts install everything inside Docker images. The host itself is not set up by them:

| Needed on the host | Strategies and concurrency (Oct 17–19, 2026) | Coding agents and memory probe (Oct 10–11, 2026) |
|---|---|---|
| NVIDIA DGX Spark (GB10, 128 GB unified memory) | one machine, single user, monitor on | same |
| DGX OS with its NVIDIA driver | kernel 7.0.0-1019-nvidia, driver 580.178.04 | kernel 6.17.0-1031-nvidia, driver 580.173.02 (Part 3's) |
| Docker with the NVIDIA Container Toolkit | Docker 29.6.2, Toolkit 1.20.1 | Docker 29.2.1, Toolkit 1.20.0 |
| `git`, `curl`, `bc`, `ss`, `python3` ≥ 3.10 (standard library only) | Python 3.12.3 | same |
| Hugging Face CLI: `pip install -U huggingface_hub` | for the checkpoint download | |
| Disk | about 25 GB checkpoints, 30 GB images | |

`scripts/setup.sh` checks all of these first and stops with a message if something is missing. Pulling the NVIDIA PyTorch base image may need `docker login nvcr.io` with an NGC API key.

## 1. Setup (one time)

```bash
scripts/setup.sh
```

- Checkpoints, at pinned revisions, into `HF_DIR` (default `~/.cache/huggingface`; `SKIP_MODELS=1` if you already have them): `nvidia/Qwen3.8-27B-NVFP4` @ `482ca0f3…` and the drafter `z-lab/Qwen3.8-27B-DFlash2` @ `50307d4c…`.
- `qwen38-tensorfold-repro:0.6.0-c464617`: TensorFold v0.6.0 and the recipe's five patches at fixed commits, hash-checked (`tensorfold/fetch_sources.sh`), on a base image fixed by digest. Every Python package is installed at the version in `tensorfold/pip-freeze.txt` (the image used for the results), and the build fails if any version differs. The files in `tensorfold/` are Part 3's plus this version lock.
- `qwen38-tensorfold-repro:0.6.0-trace`: adds timing fields to the stream, measurement only (Part 3).
- `agent-harness:0.1`: Node 22.23.3 (Debian 12) with Qwen Code 0.25.0, OpenCode 1.18.35, Codex CLI 0.162.1 and Claude Code 2.1.296; also the sandbox for running and grading code (Python 3.11.2, g++ 12.2, pytest 7.2.1).
- LiteLLM 1.104.2 by image digest (the gateway for Claude Code).

## 2. Acceptance check on the example tasks

```bash
scripts/smoke_test.sh
```

About 15 minutes (11 on ours). It needs the GPU free, so stop any other model server first. It runs three methods (low thinking D2, the 4-branch vote CL4 and deep direction + low execution H) on the 5 example tasks in `examples/tasks`, with the example grader. Then it runs the 4 coding agents on the example repository task in `examples/agent_tasks`. It prints a table and `SMOKE TEST PASSED` when every step ran and every answer was graded. The number of correct answers is printed, not checked. Ours: D2 4/5, CL4 5/5, H 5/5, every agent 1/1. `SMOKE_AGENTS=opencode` runs fewer agents; `SMOKE_AGENTS=` skips them.

## 3. Runs on your own tasks

```bash
TASKS_DIR=/path/to/tasks GRADER_DIR=/path/to/grader scripts/run.sh strategies my confirm
```

This runs deep thinking (A65), low thinking (D2), deep direction + low execution (H) and the 4-branch vote (CL4) with the confirmation-phase settings (output limit 65,536, context 81,920, 40 minutes per task). Then it computes the five tests C1–C5 into `code/strategies/results/my-tests.json`. Run one method at a time with `code/strategies/run_method.py <method> <run_id>`. Methods: `A`, `A65`, `D1`, `D2`, `D3`, `C --k 4|8`, `CL --k 4`, `B` (agent that runs code), `P1`, `P2`, `H`. The exploration settings are the defaults: 25 minutes, context 32,768, output limit 24,576.

- **Tasks:** `TASKS_DIR/<split>.jsonl`, with split `dev`, `hard`, `confirm` or `sealed` (`--split`). Each line is one JSON object with `id`, `category` (`coding` or `reasoning`), `messages` (the user turn) and `grader`, which holds whatever your grader needs. The runner never shows `grader` to the model. See `examples/tasks/dev.jsonl`.
- **Grader:** `GRADER_DIR/grading.py` (or a package `grading/`) providing `grade(grader: dict, text: str, thinking: bool) -> (score, passed, detail)`. With `thinking=True`, `text` is the thinking, then `</think>`, then the answer. The grader runs in the sandbox image with no network. `examples/grader/grading.py` is a working example.
- **Answer formats:** the vote (C, CL) and P2 group answers by `grader["type"]`:
  - `numeric`: the number after the last `Final answer:`;
  - `exact_output`: the last code block;
  - `python_check`: the last `Final answer:` line;
  - `python_tests` and `cpp_tests`: coding tasks, no vote.

  Use these type names, or edit `answer_key` in `code/strategies/common.py`.
- **Other environment variables:**
  - `HF_DIR`;
  - `TF_CACHE_DIR`: the kernel cache, default `~/.cache/qwen38-tensorfold-cache`. The first start compiles kernels and takes a few minutes longer;
  - `SANDBOX_IMAGE`: any image with `python3` and `g++`.
- **Server lifecycle:** the runner starts and stops the model server itself, on port 8114.

## 4. Coding agents

```bash
AGENT_TASKS_DIR=/path/to/agent_tasks scripts/run.sh agents my-agents
```

- **Task format:** `AGENT_TASKS_DIR` holds one folder per task, each with:
  - `repo/`: the starting code, which becomes a fresh git repository for every run;
  - `prompt.txt`: what the agent is asked;
  - `hidden/`: tests the agent never sees. Python uses pytest with the repo on `PYTHONPATH`; JavaScript uses `hidden/hidden.test.js` with `node --test`.

  See `examples/agent_tasks/ex_median`.
- **What `code/coding_agents/run_all.py` does:**
  1. starts the model server with `serve_model.sh`: the settings of the published runs, on 127.0.0.1:8091, in container `TF_CONTAINER`, default `qwen38-tensorfold`;
  2. starts the normalizing proxy (`normalize_proxy.py` on 8099), so every agent gets the same settings: thinking at the default effort, greedy decoding, a 32,768-token output limit;
  3. for Claude Code, starts LiteLLM with `litellm_ctl.sh` on 4000;
  4. for each task and each agent in a shuffled order:
     - restarts the model server, so the cache starts empty;
     - runs the agent in a fresh container with a 30-minute limit (`--limit`);
     - grades with the hidden tests in a container with no network;
  5. at the end, stops everything it started.
- **Network:** the agents run with the host network, which they need to reach the local proxy. OpenCode's `webfetch` is denied in its config, and Codex's hosted web search is dropped by the proxy. The other agents' own fetch tools were not blocked.
- **Claude Code:** goes through LiteLLM with a local-only token. Anthropic does not support using Claude Code with non-Claude models.
- **Results:** `code/coding_agents/results/<run_id>/results.jsonl`. `code/coding_agents/analyze.py` builds the summary table.

## 5. Memory

All model servers here start with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and a 40 GiB host memory reserve.

- `code/memory/memwatch.py` logs memory every 5 s and stops the server if available memory drops under 12 GiB. The agent runner uses it.
- `code/memory/leak_probe.py` reproduces the allocator growth with and without the setting.
