# Qwen3.8 on one NVIDIA DGX Spark

[English](#english) | [中文](#中文)

## English

Seven studies of running Qwen3.8 locally on one **DGX Spark** (GB10, 128 GB) for **one user**, with thinking on. One question: **how do you get correct answers fastest?** The measure is **minutes per correct answer** (all time, including wrong answers, divided by the number of correct answers), not tokens/s.

| Part | Question | Folder |
|---|---|---|
| 1 | Which engine, weight format and drafter is fastest for one user? | [speed-comparison-display-on](qwen38-spark-speed-comparison-display-on/README.md) |
| 2 | How much does the monitor slow decoding? Where does vLLM MTP depth stop helping? | [speed-comparison-display-off](qwen38-spark-speed-comparison-display-off/README.md) |
| 3 | Final engine test on 48 unseen prompts | [speed-comparison-final](qwen38-spark-speed-comparison-final/README.md) |
| 4 | Deep vs low thinking, parallel vote, agents | [test-time-strategies](qwen38-spark-test-time-strategies/README.md) |
| 5 | Does it hold for a fine-tune and 400 new tasks? | [model-swap](qwen38-spark-model-swap/README.md) |
| 6 | Qwen3.8-Flash-Next (125B MoE) vs Qwen3.8-27B | [flash-next](qwen38-spark-flash-next/README.md) |
| 7 | Reasoning loops, and a thinking budget that removes them | [reasoning-loops](qwen38-spark-reasoning-loops/README.md) |

**Key findings**

- **Engine:** TensorFold 0.6.0 + NVIDIA NVFP4 + DFlash2: 6.0 min per correct answer vs 11.2 for the best vLLM setup (Part 3). The monitor alone costs 9-18% of decode speed (Part 2).
- **Think less:** deep thinking 35/48 at 9.61 min per correct answer; low thinking 38/48 at 2.96; 4 low-thinking branches + vote 44/48 at 2.82 (Part 4). The same pattern holds on 400 new tasks and for a fine-tune (Part 5).
- **Bigger model:** Flash Next with the vote gets 391/400 vs 373/400 for 27B, but takes 2.4x as long per correct answer (Part 6).
- **Loops:** a 32,768-token thinking budget removes every output-limit hit in deep thinking: Flash Next 364 → 387/400, 27B 372 → 383/400, and both get faster (Part 7).

**Recommended setup**

- TensorFold 0.6.0 + `nvidia/Qwen3.8-27B-NVFP4` + `z-lab/Qwen3.8-27B-DFlash2`, FP8 KV cache.
- `reasoning_effort: low` by default; 4 branches + majority vote for checkable answers; deep thinking only with `"thinking_budget": 32768`.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for long sessions and agents.

**Reproduce:** each folder has a `REPRODUCE.md` with pinned versions (images by digest, checkpoints by commit). Tables can be recomputed without a GPU (`scripts/recheck.sh` in Parts 5-7). Task prompts and graders are private; per-task results are published by task id.

Results are specific to these model snapshots, software versions, tasks and one machine. Code: MIT.

## 中文

在一台 **DGX Spark**（GB10，128 GB）上本地运行 Qwen3.8 的七项研究，单用户，开启思考。只问一个问题：**怎样最快得到正确答案？** 指标是**每答对一题的分钟数**（全部时间，包括答错的题，除以答对题数），而不是 tok/s。

| 部分 | 问题 | 文件夹 |
|---|---|---|
| 1 | 哪种引擎、权重格式和草稿模型对单用户最快？ | [speed-comparison-display-on](qwen38-spark-speed-comparison-display-on/README.md) |
| 2 | 显示器让生成慢多少？vLLM 的 MTP 深度到几就不再变快？ | [speed-comparison-display-off](qwen38-spark-speed-comparison-display-off/README.md) |
| 3 | 在 48 道没用过的题上做引擎的最终检验 | [speed-comparison-final](qwen38-spark-speed-comparison-final/README.md) |
| 4 | 深度思考、低思考、并行投票、代理，怎么用最快 | [test-time-strategies](qwen38-spark-test-time-strategies/README.md) |
| 5 | 换成微调模型、换成 400 道新题，结论还成立吗？ | [model-swap](qwen38-spark-model-swap/README.md) |
| 6 | Qwen3.8-Flash-Next（125B MoE）对比 Qwen3.8-27B | [flash-next](qwen38-spark-flash-next/README.md) |
| 7 | 推理循环，以及消除循环的思考预算 | [reasoning-loops](qwen38-spark-reasoning-loops/README.md) |

**主要结论**

- **引擎**：TensorFold 0.6.0 + NVIDIA NVFP4 + DFlash2 每答对一题 6.0 分钟，最好的 vLLM 配置是 11.2 分钟（Part 3）。光是开着显示器，生成就慢 9-18%（Part 2）。
- **少想更快**：深度思考 35/48，每答对一题 9.61 分钟；低思考 38/48，2.96 分钟；4 条低思考分支并行投票 44/48，2.82 分钟（Part 4）。在 400 道新题和微调模型上结论相同（Part 5）。
- **更大的模型**：Flash Next 投票答对 391/400，27B 是 373/400，但每答对一题的时间是 27B 的 2.4 倍（Part 6）。
- **循环**：32,768 token 的思考预算消除了深度思考里所有写满上限的情况：Flash Next 从 364/400 提高到 387/400，27B 从 372/400 提高到 383/400，而且两者都更快（Part 7）。

**推荐设置**

- TensorFold 0.6.0 + `nvidia/Qwen3.8-27B-NVFP4` + `z-lab/Qwen3.8-27B-DFlash2`，FP8 KV 缓存。
- 默认 `reasoning_effort: low`；答案可以核对的题用 4 条分支多数投票；只在需要时用深度思考，并加 `"thinking_budget": 32768`。
- 长对话和代理设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。

**复现**：每个文件夹都有 `REPRODUCE.md`，版本全部固定（镜像按 digest，检查点按提交）。表格不用 GPU 就能重算（Part 5-7 的 `scripts/recheck.sh`）。题目和评分器不公开，逐题结果按题号公开。

结果只针对这些模型版本、软件版本、题目和这一台机器。代码采用 MIT 协议。
