# Qwen3.8 on one NVIDIA DGX Spark

[English](#english) | [中文](#中文)

## English

Ten studies of running Qwen3.8 locally on one **DGX Spark** (GB10, 128 GB) for **one user**, with thinking on. One question: **how do you get correct answers fastest?** The measure is **minutes per correct answer** (all time, including wrong answers, divided by the number of correct answers), not tokens/s.

| Part | Question | Folder |
|---|---|---|
| 1 | Which engine, weight format and drafter is fastest for one user? | [speed-comparison-display-on](qwen38-spark-speed-comparison-display-on/README.md) |
| 2 | How much does the monitor slow decoding? Where does vLLM MTP depth stop helping? | [speed-comparison-display-off](qwen38-spark-speed-comparison-display-off/README.md) |
| 3 | Final engine test on 48 unseen prompts | [speed-comparison-final](qwen38-spark-speed-comparison-final/README.md) |
| 4 | Deep vs low thinking, parallel vote, agents | [test-time-strategies](qwen38-spark-test-time-strategies/README.md) |
| 5 | Does it hold for a fine-tune and 400 new tasks? | [model-swap](qwen38-spark-model-swap/README.md) |
| 6 | Qwen3.8-Flash-Next (125B MoE) vs Qwen3.8-27B | [flash-next](qwen38-spark-flash-next/README.md) |
| 7 | Reasoning loops, and a thinking budget that removes them | [reasoning-loops](qwen38-spark-reasoning-loops/README.md) |
| 8 | Harder tasks: 250 new private tasks and 110 recent arXiv math problems | [harder-tasks](qwen38-spark-harder-tasks/README.md) |
| 9 | Can the 27B check its own code with its own tests? | [self-verification](qwen38-spark-self-verification/README.md) |
| 10 | TensorFold 1.2.5 (the new native engine) vs 0.6.0 | [engine-upgrade](qwen38-spark-engine-upgrade/README.md) |

**Key findings**

- **Engine:** TensorFold 0.6.0 + NVIDIA NVFP4 + DFlash2: 6.0 min per correct answer vs 11.2 for the best vLLM setup (Part 3). The monitor alone costs 9-18% of decode speed (Part 2).
- **Think less:** deep thinking 35/48 at 9.61 min per correct answer; low thinking 38/48 at 2.96; 4 low-thinking branches + vote 44/48 at 2.82 (Part 4). The same pattern holds on 400 new tasks and for a fine-tune (Part 5).
- **Bigger model:** Flash Next with the vote gets 391/400 vs 373/400 for 27B, but takes 2.4x as long per correct answer (Part 6).
- **Loops:** a 32,768-token thinking budget removes every output-limit hit in deep thinking: Flash Next 364 → 387/400, 27B 372 → 383/400, and both get faster (Part 7).
- **Harder tasks:** on 250 new private tasks, Flash Next with deep thinking and the budget answers 227/250 vs 208/250 for the 27B vote, but takes 2.7x as long per correct answer. On 110 recent arXiv math problems both nearly fail (10/110 and 2/110) (Part 8).
- **Self-verification:** the 27B writing its own tests and choosing the branch that passes gets 139/150 vs 134/150 on held-out coding tasks, not a confirmed gain, at about 2x the time. Its "verified" label is reliable: 1 wrong of 85 (Part 9).
- **Engine upgrade:** TensorFold 1.2.5 against 0.6.0 with the 27B vote: 377/400 vs 373/400 at 1.005 vs 0.958 min per correct answer, no detectable difference. 1.2.5 decodes 5-8% faster with 4 streams but processes long prompts up to 27% slower (Part 10).

**Recommended setup**

- TensorFold 0.6.0 + `nvidia/Qwen3.8-27B-NVFP4` + `z-lab/Qwen3.8-27B-DFlash2`, FP8 KV cache (1.2.5 gave no measured gain, Part 10).
- `reasoning_effort: low` by default; 4 branches + majority vote for checkable answers; deep thinking only with `"thinking_budget": 32768`.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for long sessions and agents.

**Reproduce:** each folder has a `REPRODUCE.md` with pinned versions (images by digest, checkpoints by commit). Tables can be recomputed without a GPU (`scripts/recheck.sh` in Parts 5-10). Task prompts and graders are private; per-task results are published by task id.

Results are specific to these model snapshots, software versions, tasks and one machine. Code: MIT.

## 中文

在一台 **DGX Spark**（GB10，128 GB）上本地运行 Qwen3.8 的十项研究，单用户，开启思考。只问一个问题：**怎样最快得到正确答案？** 指标是**每答对一题的分钟数**（全部时间，包括答错的题，除以答对题数），而不是 tok/s。

| 部分 | 问题 | 文件夹 |
|---|---|---|
| 1 | 哪种引擎、权重格式和草稿模型对单用户最快？ | [speed-comparison-display-on](qwen38-spark-speed-comparison-display-on/README.md) |
| 2 | 显示器让生成慢多少？vLLM 的 MTP 深度到几就不再变快？ | [speed-comparison-display-off](qwen38-spark-speed-comparison-display-off/README.md) |
| 3 | 在 48 道没用过的题上做引擎的最终检验 | [speed-comparison-final](qwen38-spark-speed-comparison-final/README.md) |
| 4 | 深度思考、低思考、并行投票、代理，怎么用最快 | [test-time-strategies](qwen38-spark-test-time-strategies/README.md) |
| 5 | 换成微调模型、换成 400 道新题，结论还成立吗？ | [model-swap](qwen38-spark-model-swap/README.md) |
| 6 | Qwen3.8-Flash-Next（125B MoE）对比 Qwen3.8-27B | [flash-next](qwen38-spark-flash-next/README.md) |
| 7 | 推理循环，以及消除循环的思考预算 | [reasoning-loops](qwen38-spark-reasoning-loops/README.md) |
| 8 | 更难的题：250 道新的私有题和 110 道近期 arXiv 数学题 | [harder-tasks](qwen38-spark-harder-tasks/README.md) |
| 9 | 27B 能不能用自己写的测试检查自己的代码？ | [self-verification](qwen38-spark-self-verification/README.md) |
| 10 | 新的原生引擎 TensorFold 1.2.5 对比 0.6.0 | [engine-upgrade](qwen38-spark-engine-upgrade/README.md) |

**主要结论**

- **引擎**：TensorFold 0.6.0 + NVIDIA NVFP4 + DFlash2 每答对一题 6.0 分钟，最好的 vLLM 配置是 11.2 分钟（Part 3）。光是开着显示器，生成就慢 9-18%（Part 2）。
- **少想更快**：深度思考 35/48，每答对一题 9.61 分钟；低思考 38/48，2.96 分钟；4 条低思考分支并行投票 44/48，2.82 分钟（Part 4）。在 400 道新题和微调模型上结论相同（Part 5）。
- **更大的模型**：Flash Next 投票答对 391/400，27B 是 373/400，但每答对一题的时间是 27B 的 2.4 倍（Part 6）。
- **循环**：32,768 token 的思考预算消除了深度思考里所有写满上限的情况：Flash Next 从 364/400 提高到 387/400，27B 从 372/400 提高到 383/400，而且两者都更快（Part 7）。
- **更难的题**：在 250 道新的私有题上，Flash Next 深度思考加预算答对 227/250，27B 投票是 208/250，但每答对一题的时间是 27B 的 2.7 倍。在 110 道近期 arXiv 数学题上两者几乎都答不出（10/110 和 2/110）（Part 8）。
- **自我验证**：27B 自己写测试、选通过测试的分支，在没用来调方法的编程题上答对 139/150，第一份代码是 134/150，没有确认更准，时间约 2 倍。它的"已验证"标记可靠：85 题里只错 1 题（Part 9）。
- **引擎升级**：27B 投票换用 TensorFold 1.2.5，答对 377/400，0.6.0 是 373/400；每答对一题 1.005 对 0.958 分钟，没有检出差别。1.2.5 四路并发时生成快 5-8%，但长提示词的处理最多慢 27%（Part 10）。

**推荐设置**

- TensorFold 0.6.0 + `nvidia/Qwen3.8-27B-NVFP4` + `z-lab/Qwen3.8-27B-DFlash2`，FP8 KV 缓存（1.2.5 没有测出提升，Part 10）。
- 默认 `reasoning_effort: low`；答案可以核对的题用 4 条分支多数投票；只在需要时用深度思考，并加 `"thinking_budget": 32768`。
- 长对话和代理设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。

**复现**：每个文件夹都有 `REPRODUCE.md`，版本全部固定（镜像按 digest，检查点按提交）。表格不用 GPU 就能重算（Part 5-10 的 `scripts/recheck.sh`）。题目和评分器不公开，逐题结果按题号公开。

结果只针对这些模型版本、软件版本、题目和这一台机器。代码采用 MIT 协议。
