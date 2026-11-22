# Reproduce

## What can and cannot be reproduced

| | How |
|---|---|
| Every table and test in the README, from the published data | `scripts/recheck.sh` (no GPU) |
| The figures | `figures/make_figures.py` (needs matplotlib) |
| PLAN_22 M1 and M3: tuned read bandwidth and cuBLAS, no model loaded | step 2 |
| PLAN_16 B0 and PLAN_22 M2: tree widths on public prompts and Part 4's example tasks | step 3 |
| PLAN_14's profile, PLAN_15's quantizer, PLAN_16's drafter code, PLAN_17's runner | in `code/`, as they ran (paths made configurable); not re-run for this page, see step 4 |
| PLAN_15's and PLAN_17's scores on the 400 tasks | **not possible from this repository**: the tasks are private (kept unseen for future model comparisons) |

## 1. Prerequisites

- **Part 4's setup** ([REPRODUCE](https://github.com/NautilusWu/qwen3.8_study/blob/main/qwen38-spark-test-time-strategies/REPRODUCE.md)): a DGX Spark, Docker with the NVIDIA Container Toolkit, the TensorFold 0.6.0 images (`qwen38-tensorfold-repro:0.6.0-trace` is the one used here, with `nvcc` 13.3 and PyTorch inside), and the two checkpoints (`nvidia/Qwen3.8-27B-NVFP4` @`482ca0f3`, `z-lab/Qwen3.8-27B-DFlash2` @`50307d4c`).
- Check out Part 4's folder next to this one, as in the repository.
- Nothing else may run on the GPU. Turn the display off and leave the desktop idle if you want numbers comparable to ours: with the display on, the same runs were up to 16% slower (README).

## 2. Read bandwidth and cuBLAS (PLAN_22 M1, M3)

No model is loaded. `code/ceiling/sizes.txt` and `tensors.json` list the 850 weight groups of one forward pass, from the checkpoint's file headers (`tensor_sizes.py <snapshot folder>` regenerates them).

```bash
python3 code/ceiling/run_ceiling.py m1 out-ceiling
```
```bash
python3 code/ceiling/run_ceiling.py m3 out-ceiling
```

- `m1` compiles `bw_tuned.cu` for sm_121 inside the image, sweeps 288 settings at 2 and 8 GB, runs the 401 matmul weight groups in layer order for the best 12 settings, and confirms the best in a fresh process. It prints `F_ms` (our run: 86.40) and writes `out-ceiling/m1.json`. About 5 minutes.
- `m3` runs the same 401 matmuls through cuBLAS (`torch._scaled_mm`, NVFP4 and FP8) at 16-128 rows and writes `out-ceiling/m3.json`. About 15 seconds.
- Expect a few percent of variation between runs and days; the confirmation step exists because the best of 288 settings is biased upward.

**Ours** (the published code in a clean copy, 2026-11-21 19:55-20:00, display on):
- `m1`: the candidate round's best was 86.33 ms (setting 2,2,0,256,1, one arena); the fresh confirmation gave **F = 88.87 ms** (197.5 GB/s), against 86.40 in the formal run. B\* 200.7 GB/s. All 850 groups: 86.29 ms.
- `m3`: 102.0 ms at 16 rows, 106.3 at 64 (+4%), 111.5 at 128 (formal run: 99.0, 103.9, 107.8).
- So the floor moved by about 3% between two runs on the same evening, which is the size of the whole matmul gap.

## 3. Tree widths (PLAN_16 B0, PLAN_22 M2)

```bash
export HF_DIR=~/.cache/huggingface TF_CACHE_DIR=~/.cache/qwen38-tensorfold-cache   # as in Part 4
python3 code/tree/run_tree_width.py ../qwen38-spark-test-time-strategies/code/strategies out-tree --order 16,64
```

- Each start runs TensorFold 0.6.0 at `--parallel 1` with the tree width set by `code/tree/serve_rows.py` (one constructor argument, nothing on disk changes) and the per-round hook `code/profile/hook/serve_hooked.py`.
- Then the warm-up, three public prompts and Part 4's 5 example tasks as D2. About 2 minutes per start after the kernels are compiled; the first start with an empty kernel cache takes longer.
- PLAN_22 used the default order `16,32,48,64,96,128,128,96,64,48,32,16`; PLAN_16 B0 used `16,24,32,32,24,16`.
- Every reply must be the same token for token across all starts (verification is exact); otherwise it stops.
- `out-tree/runs.json` has, per start: tokens per round (exact: 8.047 at 16 rows and 9.661 at 64 rows in ours), ms per round and the stage times, which depend on the machine and the day.

**Ours** (2026-11-21 20:00-20:06, a clean copy with an empty kernel cache, display on, `--order 16,64`):
- exit code 0; both starts took 18.1 s;
- **all 8 replies identical to PLAN_22's at both widths**; tokens per round exactly 8.047 and 9.661;
- ms per round 133.5 (16 rows, the first start, compiling kernels into the empty cache) and 149.5 (64 rows). The formal run had 115.0 and 146.4; timing depends on the day, the display and the cache.

## 4. The rest of the code, as it ran

These files are what ran, with local paths replaced by environment variables. They were not re-run for this page.

| Path | Plan | Needs |
|---|---|---|
| `code/profile/profile_runs.py`, `analyze_profile.py`, `hook/bw_bench.py` | PLAN_14 | `PART4_STRATEGIES=<part4>/code/strategies`; Nsight Systems inside the trace image; `analyze_profile.py` reads the exported SQLite traces |
| `code/fp4/quantize_attn.py` | PLAN_15 | the BF16 checkpoint `Qwen/Qwen3.8-27B` @`1d4bf0f2` (about 54 GB) and the NVFP4 checkpoint; writes a new checkpoint folder |
| `code/drafter/dflash_twin.py`, `b1c_train.py`, `b1b_seeds.json` | PLAN_16 B1 | the trainable drafter copy, the training script and the topic seeds; the generated training data (3.5 million tokens) is not published |
| `code/engine/p17_engine.py`, `p17_d2.py` | PLAN_17 | Part 10's `code/native_engine.py` on the Python path, `TF127_DIR` with the 1.2.7 release, DSpine under `HF_DIR/extra/tencent/...`, and your own 400 tasks (`TASKS_DIR`, `TASK_SPLIT`) |

PLAN_17's speed passes replayed 32 recorded coding-agent turns. Those recordings contain agent transcripts and are not published.

## 5. Recompute the tables

```bash
scripts/recheck.sh
```

Expected last lines:

```
summary.csv: all 7 rows match
tests: all match
reported only: all match
```

- The tests are recomputed with `code/stats.py`, called the way the locked analyses called it. The task order (T001-T400) is the order the locked analyses used, so the bootstrap intervals match to the last digit.
- The cross-plan rows also read Part 10's and Part 11's per-task files from the sibling folders.

## Versions

- TensorFold 0.6.0 image and the checkpoints as in Part 4 (images by digest, checkpoints by commit).
- TensorFold 1.2.7 (PLAN_17): release `v1.2.7`, commit `93a2e93ecebff8aa52b88e58c5dd222516e69c5f`, archive sha256 `250b7cd3d1aa2685cbf011f501c4f10bbff44c6ccd876dc63a7a4bbf337ed01b`, run in the base image of Part 4's TensorFold image, as in Part 10.
- DSpine (PLAN_17): `tencent/Qwen3.8-27B-DSpine-b16` @`ea909c0f5fab8865f9e14600334f55f7ce47308f`, `model.safetensors` sha256 `ac90f1df842ea66b23e4224f3e86624add547526f689ffe9e9c6b7a5937116b1`.
- Machine: DGX Spark (GB10), kernel 7.0.0-1019-nvidia, driver 580.178.04.
