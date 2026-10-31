# Reproduce

## What can and cannot be reproduced

| | How |
|---|---|
| The tables in the README from the per-task results | `scripts/recheck.sh 72` and `scripts/recheck.sh 400` (no GPU) |
| The fine-tune checkpoint with an NVFP4 lm_head, and the format checks | steps 1-3 below |
| Running the comparison with your own tasks | step 4 (uses Part 4's runner) |
| Re-running our scores | **not possible from this repository**: Part 4's task sets and the 400 new tasks are private (kept unseen for future model comparisons) |

## 0. Prerequisites

Part 4's setup: a DGX Spark (or another machine Part 4's images support), Docker with the NVIDIA Container Toolkit, and [Part 4's `scripts/setup.sh`](https://github.com/NautilusWu/qwen3.8_study/blob/main/qwen38-spark-test-time-strategies/REPRODUCE.md), which downloads the original checkpoint and the DFlash2 drafter and builds the TensorFold images. Machine and versions used here are in the README's Conditions.

## 1. Download the fine-tune (about 23.8 GB)

Only the `NVFP4/W4A4-W8A8` folder at the commit used, without its own drafter (we use z-lab DFlash2 for both models):

```bash
hf download nerkyor/Qwen3.8-27B-Coder390-EfficientThink-Opus5.5-GPT6Astra-Grok4.7-DSV4Pro-K3-SFT-RLOO-MTP-DFlash2 \
  --revision 57c00f892e6b2ac425eb34b3ab12b3025901fb66 \
  --include "NVFP4/W4A4-W8A8/*" --exclude "NVFP4/W4A4-W8A8/DFlash2-FP8/*"
```

Then check the files against the folder's own `SHA256SUMS` (`sha256sum -c SHA256SUMS` inside the folder).

## 2. Convert lm_head to NVFP4 and build the checkpoint folder

`code/quantize_head.py` uses NVIDIA ModelOpt from the TensorFold image (ModelOpt asks for the GPU's compute capability, so the GPU must be visible; the computation runs on the CPU, no network):

```bash
SRC=<downloaded snapshot>/NVFP4/W4A4-W8A8
mkdir -p "$HF_DIR/extra/head" && docker run --rm --network none --gpus all --user "$(id -u):$(id -g)" \
  -v "$(readlink -f "$SRC/text-06.safetensors")":/in/text-06.safetensors:ro -v "$HF_DIR/extra/head":/out -v "$PWD/code":/tools:ro \
  --entrypoint python qwen38-tensorfold-repro:0.6.0-c464617 /tools/quantize_head.py /in/text-06.safetensors /out/text-06.safetensors
python3 code/make_overlay.py "$SRC" "$HF_DIR/extra/coder390-nvfp4head" "$HF_DIR/extra/head/text-06.safetensors"
```

`quantize_head.py` prints the relative error against the BF16 weight (expected about 0.095) and checks the packing against TensorFold's own dequantisation. The new folder must live inside `HF_DIR`, which the runner mounts as `/hf`; its symlinks are relative.

## 3. Check the format against NVIDIA's checkpoint

```bash
python3 code/compare_tensors.py "$HF_DIR/hub/models--nvidia--Qwen3.8-27B-NVFP4/snapshots/482ca0f3832238542f8f5295dde86b5f22711d80" "$HF_DIR/extra/coder390-nvfp4head"
```

Expected: 2194 vs 2193 tensors, only `lm_head.input_scale` missing on the right, 0 dtype/shape differences.

## 4. Run with your own tasks

Use Part 4's runner unchanged; `code/run_with_target.py` only swaps the target checkpoint (a path inside `/hf`). With Part 4's repository checked out next to this one, and your tasks and grader set up as in Part 4's REPRODUCE.md:

```bash
python3 code/run_with_target.py ../qwen38-spark-test-time-strategies/code/strategies /hf/extra/coder390-nvfp4head \
  A65 ft-A65 --split confirm --limit-min 40 --context 81920 --max-tokens 65536
```

Methods and settings as in Part 4 (`D2` low thinking, `CL --k 4` vote, `H` plan + low execution, `A65` deep thinking). Run the same methods without the wrapper for the original model, then compare with `code/stats.py NAME=results/<run>/graded.jsonl ...` (paired by task id).

## 5. Recompute the tables

```bash
scripts/recheck.sh 72
scripts/recheck.sh 400
```

`code/stats.py` reads the per-task CSV files and prints totals and, for every pair of configurations, the tasks only one side solved, the exact McNemar p (Holm-adjusted over all pairs printed), and paired bootstrap 95% intervals (10,000 resamples, seed 2026) of the accuracy difference and the time ratio. It reproduces every number in the README. Note: the Holm adjustment in its output covers all pairs it prints (28 for the 72-task set, 15 for the 400-task set), which differs from the adjustment over the three locked pairs in `data/tables_400_preregistered.json`.
