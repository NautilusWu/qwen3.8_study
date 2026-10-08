# Reproducing the setup

This sets up the same engines, checkpoints and launch settings on a DGX Spark, and measures them the same way. The prompts used for the published results are not included; bring your own (format below).

## 0. Machine

What the published results ran on:

| Item | Version |
|---|---|
| Hardware | NVIDIA DGX Spark (GB10, 128 GB unified memory) |
| OS | DGX OS 7.2.3 (Ubuntu 24.04.4 LTS), kernel 6.17.0-1031-nvidia |
| GPU driver | 580.173.02 |
| Docker | 29.2.1 |
| NVIDIA Container Toolkit | 1.20.0 |

You also need about 75 GB of free disk for the checkpoints and about 75 GB for the container images. Pulling from `nvcr.io` may require `docker login nvcr.io` with an NGC API key.

All scripts take the Hugging Face cache from `HF_DIR` (default `~/.cache/huggingface`, the directory that contains `hub/`).

## 1. Download the checkpoints

```bash
scripts/download_models.sh
```

This downloads, at the exact revisions used:

| Checkpoint | Revision | Used by | Size |
|---|---|---|---|
| `Qwen/Qwen3.8-27B` (tokenizer and chat template only) | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | all vLLM rows | < 20 MB |
| `nvidia/Qwen3.8-27B-NVFP4` | `482ca0f3832238542f8f5295dde86b5f22711d80` | vLLM NVFP4 rows, TensorFold NVFP4 | ~21 GB |
| `Qwen/Qwen3.8-27B-FP8` | `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a` | vLLM FP8 rows | ~29 GB |
| `TensorFold/Qwen3.8-27B-MLX-4bit` | `22d8d538154e0e5b6f8dcb1fcc74a60a0798104c` | TensorFold MLX 4-bit | ~15 GB |
| `z-lab/Qwen3.8-27B-DFlash2` | `50307d4c4cde6860d4eee73e2547cd786fe8e8a4` | vLLM FP8 DFlash2, both TensorFold rows | ~3.6 GB |

To download only some of them: `scripts/download_models.sh nvfp4 dflash2`.

## 2. Prompts

Put your prompts in `vllm/prompts/prompts.jsonl`, one JSON object per line (see `vllm/prompts/example.jsonl`):

```json
{"id": "my_prompt_1", "category": "coding", "messages": [{"role": "user", "content": "..."}]}
```

`max_tokens` per prompt is optional (default 24,576). Both engines add the system prompt "You are a helpful assistant.", use thinking on at the chat template's default effort and temperature 0, and send the prompts one at a time in the same seeded order. Grading the answers is up to you; every answer is saved in full.

## 3. vLLM rows

### Images

```bash
vllm/scripts/build.sh
docker pull nvcr.io/nvidia/vllm:26.09-py3@sha256:557747337846eecf34486866f3d992383c6f8af869fdb0e365b5093fcaee02c2
```

- `build.sh` builds `qwen38-study:0.1.0` from `nvcr.io/nvidia/vllm:26.08-py3@sha256:4b16878dc624fc3dcef3d9cf13564c148b602698219c846f275012d49cdb3b56` (vLLM 0.27.1, PyTorch 2.14.0a0, Transformers 5.14.1). It pins `xgrammar==0.2.3` and fails the build if any library version differs from the one used. It is used for the four NVIDIA NVFP4 rows.
- The FP8 rows use NVIDIA's vLLM 26.09 image unchanged (vLLM 0.29.0, PyTorch 2.14.0a0, Transformers 5.16.1). vLLM 0.27.1 has no DFlash2 support.

### Run one configuration

From the repository root (output paths are relative to `vllm/`):

```bash
vllm/scripts/run_condition.sh configs/nvidia_nvfp4_mtp3.yaml results/nvidia_nvfp4_mtp3 0
```

For the FP8 rows, select the vLLM 0.29.0 image:

```bash
STUDY_IMAGE=nvcr.io/nvidia/vllm:26.09-py3@sha256:557747337846eecf34486866f3d992383c6f8af869fdb0e365b5093fcaee02c2 vllm/scripts/run_condition.sh configs/fp8_dflash2.yaml results/fp8_dflash2 0
```

`run_condition.sh` starts the server, waits until it is healthy, sends 6 warm-up requests (not counted), sends each prompt once, saves the server log and removes the container. It refuses to start while another CUDA process is running. Outputs in `vllm/results/<name>/`:

- `requests.jsonl`: per request, token ids, full text, timings and server speculative-decoding counters;
- `telemetry.csv`: GPU power, temperature, clock and utilization, host memory, every 0.5 s;
- `display.csv`: number of active monitors every 5 s;
- `server.log`, `server_inspect.json`, `run_meta.json`.

Configurations (`vllm/configs/`):

| File | Row | Image |
|---|---|---|
| `nvidia_nvfp4_mtp0.yaml` | NVIDIA NVFP4, no speculation | `qwen38-study:0.1.0` |
| `nvidia_nvfp4_mtp3.yaml` / `_mtp4` / `_mtp5` | NVIDIA NVFP4, MTP 3 / 4 / 5 | `qwen38-study:0.1.0` |
| `fp8_mtp3.yaml` | Qwen FP8, MTP 3 | vLLM 26.09 |
| `fp8_dflash2.yaml` | Qwen FP8, DFlash2, 7 draft tokens | vLLM 26.09 |

The server command these configs produce, for NVFP4 MTP3 (the others differ only in the checkpoint and `--speculative-config`):

```text
vllm serve nvidia/Qwen3.8-27B-NVFP4 --revision 482ca0f3832238542f8f5295dde86b5f22711d80
  --tokenizer Qwen/Qwen3.8-27B --tokenizer-revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
  --chat-template /hf/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/chat_template.jinja
  --served-model-name nvidia_nvfp4_mtp3 --host 127.0.0.1 --port 8110 --dtype bfloat16
  --max-model-len 32768 --kv-cache-dtype bfloat16 --gpu-memory-utilization 0.75
  --max-num-seqs 4 --max-num-batched-tokens 8192 --enable-chunked-prefill
  --no-enable-prefix-caching --generation-config vllm --language-model-only
  --speculative-config {"method":"mtp","num_speculative_tokens":3}
  --safetensors-load-strategy prefetch
```

The container runs with `--gpus all --ipc host --network host --cpuset-cpus 5-9,15-19 --shm-size 32g --ulimit memlock=-1 --ulimit stack=67108864`; the client runs on cores 0-3. `python -m benchmark.config serve-args <config>` inside the image prints the command for any config.

DFlash2 uses `--speculative-config {"method":"dflash","model":"/hf/hub/models--z-lab--Qwen3.8-27B-DFlash2/snapshots/50307d4c4cde6860d4eee73e2547cd786fe8e8a4","num_speculative_tokens":7}`. In vLLM it needs an unquantized LM head, which the Qwen FP8 checkpoint has and the NVIDIA NVFP4 checkpoint does not.

## 4. TensorFold rows

### Build the image

```bash
tensorfold/fetch_sources.sh
docker build -t qwen38-tensorfold-repro:0.6.0-c464617 tensorfold
docker build -f tensorfold/Dockerfile.trace -t qwen38-tensorfold-repro:0.6.0-trace tensorfold
```

- `fetch_sources.sh` fetches [TensorFold](https://github.com/ashhart/TensorFold) v0.6.0 at `c4646171139ee8a3c38103eaa1699dad226ec12b` and [MiaAI-Lab's DGX Spark recipe](https://github.com/MiaAI-Lab/Qwen3.8-27B-DGX-Spark-TensorFold) at `c11ff138871cf43d12898d79dbfa4dc952d17ca4`, then checks every file against the hashes of the tree used for the published results (`source.sha256`, `patches.sha256`).
- `Dockerfile`: base `nvcr.io/nvidia/pytorch:26.07-py3@sha256:2140e699b3beaf7f96a0081fd9c9406bc3832b435cdb60dfa2d261f7d2f34a1c`, installs TensorFold with `transformers==5.17.0` and `av`, and applies the recipe's five patches (FP8 KV cache, memory reserve, KV pool, YaRN, images/video). `pip-freeze.txt` lists the Python packages of the image used for the published results. A fresh build from these steps installed byte-identical TensorFold code and the same packages except `typer` 0.27.3 instead of 0.27.2 (a command-line helper, not used for inference), and produced token-identical outputs on test requests.
- `Dockerfile.trace` adds `instrument.py`, a measurement-only change: the stream carries the committed token ids of every event and the actual prompt token ids, and a request can ask for the prompt cache to be cleared first. It does not change logits, drafting, sampling or kernels. `bench.py` needs it.

### Run one configuration

```bash
tensorfold/serve.sh nvfp4
python3 tensorfold/bench.py --prompts vllm/prompts/prompts.jsonl --out results/tf_nvfp4
tensorfold/stop.sh results/tf_nvfp4
```

Use `mlx4` instead of `nvfp4` for the MLX 4-bit row. `serve.sh` runs:

```text
docker run --gpus all --ipc host --network host --cpuset-cpus 5-9,15-19
  --memory 80g --memory-swap 80g --ulimit memlock=-1 --ulimit stack=67108864
  -e TENSORFOLD_KV_DTYPE=fp8 -e TENSORFOLD_MEMORY_RESERVE_GIB=40 -e TENSORFOLD_KV_POOL_GIB=0
  -v $HF_DIR:/hf:ro -v <kernel cache>:/cache qwen38-tensorfold-repro:0.6.0-trace
  tensorfold serve <target> --drafter <DFlash2> --backend cuda --name Qwen3.8-27B
  --host 127.0.0.1 --port 8112 --parallel 1 --context 32768
  --temperature 0 --top-p 1 --top-k 0 --thinking --prefill-fp8
```

Differences from the recipe's defaults: context 32,768 instead of 262,144; one stream instead of 8; vision off; 40 GiB host memory reserve and no fixed KV pool; container memory capped at 80 GB.

`bench.py` sends 13 warm-up requests (not counted; the same as in the published runs), then each prompt once. Outputs: `requests.jsonl` (timings, token ids, thinking and answer text, TensorFold's round counters), `warmup.jsonl`, `telemetry.csv` (GPU temperature, power, utilization and host memory every 2 s). The server reports ready after about 20 s; on the first launch the warm-up requests also compile kernels into the kernel cache, which took about 4 minutes in total.

## 5. Metrics

Both clients record the same quantities:

| Metric | vLLM field (`requests.jsonl`) | TensorFold field |
|---|---|---|
| Output speed: committed tokens after the first token-carrying event / time from that event to the last one | `post_first_event_tok_s` | `post_first_event_tok_s` |
| Total time: request sent to stream finished | `e2e_s` | `e2e_s` |
| Time until the answer text starts (after thinking) | `ttft_answer_s` | `ttft_answer_s` |
| Tokens accepted per verification step | `sd_mean_acceptance_length` | `tokens_per_round` |
| Prompt token ids hash (compare across engines) | `prompt_token_ids_sha256` | `prompt_token_ids_sha256` |

Rejected draft tokens are never counted. In the published table, speeds are medians over prompts and "total time vs NVFP4 MTP3" is a per-prompt paired geometric mean.

## 6. Thermal protection and monitoring used for the published results

| Rows | Automatic thermal stop | Peak GPU temperature |
|---|---|---|
| All six vLLM rows | none; temperature logged every 0.5 s | 69-77 °C |
| TensorFold MLX 4-bit, first launch | stop when the GPU stays at 80 °C or more for 3 s; it triggered after 4 prompts | 80 °C |
| TensorFold MLX 4-bit, second launch (remaining 20 prompts) | disabled, temperature logged every 2 s | 79 °C |
| TensorFold NVIDIA NVFP4 | disabled, temperature logged every 2 s | 80 °C |

During the TensorFold launches a separate watchdog would also have stopped the server if host available memory fell below 32 GiB for 2 s or below 24 GiB once, if swap grew, or if the kernel log showed an NVIDIA out-of-memory or Xid error. None of these triggered, and no run logged a driver error. That watchdog is not part of this repository, and the scripts here have no automatic thermal stop; they only log temperature, so keep an eye on it.

The display was on for every run except the second TensorFold MLX 4-bit launch, where it was reportedly off (not verified). An active display lowers decode speed by about 12-22% on this machine; `display.csv` records it for vLLM runs.
