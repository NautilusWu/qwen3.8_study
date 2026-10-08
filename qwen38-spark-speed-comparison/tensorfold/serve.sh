#!/usr/bin/env bash
# Start TensorFold with the flags of the published TensorFold rows and block until it is healthy.
#   tensorfold/serve.sh nvfp4    # target nvidia/Qwen3.8-27B-NVFP4
#   tensorfold/serve.sh mlx4     # target TensorFold/Qwen3.8-27B-MLX-4bit
# Both use the z-lab/Qwen3.8-27B-DFlash2 drafter, FP8 KV cache and FP8 prefill, one stream, 32K context.
set -euo pipefail

HF_DIR="${HF_DIR:-$HOME/.cache/huggingface}"           # host Hugging Face cache (contains hub/)
CACHE_DIR="${TF_CACHE_DIR:-$HOME/.cache/qwen38-tensorfold-cache}"   # compiled kernels, reused across launches
IMAGE="${TF_IMAGE:-qwen38-tensorfold-repro:0.6.0-trace}"
NAME="${TF_CONTAINER:-q38-tensorfold}"
PORT=8112
STARTUP_TIMEOUT_S=1800

case "${1:-}" in
  nvfp4) TARGET=/hf/hub/models--nvidia--Qwen3.8-27B-NVFP4/snapshots/482ca0f3832238542f8f5295dde86b5f22711d80 ;;
  mlx4)  TARGET=/hf/hub/models--TensorFold--Qwen3.8-27B-MLX-4bit/snapshots/22d8d538154e0e5b6f8dcb1fcc74a60a0798104c ;;
  *) echo "usage: $0 nvfp4|mlx4" >&2; exit 2 ;;
esac
DRAFT=/hf/hub/models--z-lab--Qwen3.8-27B-DFlash2/snapshots/50307d4c4cde6860d4eee73e2547cd786fe8e8a4

# Single stream: the GPU must be exclusive.
others="$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null || true)"
if [ -n "$others" ]; then
  echo "other CUDA processes are running; stop them first:" >&2
  echo "$others" >&2
  exit 1
fi

mkdir -p "$CACHE_DIR"
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" \
  --gpus all --ipc host --network host --cpuset-cpus 5-9,15-19 \
  --memory 80g --memory-swap 80g --ulimit memlock=-1 --ulimit stack=67108864 \
  -e TENSORFOLD_KV_DTYPE=fp8 -e TENSORFOLD_MEMORY_RESERVE_GIB=40 -e TENSORFOLD_KV_POOL_GIB=0 \
  -v "$HF_DIR:/hf:ro" -v "$CACHE_DIR:/cache" \
  "$IMAGE" \
  tensorfold serve "$TARGET" --drafter "$DRAFT" --backend cuda --name Qwen3.8-27B \
  --host 127.0.0.1 --port "$PORT" --parallel 1 --context 32768 \
  --temperature 0 --top-p 1 --top-k 0 --thinking --prefill-fp8 >/dev/null

t0=$(date +%s)
until curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
  if [ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" != "true" ]; then
    echo "server container exited during startup:" >&2; docker logs --tail 80 "$NAME" >&2; exit 1
  fi
  if (( $(date +%s) - t0 > STARTUP_TIMEOUT_S )); then
    echo "startup timed out after ${STARTUP_TIMEOUT_S}s" >&2; docker logs --tail 80 "$NAME" >&2; exit 1
  fi
  sleep 5
done
echo "TensorFold ($1) ready on 127.0.0.1:$PORT after $(( $(date +%s) - t0 ))s"
