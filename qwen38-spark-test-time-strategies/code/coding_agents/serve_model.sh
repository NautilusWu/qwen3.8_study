#!/usr/bin/env bash
# serve_model.sh start | stop: the model server for the agent comparison, as used for the results:
# TensorFold + NVIDIA NVFP4 + DFlash2 on 127.0.0.1:8091, one stream, 262,144-token context, thinking on, greedy,
# FP8 KV cache and prefill, 40 GiB memory reserve, expandable_segments. run_all.py restarts it before every run.
# Container name: TF_CONTAINER (default qwen38-tensorfold); checkpoints from HF_DIR, kernel cache in TF_CACHE_DIR.
set -euo pipefail
NAME=${TF_CONTAINER:-qwen38-tensorfold}
HF_DIR=${HF_DIR:-$HOME/.cache/huggingface}
TF_CACHE_DIR=${TF_CACHE_DIR:-$HOME/.cache/qwen38-tensorfold-cache}
MODEL=/hf/hub/models--nvidia--Qwen3.8-27B-NVFP4/snapshots/482ca0f3832238542f8f5295dde86b5f22711d80
DRAFTER=/hf/hub/models--z-lab--Qwen3.8-27B-DFlash2/snapshots/50307d4c4cde6860d4eee73e2547cd786fe8e8a4

docker rm -f "$NAME" >/dev/null 2>&1 || true
[ "${1:-}" = stop ] && exit 0
[ "${1:-}" = start ] || { echo "usage: $0 start|stop" >&2; exit 2; }
mkdir -p "$TF_CACHE_DIR"
docker run -d --name "$NAME" --gpus all --network host --ipc host \
    --memory 100g --cpuset-cpus 5-9,15-19 --ulimit memlock=-1 --ulimit stack=67108864 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e TENSORFOLD_MEMORY_RESERVE_GIB=40 \
    -e TENSORFOLD_KV_POOL_GIB=0 -e TENSORFOLD_KV_DTYPE=fp8 \
    -v "$TF_CACHE_DIR":/cache -v "$HF_DIR":/hf:ro \
    qwen38-tensorfold-repro:0.6.0-c464617 \
    tensorfold serve "$MODEL" --drafter "$DRAFTER" --backend cuda --name qwen3.8-27b-tensorfold-nvfp4 \
    --host 127.0.0.1 --port 8091 --parallel 1 --context 262144 --max-tokens 65536 \
    --temperature 0 --top-p 1 --top-k 0 --thinking --prefill-fp8 >/dev/null
for _ in $(seq 600); do
    curl -sf --max-time 2 http://127.0.0.1:8091/health >/dev/null && { echo "model server ready on 127.0.0.1:8091"; exit 0; }
    if [ "$(docker inspect -f '{{.State.Running}}' "$NAME")" != true ]; then
        docker logs --tail 20 "$NAME" 2>&1; echo "model server failed to start" >&2; exit 1
    fi
    sleep 1
done
echo "model server not healthy after 600 s" >&2
exit 1
