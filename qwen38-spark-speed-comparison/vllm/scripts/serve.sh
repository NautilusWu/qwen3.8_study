#!/usr/bin/env bash
# Start the vLLM server for one condition and block until it is healthy.
#   scripts/serve.sh configs/nvidia_nvfp4_mtp3.yaml
source "$(dirname "$0")/lib.sh"
cfg="$1"
mapfile -d '' -t ARGS < <(in_image -m benchmark.config serve-args "$cfg")
mapfile -d '' -t ENVS < <(in_image -m benchmark.config serve-env "$cfg")
ENV_FLAGS=(); for e in "${ENVS[@]}"; do ENV_FLAGS+=(-e "$e"); done
name="$(cfg_get "$cfg" server.container_name)"
port="$(cfg_get "$cfg" server.port)"
cpuset="$(cfg_get "$cfg" server.cpuset)"
shm="$(cfg_get "$cfg" server.shm_size)"
timeout_s="$(cfg_get "$cfg" server.startup_timeout_s)"
condition="$(cfg_get "$cfg" condition)"

mkdir -p "$CACHE_DIR"
docker rm -f "$name" >/dev/null 2>&1 || true
if curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
  log "port $port already serving something else; refusing to start"; exit 1
fi

# Single stream: the GPU must be exclusive. Refuse if any other CUDA process is alive
# (e.g. a dev vLLM container), since it would both steal unified memory and perturb timings.
others="$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null || true)"
if [ -n "$others" ] && [ -z "${STUDY_ALLOW_SHARED_GPU:-}" ]; then
  log "other CUDA processes are running; stop them first (or set STUDY_ALLOW_SHARED_GPU=1):"
  echo "$others" >&2
  docker ps --format '  {{.Names}}  {{.Image}}  {{.Status}}' >&2
  exit 1
fi

log "starting $condition: ${ENVS[*]} ${ARGS[*]}"
docker run -d --name "$name" \
  --label study=qwen38 --label condition="$condition" \
  --gpus all --ipc host --network host \
  --cpuset-cpus "$cpuset" --shm-size "$shm" \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$HF_DIR:/hf" -v "$CACHE_DIR:/root/.cache" \
  -e TRITON_CACHE_DIR=/root/.cache/triton "${ENV_FLAGS[@]}" \
  "$IMAGE" "${ARGS[@]}" >/dev/null

t0=$(date +%s)
until curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1; do
  if [ "$(docker inspect -f '{{.State.Running}}' "$name" 2>/dev/null)" != "true" ]; then
    log "server container exited during startup:"; docker logs --tail 80 "$name" >&2; exit 1
  fi
  if (( $(date +%s) - t0 > timeout_s )); then
    log "startup timed out after ${timeout_s}s"; docker logs --tail 80 "$name" >&2; exit 1
  fi
  sleep 5
done
log "$condition ready after $(( $(date +%s) - t0 ))s"
