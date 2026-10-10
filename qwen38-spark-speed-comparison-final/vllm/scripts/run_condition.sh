#!/usr/bin/env bash
# One server launch = one (condition, block) cell:
#   start server -> warm-up -> measured requests -> save server log -> stop server.
#   scripts/run_condition.sh configs/nvidia_nvfp4_mtp3.yaml results/<run>/nvidia_nvfp4_mtp3 0 [run.py args...]
source "$(dirname "$0")/lib.sh"
cfg="$1"; out="$2"; block="$3"; shift 3
name="$(cfg_get "$cfg" server.container_name)"
client_cpuset="$(cfg_get "$cfg" client.cpuset)"
mkdir -p "$out"
git_meta

# Sample display state for the whole cell (see display_count in lib.sh).
( while true; do printf '%s,%s\n' "$(date +%s.%N)" "$(display_count)"; sleep 5; done ) > "$out/display.csv" &
display_sampler=$!

cleanup() {
  kill "$display_sampler" 2>/dev/null || true
  docker logs --timestamps "$name" > "$out/server.log" 2>&1 || true
  docker inspect "$name" > "$out/server_inspect.json" 2>/dev/null || true
  docker rm -f "$name" >/dev/null 2>&1 || true
}
trap cleanup EXIT

launch=$(date +%s.%N)
"$REPO/scripts/serve.sh" "$cfg"
ready=$(date +%s.%N)
printf '{"launch_wall": %s, "ready_wall": %s}\n' "$launch" "$ready" > "$out/startup.json"

docker run --rm --network host --gpus all --user "$(id -u):$(id -g)" \
  --cpuset-cpus "$client_cpuset" \
  -v "$REPO:/study" -w /study \
  -e STUDY_GIT_COMMIT -e STUDY_GIT_DIRTY -e STUDY_IMAGE_ID -e STUDY_RUN_ID \
  --entrypoint python "$IMAGE" \
  -m benchmark.run --config "$cfg" --out "$out" --block "$block" "$@" 2>&1 | tee "$out/client.log"
exit "${PIPESTATUS[0]}"
