# Shared settings for the scripts. Source, don't execute.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${STUDY_IMAGE:-qwen38-study:0.1.0}"
# Hugging Face cache on the host (the directory that contains hub/); mounted read-write at /hf.
HF_DIR="${HF_DIR:-$HOME/.cache/huggingface}"
# Persistent vLLM / torch.compile / Triton caches for these runs only.
CACHE_DIR="${STUDY_CACHE_DIR:-$HOME/.cache/qwen38-vllm-cache}"
# All paths given to the scripts are relative to the repo root (same inside the containers, where it is /study).
cd "$REPO"

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" >&2; }

# Run a Python module from the working tree inside the study image (no GPU, no network needed).
in_image() {
  docker run --rm --network none --user "$(id -u):$(id -g)" \
    -v "$REPO:/study" -w /study --entrypoint python "$IMAGE" "$@"
}

cfg_get() { in_image -m benchmark.config get "$1" "$2"; }

git_meta() {
  STUDY_GIT_COMMIT="$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo none)"
  if [ -n "$(git -C "$REPO" status --porcelain -- benchmark configs prompts scripts Dockerfile 2>/dev/null)" ]; then
    STUDY_GIT_DIRTY=1
  else
    STUDY_GIT_DIRTY=0
  fi
  STUDY_IMAGE_ID="$(docker image inspect "$IMAGE" --format '{{.Id}}')"
  export STUDY_GIT_COMMIT STUDY_GIT_DIRTY STUDY_IMAGE_ID
}

# Number of monitors the GPU is actively driving on any X display this user can reach
# (0 = headless, unplugged/disconnected, or DPMS off). On our GB10, decode throughput was
# 12-22% lower whenever the desktop monitor (2560x1440 @ 165 Hz) was active.
# Uses only non-probing queries: `xrandr --listmonitors` re-probes outputs (EDID reads,
# X log spam) on every call, `xrandr --current` and `xset q` do not.
display_count() {
  local n=0 sock d c
  for sock in /tmp/.X11-unix/X*; do
    [ -e "$sock" ] || continue
    d=":${sock##*/X}"
    DISPLAY="$d" timeout 5 xset q 2>/dev/null | grep -q "Monitor is Off" && continue
    c="$(DISPLAY="$d" timeout 5 xrandr --current 2>/dev/null | grep -cE ' connected( primary)? [0-9]+x[0-9]+\+')"
    n=$(( n + ${c:-0} ))
  done
  echo "$n"
}
