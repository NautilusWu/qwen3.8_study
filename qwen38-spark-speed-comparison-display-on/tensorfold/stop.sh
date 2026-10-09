#!/usr/bin/env bash
# Save the server log next to your results (optional) and remove the TensorFold container.
#   tensorfold/stop.sh [results_dir]
set -euo pipefail
NAME="${TF_CONTAINER:-q38-tensorfold}"
if [ -n "${1:-}" ]; then
  mkdir -p "$1"
  docker logs --timestamps "$NAME" > "$1/server.log" 2>&1 || true
fi
docker rm -f "$NAME" >/dev/null 2>&1 || true
