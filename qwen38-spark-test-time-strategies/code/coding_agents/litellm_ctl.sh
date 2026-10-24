#!/usr/bin/env bash
# litellm_ctl.sh start | stop: the LiteLLM gateway Claude Code talks to (Anthropic Messages API on 127.0.0.1:4000,
# forwarded to the normalizing proxy on 8099). LiteLLM 1.104.2, the image used for the results; local-only token.
set -euo pipefail
cd "$(dirname "$0")"
NAME=harness-litellm
IMAGE=ghcr.io/berriai/litellm@sha256:3d88bd757134a00e8c00237b85a021ab1f6d54abb9c1df750f411dbe7256a500
docker rm -f "$NAME" >/dev/null 2>&1 || true
[ "${1:-}" = stop ] && exit 0
[ "${1:-}" = start ] || { echo "usage: $0 start|stop" >&2; exit 2; }
docker run -d --name "$NAME" --network host -e LITELLM_TELEMETRY=False \
    -v "$PWD/configs/litellm.yaml:/app/config.yaml:ro" "$IMAGE" --config /app/config.yaml --host 127.0.0.1 --port 4000 >/dev/null
for _ in $(seq 90); do
    curl -sf -m 2 http://127.0.0.1:4000/health/liveliness >/dev/null && { echo "LiteLLM ready on 127.0.0.1:4000"; exit 0; }
    sleep 2
done
docker logs --tail 20 "$NAME" 2>&1; echo "LiteLLM did not start" >&2; exit 1
