#!/usr/bin/env bash
# Build the vLLM 0.27.1 runtime image and record its identity.
source "$(dirname "$0")/lib.sh"
cd "$REPO"
docker build -t "$IMAGE" "$@" .
{
  echo "image: $IMAGE"
  echo "image_id: $(docker image inspect "$IMAGE" --format '{{.Id}}')"
  echo "base_image: $(grep -m1 '^FROM' Dockerfile | awk '{print $2}')"
  echo "built_at: $(date -u +%FT%TZ)"
} > image-digest.txt
cat image-digest.txt
