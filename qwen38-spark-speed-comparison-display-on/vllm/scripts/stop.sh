#!/usr/bin/env bash
# Stop and remove the server container.
source "$(dirname "$0")/lib.sh"
docker rm -f "$(cfg_get configs/nvidia_nvfp4_mtp3.yaml server.container_name)" >/dev/null 2>&1 || true
