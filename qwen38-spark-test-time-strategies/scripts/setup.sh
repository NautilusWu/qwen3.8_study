#!/usr/bin/env bash
# One-time setup: check the host, download the checkpoints, build the images. Run from anywhere.
#   scripts/setup.sh                 # everything
#   SKIP_MODELS=1 scripts/setup.sh   # images only (checkpoints already in HF_DIR)
# Not installed by this script (the host itself): a DGX Spark with DGX OS and its NVIDIA driver, Docker with the
# NVIDIA Container Toolkit, git, curl, bc, ss, python3 >= 3.10, and the Hugging Face CLI (pip install -U huggingface_hub).
# Disk: about 25 GB for the checkpoints and 30 GB for the images. Pulling the NVIDIA PyTorch base image may need
# `docker login nvcr.io` with an NGC API key.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_DIR="${HF_DIR:-$HOME/.cache/huggingface}"

# --- host check -----------------------------------------------------------------------------------------------
missing=()
for c in docker git curl bc ss python3 nvidia-smi; do command -v "$c" >/dev/null 2>&1 || missing+=("$c"); done
[ -n "${SKIP_MODELS:-}" ] || command -v hf >/dev/null 2>&1 || missing+=("hf (pip install -U huggingface_hub)")
[ ${#missing[@]} -eq 0 ] || { echo "missing on this host: ${missing[*]}" >&2; exit 1; }
python3 -c 'import sys; sys.exit(sys.version_info < (3, 10))' || { echo "python3 >= 3.10 needed" >&2; exit 1; }
docker info --format '{{json .Runtimes}}' | grep -q nvidia || { echo "Docker has no NVIDIA runtime: install the NVIDIA Container Toolkit" >&2; exit 1; }
gpu="$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
echo "host: $(uname -m), $(uname -r), GPU/driver: $gpu"
[ "$(uname -m)" = aarch64 ] || echo "warning: the images are for aarch64 (DGX Spark); this host is $(uname -m)" >&2
case "$gpu" in *GB10*) ;; *) echo "warning: results were measured on a DGX Spark (GB10); this GPU is different" >&2 ;; esac

# --- checkpoints (pinned revisions) ------------------------------------------------------------------------------
[ -n "${SKIP_MODELS:-}" ] || scripts/download_models.sh nvfp4 dflash2

# --- images ------------------------------------------------------------------------------------------------------
tensorfold/fetch_sources.sh                            # TensorFold v0.6.0 + recipe patches, hash-checked
docker build -t qwen38-tensorfold-repro:0.6.0-c464617 tensorfold      # fails if any Python package version differs
docker build -f tensorfold/Dockerfile.trace -t qwen38-tensorfold-repro:0.6.0-trace tensorfold
docker build -t agent-harness:0.1 code/coding_agents/image            # pinned agents; python3 + g++ for the sandbox
docker pull ghcr.io/berriai/litellm@sha256:3d88bd757134a00e8c00237b85a021ab1f6d54abb9c1df750f411dbe7256a500

echo "setup done (HF_DIR=$HF_DIR). Next: scripts/smoke_test.sh"
