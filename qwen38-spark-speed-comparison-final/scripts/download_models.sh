#!/usr/bin/env bash
# Download the checkpoints at the exact revisions used into the Hugging Face cache ($HF_DIR/hub).
#   scripts/download_models.sh              # everything (~69 GB)
#   scripts/download_models.sh nvfp4 dflash2
# Parts: tokenizer (always), nvfp4 (~21 GB), fp8 (~29 GB), mlx4 (~15 GB), dflash2 (~3.6 GB).
# Uses the `hf` CLI on the host if present (pip install -U huggingface_hub), otherwise the one in
# the vLLM runtime image (vllm/scripts/build.sh builds it).
set -euo pipefail
HF_DIR="${HF_DIR:-$HOME/.cache/huggingface}"
mkdir -p "$HF_DIR"

hf_cli() {
  if command -v hf >/dev/null 2>&1; then
    HF_HOME="$HF_DIR" HF_HUB_OFFLINE=0 hf "$@"
  else
    docker run --rm --user "$(id -u):$(id -g)" -e HF_HOME=/hf -e HF_HUB_OFFLINE=0 -e HF_TOKEN \
      -v "$HF_DIR:/hf" --entrypoint hf "${STUDY_IMAGE:-qwen38-study:0.1.0}" "$@"
  fi
}

parts=("$@")
[ ${#parts[@]} -gt 0 ] || parts=(nvfp4 fp8 mlx4 dflash2)

# Official Qwen tokenizer and chat template, used by every vLLM row; the BF16 weights are not needed.
hf_cli download Qwen/Qwen3.8-27B --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 --exclude "*.safetensors"

for p in "${parts[@]}"; do
  case "$p" in
    nvfp4)   hf_cli download nvidia/Qwen3.8-27B-NVFP4 --revision 482ca0f3832238542f8f5295dde86b5f22711d80 ;;
    fp8)     hf_cli download Qwen/Qwen3.8-27B-FP8 --revision 017b9c7af6b5689d5dd426a76e0bc077eb5ca20a ;;
    mlx4)    hf_cli download TensorFold/Qwen3.8-27B-MLX-4bit --revision 22d8d538154e0e5b6f8dcb1fcc74a60a0798104c ;;
    dflash2) hf_cli download z-lab/Qwen3.8-27B-DFlash2 --revision 50307d4c4cde6860d4eee73e2547cd786fe8e8a4 ;;
    *) echo "unknown part: $p (use nvfp4 fp8 mlx4 dflash2)" >&2; exit 2 ;;
  esac
done
echo "downloaded into $HF_DIR/hub"
