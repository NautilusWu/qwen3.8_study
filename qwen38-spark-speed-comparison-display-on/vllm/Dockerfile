# Qwen3.8-27B single-stream runs on vLLM 0.27.1 -- immutable runtime image.
#
# Used for the NVIDIA NVFP4 rows (no speculation, MTP3, MTP4, MTP5). The FP8 rows
# use NVIDIA's vLLM 26.09 image unchanged (see REPRODUCE.md). Model weights are NOT
# baked in; they come from the Hugging Face cache mounted at /hf (see scripts/serve.sh).
#
# Base: NGC vLLM 26.08 (vLLM 0.27.1, PyTorch 2.14.0a0, CUDA 13.4, Triton 3.6.0,
# Transformers 5.14.1, FlashInfer 0.6.17). Pinned by digest so a re-pushed tag
# cannot silently change the stack.
FROM nvcr.io/nvidia/vllm:26.08-py3@sha256:4b16878dc624fc3dcef3d9cf13564c148b602698219c846f275012d49cdb3b56

# The 26.07 dev container needed a manual `pip install xgrammar==0.2.3`
# (0.2.0 lacks normalize_tool_choice). 26.08 already ships 0.2.3; pin it
# explicitly anyway so a base-image change fails the build instead of drifting.
RUN python -m pip install --no-deps xgrammar==0.2.3

# Analysis / quality-evaluation dependencies (not used by the server).
# Installed under the base image's /etc/pip/constraint.txt so numpy stays <=2.1.
RUN python -m pip install \
        pandas==2.2.3 \
        matplotlib==3.10.0 \
        statsmodels==0.14.5

# Fail the build if the runtime stack is not exactly the documented one.
RUN python - <<'PY'
from importlib.metadata import version
expected = {
    "vllm": "0.27.1+93523f72.nv26.8.64249418",
    "torch": "2.14.0a0+4fdf77b940.nv26.8.63802676",
    "triton": "3.6.0+git5d72932fc5.nv26.3",
    "transformers": "5.14.1",
    "xgrammar": "0.2.3",
    "flashinfer-python": "0.6.17+1b4edb7f.nv26.8.cu.63843061",
    "compressed-tensors": "0.17.0",
    "numpy": "2.1.0",
    "pandas": "2.2.3",
    "matplotlib": "3.10.0",
    "statsmodels": "0.14.5",
}
bad = {k: (version(k), v) for k, v in expected.items() if version(k) != v}
assert not bad, f"version drift (installed, expected): {bad}"
import xgrammar
assert hasattr(xgrammar, "normalize_tool_choice")
import statsmodels.formula.api  # noqa: F401  (0.14.4 breaks against the base image's scipy 1.18)
print("runtime stack OK")
PY

ENV HF_HOME=/hf \
    HF_HUB_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1 \
    DO_NOT_TRACK=1 \
    PYTHONUNBUFFERED=1

# Code is NOT copied in: the image is the environment only. Scripts bind-mount
# this directory at /study and every run records the git commit it ran from.
WORKDIR /study

LABEL org.opencontainers.image.title="qwen38-dgx-spark-study" \
      org.opencontainers.image.description="Qwen3.8-27B single-stream runtime (vLLM 0.27.1 / NGC 26.08)"
