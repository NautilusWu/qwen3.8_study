FROM nvcr.io/nvidia/pytorch@sha256:2140e699b3beaf7f96a0081fd9c9406bc3832b435cdb60dfa2d261f7d2f34a1c
COPY source /opt/tf-src
COPY pip-freeze.txt /opt/pip-freeze.txt
# Version lock: every package from PyPI at the version of the image used for the published results (the lines
# with "@" are the base image's own builds); the build fails if the installed set differs in any way.
RUN grep -v ' @ ' /opt/pip-freeze.txt > /tmp/constraints.txt \
    && python -m pip install --no-cache-dir -c /tmp/constraints.txt /opt/tf-src 'transformers==5.17.0' 'av==19.0.1' \
    && python -m pip freeze | grep -v ' @ ' | diff /tmp/constraints.txt - \
    && echo "installed packages match pip-freeze.txt"
COPY recipe/patches /opt/tf-patches
RUN cd /usr/local/lib/python3.12/dist-packages && for p in /opt/tf-patches/*.patch; do patch -p0 --forward < "$p" || exit 1; done
RUN python -c "import tensorfold.cuda.server, tensorfold.vision.qwen_cuda, tensorfold.vision.videos, av; print('fixed TensorFold and five recipe patches imported')"
ENV HF_HUB_OFFLINE=1 TENSORFOLD_NO_UPDATE_CHECK=1 TORCH_EXTENSIONS_DIR=/cache/torch_extensions TRITON_CACHE_DIR=/cache/triton MAX_JOBS=2
WORKDIR /workspace
LABEL experiment="tf-single-stream-32k" tensorfold.commit="c4646171139ee8a3c38103eaa1699dad226ec12b" recipe.commit="c11ff138871cf43d12898d79dbfa4dc952d17ca4"
