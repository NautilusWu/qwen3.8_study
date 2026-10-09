#!/usr/bin/env bash
# Fetch TensorFold v0.6.0 and MiaAI-Lab's DGX Spark recipe (its five patches) at the exact
# commits used, into ./source and ./recipe, which the Dockerfile copies. Then check every file
# against the hashes of the tree the published results were built from.
#   tensorfold/fetch_sources.sh
set -euo pipefail
cd "$(dirname "$0")"

TF_REPO=https://github.com/ashhart/TensorFold.git
TF_COMMIT=c4646171139ee8a3c38103eaa1699dad226ec12b          # v0.6.0
RECIPE_REPO=https://github.com/MiaAI-Lab/Qwen3.8-27B-DGX-Spark-TensorFold.git
RECIPE_COMMIT=c11ff138871cf43d12898d79dbfa4dc952d17ca4

fetch() {  # <repo> <commit> <dest>: the commit's files only, no .git
  local tmp
  tmp="$(mktemp -d)"
  git -C "$tmp" init -q
  git -C "$tmp" remote add origin "$1"
  git -C "$tmp" fetch -q --depth 1 origin "$2"
  rm -rf "$3"
  mkdir -p "$3"
  git -C "$tmp" archive FETCH_HEAD | tar -x -C "$3"
  rm -rf "$tmp"
}

fetch "$TF_REPO" "$TF_COMMIT" source
fetch "$RECIPE_REPO" "$RECIPE_COMMIT" recipe

sha256sum -c --quiet source.sha256
n="$(find source -type f | wc -l)"
[ "$n" -eq "$(wc -l < source.sha256)" ] || { echo "source has $n files, expected $(wc -l < source.sha256)" >&2; exit 1; }
sha256sum -c --quiet patches.sha256
echo "source/ and recipe/patches/ match the tree used for the published results"
