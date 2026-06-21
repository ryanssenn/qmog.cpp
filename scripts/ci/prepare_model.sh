#!/usr/bin/env bash
#
# Download the pre-exported Q8F16 .mog model from Hugging Face for CI.
#
# Usage: ./scripts/ci/prepare_model.sh
#
# Env overrides:
#   MODEL_MOG     local output path (default: ./mistral-7B-Q8F16.mog)
#   HF_MOG_REPO   Hugging Face repo id (default: QmogAI/Mistral-7B-Q8F16)
#   HF_MOG_FILE   filename within the repo (default: mistral-7B-Q8F16.mog)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

MODEL="${MODEL_MOG:-./mistral-7B-Q8F16.mog}"
HF_REPO="${HF_MOG_REPO:-QmogAI/Mistral-7B-Q8F16}"
HF_FILE="${HF_MOG_FILE:-mistral-7B-Q8F16.mog}"

if [ -f "$MODEL" ]; then
    echo "Model already present: $MODEL ($(du -h "$MODEL" | cut -f1))"
    exit 0
fi

echo "Downloading $HF_REPO/$HF_FILE -> $MODEL ..."

VENV="${CI_VENV:-.ci-venv}"
if [ ! -d "$VENV" ]; then
    python3 -m venv "$VENV"
fi
# shellcheck source=/dev/null
source "$VENV/bin/activate"
pip install --quiet huggingface_hub

python3 - <<PY
import shutil
from huggingface_hub import hf_hub_download

cached = hf_hub_download(repo_id="${HF_REPO}", filename="${HF_FILE}")
shutil.copy2(cached, "${MODEL}")
PY
deactivate 2>/dev/null || true

echo "Download complete: $MODEL ($(du -h "$MODEL" | cut -f1))"
