#!/usr/bin/env bash
# Download tencent/Hy-MT2-1.8B weights + tokenizer into the repo root.
# Requires `huggingface-cli` from `pip install huggingface_hub`.
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DEST="${HY_MT2_MODEL_DIR:-$REPO_ROOT/Hy-MT2-1.8B}"

mkdir -p "$DEST"
huggingface-cli download tencent/Hy-MT2-1.8B \
    --local-dir "$DEST" \
    --exclude "train/*" "imgs/*"
echo "Hy-MT2-1.8B fetched to $DEST"
