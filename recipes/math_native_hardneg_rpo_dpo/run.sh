#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

: "${MODEL_PATH:?MODEL_PATH must point to the read-only base model}"
: "${OUTPUT_DIR:?OUTPUT_DIR must point to the checkpoint output directory}"

uv sync
uv run python recipes/math_native_hardneg_rpo_dpo/train.py \
  --model_path "$MODEL_PATH" \
  --output_dir "$OUTPUT_DIR"
