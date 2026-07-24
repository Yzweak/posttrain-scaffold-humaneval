#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

: "${MODEL_PATH:?MODEL_PATH must point to the read-only base model}"
: "${OUTPUT_DIR:?OUTPUT_DIR must point to the output checkpoint directory}"

uv sync
uv run python recipes/mbpp_lora_sft/train.py --model-path "$MODEL_PATH" --output-dir "$OUTPUT_DIR"
