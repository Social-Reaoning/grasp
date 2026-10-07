#!/usr/bin/env bash
set -euo pipefail

# ---------------- user configuration ----------------
MODEL_PATH=${MODEL_PATH:-"interlive/GRASP-Qwen3-VL-8B"}
MODEL_TYPE=${MODEL_TYPE:-"qwen3_vl"}            # qwen3_vl | qwen3_5
BENCH_DIR=${BENCH_DIR:?"Set BENCH_DIR to the extracted GRASP-Bench directory (json/ + video/)"}
OUTPUT=${OUTPUT:-"eval/results/$(basename "$MODEL_PATH").json"}
# -----------------------------------------------------

uv run python -m eval.eval_social_qa \
    --model_type "$MODEL_TYPE" \
    --model_path "$MODEL_PATH" \
    --data_path "$BENCH_DIR" \
    --output "$OUTPUT" \
    --reasoning \
    "$@"
