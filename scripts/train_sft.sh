#!/usr/bin/env bash
set -euo pipefail

# ---------------- user configuration ----------------
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-VL-8B-Instruct"}
MODEL_TYPE=${MODEL_TYPE:-"qwen3_vl"}            # qwen3_vl | qwen3_5
DATA_PATH=${DATA_PATH:?"Set DATA_PATH to train/data_sft.jsonl"}
VIDEO_ROOT=${VIDEO_ROOT:?"Set VIDEO_ROOT to the directory the video tars were extracted into"}
OUTPUT_DIR=${OUTPUT_DIR:-"checkpoints/grasp_sft"}
NUM_GPUS=${NUM_GPUS:-8}
# -----------------------------------------------------

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA_PATH=$(realpath "$DATA_PATH")
OUTPUT_DIR=$(realpath -m "$OUTPUT_DIR")
cd "$VIDEO_ROOT"

uv run --project "$REPO_DIR" torchrun --nproc_per_node "$NUM_GPUS" -m socialmllm.train.sft.run \
    --deepspeed "$REPO_DIR/configs/deepspeed_zero2.json" \
    --model_name_or_path "$MODEL_PATH" \
    --model_type "$MODEL_TYPE" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --tune_lang true \
    --bf16 true \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps $((32 / NUM_GPUS)) \
    --learning_rate 2e-6 \
    --weight_decay 1e-3 \
    --max_grad_norm 1.0 \
    --warmup_steps 200 \
    --lr_scheduler_type cosine \
    --video_max_fps 2 \
    --gradient_checkpointing true \
    --logging_steps 1 \
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 2 \
    "$@"
