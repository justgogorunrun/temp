#!/usr/bin/env bash
set -euo pipefail

NUM_GPUS=${NUM_GPUS:-2}
MODEL_PATH=${MODEL_PATH:-"/path/to/Qwen2.5-VL-7B-Instruct"}
MODEL_TYPE=${MODEL_TYPE:-"qwen2.5vl"}
VIDEO_DIR=${VIDEO_DIR:-"/path/to/MMR-V/videos"}
OUTPUT_PATH=${OUTPUT_PATH:-"/path/to/results/qwenvl_series_mmr.json"}
ANNOTATION_PATH=${ANNOTATION_PATH:-""}
MAX_FRAMES=${MAX_FRAMES:-8}
FPS=${FPS:-1.0}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-256}
ATTN_IMPL=${ATTN_IMPL:-"flash_attention_2"}

EXTRA_ARGS=()
if [[ -n "${ANNOTATION_PATH}" ]]; then
  EXTRA_ARGS+=("--annotation-path" "${ANNOTATION_PATH}")
fi

accelerate launch \
  --num_processes "${NUM_GPUS}" \
  MMR-V/evaluation/qwenvl_series_on_MMR_local.py \
  --model-path "${MODEL_PATH}" \
  --model-type "${MODEL_TYPE}" \
  --video-dir "${VIDEO_DIR}" \
  --output-path "${OUTPUT_PATH}" \
  --max-frames "${MAX_FRAMES}" \
  --fps "${FPS}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --attn-impl "${ATTN_IMPL}" \
  --merge-outputs \
  "${EXTRA_ARGS[@]}"
