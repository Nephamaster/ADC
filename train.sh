#!/usr/bin/env bash
set -euo pipefail

NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
MODEL_PATH="${MODEL_PATH:-./checkpoints/Qwen3.5-9B-Base-34M}"
# MODEL_PATH="${MODEL_PATH:-../../data/models/Qwen3.5-9B-Base}"
DATASET_PATH="${DATASET_PATH:-./data/train/csc_mix.jsonl}"
CACHE_DIR="${CACHE_DIR:-./cache}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/Qwen3.5-9B-Adapter-CSCMIX}"
PLUG_IDX="${PLUG_IDX:--4}"
LEARNING_RATE="${LEARNING_RATE:-7e-5}"
UNFREEZE_FIRST_LAYERS="${UNFREEZE_FIRST_LAYERS:-0}"
UNFREEZE_LAST_LAYERS="${UNFREEZE_LAST_LAYERS:-1}"

echo "[train.sh] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "[train.sh] NPROC_PER_NODE=${NPROC_PER_NODE} NNODES=${NNODES}"

if [[ "${NNODES}" == "1" ]]; then
  exec torchrun --standalone \
    --nproc_per_node "${NPROC_PER_NODE}" \
    train.py \
    --model "${MODEL_PATH}" \
    --dataset "${DATASET_PATH}" \
    --cache "${CACHE_DIR}" \
    --output "${OUTPUT_DIR}" \
    --plug_idx $PLUG_IDX \
    --bf16
else
  exec torchrun \
    --nproc_per_node "${NPROC_PER_NODE}" \
    --nnodes "${NNODES}" \
    --node_rank "${NODE_RANK}" \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${MASTER_PORT}" \
    train.py \
    --model "${MODEL_PATH}" \
    --dataset "${DATASET_PATH}" \
    --cache "${CACHE_DIR}" \
    --output "${OUTPUT_DIR}" \
    --plug_idx $PLUG_IDX \
    --learning_rate "${LEARNING_RATE}" \
    --unfreeze_first_layers "${UNFREEZE_FIRST_LAYERS}" \
    --unfreeze_last_layers "${UNFREEZE_LAST_LAYERS}" \
    --bf16
fi
