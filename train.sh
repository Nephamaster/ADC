#!/usr/bin/env bash
set -euo pipefail

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

MODEL_PATH="${MODEL_PATH:-/share/project/wuhaiming/spaces/ADC/charize/Qwen3-8B-Base-Char}"
DATASET_PATH="${DATASET_PATH:-/share/project/wuhaiming/spaces/LlamaFactory/data/twnlp_csc.jsonl}"
CACHE_DIR="${CACHE_DIR:-/share/project/wuhaiming/spaces/ADC/cache/}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/Qwen3-8B-Char-Adapter-twnlp}"
PLUG_IDX="${PLUG_IDX:-28}"
LEARNING_RATE="${LEARNING_RATE:-7e-5}"

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
    --plug_idx "${PLUG_IDX}" \
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
    --plug_idx "${PLUG_IDX}" \
    --learning_rate "${LEARNING_RATE}" \
    --bf16
fi