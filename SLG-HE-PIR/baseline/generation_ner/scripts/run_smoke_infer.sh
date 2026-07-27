#!/bin/bash
# NER 推理烟雾测试：5 个样本
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${TASK_DIR}/../.." && pwd)"

MODEL_PATH="/root/autodl-tmp/hf_cache/Llama-3-1-8B-I"
DATA_PATH="${REPO_ROOT}/datasets/botriplex/Preprocessed BioTriplex/"
OUTPUT_DIR="${TASK_DIR}/checkpoints"

cd "${TASK_DIR}"
unset OMP_NUM_THREADS
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python scripts/ner_infer.py \
    --model_name "${MODEL_PATH}" \
    --peft_model "${OUTPUT_DIR}" \
    --data_path "${DATA_PATH}" \
    --output_path "${TASK_DIR}/logs/smoke_ner_infer.json" \
    --bf16 \
    --max_eval_samples 5 \
    --max_new_tokens 200 2>&1 | tail -30
