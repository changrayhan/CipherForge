#!/bin/bash
# scripts/biotriplex_generation_ner.sh
# BioTriplex Task B — NER JSON generation under SLG-HE-PIR.
#
# Mirrors baseline/generation_ner/scripts/run_finetune.sh but routes
# training through the three-party privacy-preserving runtime.
# All hyperparameters are aligned with docs/BIOTRIPLEX_FINETUNE_README.md.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ----------------------------------------------------------------------------
# Task-specific paths (parallel to baseline/generation_ner/)
# ----------------------------------------------------------------------------
TASK_DIR="${REPO_ROOT}/baseline/generation_ner"
CHECKPOINT_DIR="${TASK_DIR}/checkpoints"
ADAPTER_DIR="${TASK_DIR}/adapter"
LOG_DIR="${TASK_DIR}/logs"
mkdir -p "${CHECKPOINT_DIR}" "${ADAPTER_DIR}" "${LOG_DIR}"

MODEL_PATH="/root/autodl-tmp/hf_cache/Llama-3-1-8B-I"
DATA_PATH="${REPO_ROOT}/datasets/botriplex/Preprocessed BioTriplex"
BFV_CACHE_DIR="/root/autodl-tmp/slg-bfv-cache"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# ----------------------------------------------------------------------------
# OOM mitigation (same as baseline)
# ----------------------------------------------------------------------------
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset OMP_NUM_THREADS

# ----------------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------------
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [NER] start, timestamp=${TIMESTAMP}"
echo "[NER] data_path=${DATA_PATH}"
echo "[NER] output_dir=${CHECKPOINT_DIR}"
echo "[NER] log_dir=${LOG_DIR}"

cd "${REPO_ROOT}"

python src/scripts/biotriplex_finetune.py \
    --task_type generation \
    --stage all \
    --data_path "${DATA_PATH}" \
    --hf_model "${MODEL_PATH}" \
    --bfv_cache_dir "${BFV_CACHE_DIR}" \
    --output_dir "${CHECKPOINT_DIR}" \
    --log_dir "${LOG_DIR}" \
    --adapter_dir "${ADAPTER_DIR}" \
    --max_epochs 10 \
    --batch_size 1 \
    --max_seq_length 10000 \
    --learning_rate 1e-4 \
    --weight_decay 0.2 \
    --warmup_steps 200 \
    --gradient_clip_norm 1.0 \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --use_chunked_pipeline True \
    --chunk_tokens 1536 \
    --seed 42 \
    --log_freq 10 \
    --save_freq 1 \
    --do_test_eval \
    --dp_enable \
    --dp_alpha 0.15 \
    --dp_answer_beta 0.5 \
    --dp_calibration_steps 5 \
    --dp_dump_audit \
    --dp_num_classes 7 \
    2>&1 | tee "${LOG_DIR}/train_${TIMESTAMP}.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] [NER] all done; metrics → ${LOG_DIR}/ner_*_evaluate_metrics.json"