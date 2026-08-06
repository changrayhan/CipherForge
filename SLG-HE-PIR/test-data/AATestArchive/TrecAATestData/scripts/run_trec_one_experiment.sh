#!/usr/bin/env bash
# run_trec_one_experiment.sh — run a single TREC-QC baseline experiment.
#
# Usage:
#   ./run_trec_one_experiment.sh <exp_name> <lora_target> <dp_alpha> <dp_beta> <seed>
#
# Example:
#   ./run_trec_one_experiment.sh B-T_dpa15 "q,v" 0.15 0.5 42
#
# This bypasses the runner and just calls trec_baseline_trainer.py directly.
# Useful for debugging a single experiment.

set -euo pipefail

if [[ $# -ne 5 ]]; then
    echo "Usage: $0 <exp_name> <lora_target> <dp_alpha> <dp_beta> <seed>"
    exit 1
fi

EXP_NAME="$1"
LORA_TARGET="$2"
DP_ALPHA="$3"
DP_BETA="$4"
SEED="$5"

TREC_ROOT="/root/autodl-tmp/CipherForgeCode/SLG-HE-PIR/test-data/TrecAATestData"
REPO="/root/autodl-tmp/CipherForgeCode/SLG-HE-PIR"
SNAPSHOT_DIR=$(ls /root/autodl-tmp/CipherForgeCode/hf_cache/models--unsloth--Llama-3.2-1B/snapshots/ 2>/dev/null | head -1)
HF_MODEL_DEFAULT="/root/autodl-tmp/CipherForgeCode/hf_cache/models--unsloth--Llama-3.2-1B/snapshots/${SNAPSHOT_DIR}"
HF_MODEL="${HF_MODEL:-$HF_MODEL_DEFAULT}"
TREC_DATA_DIR="${REPO}/datasets/trec-qc"
GOLD_FILE="${TREC_ROOT}/gold/test_gold_general_qa.txt"

EXP_DIR="${TREC_ROOT}/runs/baseline/${EXP_NAME}/seed${SEED}"
LOG_DIR="${EXP_DIR}/logs"
mkdir -p "${EXP_DIR}" "${LOG_DIR}"

cd "${REPO}"
python3 "${TREC_ROOT}/scripts/trec_baseline_trainer.py" \
    --data_dir "${TREC_DATA_DIR}" \
    --gold_path "${GOLD_FILE}" \
    --hf_model "${HF_MODEL}" \
    --output_dir "${EXP_DIR}" \
    --log_dir "${LOG_DIR}" \
    --max_epochs 5 \
    --batch_size 8 \
    --max_seq_length 256 \
    --learning_rate 1e-4 \
    --weight_decay 0.0 \
    --warmup_steps 50 \
    --gradient_clip_norm 1.0 \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target "${LORA_TARGET}" \
    --seed "${SEED}" \
    --dp_alpha "${DP_ALPHA}" \
    --dp_answer_beta "${DP_BETA}" \
    --dp_calibration_steps 5