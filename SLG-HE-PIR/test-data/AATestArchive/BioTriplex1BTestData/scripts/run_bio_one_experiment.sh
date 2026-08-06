#!/usr/bin/env bash
# run_bio_one_experiment.sh — single-experiment launcher for BioTriplex 1B.
#
# Usage:
#   bash run_bio_one_experiment.sh <exp_name> <seed> [dp_alpha] [dp_beta] [lora_target]
#
# Defaults:
#   dp_alpha=0.00  dp_beta=0.5  lora_target=q,v  epochs=8

set -euo pipefail

EXP_NAME="${1:?usage: run_bio_one_experiment.sh <exp_name> <seed> [dp_alpha] [dp_beta] [lora_target]}"
SEED="${2:?usage: run_bio_one_experiment.sh <exp_name> <seed> [dp_alpha] [dp_beta] [lora_target]}"
DP_ALPHA="${3:-0.00}"
DP_BETA="${4:-0.5}"
LORA_TARGET="${5:-q,v}"
EPOCHS="${EPOCHS:-8}"

BIO_ROOT="${BIO_ROOT:-/root/autodl-tmp/CipherForgeCode/SLG-HE-PIR/test-data/BioTriplex1BTestData}"
DATA_DIR="${BIO_ROOT}/data"
GOLD_TEST="${DATA_DIR}/test_gold_general_qa.txt"
HF_MODEL_DEFAULT="/root/autodl-tmp/CipherForgeCode/hf_cache/models--unsloth--Llama-3.2-1B/snapshots/9535bd9b1d1dea6acafbdc4813b728796aeb28da"

EXP_DIR="${BIO_ROOT}/runs/baseline/${EXP_NAME}/seed_${SEED}"
LOG_DIR="${EXP_DIR}/logs"
OUT_DIR="${EXP_DIR}/output"
DONE_FLAG="${EXP_DIR}/DONE.flag"
mkdir -p "${LOG_DIR}" "${OUT_DIR}"

if [ -f "${DONE_FLAG}" ]; then
    echo "[skip] ${EXP_NAME} seed=${SEED} (DONE.flag exists)"
    exit 0
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] >>> exp=${EXP_NAME} seed=${SEED} lora=${LORA_TARGET} dp_alpha=${DP_ALPHA} dp_beta=${DP_BETA} epochs=${EPOCHS}"

cd /root/autodl-tmp/CipherForgeCode/SLG-HE-PIR
PYTHONPATH=/root/autodl-tmp/CipherForgeCode/SLG-HE-PIR python3 \
    ${BIO_ROOT}/scripts/bio_baseline_trainer.py \
    --data_dir "${DATA_DIR}" \
    --gold_path "${GOLD_TEST}" \
    --hf_model "${HF_MODEL:-${HF_MODEL_DEFAULT}}" \
    --output_dir "${OUT_DIR}" \
    --log_dir "${LOG_DIR}" \
    --max_epochs "${EPOCHS}" \
    --batch_size 1 \
    --max_seq_length 1024 \
    --learning_rate 5e-5 \
    --lora_target "${LORA_TARGET}" \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --dp_alpha "${DP_ALPHA}" \
    --dp_answer_beta "${DP_BETA}" \
    --seed "${SEED}"

touch "${DONE_FLAG}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] <<< exp=${EXP_NAME} seed=${SEED} DONE"