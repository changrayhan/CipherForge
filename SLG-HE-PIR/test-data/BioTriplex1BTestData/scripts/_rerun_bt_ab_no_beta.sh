#!/usr/bin/env bash
# _rerun_bt_ab_no_beta.sh — re-run the B-T_ab_no_beta experiment after the
# dp_answer_beta dead-parameter bug fix.
#
# Config: dp_alpha=0.15, dp_answer_beta=0.0 (no answer noise → clean DP)
# Seeds: 42, 123, 2025 — same as the original Phase 1.5 run so results
# remain directly comparable to the other Phase 1.5 entries.
# Epochs: 8 (same as the rest of Phase 1).
#
# Usage:
#   bash scripts/_rerun_bt_ab_no_beta.sh

set -euo pipefail

BIO_ROOT="/root/autodl-tmp/CipherForgeCode/SLG-HE-PIR/test-data/BioTriplex1BTestData"
SCRIPT_DIR="${BIO_ROOT}/scripts"
LOG_ROOT="${BIO_ROOT}/runs/baseline/_runner_logs"
mkdir -p "${LOG_ROOT}"

SEEDS="42,123,2025"
EPOCHS=8
LORA_TARGET="q_proj,v_proj"
DP_ALPHA="0.15"
DP_BETA="0.0"

DATA_DIR="${BIO_ROOT}/data"
GOLD_TEST="${DATA_DIR}/test_gold_general_qa.txt"
HF_MODEL_DEFAULT="/root/autodl-tmp/CipherForgeCode/hf_cache/models--unsloth--Llama-3.2-1B/snapshots/9535bd9b1d1dea6acafbdc4813b728796aeb28da"

ts_now() { date '+%Y-%m-%d %H:%M:%S'; }
log_line() {
    echo "[$(ts_now)] $*" | tee -a "${LOG_ROOT}/rerun_ab_no_beta.log"
}

log_line "============================================================="
log_line "B-T_ab_no_beta RE-RUN STARTED (post dp_answer_beta fix)"
log_line "SEEDS=${SEEDS} EPOCHS=${EPOCHS} dp_alpha=${DP_ALPHA} dp_answer_beta=${DP_BETA}"
log_line "============================================================="

EXP_NAME="B-T_ab_no_beta"
RUNS_ROOT="${BIO_ROOT}/runs/baseline"

for SEED in ${SEEDS//,/ }; do
    EXP_DIR="${RUNS_ROOT}/${EXP_NAME}/seed_${SEED}"
    LOG_DIR="${EXP_DIR}/logs"
    OUT_DIR="${EXP_DIR}/output"
    DONE_FLAG="${EXP_DIR}/DONE.flag"
    mkdir -p "${LOG_DIR}" "${OUT_DIR}"

    if [ -f "${DONE_FLAG}" ]; then
        log_line "[skip] ${EXP_NAME} seed=${SEED} (DONE.flag exists)"
        continue
    fi

    log_line ">>> rerun ${EXP_NAME} seed=${SEED} lora=${LORA_TARGET} dp_alpha=${DP_ALPHA} dp_answer_beta=${DP_BETA} epochs=${EPOCHS}"

    cd /root/autodl-tmp/CipherForgeCode/SLG-HE-PIR

    PYTHONPATH=/root/autodl-tmp/CipherForgeCode/SLG-HE-PIR python3 \
        ${SCRIPT_DIR}/bio_baseline_trainer.py \
        --data_dir "${DATA_DIR}" \
        --gold_path "${GOLD_TEST}" \
        --hf_model "${HF_MODEL_DEFAULT}" \
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
        --seed "${SEED}" \
        > "${LOG_DIR}/train_stdout.log" 2>&1 || {
        log_line "!!! rerun ${EXP_NAME} seed=${SEED} FAILED (see ${LOG_DIR}/train_stdout.log)"
        continue
    }

    touch "${DONE_FLAG}"
    log_line "<<< rerun ${EXP_NAME} seed=${SEED} DONE"
done

log_line "============================================================="
log_line "B-T_ab_no_beta RE-RUN COMPLETE"
log_line "============================================================="
