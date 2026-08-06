#!/usr/bin/env bash
# bio_phase15C_runner.sh — BioTriplex 1B Phase 1.5-C.
#
# Goal: a single in-silico "full protocol stack" baseline that mirrors what the
#       Phase 4 SLG-HE-PIR path will run, but stays in plaintext. This gives a
#       ground-truth anchor for the Phase 4 numbers (i.e. how much of the SLG
#       total tax is BFV/Gold-only protocol vs the simulated quantization+DP).
#
# One config: B-T_dpa15 + scale=10000 + g_H_dtype=bf16
#   dp_alpha         = 0.15
#   dp_answer_beta   = 0.5
#   scale            = 10000 (default CipherForge round-trip)
#   g_H_dtype        = bf16  (matches party_m gradient injection dtype)
#   lora_target      = "q,v"
#   seeds            = 42, 123, 2025
#
# Total: 1 config × 3 seeds = 3 runs.
#
# Output root: ${BIO_ROOT}/runs/fullstack_baseline/
#
# Env overrides:
#   SEEDS       : "42,123,2025"
#   MAX_EPOCHS  : default 8

set -euo pipefail

BIO_ROOT="${BIO_ROOT:-/root/autodl-tmp/CipherForgeCode/SLG-HE-PIR/test-data/BioTriplex1BTestData}"
SCRIPT_DIR="${BIO_ROOT}/scripts"
LOG_ROOT="${BIO_ROOT}/runs/fullstack_baseline/_runner_logs"

SEEDS="${SEEDS:-42,123,2025}"
MAX_EPOCHS="${MAX_EPOCHS:-8}"

DATA_DIR="${BIO_ROOT}/data"
GOLD_TEST="${DATA_DIR}/test_gold_general_qa.txt"
HF_MODEL="/root/autodl-tmp/CipherForgeCode/hf_cache/models--unsloth--Llama-3.2-1B/snapshots/9535bd9b1d1dea6acafbdc4813b728796aeb28da"

mkdir -p "${LOG_ROOT}"

ts_now() { date '+%Y-%m-%d %H:%M:%S'; }
log_line() {
    echo "[$(ts_now)] $*" | tee -a "${LOG_ROOT}/runner.log"
}

log_line "==============================================================="
log_line "BioTriplex 1B Phase 1.5-C (full-stack baseline) STARTED"
log_line "SEEDS=${SEEDS} MAX_EPOCHS=${MAX_EPOCHS}"
log_line "BIO_ROOT=${BIO_ROOT} DATA_DIR=${DATA_DIR}"
log_line "==============================================================="

RUNS_ROOT="${BIO_ROOT}/runs/fullstack_baseline"
mkdir -p "${RUNS_ROOT}"

# Single config: B-T-fullstack.
EXP_NAME="B-T-fullstack"

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

    log_line ">>> phase15C exp=${EXP_NAME} seed=${SEED} dp_alpha=0.15 scale=10000 g_H_dtype=bf16 epochs=${MAX_EPOCHS}"

    cd /root/autodl-tmp/CipherForgeCode/SLG-HE-PIR

    PYTHONPATH=/root/autodl-tmp/CipherForgeCode/SLG-HE-PIR python3 \
        ${SCRIPT_DIR}/bio_baseline_trainer.py \
        --data_dir "${DATA_DIR}" \
        --gold_path "${GOLD_TEST}" \
        --hf_model "${HF_MODEL}" \
        --output_dir "${OUT_DIR}" \
        --log_dir "${LOG_DIR}" \
        --max_epochs "${MAX_EPOCHS}" \
        --batch_size 1 \
        --max_seq_length 1024 \
        --learning_rate 5e-5 \
        --lora_target "q_proj,v_proj" \
        --lora_rank 8 \
        --lora_alpha 16 \
        --lora_dropout 0.05 \
        --dp_alpha 0.15 \
        --dp_answer_beta 0.5 \
        --scale 10000 \
        --g_H_dtype bf16 \
        --seed "${SEED}" \
        > "${LOG_DIR}/train_stdout.log" 2>&1 || {
        log_line "!!! phase15C exp=${EXP_NAME} seed=${SEED} FAILED (see ${LOG_DIR}/train_stdout.log)"
        continue
    }

    touch "${DONE_FLAG}"
    log_line "<<< phase15C exp=${EXP_NAME} seed=${SEED} DONE"
done

log_line "==============================================================="
log_line "BioTriplex 1B Phase 1.5-C runner EXITED"
[ -n "${CHAINED_SENTINEL:-}" ] && touch "$CHAINED_SENTINEL"
log_line "==============================================================="
