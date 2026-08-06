#!/usr/bin/env bash
# bio_phase15B_runner.sh — BioTriplex 1B Phase 1.5-B.
#
# Goal: capture the DP × scale × g_H_dtype interaction (the strict superset
# of Phase 1.5's "scale×g_H_dtype, dp_alpha=0" sweep).
#
# Design:
#   - Hold dp_alpha = 0.15 (CipherForge best DP setting per Phase 1).
#   - Hold dp_answer_beta = 0.5.
#   - Hold lora_target = "q,v" (2-layer, smallest variance config).
#   - Sweep scale ∈ {100, 10k, 100k} × g_H_dtype ∈ {bf16, fp32} = 6 configs.
#   - 3 seeds (42, 123, 2025).
#   Total: 6 × 3 = 18 runs.
#
# Output root: ${BIO_ROOT}/runs/quant_dp15/
#
# Env overrides:
#   SEEDS      : "42,123,2025"
#   MAX_EPOCHS : default 8

set -euo pipefail

BIO_ROOT="${BIO_ROOT:-/root/autodl-tmp/CipherForgeCode/SLG-HE-PIR/test-data/BioTriplex1BTestData}"
SCRIPT_DIR="${BIO_ROOT}/scripts"
LOG_ROOT="${BIO_ROOT}/runs/quant_dp15/_runner_logs"

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
log_line "BioTriplex 1B Phase 1.5-B (DP=0.15 × scale × dtype) STARTED"
log_line "SEEDS=${SEEDS} MAX_EPOCHS=${MAX_EPOCHS}"
log_line "BIO_ROOT=${BIO_ROOT} DATA_DIR=${DATA_DIR}"
log_line "==============================================================="

RUNS_ROOT="${BIO_ROOT}/runs/quant_dp15"
mkdir -p "${RUNS_ROOT}"

# Config grid: 3 scales × 2 g_H_dtype = 6 experiments.
# Format: exp_name | scale | g_H_dtype
configs=(
    "B-dpa15-s100-bf16|100|bf16"
    "B-dpa15-s100-fp32|100|fp32"
    "B-dpa15-s10k-bf16|10000|bf16"
    "B-dpa15-s10k-fp32|10000|fp32"
    "B-dpa15-s100k-bf16|100000|bf16"
    "B-dpa15-s100k-fp32|100000|fp32"
)

for cfg_line in "${configs[@]}"; do
    IFS='|' read -r EXP_NAME SCALE G_H_DTYPE <<< "${cfg_line}"

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

        log_line ">>> phase15B exp=${EXP_NAME} seed=${SEED} scale=${SCALE} g_H_dtype=${G_H_DTYPE} epochs=${MAX_EPOCHS}"

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
            --scale "${SCALE}" \
            --g_H_dtype "${G_H_DTYPE}" \
            --seed "${SEED}" \
            > "${LOG_DIR}/train_stdout.log" 2>&1 || {
            log_line "!!! phase15B exp=${EXP_NAME} seed=${SEED} FAILED (see ${LOG_DIR}/train_stdout.log)"
            continue
        }

        touch "${DONE_FLAG}"
        log_line "<<< phase15B exp=${EXP_NAME} seed=${SEED} DONE"
    done
done

log_line "==============================================================="
log_line "BioTriplex 1B Phase 1.5-B runner EXITED"
log_line "==============================================================="
