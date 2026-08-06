#!/usr/bin/env bash
# bio_phase1D_runner.sh — BioTriplex 1B Phase 1D.
#
# Goal: increase seed count for the Phase 1 baseline grid (8 configs) so the
#       DP-α inverted-U can be quantified with statistical power (n=3 -> n>=5).
#
# Design:
#   - Same 8 Phase-1 configs (B-T, B-T7, B-T_dpa05, B-T_dpa15, B-T_dpa30,
#     B-T_dpa50, B-T7_dpa15, B-T_ab_no_beta).
#   - New seeds: 7, 2026, 2027 (added to {42, 123, 2025} which are already done).
#   - We keep the old `runs/baseline/` directory unchanged; new seeds land in
#     a sibling subtree `runs/baseline_extra_seeds/` for bookkeeping.
#   Total new runs: 8 configs × 3 seeds = 24.
#
# Output root: ${BIO_ROOT}/runs/baseline_extra_seeds/
#
# Env overrides:
#   SEEDS       : "7,2026,2027"
#   MAX_EPOCHS  : default 8

set -euo pipefail

BIO_ROOT="${BIO_ROOT:-/root/autodl-tmp/CipherForgeCode/SLG-HE-PIR/test-data/BioTriplex1BTestData}"
SCRIPT_DIR="${BIO_ROOT}/scripts"
LOG_ROOT="${BIO_ROOT}/runs/baseline_extra_seeds/_runner_logs"

SEEDS="${SEEDS:-7,2026,2027}"
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
log_line "BioTriplex 1B Phase 1D (extra seeds for Phase 1) STARTED"
log_line "SEEDS=${SEEDS} MAX_EPOCHS=${MAX_EPOCHS}"
log_line "BIO_ROOT=${BIO_ROOT} DATA_DIR=${DATA_DIR}"
log_line "==============================================================="

RUNS_ROOT="${BIO_ROOT}/runs/baseline_extra_seeds"
mkdir -p "${RUNS_ROOT}"

# Config grid (8 Phase-1 baseline configs).
# Format: exp_name | lora_target | dp_alpha | dp_beta
configs=(
    "B-T           |q_proj,v_proj      |0.00|0.5"
    "B-T7          |q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj|0.00|0.5"
    "B-T_dpa05     |q_proj,v_proj      |0.05|0.5"
    "B-T_dpa15     |q_proj,v_proj      |0.15|0.5"
    "B-T_dpa30     |q_proj,v_proj      |0.30|0.5"
    "B-T_dpa50     |q_proj,v_proj      |0.50|0.5"
    "B-T7_dpa15    |q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj|0.15|0.5"
    "B-T_ab_no_beta|q_proj,v_proj      |0.15|0.0"
)

for cfg_line in "${configs[@]}"; do
    IFS='|' read -r EXP_NAME LORA_TARGET DP_ALPHA DP_BETA <<< "${cfg_line}"
    EXP_NAME="$(echo "${EXP_NAME}" | xargs)"
    LORA_TARGET="$(echo "${LORA_TARGET}" | xargs)"
    DP_ALPHA="$(echo "${DP_ALPHA}" | xargs)"
    DP_BETA="$(echo "${DP_BETA}" | xargs)"

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

        log_line ">>> phase1D exp=${EXP_NAME} seed=${SEED} lora=${LORA_TARGET} dp_alpha=${DP_ALPHA} dp_beta=${DP_BETA} epochs=${MAX_EPOCHS}"

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
            --lora_target "${LORA_TARGET}" \
            --lora_rank 8 \
            --lora_alpha 16 \
            --lora_dropout 0.05 \
            --dp_alpha "${DP_ALPHA}" \
            --dp_answer_beta "${DP_BETA}" \
            --seed "${SEED}" \
            > "${LOG_DIR}/train_stdout.log" 2>&1 || {
            log_line "!!! phase1D exp=${EXP_NAME} seed=${SEED} FAILED (see ${LOG_DIR}/train_stdout.log)"
            continue
        }

        touch "${DONE_FLAG}"
        log_line "<<< phase1D exp=${EXP_NAME} seed=${SEED} DONE"
    done
done

log_line "==============================================================="
log_line "BioTriplex 1B Phase 1D runner EXITED"
[ -n "${CHAINED_SENTINEL:-}" ] && touch "$CHAINED_SENTINEL"
log_line "==============================================================="
