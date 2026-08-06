#!/usr/bin/env bash
# bio_phase1E_runner.sh — BioTriplex 1B Phase 1E.
#
# Goal: fill the gaps in the Phase 1 DP-α curve so the inverted-U shape can be
#       fitted with reasonable density. Phase 1 currently samples α ∈ {0, 0.05,
#       0.15, 0.30, 0.50}. Phase 1E adds the missing mid-points so the curve
#       becomes α ∈ {0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50}.
#
# Design:
#   - Hold lora_target = "q,v" (same as the rest of Phase 1).
#   - Hold dp_answer_beta = 0.5.
#   - Add α ∈ {0.02, 0.10, 0.20} × 3 seeds = 9 runs.
#     (α=0.05 / 0.15 / 0.30 / 0.50 already in Phase 1; new configs limited
#      to the 3 point-fill values the user picked.)
#
# Output root: ${BIO_ROOT}/runs/dp_alpha_scan/
#
# Env overrides:
#   SEEDS       : "42,123,2025"
#   MAX_EPOCHS  : default 8

set -euo pipefail

BIO_ROOT="${BIO_ROOT:-/root/autodl-tmp/CipherForgeCode/SLG-HE-PIR/test-data/BioTriplex1BTestData}"
SCRIPT_DIR="${BIO_ROOT}/scripts"
LOG_ROOT="${BIO_ROOT}/runs/dp_alpha_scan/_runner_logs"

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
log_line "BioTriplex 1B Phase 1E (DP-α gap fill) STARTED"
log_line "SEEDS=${SEEDS} MAX_EPOCHS=${MAX_EPOCHS}"
log_line "BIO_ROOT=${BIO_ROOT} DATA_DIR=${DATA_DIR}"
log_line "==============================================================="

RUNS_ROOT="${BIO_ROOT}/runs/dp_alpha_scan"
mkdir -p "${RUNS_ROOT}"

# Config grid: 3 alpha values, q,v LoRA, dp_beta=0.5.
# Format: exp_name | dp_alpha
configs=(
    "B-T_dpa02|0.02"
    "B-T_dpa10|0.10"
    "B-T_dpa20|0.20"
)

for cfg_line in "${configs[@]}"; do
    IFS='|' read -r EXP_NAME DP_ALPHA <<< "${cfg_line}"
    EXP_NAME="$(echo "${EXP_NAME}" | xargs)"
    DP_ALPHA="$(echo "${DP_ALPHA}" | xargs)"

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

        log_line ">>> phase1E exp=${EXP_NAME} seed=${SEED} dp_alpha=${DP_ALPHA} epochs=${MAX_EPOCHS}"

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
            --dp_alpha "${DP_ALPHA}" \
            --dp_answer_beta 0.5 \
            --seed "${SEED}" \
            > "${LOG_DIR}/train_stdout.log" 2>&1 || {
            log_line "!!! phase1E exp=${EXP_NAME} seed=${SEED} FAILED (see ${LOG_DIR}/train_stdout.log)"
            continue
        }

        touch "${DONE_FLAG}"
        log_line "<<< phase1E exp=${EXP_NAME} seed=${SEED} DONE"
    done
done

log_line "==============================================================="
log_line "BioTriplex 1B Phase 1E runner EXITED"
[ -n "${CHAINED_SENTINEL:-}" ] && touch "$CHAINED_SENTINEL"
log_line "==============================================================="
