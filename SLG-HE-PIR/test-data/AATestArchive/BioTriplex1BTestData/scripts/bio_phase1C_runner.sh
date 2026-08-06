#!/usr/bin/env bash
# bio_phase1C_runner.sh — BioTriplex 1B Phase 1C: cumulative module ablation.
#
# Goal: Quantify each protocol-stack module's marginal accuracy contribution
#       by stepping through the full pipeline one module at a time.
#
# Modules (and their knobs):
#   - DP noise           : --dp_alpha
#   - BFV round-trip tax : --scale     (scale=0  -> off; otherwise round-trip noise std = 1/(2*scale))
#   - bf16 injection tax : --g_H_dtype (bf16 -> on; fp32 -> off)
#
# Pipeline stacking order matches production CipherForge path:
#       loss cast  ->  scale round-trip noise  ->  DP gradient noise
#
# 6 cumulative configurations × 3 seeds (42, 123, 2025) = 18 runs.
#
# Config name | dp_alpha | scale   | g_H_dtype  | meaning
# ----------|----------|---------|-----------|-------------------------------------
# B-Base     | 0.00     | 0       | fp32      | clean LoRA baseline (control)
# B-qOnly    | 0.00     | 10000   | fp32      | + quantization only (default scale)
# B-dOnly    | 0.15     | 0       | fp32      | + DP only
# B-dqOnly   | 0.15     | 10000   | fp32      | + DP + quantization
# B-dhOnly   | 0.15     | 0       | bf16      | + DP + bf16 injection
# B-dqh      | 0.15     | 10000   | bf16      | + DP + quantization + bf16 (= protocol stack)
#
# Note: scale=0 is wired to the "skip round-trip tax" branch (see
# bio_baseline_trainer.py line ~190: ``if args.scale > 0``).
#
# Output root: ${BIO_ROOT}/runs/cumulative/
#
# Env overrides:
#   SEEDS       : "42,123,2025"
#   MAX_EPOCHS  : default 8

set -euo pipefail

BIO_ROOT="${BIO_ROOT:-/root/autodl-tmp/CipherForgeCode/SLG-HE-PIR/test-data/BioTriplex1BTestData}"
SCRIPT_DIR="${BIO_ROOT}/scripts"
LOG_ROOT="${BIO_ROOT}/runs/cumulative/_runner_logs"

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
log_line "BioTriplex 1B Phase 1C (cumulative ablation) STARTED"
log_line "SEEDS=${SEEDS} MAX_EPOCHS=${MAX_EPOCHS}"
log_line "BIO_ROOT=${BIO_ROOT} DATA_DIR=${DATA_DIR}"
log_line "==============================================================="

RUNS_ROOT="${BIO_ROOT}/runs/cumulative"
mkdir -p "${RUNS_ROOT}"

# Config grid: 6 cumulative configurations.
# Format: exp_name | dp_alpha | scale | g_H_dtype
configs=(
    "B-Base  |0.00|0     |fp32"
    "B-qOnly |0.00|10000 |fp32"
    "B-dOnly |0.15|0     |fp32"
    "B-dqOnly|0.15|10000 |fp32"
    "B-dhOnly|0.15|0     |bf16"
    "B-dqh   |0.15|10000 |bf16"
)

for cfg_line in "${configs[@]}"; do
    # Trim leading whitespace because the heredoc config lines have padding.
    cfg_line="${cfg_line## }"
    IFS='|' read -r EXP_NAME DP_ALPHA SCALE G_H_DTYPE <<< "${cfg_line}"
    # Trim each field.
    EXP_NAME="$(echo "${EXP_NAME}" | xargs)"
    DP_ALPHA="$(echo "${DP_ALPHA}" | xargs)"
    SCALE="$(echo "${SCALE}" | xargs)"
    G_H_DTYPE="$(echo "${G_H_DTYPE}" | xargs)"

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

        log_line ">>> phase1C exp=${EXP_NAME} seed=${SEED} dp_alpha=${DP_ALPHA} scale=${SCALE} g_H_dtype=${G_H_DTYPE} epochs=${MAX_EPOCHS}"

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
            --scale "${SCALE}" \
            --g_H_dtype "${G_H_DTYPE}" \
            --seed "${SEED}" \
            > "${LOG_DIR}/train_stdout.log" 2>&1 || {
            log_line "!!! phase1C exp=${EXP_NAME} seed=${SEED} FAILED (see ${LOG_DIR}/train_stdout.log)"
            continue
        }

        touch "${DONE_FLAG}"
        log_line "<<< phase1C exp=${EXP_NAME} seed=${SEED} DONE"
    done
done

log_line "==============================================================="
log_line "BioTriplex 1B Phase 1C runner EXITED"
[ -n "${CHAINED_SENTINEL:-}" ] && touch "$CHAINED_SENTINEL"
log_line "==============================================================="
