#!/usr/bin/env bash
# bio_quant_runner.sh — BioTriplex 1B Phase 1.5: Intermediate quantization level ablation.
#
# What it does:
#   Single control variable: cfg.scale ∈ {100, 1000, 10000, 100000} (round-trip quantization)
#   Cross variable:         g_H_dtype ∈ {bf16, fp32}            (post-decimal injection precision)
#   ============================
#   Total: 4 × 2 = 8 configs × 3 seeds × 8 epochs = 24 training runs.
#
# Each config is a separate plaintext Baseline run that simulates the BFV quantization
# path WITHOUT actual BFV encryption. The vector V_y gets the same `scale`-rounded
# treatment as in the BFV path, but the linear algebra stays in cleartext (fast).
#
# Output:
#   ${BIO_ROOT}/runs/quant/<exp>/seed_<N>/...
#
# Env overrides:
#   SEEDS          : "42,123,2025" (default)
#   MAX_EPOCHS     : override epoch count (default 8)
#   PHASES         : comma-separated list of sub-phases (default "1.5")

set -euo pipefail

BIO_ROOT="${BIO_ROOT:-/root/autodl-tmp/CipherForgeCode/SLG-HE-PIR/test-data/BioTriplex1BTestData}"
SCRIPT_DIR="${BIO_ROOT}/scripts"
LOG_ROOT="${BIO_ROOT}/runs/quant/_runner_logs"

PHASES="${PHASES:-1.5}"
SEEDS="${SEEDS:-42,123,2025}"
PHASE15_EPOCHS="${PHASE15_EPOCHS:-8}"

DATA_DIR="${BIO_ROOT}/data"
GOLD_TEST="${DATA_DIR}/test_gold_general_qa.txt"
HF_MODEL="/root/autodl-tmp/CipherForgeCode/hf_cache/models--unsloth--Llama-3.2-1B/snapshots/9535bd9b1d1dea6acafbdc4813b728796aeb28da"

mkdir -p "${LOG_ROOT}"

ts_now() { date '+%Y-%m-%d %H:%M:%S'; }
log_line() {
    echo "[$(ts_now)] $*" | tee -a "${LOG_ROOT}/runner.log"
}

log_line "==============================================================="
log_line "BioTriplex 1B Phase 1.5 (quantization ablation) STARTED"
log_line "PHASES=${PHASES} SEEDS=${SEEDS} PHASE15_EPOCHS=${PHASE15_EPOCHS}"
log_line "BIO_ROOT=${BIO_ROOT} DATA_DIR=${DATA_DIR}"
log_line "==============================================================="

run_phase15() {
    local EPOCHS=$PHASE15_EPOCHS
    local RUNS_ROOT="${BIO_ROOT}/runs/quant"
    mkdir -p "${RUNS_ROOT}"

    # Config grid: 4 scales × 2 g_H_dtype = 8 experiments.
    # Format: exp_name | scale | g_H_dtype
    local configs=(
        "B-q-s100-bf16|100|bf16"
        "B-q-s1k-bf16|1000|bf16"
        "B-q-s10k-bf16|10000|bf16"
        "B-q-s100k-bf16|100000|bf16"
        "B-q-s100-fp32|100|fp32"
        "B-q-s1k-fp32|1000|fp32"
        "B-q-s10k-fp32|10000|fp32"
        "B-q-s100k-fp32|100000|fp32"
    )

    for cfg_line in "${configs[@]}"; do
        IFS='|' read -r EXP_NAME SCALE G_H_DTYPE <<< "${cfg_line}"

        for SEED in ${SEEDS//,/ }; do
            local EXP_DIR="${RUNS_ROOT}/${EXP_NAME}/seed_${SEED}"
            local LOG_DIR="${EXP_DIR}/logs"
            local OUT_DIR="${EXP_DIR}/output"
            local DONE_FLAG="${EXP_DIR}/DONE.flag"
            mkdir -p "${LOG_DIR}" "${OUT_DIR}"

            if [ -f "${DONE_FLAG}" ]; then
                log_line "[skip] ${EXP_NAME} seed=${SEED} (DONE.flag exists)"
                continue
            fi

            log_line ">>> quant exp=${EXP_NAME} seed=${SEED} scale=${SCALE} g_H_dtype=${G_H_DTYPE} epochs=${EPOCHS}"

            cd /root/autodl-tmp/CipherForgeCode/SLG-HE-PIR

            PYTHONPATH=/root/autodl-tmp/CipherForgeCode/SLG-HE-PIR python3 \
                ${SCRIPT_DIR}/bio_baseline_trainer.py \
                --data_dir "${DATA_DIR}" \
                --gold_path "${GOLD_TEST}" \
                --hf_model "${HF_MODEL}" \
                --output_dir "${OUT_DIR}" \
                --log_dir "${LOG_DIR}" \
                --max_epochs "${EPOCHS}" \
                --batch_size 1 \
                --max_seq_length 1024 \
                --learning_rate 5e-5 \
                --lora_target "q_proj,v_proj" \
                --lora_rank 8 \
                --lora_alpha 16 \
                --lora_dropout 0.05 \
                --dp_alpha 0.0 \
                --dp_answer_beta 0.5 \
                --scale "${SCALE}" \
                --g_H_dtype "${G_H_DTYPE}" \
                --seed "${SEED}" \
                > "${LOG_DIR}/train_stdout.log" 2>&1 || {
                log_line "!!! quant exp=${EXP_NAME} seed=${SEED} FAILED (see ${LOG_DIR}/train_stdout.log)"
                continue
            }

            touch "${DONE_FLAG}"
            log_line "<<< quant exp=${EXP_NAME} seed=${SEED} DONE"
        done
    done
}

for PHASE in ${PHASES//,/ }; do
    case "${PHASE}" in
        1.5)
            log_line "===== PHASE 1.5 START ====="
            run_phase15
            log_line "===== PHASE 1.5 DONE ====="
            ;;
        *)
            log_line "!!! unknown phase: ${PHASE}"
            ;;
    esac
done

log_line "==============================================================="
log_line "BioTriplex 1B Phase 1.5 runner EXITED"
log_line "==============================================================="