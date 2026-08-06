#!/usr/bin/env bash
# _bio_internal_runner.sh — BioTriplex 1B accuracy ablation runner.
#
# Pipeline for the BioTriplex / Llama-3.2-1B migration:
#   Phase 1: plaintext LoRA baseline with DP noise scans (5 + 2 ablation configs)
#            × 3 seeds × 8 epochs.
#   Phase 4: full SLG-HE-PIR encrypted protocol path (1 config × 3 seeds × 10 epochs).
#
# Outputs:
#   ${BIO_ROOT:-/root/autodl-tmp/CipherForgeCode/SLG-HE-PIR/test-data/BioTriplex1BTestData}/runs/baseline/<exp>/seed_<N>/...
#   ${BIO_ROOT:-...}/runs/slg/<exp>/seed_<N>/...
#
# Env overrides:
#   PHASES     : "1,4"   (default both)
#   SEEDS      : "42,123,2025"  (default)
#   MAX_EPOCHS : override epoch count (default 8 for phase 1, 10 for phase 4)

set -euo pipefail

BIO_ROOT="${BIO_ROOT:-/root/autodl-tmp/CipherForgeCode/SLG-HE-PIR/test-data/BioTriplex1BTestData}"
SCRIPT_DIR="${BIO_ROOT}/scripts"
LOG_ROOT="${BIO_ROOT}/runs/baseline/_runner_logs"
mkdir -p "${LOG_ROOT}"

PHASES="${PHASES:-1,4}"
SEEDS="${SEEDS:-42,123,2025}"
PHASE1_EPOCHS="${PHASE1_EPOCHS:-8}"
PHASE4_EPOCHS="${PHASE4_EPOCHS:-10}"
LORA_TARGET_SL7="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
LORA_TARGET_DEFAULT="q_proj,v_proj"

DATA_DIR="${BIO_ROOT}/data"
GOLD_TEST="${DATA_DIR}/test_gold_general_qa.txt"
HF_MODEL_DEFAULT="/root/autodl-tmp/CipherForgeCode/hf_cache/models--unsloth--Llama-3.2-1B/snapshots/9535bd9b1d1dea6acafbdc4813b728796aeb28da"

ts_now() { date '+%Y-%m-%d %H:%M:%S'; }
log_line() {
    echo "[$(ts_now)] $*" | tee -a "${LOG_ROOT}/runner.log"
}

log_line "==============================================================="
log_line "BioTriplex 1B accuracy-ablation runner STARTED"
log_line "PHASES=${PHASES} SEEDS=${SEEDS} PHASE1_EPOCHS=${PHASE1_EPOCHS} PHASE4_EPOCHS=${PHASE4_EPOCHS}"
log_line "BIO_ROOT=${BIO_ROOT} DATA_DIR=${DATA_DIR}"
log_line "==============================================================="

# Phase 1: plaintext baseline (LoRA + optional DP proxy)
run_phase1() {
    local EPOCHS=$PHASE1_EPOCHS
    local RUNS_ROOT="${BIO_ROOT}/runs/baseline"
    mkdir -p "${RUNS_ROOT}"

    # Format: exp_name | lora_target | dp_alpha | dp_beta | extra_args
    local configs=(
        "B-T           |q_proj,v_proj      |0.00|0.5|"
        "B-T7          |q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj|0.00|0.5|"
        "B-T_dpa05     |q_proj,v_proj      |0.05|0.5|"
        "B-T_dpa15     |q_proj,v_proj      |0.15|0.5|"
        "B-T_dpa30     |q_proj,v_proj      |0.30|0.5|"
        "B-T_dpa50     |q_proj,v_proj      |0.50|0.5|"
        "B-T7_dpa15    |q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj|0.15|0.5|"
        "B-T_ab_no_beta|q_proj,v_proj      |0.15|0.0|"
    )

    IFS='|' read -ra parts <<< "B-T | q,v | 0.00 | 0.5 |"

    for cfg_line in "${configs[@]}"; do
        IFS='|' read -r EXP_NAME LORA_TARGET DP_ALPHA DP_BETA _ <<< "${cfg_line}"
        EXP_NAME=$(echo "${EXP_NAME}" | xargs)  # trim
        LORA_TARGET=$(echo "${LORA_TARGET}" | xargs)  # trim

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

            log_line ">>> baseline exp=${EXP_NAME} seed=${SEED} lora=${LORA_TARGET} dp_alpha=${DP_ALPHA} dp_beta=${DP_BETA} epochs=${EPOCHS}"

            cd /root/autodl-tmp/CipherForgeCode/SLG-HE-PIR

            PYTHONPATH=/root/autodl-tmp/CipherForgeCode/SLG-HE-PIR python3 \
                ${SCRIPT_DIR}/bio_baseline_trainer.py \
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
                --seed "${SEED}" \
                > "${LOG_DIR}/train_stdout.log" 2>&1 || {
                log_line "!!! baseline exp=${EXP_NAME} seed=${SEED} FAILED (see ${LOG_DIR}/train_stdout.log)"
                continue
            }

            touch "${DONE_FLAG}"
            log_line "<<< baseline exp=${EXP_NAME} seed=${SEED} DONE"
        done
    done
}

# Phase 4: full SLG-HE-PIR encrypted protocol path.
# NOTE: Phase 4 requires the full BFV/S3PIR pipeline — it is much slower than
# phase 1 (each step goes through encrypted V mat-vec). We invoke the existing
# biotriplex_finetune.py entrypoint with task_type=classification.
run_phase4() {
    # Phase 4 参数与 Phase 1.5 (B-T_ab_no_beta) 对齐:
    #   max_seq_length=1024 (与 baseline 一致, 避免 4096 的 4x padding 开销)
    #   epochs=8 (与 baseline 一致)
    #   seed=42 (单 seed 测试, 3 seed 留后续)
    #   dp_alpha=0.15, dp_beta=0.5 (与 Phase 1.5 B-T_ab_no_beta 一致)
    local EPOCHS=8
    local MAX_SEQ=1024
    local RUNS_ROOT="${BIO_ROOT}/runs/slg"
    mkdir -p "${RUNS_ROOT}"

    local EXP_NAME="SLG-T_dpa15"
    local LORA_TARGET="q_proj,v_proj"
    local DP_ALPHA="0.15"
    local DP_BETA="0.5"
    local SEED="42"

    local EXP_DIR="${RUNS_ROOT}/${EXP_NAME}/seed_${SEED}"
    local LOG_DIR="${EXP_DIR}/logs"
    local OUT_DIR="${EXP_DIR}/output"
    local DONE_FLAG="${EXP_DIR}/DONE.flag"
    mkdir -p "${LOG_DIR}" "${OUT_DIR}"

    if [ -f "${DONE_FLAG}" ]; then
        log_line "[skip] SLG exp=${EXP_NAME} seed=${SEED} (DONE.flag exists)"
        return
    fi

    log_line ">>> SLG exp=${EXP_NAME} seed=${SEED} lora=${LORA_TARGET} dp_alpha=${DP_ALPHA} epochs=${EPOCHS} max_seq=${MAX_SEQ}"

    cd /root/autodl-tmp/CipherForgeCode/SLG-HE-PIR

    PYTHONPATH=/root/autodl-tmp/CipherForgeCode/SLG-HE-PIR python3 \
        /root/autodl-tmp/CipherForgeCode/SLG-HE-PIR/src/scripts/biotriplex_finetune.py \
        --task_type classification \
        --stage all \
        --data_path "${DATA_DIR}" \
        --hf_model "${HF_MODEL:-${HF_MODEL_DEFAULT}}" \
        --output_dir "${OUT_DIR}" \
        --log_dir "${LOG_DIR}" \
        --max_epochs "${EPOCHS}" \
        --batch_size 1 \
        --max_seq_length "${MAX_SEQ}" \
        --learning_rate 5e-5 \
        --lora_rank 8 \
        --lora_alpha 16 \
        --lora_dropout 0.05 \
        --u_layers 8 \
        --m_layers 8 \
        --dp_enable \
        --dp_alpha "${DP_ALPHA}" \
        --dp_answer_beta "${DP_BETA}" \
        --dp_num_classes 7 \
        --seed "${SEED}" \
        > "${LOG_DIR}/slg_stdout.log" 2>&1 || {
        log_line "!!! SLG exp=${EXP_NAME} seed=${SEED} FAILED (see ${LOG_DIR}/slg_stdout.log)"
        return
    }

    touch "${DONE_FLAG}"
    log_line "<<< SLG exp=${EXP_NAME} seed=${SEED} DONE"
}

# Dispatch
for PHASE in ${PHASES//,/ }; do
    case "${PHASE}" in
        1)
            log_line "===== PHASE 1 START ====="
            run_phase1
            log_line "===== PHASE 1 DONE ====="
            ;;
        4)
            log_line "===== PHASE 4 START ====="
            run_phase4
            log_line "===== PHASE 4 DONE ====="
            ;;
        *)
            log_line "!!! unknown phase: ${PHASE}"
            ;;
    esac
done

log_line "==============================================================="
log_line "BioTriplex 1B runner EXITED"
log_line "==============================================================="