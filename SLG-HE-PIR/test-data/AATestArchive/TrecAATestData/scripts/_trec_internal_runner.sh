#!/usr/bin/env bash
# _trec_internal_runner.sh — internal scheduler for TREC-QC accuracy ablation.
#
# Drives the full pipeline for the TREC-QC / Llama-3.2-1B migration:
#   Phase 1 (baseline): 9 experiments × 3 seeds × 5 epochs on plaintext
#   Phase 4 (SLG):      SLG-fixed with BFV encryption (1 experiment × 3 seeds × 10 epochs)
#
# All outputs go to ${TREC_ROOT:-/root/autodl-tmp/CipherForgeCode/SLG-HE-PIR/test-data/TrecAATestData}
#
# NOTE: This script is meant to be invoked from `run_trec_full_background.sh`
# so that it can be detached via setsid/nohup. It writes to `${LOG_ROOT}/runner.log`.
#
# Env overrides:
#   PHASES    : "1,4"  (default both)
#   SEEDS     : "42,123,2025"  (default)
#   EXPERIMENT_SET : "full" | "smoke"  (default full)

set -euo pipefail

TREC_ROOT="${TREC_ROOT:-/root/autodl-tmp/CipherForgeCode/SLG-HE-PIR/test-data/TrecAATestData}"
SCRIPT_DIR="${TREC_ROOT}/scripts"
LOG_ROOT="${TREC_ROOT}/runs/baseline/_runner_logs"
mkdir -p "${LOG_ROOT}"

PHASES="${PHASES:-1,4}"
SEEDS="${SEEDS:-42,123,2025}"
EXPERIMENT_SET="${EXPERIMENT_SET:-full}"

ts_now() { date '+%Y-%m-%d %H:%M:%S'; }
log_line() {
    echo "[$(ts_now)] $*" | tee -a "${LOG_ROOT}/runner.log"
}

log_line "==============================================================="
log_line "TREC-QC accuracy-ablation runner STARTED"
log_line "PHASES=${PHASES} SEEDS=${SEEDS} EXPERIMENT_SET=${EXPERIMENT_SET}"
log_line "TREC_ROOT=${TREC_ROOT}"
log_line "==============================================================="

# --- Shared paths ---
REPO="/root/autodl-tmp/CipherForgeCode/SLG-HE-PIR"
SNAPSHOT_DIR=$(ls /root/autodl-tmp/CipherForgeCode/hf_cache/models--unsloth--Llama-3.2-1B/snapshots/ 2>/dev/null | head -1)
HF_MODEL="${HF_MODEL:-/root/autodl-tmp/CipherForgeCode/hf_cache/models--unsloth--Llama-3.2-1B/snapshots/${SNAPSHOT_DIR}}"
TREC_DATA_DIR="/root/autodl-tmp/CipherForgeCode/SLG-HE-PIR/datasets/trec-qc"
GOLD_FILE="${TREC_ROOT}/gold/test_gold_general_qa.txt"

# --- Run a single baseline experiment ---
# Args: $1=experiment_name $2=lora_target $3=dp_alpha $4=dp_beta $5=seed
run_one_baseline() {
    local exp_name="$1"
    local lora_target="$2"
    local dp_alpha="$3"
    local dp_beta="$4"
    local seed="$5"
    local max_epochs=5
    if [[ "${EXPERIMENT_SET}" == "smoke" ]]; then
        max_epochs=1
    fi

    local exp_dir="${TREC_ROOT}/runs/baseline/${exp_name}/seed${seed}"
    local log_dir="${exp_dir}/logs"
    mkdir -p "${exp_dir}" "${log_dir}"

    if [[ -f "${exp_dir}/DONE.flag" ]]; then
        log_line "[skip] ${exp_name}/seed${seed} already done"
        return 0
    fi

    log_line ">>> baseline exp=${exp_name} seed=${seed} lora=${lora_target} dp_alpha=${dp_alpha} dp_beta=${dp_beta}"

    cd "${REPO}"
    python3 "${SCRIPT_DIR}/trec_baseline_trainer.py" \
        --data_dir "${TREC_DATA_DIR}" \
        --gold_path "${GOLD_FILE}" \
        --hf_model "${HF_MODEL}" \
        --output_dir "${exp_dir}" \
        --log_dir "${log_dir}" \
        --max_epochs ${max_epochs} \
        --batch_size 8 \
        --max_seq_length 256 \
        --learning_rate 1e-4 \
        --weight_decay 0.0 \
        --warmup_steps 50 \
        --gradient_clip_norm 1.0 \
        --lora_rank 8 \
        --lora_alpha 16 \
        --lora_dropout 0.05 \
        --lora_target "${lora_target}" \
        --seed ${seed} \
        --dp_alpha ${dp_alpha} \
        --dp_answer_beta ${dp_beta} \
        --dp_calibration_steps 5 \
        2>&1 | tee -a "${log_dir}/runner.log"

    if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
        log_line "ERROR: ${exp_name}/seed${seed} failed"
        return 1
    fi

    touch "${exp_dir}/DONE.flag"
    log_line "<<< baseline exp=${exp_name} seed=${seed} DONE"
}

# --- Phase 1: 9 baseline experiments × 3 seeds ---
declare -a BASELINE_EXPS=(
    # name           lora_target       dp_alpha   dp_beta
    "B-T_se_2target    q_proj,v_proj               0.0  0.5"
    "B-T7_se_7target   q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj  0.0  0.5"
    "B-T_dpa00         q_proj,v_proj               0.00 0.5"
    "B-T_dpa05         q_proj,v_proj               0.05 0.5"
    "B-T_dpa15         q_proj,v_proj               0.15 0.5"
    "B-T_dpa30         q_proj,v_proj               0.30 0.5"
    "B-T_dpa50         q_proj,v_proj               0.50 0.5"
    "B-T7_dpa15        q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj  0.15 0.5"
    "B-T_ab_no_beta    q_proj,v_proj               0.15 0.0"
)

if [[ "${PHASES}" == *1* ]]; then
    log_line "###### PHASE 1: BASELINE (9 experiments × ${SEEDS//,/+} seeds) ######"
    IFS=',' read -ra SEED_ARR <<< "${SEEDS}"
    for cfg in "${BASELINE_EXPS[@]}"; do
        # Parse "name target alpha beta"
        read -r name target alpha beta <<< "${cfg}"
        for seed in "${SEED_ARR[@]}"; do
            run_one_baseline "${name}" "${target}" "${alpha}" "${beta}" "${seed}" \
                || { log_line "PHASE 1 ABORT"; exit 1; }
        done
    done
    log_line "###### PHASE 1 COMPLETE ######"
fi

# --- Phase 4: SLG-fixed (1 experiment × 3 seeds × 10 epochs) ---
if [[ "${PHASES}" == *4* ]]; then
    log_line "###### PHASE 4: SLG-FIXED (1 experiment × ${SEEDS//,/+} seeds × 10 epochs) ######"
    IFS=',' read -ra SEED_ARR <<< "${SEEDS}"
    for seed in "${SEED_ARR[@]}"; do
        exp_name="SLG-T_dpa15"
        exp_dir="${TREC_ROOT}/runs/slg/${exp_name}/seed${seed}"
        log_dir="${exp_dir}/logs"
        adapter_dir="${exp_dir}/adapter"
        mkdir -p "${exp_dir}" "${log_dir}" "${adapter_dir}"
        if [[ -f "${exp_dir}/DONE.flag" ]]; then
            log_line "[skip] SLG ${exp_name}/seed${seed} already done"
            continue
        fi
        log_line ">>> SLG exp=${exp_name} seed=${seed}"
        cd "${REPO}"
        python3 src/scripts/biotriplex_finetune.py \
            --task_type trec-qc \
            --stage all \
            --data_path "${TREC_DATA_DIR}" \
            --hf_model "${HF_MODEL}" \
            --bfv_cache_dir /root/autodl-tmp/CipherForgeCode/slg-bfv-cache-trec \
            --output_dir "${exp_dir}" \
            --log_dir "${log_dir}" \
            --adapter_dir "${adapter_dir}" \
            --max_epochs 10 \
            --batch_size 1 \
            --max_seq_length 256 \
            --learning_rate 1e-4 \
            --weight_decay 0.0 \
            --warmup_steps 50 \
            --gradient_clip_norm 1.0 \
            --lora_rank 8 \
            --lora_alpha 16 \
            --lora_dropout 0.05 \
            --lora_target "q_proj,v_proj" \
            --use_chunked_pipeline True \
            --chunk_tokens 256 \
            --seed ${seed} \
            --log_freq 10 \
            --save_freq 5 \
            --do_test_eval \
            --dp_enable \
            --dp_alpha 0.15 \
            --dp_answer_beta 0.5 \
            --dp_calibration_steps 5 \
            --dp_num_classes 6 \
            --scale 10000 \
            --vocab_size 128256 \
            --hidden_dim 2048 \
            --poly_degree 4096 \
            --plain_bits 30 \
            --lam 80 \
            --u_layers 8 \
            --m_layers 8 \
            2>&1 | tee -a "${log_dir}/runner.log"
        if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
            log_line "ERROR: SLG ${exp_name}/seed${seed} failed"
            exit 1
        fi
        touch "${exp_dir}/DONE.flag"
        log_line "<<< SLG ${exp_name}/seed${seed} DONE"
    done
    log_line "###### PHASE 4 COMPLETE ######"
fi

log_line "==============================================================="
log_line "ALL DONE"
log_line "==============================================================="