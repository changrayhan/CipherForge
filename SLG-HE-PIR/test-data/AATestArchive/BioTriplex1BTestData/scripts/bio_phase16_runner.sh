#!/usr/bin/env bash
# bio_phase16_runner.sh — BioTriplex 1B Phase 1.6: Quantization ablation on gradients.
#
# What it does (REPLACES Phase 1.5 in-silico design, 2026-08-02):
#   scale ∈ {100, 1k, 10k, 100k}     (BFV round-trip quantization noise)
#   g_H_dtype ∈ {bf16, fp32, none}   (party_m gradient injection precision)
#   seeds = {42, 123, 2025}
#   + 1 control group: --quant_off (zero quant), 3 seeds
#   ============================
#   Total: 4 × 3 × 3 = 36 main runs + 3 control = 39 runs × ~12 min ≈ 7.8h
#
# Phase 1.5 (B-q-s*-bf16/fp32 in runs/quant/) was **inert** because scale noise
# and g_H_dtype cast were applied on the **loss scalar** (0-d leaf tensor):
#   * torch.randn_like(loss) is a leaf tensor with no dependency on params
#     → noise vanishes from d(loss)/dθ. gradient is bit-exact identical.
#   * bf16/fp32 cast of a single scalar (loss ≈ 1.0) is bit-exact — no effect.
#
# Phase 1.6 (runs/quant_v2/) uses `bio_baseline_trainer_v2.py` which moves
# both injections onto the **trainable gradients** after `loss.backward()`
# and before `clip_grad_norm_`, matching the DP-SGD convention used by
# `_add_dp_noise_to_grads`.
#
# Output:
#   ${BIO_ROOT}/runs/quant_v2/<exp>/seed_<N>/...
#
# Env overrides:
#   SEEDS          : "42,123,2025" (default)
#   MAX_EPOCHS     : override epoch count (default 8)
#   PHASES         : comma-separated list (default "1.6")

set -euo pipefail

BIO_ROOT="${BIO_ROOT:-/root/autodl-tmp/CipherForgeCode/SLG-HE-PIR/test-data/BioTriplex1BTestData}"
SCRIPT_DIR="${BIO_ROOT}/scripts"
LOG_ROOT="${BIO_ROOT}/runs/quant_v2/_runner_logs"
TRAINER_V2="${SCRIPT_DIR}/bio_baseline_trainer_v2.py"

PHASES="${PHASES:-1.6}"
SEEDS="${SEEDS:-42,123,2025}"
PHASE16_EPOCHS="${PHASE16_EPOCHS:-8}"

DATA_DIR="${BIO_ROOT}/data"
GOLD_TEST="${DATA_DIR}/test_gold_general_qa.txt"
HF_MODEL="/root/autodl-tmp/CipherForgeCode/hf_cache/models--unsloth--Llama-3.2-1B/snapshots/9535bd9b1d1dea6acafbdc4813b728796aeb28da"

mkdir -p "${LOG_ROOT}"

ts_now() { date '+%Y-%m-%d %H:%M:%S'; }
log_line() {
    echo "[$(ts_now)] $*" | tee -a "${LOG_ROOT}/runner.log"
}

log_line "==============================================================="
log_line "BioTriplex 1B Phase 1.6 (gradient-path quant ablation) STARTED"
log_line "PHASES=${PHASES} SEEDS=${SEEDS} PHASE16_EPOCHS=${PHASE16_EPOCHS}"
log_line "BIO_ROOT=${BIO_ROOT}  TRAINER=${TRAINER_V2}"
log_line "==============================================================="

# Sanity check: trainer v2 must exist and be importable
if [ ! -f "${TRAINER_V2}" ]; then
    log_line "!!! ERROR: trainer v2 not found at ${TRAINER_V2}"
    exit 1
fi
python3 -c "import ast; ast.parse(open('${TRAINER_V2}').read())" || {
    log_line "!!! ERROR: trainer v2 has syntax errors"; exit 1;
}
log_line "[ok] trainer v2 syntax check passed"


run_phase16() {
    local EPOCHS=$PHASE16_EPOCHS
    local RUNS_ROOT="${BIO_ROOT}/runs/quant_v2"
    mkdir -p "${RUNS_ROOT}"

    # Config grid: 4 scales × 3 dtypes = 12 main experiments + 1 control × 3 seeds.
    # Format: exp_name|scale|g_H_dtype|quant_off_flag
    local configs=(
        # --- main grid: scale × g_H_dtype ---
        "B-q16-s100-bf16|100|bf16|0"
        "B-q16-s1k-bf16|1000|bf16|0"
        "B-q16-s10k-bf16|10000|bf16|0"
        "B-q16-s100k-bf16|100000|bf16|0"
        "B-q16-s100-fp32|100|fp32|0"
        "B-q16-s1k-fp32|1000|fp32|0"
        "B-q16-s10k-fp32|10000|fp32|0"
        "B-q16-s100k-fp32|100000|fp32|0"
        "B-q16-s100-none|100|none|0"
        "B-q16-s1k-none|1000|none|0"
        "B-q16-s10k-none|10000|none|0"
        "B-q16-s100k-none|100000|none|0"
        # --- control group: quant fully off ---
        "B-q16-control-quant_off|10000|none|1"
    )

    for cfg_line in "${configs[@]}"; do
        IFS='|' read -r EXP_NAME SCALE G_H_DTYPE QUANT_OFF <<< "${cfg_line}"

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

            # Build the --quant_off flag conditionally
            local QOFF_FLAG=""
            if [ "${QUANT_OFF}" = "1" ]; then
                QOFF_FLAG="--quant_off"
            fi

            log_line ">>> quant_v2 exp=${EXP_NAME} seed=${SEED} scale=${SCALE} g_H_dtype=${G_H_DTYPE} quant_off=${QUANT_OFF} epochs=${EPOCHS}"

            cd /root/autodl-tmp/CipherForgeCode/SLG-HE-PIR

            PYTHONPATH=/root/autodl-tmp/CipherForgeCode/SLG-HE-PIR python3 \
                "${TRAINER_V2}" \
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
                ${QOFF_FLAG} \
                --seed "${SEED}" \
                > "${LOG_DIR}/train_stdout.log" 2>&1 || {
                log_line "!!! quant_v2 exp=${EXP_NAME} seed=${SEED} FAILED (see ${LOG_DIR}/train_stdout.log)"
                continue
            }

            touch "${DONE_FLAG}"
            log_line "<<< quant_v2 exp=${EXP_NAME} seed=${SEED} DONE"
        done
    done
}

for PHASE in ${PHASES//,/ }; do
    case "${PHASE}" in
        1.6)
            log_line "===== PHASE 1.6 START ====="
            run_phase16
            log_line "===== PHASE 1.6 DONE ====="
            ;;
        *)
            log_line "!!! unknown phase: ${PHASE}"
            ;;
    esac
done

log_line "==============================================================="
log_line "BioTriplex 1B Phase 1.6 runner EXITED"
log_line "==============================================================="