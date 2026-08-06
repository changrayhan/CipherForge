#!/usr/bin/env bash
# bio_slg_runner.sh — BioTriplex 1B Phase 4: SLG-fixed encrypted training.
#
# What it does:
#   Stage 0 (once) : build BFV encrypted DB for Llama-3.2-1B (d=2048)
#                    → writes to ${BFV_CACHE_DIR}/bfv_ct_db_*.bin
#   Stage 1 × 3 seeds : three-party LoRA fine-tuning (DP alpha=0.15, 10 epochs)
#
# Output:
#   ${BIO_ROOT}/runs/slg/SLG-T_dpa15/seed_{42,123,2025}/
#
# Env overrides:
#   SEEDS       : "42,123,2025" (default)
#   BFV_CACHE   : default /root/autodl-tmp/slg-bfv-cache
#   BIO_ROOT    : default /root/autodl-tmp/CipherForgeCode/SLG-HE-PIR/test-data/BioTriplex1BTestData

set -euo pipefail

BIO_ROOT="${BIO_ROOT:-/root/autodl-tmp/CipherForgeCode/SLG-HE-PIR/test-data/BioTriplex1BTestData}"
SCRIPT_DIR="/root/autodl-tmp/CipherForgeCode/SLG-HE-PIR/src/scripts"
BFV_CACHE="${BFV_CACHE:-/root/autodl-tmp/slg-bfv-cache}"
DATA_PATH="${BIO_ROOT}/data"
LOG_ROOT="${BIO_ROOT}/runs/slg/_runner_logs"

SEEDS="${SEEDS:-42,123,2025}"
EPOCHS="${EPOCHS:-10}"

HF_MODEL="/root/autodl-tmp/CipherForgeCode/hf_cache/models--unsloth--Llama-3.2-1B/snapshots/9535bd9b1d1dea6acafbdc4813b728796aeb28da"

mkdir -p "${LOG_ROOT}"

ts_now() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts_now)] $*" | tee -a "${LOG_ROOT}/slg_runner.log"; }

# ---------------------------------------------------------------------------
# Stage 0 — build BFV encrypted DB once (d=2048 for Llama-3.2-1B)
# ---------------------------------------------------------------------------
log "============================================================="
log "BIO_ROOT=${BIO_ROOT}"
log "SLG Phase 4: SLG-fixed (encrypted, DP alpha=0.15, 10 epochs)"
log "============================================================="

BFV_DB_PATTERN="${BFV_CACHE}/bfv_ct_db_n128256_d2048_p*.bin"
EXISTING_BFV=$(ls ${BFV_DB_PATTERN} 2>/dev/null | head -1)

if [ -n "${EXISTING_BFV}" ]; then
    log "[Stage 0] BFV DB already exists — skipping Stage 0: ${EXISTING_BFV}"
    PK_PATH="${BFV_CACHE}/bfv_pk.bin"
else
    log "[Stage 0] Building BFV encrypted DB for d=2048 (Llama-3.2-1B) ..."
    log "[Stage 0] This takes ~3-5 min. First time only."
    cd /root/autodl-tmp/CipherForgeCode/SLG-HE-PIR

    PYTHONPATH=/root/autodl-tmp/CipherForgeCode/SLG-HE-PIR python3 \
        "${SCRIPT_DIR}/biotriplex_finetune.py" \
        --task_type classification \
        --stage 0 \
        --data_path "${DATA_PATH}" \
        --hf_model "${HF_MODEL}" \
        --bfv_cache_dir "${BFV_CACHE}" \
        --output_dir "${BIO_ROOT}/runs/slg/_stage0_tmp" \
        --hidden_dim 2048 \
        --u_layers 8 \
        --m_layers 8 \
        --vocab_size 128256 \
        --poly_degree 4096 \
        --plain_bits 30 \
        --scale 10000 \
        > "${LOG_ROOT}/slg_stage0_stdout.log" 2>&1 || {
        log "!!! Stage 0 FAILED — see ${LOG_ROOT}/slg_stage0_stdout.log"
        exit 1
    }
    log "[Stage 0] BFV DB built successfully."
    PK_PATH="${BFV_CACHE}/bfv_pk.bin"
fi

# ---------------------------------------------------------------------------
# Stage 1 — three-party LoRA fine-tuning × 3 seeds
# ---------------------------------------------------------------------------
for SEED in ${SEEDS//,/ }; do
    EXP_DIR="${BIO_ROOT}/runs/slg/SLG-T_dpa15/seed_${SEED}"
    LOG_DIR="${EXP_DIR}/logs"
    OUT_DIR="${EXP_DIR}/output"
    DONE_FLAG="${EXP_DIR}/DONE.flag"
    mkdir -p "${LOG_DIR}" "${OUT_DIR}"

    if [ -f "${DONE_FLAG}" ]; then
        log "[skip] SLG-T_dpa15 seed=${SEED} (DONE.flag exists)"
        continue
    fi

    log ">>> SLG-T_dpa15 seed=${SEED} epochs=${EPOCHS}"

    cd /root/autodl-tmp/CipherForgeCode/SLG-HE-PIR

    PYTHONPATH=/root/autodl-tmp/CipherForgeCode/SLG-HE-PIR python3 \
        "${SCRIPT_DIR}/biotriplex_finetune.py" \
        --task_type classification \
        --stage 1 \
        --data_path "${DATA_PATH}" \
        --hf_model "${HF_MODEL}" \
        --bfv_cache_dir "${BFV_CACHE}" \
        --output_dir "${OUT_DIR}" \
        --log_dir "${LOG_DIR}" \
        --max_epochs "${EPOCHS}" \
        --batch_size 1 \
        --max_seq_length 4096 \
        --learning_rate 5e-5 \
        --lora_rank 8 \
        --lora_alpha 16 \
        --lora_dropout 0.05 \
        --hidden_dim 2048 \
        --u_layers 8 \
        --m_layers 8 \
        --vocab_size 128256 \
        --poly_degree 4096 \
        --plain_bits 30 \
        --scale 10000 \
        --dp_enable \
        --dp_alpha 0.15 \
        --dp_answer_beta 0.5 \
        --dp_num_classes 7 \
        --seed "${SEED}" \
        > "${LOG_DIR}/slg_stdout.log" 2>&1 || {
        log "!!! SLG-T_dpa15 seed=${SEED} FAILED at stage 1 — see ${LOG_DIR}/slg_stdout.log"
        continue
    }

    # ---- Stage 2: evaluate trained adapter via subprocess evaluator ----
    log ">>> SLG-T_dpa15 seed=${SEED} stage 2 (evaluator)"
    PYTHONPATH=/root/autodl-tmp/CipherForgeCode/SLG-HE-PIR python3 \
        "${SCRIPT_DIR}/biotriplex_finetune.py" \
        --task_type classification \
        --stage 2 \
        --data_path "${DATA_PATH}" \
        --hf_model "${HF_MODEL}" \
        --bfv_cache_dir "${BFV_CACHE}" \
        --output_dir "${OUT_DIR}" \
        --log_dir "${LOG_DIR}" \
        --adapter_dir "${OUT_DIR}/adapter" \
        --eval_max_seq_length 1024 \
        --seed "${SEED}" \
        > "${LOG_DIR}/slg_eval_stdout.log" 2>&1 || {
        log "!!! SLG-T_dpa15 seed=${SEED} FAILED at stage 2 — see ${LOG_DIR}/slg_eval_stdout.log"
        continue
    }

    # ---- Adapter: convert SLG trainer schema → baseline schema ----
    EPOCH_JSONL="${LOG_DIR}/epoch_metrics.jsonl"
    STAGE2_PATTERN="${LOG_DIR}/genrel_*_evaluate_metrics.json"
    METRICS_HIST="${LOG_DIR}/metrics_history.json"
    if [ -f "${EPOCH_JSONL}" ]; then
        python3 "${SCRIPT_DIR}/slg_metrics_adapter.py" \
            --epoch_jsonl "${EPOCH_JSONL}" \
            --stage2_json "${STAGE2_PATTERN}" \
            --output "${METRICS_HIST}" \
            >> "${LOG_DIR}/slg_adapter_stdout.log" 2>&1 || {
            log "!!! SLG-T_dpa15 seed=${SEED} metrics adapter FAILED"
        }
    else
        log "!!! ${EPOCH_JSONL} missing — metrics adapter skipped"
    fi

    touch "${DONE_FLAG}"
    log "<<< SLG-T_dpa15 seed=${SEED} DONE"
done

log "============================================================="
log "BioTriplex SLG Phase 4 COMPLETE"
log "============================================================="
