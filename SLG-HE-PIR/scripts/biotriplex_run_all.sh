#!/bin/bash
# scripts/biotriplex_run_all.sh
# Run both BioTriplex tasks sequentially (classification → generation).
# Mirrors baseline/run_all.sh but routes through SLG-HE-PIR.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_ROOT="${REPO_ROOT}/baseline/_run_all_logs"
mkdir -p "${LOG_ROOT}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="${LOG_ROOT}/biotriplex_run_all_${TIMESTAMP}.log"

# OOM mitigation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset OMP_NUM_THREADS

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "${RUN_LOG}"
}

log "========== Starting SLG-HE-PIR BioTriplex Pipeline =========="
log "  Task A (classification_genrel): 6 epochs  (paper value)"
log "  Task B (generation_ner)        : 10 epochs (paper value)"
log "  Run log: ${RUN_LOG}"

# ----------------------------------------------------------------------------
# Task A: Classification
# ----------------------------------------------------------------------------
log ">>> [1/2] Starting Classification (GenRel QA) ..."
cd "${REPO_ROOT}"
bash scripts/biotriplex_classification_genrel.sh 2>&1 | tee -a "${RUN_LOG}"
CLASSIFY_EXIT=${PIPESTATUS[0]}
if [ ${CLASSIFY_EXIT} -ne 0 ]; then
    log "ERROR: Classification task failed with exit code ${CLASSIFY_EXIT}"
    exit 1
fi
log ">>> [1/2] Classification done."

# ----------------------------------------------------------------------------
# Task B: Generation (NER)
# ----------------------------------------------------------------------------
log ">>> [2/2] Starting Generation (NER) ..."
cd "${REPO_ROOT}"
bash scripts/biotriplex_generation_ner.sh 2>&1 | tee -a "${RUN_LOG}"
GENERATE_EXIT=${PIPESTATUS[0]}
if [ ${GENERATE_EXIT} -ne 0 ]; then
    log "ERROR: Generation task failed with exit code ${GENERATE_EXIT}"
    exit 1
fi
log ">>> [2/2] Generation done."

# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------
log "========== All BioTriplex Fine-tuning Completed =========="
log "Metrics JSON files:"
log "  classification: ${REPO_ROOT}/baseline/classification_genrel/logs/genrel_*_evaluate_metrics.json"
log "  generation    : ${REPO_ROOT}/baseline/generation_ner/logs/ner_*_evaluate_metrics.json"
log ""
log "Per-task logs:"
log "  classification: ${REPO_ROOT}/baseline/classification_genrel/logs/"
log "  generation    : ${REPO_ROOT}/baseline/generation_ner/logs/"