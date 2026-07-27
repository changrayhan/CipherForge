#!/bin/bash
# run_all.sh — 串行执行两个 BioTriplex 微调任务
# 顺序：分类（GenRel QA）→ 生成（NER）
# 每个任务：训练 → 推理 → 评估，指标写入各自 logs/evaluate_metrics.json
# 不修改 papers/ 下的任何文件

set -uo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${BASE_DIR}/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

# 让 NER / LlamaForCausalLM 不爆 OOM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 防止 libgomp 报错
unset OMP_NUM_THREADS

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "${LOG_DIR}/run_all_${TIMESTAMP}.log"
}

log "========== Starting BioTriplex Fine-tuning Pipeline =========="
log "step 1/2: classification_genrel (epochs=6, paper value)"
log "step 2/2: generation_ner (epochs=10, paper value)"

# === Step 1: Classification ===
log ">>> [1/2] Starting Classification (GenRel QA)..."
cd "${BASE_DIR}/classification_genrel"
bash scripts/run_finetune.sh 2>&1 | tee -a "${LOG_DIR}/run_all_${TIMESTAMP}.log"
CLASSIFY_EXIT=${PIPESTATUS[0]}
if [ ${CLASSIFY_EXIT} -ne 0 ]; then
    log "ERROR: Classification task failed with exit code ${CLASSIFY_EXIT}"
    exit 1
fi
log ">>> [1/2] Classification GenRel QA done."

# === Step 2: Generation ===
log ">>> [2/2] Starting Generation (NER)..."
cd "${BASE_DIR}/generation_ner"
bash scripts/run_finetune.sh 2>&1 | tee -a "${LOG_DIR}/run_all_${TIMESTAMP}.log"
GENERATE_EXIT=${PIPESTATUS[0]}
if [ ${GENERATE_EXIT} -ne 0 ]; then
    log "ERROR: Generation task failed with exit code ${GENERATE_EXIT}"
    exit 1
fi
log ">>> [2/2] Generation NER done."

log "========== All BioTriplex Fine-tuning Completed =========="
log "Metrics JSON files:"
log "  classification: ${BASE_DIR}/classification_genrel/logs/*_evaluate_metrics.json"
log "  generation    : ${BASE_DIR}/generation_ner/logs/*_evaluate_metrics.json"