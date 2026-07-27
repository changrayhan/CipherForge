#!/bin/bash
# 单独的 NER 推理 + 评估脚本，在 checkpoint 已训好后继续
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${TASK_DIR}/../.." && pwd)"

MODEL_PATH="/root/autodl-tmp/hf_cache/Llama-3-1-8B-I"
DATA_PATH="${REPO_ROOT}/datasets/botriplex/Preprocessed BioTriplex/"
OUTPUT_DIR="${TASK_DIR}/checkpoints"
LOG_DIR="${TASK_DIR}/logs"

TIMESTAMP="${1:-$(ls -t ${OUTPUT_DIR}/metrics_data_*.json | head -1 | grep -oE '2026-[0-9-]+_[0-9-]+' | head -1)}"
if [ -z "${TIMESTAMP}" ]; then
    echo "ERROR: cannot infer timestamp"
    exit 1
fi

cd "${TASK_DIR}"
unset OMP_NUM_THREADS
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

INFER_OUT="${LOG_DIR}/ner_infer_outputs_${TIMESTAMP}.json"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [NER] starting inference for ${TIMESTAMP}"

python scripts/ner_infer.py \
    --model_name "${MODEL_PATH}" \
    --peft_model "${OUTPUT_DIR}" \
    --data_path "${DATA_PATH}" \
    --output_path "${INFER_OUT}" \
    --bf16 \
    --max_new_tokens 2000 2>&1 | tee "${LOG_DIR}/infer_${TIMESTAMP}.log"

GOLD_FILE="${DATA_PATH}test_gold_ner.txt"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [NER] inference done, starting evaluation"

python scripts/evaluate_metrics.py \
    --outputs_json "${INFER_OUT}" \
    --gold_jsonl "${GOLD_FILE}" \
    --results_dir "${LOG_DIR}" \
    --save_prefix "ner_${TIMESTAMP}_" 2>&1 | tee "${LOG_DIR}/evaluate_${TIMESTAMP}.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] [NER] all done"
