#!/bin/bash
# 单独的推理 + 评估脚本，用于在 checkpoint 已训练好之后继续
# 用法：bash run_inference_and_eval.sh <TIMESTAMP>
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${TASK_DIR}/../.." && pwd)"

MODEL_PATH="/root/autodl-tmp/hf_cache/Llama-3-1-8B-I"
DATA_PATH="${REPO_ROOT}/datasets/botriplex/Preprocessed BioTriplex/"
OUTPUT_DIR="${TASK_DIR}/checkpoints"
LOG_DIR="${TASK_DIR}/logs"

# 沿用最近的训练时间戳
TIMESTAMP="${1:-$(ls -t ${OUTPUT_DIR}/metrics_data_*.json | head -1 | grep -oE '2026-[0-9-]+_[0-9-]+' | head -1)}"
if [ -z "${TIMESTAMP}" ]; then
    echo "ERROR: cannot infer timestamp"
    exit 1
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] [GenRel] starting inference for ${TIMESTAMP}"
GOLD_FILE="${DATA_PATH}test_gold_general_qa.txt"
INFER_OUT="${LOG_DIR}/infer_outputs_${TIMESTAMP}.json"

cd "${TASK_DIR}"
python scripts/infer_and_save.py \
    --model_name "${MODEL_PATH}" \
    --peft_model "${OUTPUT_DIR}" \
    --data_path "${DATA_PATH}" \
    --output_path "${INFER_OUT}" 2>&1 | tee "${LOG_DIR}/infer_${TIMESTAMP}.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] [GenRel] inference done, starting evaluation"

python scripts/evaluate_metrics.py \
    --outputs_json "${INFER_OUT}" \
    --gold_jsonl "${GOLD_FILE}" \
    --results_dir "${LOG_DIR}" \
    --save_prefix "genrel_${TIMESTAMP}_" 2>&1 | tee "${LOG_DIR}/evaluate_${TIMESTAMP}.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] [GenRel] all done"
