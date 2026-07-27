#!/bin/bash
# NER 烟雾测试：epochs=1, context=2048, no fast kernels → 跑出第一条 loss
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${TASK_DIR}/../.." && pwd)"
MODEL_PATH="/root/autodl-tmp/hf_cache/Llama-3-1-8B-I"
DATA_PATH="${REPO_ROOT}/datasets/botriplex/Preprocessed BioTriplex/"
OUTPUT_DIR="${TASK_DIR}/checkpoints_smoke"
LOG_DIR="${TASK_DIR}/logs_smoke"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

cd "${REPO_ROOT}/baseline/llama-rec"
export PYTHONPATH="${REPO_ROOT}/baseline/llama-rec/src:${REPO_ROOT}/baseline/llama-rec/_compat:${PYTHONPATH:-}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [NER SMOKE] start"

python -c "
import sys, runpy
sys.path.insert(0, '${REPO_ROOT}/baseline/llama-rec/src')
sys.path.insert(0, '${REPO_ROOT}/baseline/llama-rec/_compat')
import transformers_59_patch
sys.argv = ['finetuning.py'] + sys.argv[1:]
runpy.run_path('recipes/quickstart/finetuning/finetuning.py', run_name='__main__')
" \
    --use_peft \
    --peft_method lora \
    --model_name "${MODEL_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --batch_size_training 1 \
    --batching_strategy "padding" \
    --weight_decay 0.2 \
    --num_epochs 1 \
    --dataset biotriplex_ner_dataset \
    --context_length 2048 \
    --data_path "${DATA_PATH}" \
    --use_entity_tokens_as_targets False \
    --entity_special_tokens False \
    --use_fast_kernels False \
    --bidirectional_attention_in_entity_tokens False \
    --run_validation False \
    --save_model False \
    --save_metrics False 2>&1 | tee "${LOG_DIR}/smoke.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] [NER SMOKE] done"
