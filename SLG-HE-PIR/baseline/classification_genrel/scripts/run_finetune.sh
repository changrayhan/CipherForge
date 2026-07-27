#!/bin/bash
# GenRel QA 微调：复刻 papers/BioTriplex/code/llama-rec/scripts/run_finetune_biotriplex_genrel_qa_.sh
# 训练目标：分类任务（7 类 BioTriplex 关系）
# 评估目标：多标签 F1 + Macro F1 + Macro ROC AUC
# 注：不修改 papers/ 下的任何文件，所有工作都在 baseline/ 下进行

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${TASK_DIR}/../.." && pwd)"

MODEL_PATH="/root/autodl-tmp/hf_cache/Llama-3-1-8B-I"
DATA_PATH="${REPO_ROOT}/datasets/botriplex/Preprocessed BioTriplex/"
OUTPUT_DIR="${TASK_DIR}/checkpoints"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${TASK_DIR}/logs"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

# OOM mitigation: 让 contiguous allocator 不把 GPU 内存切碎
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset OMP_NUM_THREADS

cd "${REPO_ROOT}/baseline/llama-rec"

export PYTHONPATH="${REPO_ROOT}/baseline/llama-rec/src:${PYTHONPATH:-}"

# 加载 transformers 5.x 兼容补丁（不改源码，只在 import 前注入）
export PYTHONPATH="${REPO_ROOT}/baseline/llama-rec/_compat:${PYTHONPATH:-}"
export PYTHONSTARTUP="${REPO_ROOT}/baseline/llama-rec/_compat/transformers_59_patch.py"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] [GenRel] start training, timestamp=${TIMESTAMP}"

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
    --num_epochs 6 \
    --dataset biotriplex_qakshot_dataset \
    --num_of_shots 0 \
    --context_length 10000 \
    --data_path "${DATA_PATH}" \
    --use_entity_tokens_as_targets False \
    --entity_special_tokens False \
    --use_fast_kernels True \
    --upweight_minority_class False \
    --bidirectional_attention_in_entity_tokens False \
    --enable_fsdp False \
    --return_neg_relations False \
    --use_wandb False \
    --general_relations True \
    --run_validation True \
    --save_model True \
    --save_metrics True 2>&1 | tee "${LOG_DIR}/train_${TIMESTAMP}.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] [GenRel] training done, starting inference + evaluation"

# === 推理 + 评估 ===
cd "${TASK_DIR}"

# gold 文件由训练时 dataset 自动生成在 ${DATA_PATH}/test_gold_general_qa.txt
GOLD_FILE="${DATA_PATH}/test_gold_general_qa.txt"
INFER_OUT="${TASK_DIR}/logs/infer_outputs_${TIMESTAMP}.json"

python scripts/infer_and_save.py \
    --model_name "${MODEL_PATH}" \
    --peft_model "${OUTPUT_DIR}" \
    --data_path "${DATA_PATH}" \
    --output_path "${INFER_OUT}" 2>&1 | tee "${LOG_DIR}/infer_${TIMESTAMP}.log"

python scripts/evaluate_metrics.py \
    --outputs_json "${INFER_OUT}" \
    --gold_jsonl "${GOLD_FILE}" \
    --results_dir "${TASK_DIR}/logs" \
    --save_prefix "genrel_${TIMESTAMP}_" 2>&1 | tee "${LOG_DIR}/evaluate_${TIMESTAMP}.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] [GenRel] all done"