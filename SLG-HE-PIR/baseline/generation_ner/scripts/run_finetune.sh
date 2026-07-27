#!/bin/bash
# NER JSON 微调：复刻 papers/BioTriplex/code/llama-rec/scripts/run_finetune_ner.sh
# 训练目标：生成任务（JSON 实体抽取）
# 评估目标：span-level exact-match F1（按 entity_type 分类 + 整体 Macro/Weighted/Micro）
# 注：不修改 papers/ 下的任何文件

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

echo "[$(date '+%Y-%m-%d %H:%M:%S')] [NER] start training, timestamp=${TIMESTAMP}"

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
    --num_epochs 10 \
    --dataset biotriplex_ner_dataset \
    --context_length 10000 \
    --data_path "${DATA_PATH}" \
    --use_entity_tokens_as_targets False \
    --entity_special_tokens False \
    --use_fast_kernels True \
    --bidirectional_attention_in_entity_tokens False \
    --run_validation True \
    --save_model True \
    --save_metrics True 2>&1 | tee "${LOG_DIR}/train_${TIMESTAMP}.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] [NER] training done, starting inference + evaluation"

# === 推理 + 评估 ===
cd "${TASK_DIR}"

GOLD_FILE="${DATA_PATH}/test_gold_ner.txt"
INFER_OUT="${TASK_DIR}/logs/infer_outputs_${TIMESTAMP}.json"

# 用论文原版 inference.py（不需要 logits，因为 NER 指标不依赖 logits）
cd "${REPO_ROOT}/baseline/llama-rec"

export PYTHONPATH="${REPO_ROOT}/baseline/llama-rec/src:${PYTHONPATH:-}"
# 推理阶段也需要 compat shim
export PYTHONPATH="${REPO_ROOT}/baseline/llama-rec/_compat:${PYTHONPATH:-}"

python -c "
import sys, runpy
sys.path.insert(0, '${REPO_ROOT}/baseline/llama-rec/src')
sys.path.insert(0, '${REPO_ROOT}/baseline/llama-rec/_compat')
import transformers_59_patch
sys.argv = ['inference.py'] + sys.argv[1:]
runpy.run_path('recipes/quickstart/inference/local_inference/inference.py', run_name='__main__')
" \
    --model_name "${MODEL_PATH}" \
    --peft_model "${OUTPUT_DIR}" \
    --max_new_tokens 2000 \
    --top_p 1.0 \
    --top_k 200 \
    --repetion_penalty 1.0 \
    --temperature 0.6 \
    --share_gradio False \
    --enable_salesforce_content_safety False \
    --full_dataset \
    --ner_dataset True \
    --use_entity_tokens_as_targets False \
    --entity_special_tokens False \
    --shift_entity_tokens False \
    --bidirectional_attention_in_entity_tokens False \
    --dataset_mode 'test' \
    --prefix "${TASK_DIR}/logs/infer_" 2>&1 | tee "${LOG_DIR}/infer_${TIMESTAMP}.log"

cd "${TASK_DIR}"

python scripts/evaluate_metrics.py \
    --outputs_json "${INFER_OUT}" \
    --gold_jsonl "${GOLD_FILE}" \
    --results_dir "${TASK_DIR}/logs" \
    --save_prefix "ner_${TIMESTAMP}_" 2>&1 | tee "${LOG_DIR}/evaluate_${TIMESTAMP}.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] [NER] all done"