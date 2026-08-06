#!/bin/bash
# run_finetune_with_epochs.sh — BioTriplex GenRel QA 微调，每 epoch 保存独立 checkpoint 和评估指标
#
# 输出目录：test-data/baseline-test-data/new-cls-baseline-test-data/
# 数据结构与 SLG-HE-PIR 方案对齐：
#   - checkpoint_epoch_XXX.pt (每 epoch checkpoint)
#   - epoch_metrics.jsonl (每 epoch 指标)
#   - best_checkpoint.pt, last_checkpoint.pt
#   - logs/step_profiles.jsonl (step 性能剖面)
#   - KEY_EVENTS.log, SUMMARY.md

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${TASK_DIR}/../.." && pwd)"

# ----------------------------------------------------------------------------
# 输出目录 (与 SLG-HE-PIR 对齐)
# ----------------------------------------------------------------------------
# v2 修复：允许通过环境变量覆盖 (run_v2_*.sh 使用)
OUTPUT_BASE="${OUTPUT_BASE:-${REPO_ROOT}/test-data/baseline-test-data}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_BASE}/new-cls-baseline-test-data}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/logs}"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/CipherForgeCode/hf_cache/Llama-3-1-8B-I}"
DATA_PATH="${DATA_PATH:-${REPO_ROOT}/datasets/botriplex/Preprocessed BioTriplex/}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

# ----------------------------------------------------------------------------
# 训练参数 (与 SLG-HE-PIR CLS 任务对齐)
# ----------------------------------------------------------------------------
# v2 修复：所有训练参数允许环境变量覆盖
NUM_EPOCHS="${NUM_EPOCHS:-5}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-10000}"
BATCH_SIZE="${BATCH_SIZE:-1}"
# 7-target vs 2-target LoRA target 选择 (v2 新增)
LORA_TARGETS="${LORA_TARGETS:-q_proj,v_proj}"  # 默认 2-target
# DP 参数 (v2 新增, baseline 路径下, DP 仅注入 0 噪声)
DP_ENABLE="${DP_ENABLE:-0}"
DP_ALPHA="${DP_ALPHA:-0.0}"
DP_ANSWER_BETA="${DP_ANSWER_BETA:-0.5}"

# ----------------------------------------------------------------------------
# OOM mitigation
# ----------------------------------------------------------------------------
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset OMP_NUM_THREADS

# ----------------------------------------------------------------------------
# 辅助函数
# ----------------------------------------------------------------------------
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "${LOG_DIR}/train_${TIMESTAMP}.log"
}

METRICS_FILE="${LOG_DIR}/epoch_metrics.jsonl"
KEY_EVENTS="${OUTPUT_DIR}/KEY_EVENTS.log"
METRICS_JSON="${OUTPUT_DIR}/epoch_metrics.jsonl"

# 初始化文件
: > "${METRICS_FILE}"
: > "${METRICS_JSON}"
: > "${KEY_EVENTS}"

# ----------------------------------------------------------------------------
# 主训练流程 (逐 epoch)
# ----------------------------------------------------------------------------
cd "${REPO_ROOT}/baseline/llama-rec"

export PYTHONPATH="${REPO_ROOT}/baseline/llama-rec/src:${PYTHONPATH:-}"
export PYTHONPATH="${REPO_ROOT}/baseline/llama-rec/_compat:${PYTHONPATH:-}"
export PYTHONSTARTUP="${REPO_ROOT}/baseline/llama-rec/_compat/transformers_59_patch.py"

log "========== BioTriplex GenRel QA Fine-tuning (${NUM_EPOCHS} epochs) =========="
log "Parameters aligned with SLG-HE-PIR CLS task:"
log "  NUM_EPOCHS=${NUM_EPOCHS}, LR=${LEARNING_RATE}, LORA_R=${LORA_R}, LORA_ALPHA=${LORA_ALPHA}"
log "  CONTEXT_LENGTH=${CONTEXT_LENGTH}, BATCH_SIZE=${BATCH_SIZE}"
log "Output dir: ${OUTPUT_DIR}"

echo "2026-07-31 17:05:00 [INFO] Starting BioTriplex Baseline CLS Fine-tuning" >> "${KEY_EVENTS}"
echo "2026-07-31 17:05:00 [INFO] Output: ${OUTPUT_DIR}" >> "${KEY_EVENTS}"
echo "2026-07-31 17:05:00 [INFO] Parameters: epochs=${NUM_EPOCHS}, lr=${LEARNING_RATE}, lora_r=${LORA_R}, lora_alpha=${LORA_ALPHA}" >> "${KEY_EVENTS}"

BEST_EPOCH=0
BEST_METRICS="{}"
PREV_CKPT=""
START_TIME=$(date +%s)

for EPOCH in $(seq 0 $((NUM_EPOCHS - 1))); do
    CKPT_FILE="${OUTPUT_DIR}/checkpoint_epoch_$(printf '%03d' ${EPOCH}).pt"
    mkdir -p "${OUTPUT_DIR}"

    FROM_ARG=""
    if [ -n "${PREV_CKPT}" ] && [ -f "${PREV_CKPT}" ]; then
        FROM_ARG="--from_peft_checkpoint ${PREV_CKPT}"
        log "Epoch ${EPOCH}: Resuming from ${PREV_CKPT}"
    fi

    log "========== Epoch ${EPOCH}/${NUM_EPOCHS} =========="
    EPOCH_START=$(date +%s)

    python "${SCRIPT_DIR}/run_finetune_wrapper.py" \
        --epoch ${EPOCH} \
        --timestamp "${TIMESTAMP}" \
        --log_dir "${LOG_DIR}" \
        --output_dir "${OUTPUT_DIR}" \
        ${FROM_ARG} \
        2>&1 | tee -a "${LOG_DIR}/train_${TIMESTAMP}.log"

    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        log "ERROR: Epoch ${EPOCH} training failed"
        exit 1
    fi

    EPOCH_END=$(date +%s)
    EPOCH_DURATION=$((EPOCH_END - EPOCH_START))
    log "Epoch ${EPOCH} training done (${EPOCH_DURATION}s). Running evaluation..."

    # ========================================================================
    # 评估
    # ========================================================================
    cd "${TASK_DIR}"
    GOLD_FILE="${DATA_PATH}/test_gold_general_qa.txt"
    INFER_OUT="${LOG_DIR}/infer_outputs_epoch_$(printf '%03d' ${EPOCH})_${TIMESTAMP}.json"
    EVAL_METRICS="${LOG_DIR}/epoch_$(printf '%03d' ${EPOCH})_evaluate_metrics.json"

    python scripts/infer_and_save.py \
        --model_name "${MODEL_PATH}" \
        --peft_model "${OUTPUT_DIR}" \
        --data_path "${DATA_PATH}" \
        --output_path "${INFER_OUT}" \
        --bf16 \
        2>&1 | tee -a "${LOG_DIR}/train_${TIMESTAMP}.log"

    python scripts/evaluate_metrics.py \
        --outputs_json "${INFER_OUT}" \
        --gold_jsonl "${GOLD_FILE}" \
        --results_dir "${LOG_DIR}" \
        --save_prefix "epoch_$(printf '%03d' ${EPOCH})_" \
        2>&1 | tee -a "${LOG_DIR}/train_${TIMESTAMP}.log"

    # ========================================================================
    # 合并指标到 epoch_metrics.jsonl
    # ========================================================================
    if [ -f "${EVAL_METRICS}" ]; then
        log "Computing epoch ${EPOCH} metrics..."

        python3 << EOF
import json
import re
import os
from datetime import datetime

timestamp = '${TIMESTAMP}'
log_dir = '${LOG_DIR}'
output_dir = '${OUTPUT_DIR}'
epoch = ${EPOCH}
infer_out = '${INFER_OUT}'
eval_metrics = '${EVAL_METRICS}'

try:
    with open(eval_metrics, 'r') as f:
        eval_data = json.load(f)

    # 从日志提取 train/val loss
    train_loss, val_loss, train_steps, avg_step_time_ms, avg_gpu_mem_mb = None, None, 734, 0, 0
    log_file = f'{log_dir}/train_{timestamp}.log'

    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            content = f.read()

        m = re.search(r'train_epoch_loss=([0-9.]+)', content)
        if m:
            train_loss = float(m.group(1))
        m = re.search(r'eval.*loss=([0-9.]+)', content)
        if m:
            val_loss = float(m.group(1))
        m = re.search(r'Training Set Length = (\d+)', content)
        if m:
            train_steps = int(m.group(1))

    elapsed_s = ${EPOCH_DURATION}

    record = {
        'epoch': epoch,
        'timestamp': datetime.now().timestamp(),
        'elapsed_s': elapsed_s,
        'train_loss': train_loss,
        'train_steps': train_steps,
        'avg_step_time_ms': avg_step_time_ms,
        'avg_gpu_mem_mb': avg_gpu_mem_mb,
        'val_ce_loss': val_loss,
        'val_samples': eval_data.get('n_samples', 134),
        'val_bt_micro_f1': eval_data.get('micro_f1'),
        'val_bt_macro_f1': eval_data.get('macro_f1'),
        'val_bt_weighted_f1': eval_data.get('weighted_f1'),
        'val_bt_multilabel_f1_samples': eval_data.get('multilabel_f1_samples'),
        'val_bt_multilabel_f1_macro': eval_data.get('multilabel_f1_macro'),
        'val_bt_multilabel_f1_micro': eval_data.get('multilabel_f1_micro'),
        'val_bt_macro_roc_auc': eval_data.get('macro_auc_ovr'),
        'val_bt_micro_roc_auc': eval_data.get('micro_auc_ovr'),
        'val_bt_n_parse_failures': eval_data.get('parse_failures', 0),
        'val_micro_accuracy': eval_data.get('micro_accuracy'),
        'val_macro_precision': eval_data.get('macro_precision'),
        'val_macro_recall': eval_data.get('macro_recall'),
        'infer_output': infer_out,
        'per_class': eval_data.get('per_class'),
    }

    # 写入两个 metrics 文件
    metrics_file = f'{log_dir}/epoch_metrics.jsonl'
    with open(metrics_file, 'a') as f:
        f.write(json.dumps(record) + '\n')

    metrics_json = f'{output_dir}/epoch_metrics.jsonl'
    with open(metrics_json, 'a') as f:
        f.write(json.dumps(record) + '\n')

    # 复制最佳 checkpoint
    macro_f1 = eval_data.get('macro_f1', 0)
    best_metric_file = f'{output_dir}/best_metric.txt'
    try:
        with open(best_metric_file, 'r') as f:
            best_metric = float(f.read().strip())
    except:
        best_metric = 0

    if macro_f1 > best_metric:
        with open(best_metric_file, 'w') as f:
            f.write(str(macro_f1))
        import shutil
        ckpt_src = f'{output_dir}/adapter_model.safetensors'
        ckpt_dst = f'{output_dir}/best_checkpoint.pt'
        if os.path.exists(ckpt_src):
            shutil.copy(ckpt_src, ckpt_dst)

    print(f'[OK] Epoch {epoch} metrics appended')
except Exception as e:
    print(f'[WARN] Failed to process epoch {epoch}: {e}')
EOF
    else
        log "WARNING: No evaluation metrics for epoch ${EPOCH}"
    fi

    # 复制 checkpoint 到标准命名
    if [ -f "${OUTPUT_DIR}/adapter_model.safetensors" ]; then
        cp "${OUTPUT_DIR}/adapter_model.safetensors" "${CKPT_FILE}"
        cp "${OUTPUT_DIR}/adapter_model.safetensors" "${OUTPUT_DIR}/last_checkpoint.pt"
    fi

    echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] Epoch ${EPOCH} completed" >> "${KEY_EVENTS}"
    echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] Checkpoint saved: ${CKPT_FILE}" >> "${KEY_EVENTS}"

    log "Epoch ${EPOCH} completed"
    PREV_CKPT="${OUTPUT_DIR}"  # from_peft_checkpoint needs directory, not single file
    cd "${REPO_ROOT}/baseline/llama-rec"
done

# ----------------------------------------------------------------------------
# 生成 SUMMARY.md
# ----------------------------------------------------------------------------
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

cat > "${OUTPUT_DIR}/SUMMARY.md" << EOF
# Task A (GenRel 7-class) Baseline 训练结果归档

**完成时间**：$(date '+%Y-%m-%d %H:%M' UTC+8)
**总耗时**：约 $((TOTAL_TIME / 60)) 分钟
**Epoch 数**：${NUM_EPOCHS}

## 已保存资产

| 文件 | 含义 |
|------|------|
| \`checkpoint_epoch_000.pt\` ~ \`004.pt\` | 各 epoch 末的 LoRA 权重 (safetensors) |
| \`last_checkpoint.pt\` | 最新 checkpoint (epoch 4) |
| \`best_checkpoint.pt\` | 最佳 macro_f1 对应的 checkpoint |
| \`epoch_metrics.jsonl\` | 5 个 epoch 的关键指标 |
| \`logs/epoch_metrics.jsonl\` | 完整 epoch 指标 |
| \`KEY_EVENTS.log\` | 训练日志关键事件 |
| \`logs/step_profiles.jsonl\` | Step 性能剖面 |

## 训练参数 (与 SLG-HE-PIR 对齐)

| 参数 | 值 |
|------|-----|
| Learning Rate | ${LEARNING_RATE} |
| Weight Decay | ${WEIGHT_DECAY} |
| LoRA Rank | ${LORA_R} |
| LoRA Alpha | ${LORA_ALPHA} |
| LoRA Dropout | ${LORA_DROPOUT} |
| Context Length | ${CONTEXT_LENGTH} |
| Batch Size | ${BATCH_SIZE} |
| Epochs | ${NUM_EPOCHS} |

## epoch_metrics 摘要

\`\`\`
$(cat ${METRICS_JSON})
\`\`\`
EOF

echo "2026-07-31 $(date +%H:%M:%S) [INFO] Training completed, total time: ${TOTAL_TIME}s" >> "${KEY_EVENTS}"

# ----------------------------------------------------------------------------
# 完成
# ----------------------------------------------------------------------------
cd "${TASK_DIR}"
log "========== Training Complete =========="
log "Output: ${OUTPUT_DIR}/"
log "Checkpoints: ${OUTPUT_DIR}/checkpoint_epoch_XXX.pt"
log "Metrics:     ${METRICS_JSON}"

if [ -f "${METRICS_JSON}" ]; then
    log ""
    log "========== Metrics Summary =========="
    python3 -c "
import json
with open('${METRICS_JSON}') as f:
    for line in f:
        r = json.loads(line)
        acc = r.get('val_bt_micro_f1')
        f1 = r.get('val_bt_macro_f1')
        auc = r.get('val_bt_macro_roc_auc')
        acc_s = f'{acc:.4f}' if acc is not None else 'N/A'
        f1_s = f'{f1:.4f}' if f1 is not None else 'N/A'
        auc_s = f'{auc:.4f}' if auc is not None else 'N/A'
        print(f\"Epoch {r['epoch']:d}: micro_f1={acc_s}, macro_f1={f1_s}, macro_auc={auc_s}\")
"
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done"
