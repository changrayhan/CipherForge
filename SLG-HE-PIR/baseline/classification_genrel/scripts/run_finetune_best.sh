#!/bin/bash
# BioTriplex GenRel QA 微调脚本 (分类任务) — Best-Checkpoint 版
#
# 与 scripts/run_finetune.sh 完全相同的训练超参，唯一区别：
#   后台开一个 monitor 进程，每个 epoch 结束后读取最新 metrics_data_*.json，
#   若 val_loss < 历史最佳，则把当前 adapter + adapter_config 快照到
#   ${OUTPUT_DIR}/checkpoint_best/ 单独保留，避免被下一个 epoch 的覆盖。
#
# 用法：
#   bash scripts/run_finetune_best.sh           # 跟原版完全一样，外加 best 保存
#   bash scripts/run_finetune_best.sh --epochs 6 # 可在前面加额外参数覆盖
#                       （实际未实现 argparse 直接转发；需要手动编辑下方 EXTRA_ARGS）
#
# 产物：
#   ${OUTPUT_DIR}/adapter_model.safetensors                 ← 最新 epoch (随训练覆盖)
#   ${OUTPUT_DIR}/adapter_config.json                       ← 最新 epoch (随训练覆盖)
#   ${OUTPUT_DIR}/checkpoint_best/adapter_model.safetensors  ← val_loss 最低的 epoch
#   ${OUTPUT_DIR}/checkpoint_best/adapter_config.json        ← val_loss 最低的 epoch
#   ${OUTPUT_DIR}/checkpoint_best/metrics_snapshot.json      ← 对应 epoch 的指标快照
#   ${OUTPUT_DIR}/checkpoint_best/best_log.txt               ← best 变化轨迹

set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MODEL_PATH="/root/autodl-tmp/hf_cache/Llama-3-1-8B-I"
DATA_PATH="${BASE_DIR}/../datasets/botriplex_classification/"
OUTPUT_DIR="${SCRIPT_DIR}/../checkpoints"
BEST_DIR="${OUTPUT_DIR}/checkpoint_best"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${SCRIPT_DIR}/../logs"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}" "${BEST_DIR}"

echo "=========================================="
echo "BioTriplex GenRel QA Fine-tuning (Best-Checkpoint)"
echo "Start time: $(date)"
echo "Model:      ${MODEL_PATH}"
echo "Data path:  ${DATA_PATH}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Best dir:   ${BEST_DIR}"
echo "=========================================="

cd "${BASE_DIR}/llama-rec"
export PYTHONPATH="${BASE_DIR}/llama-rec/src:${PYTHONPATH}"

# ---------------- best-checkpoint monitor (后台) ----------------
# 工作方式：每 30s 扫描一次 OUTPUT_DIR 中所有 metrics_data_*.json，
#           解析出 min val_loss，找出产生该最低 loss 的文件时间戳，
#           把那个时间点对应的 adapter 复制到 BEST_DIR。
#
# 实现细节：
#   - llama-recipes 不会单独写 "epoch-N 的 adapter"，所有 epoch 都覆盖 output_dir 同一份 adapter。
#     因此我们只能在最佳 epoch 出现后的"下一次扫描"里看到（最多 30s 延迟）。
#     这是 llama-recipes 的 API 限制，做不到精确的 epoch-end 同步。
#
#   - val_loss 取的是 metrics JSON 中的 "eval_epoch_loss" / "eval_loss"，
#     若字段名不对就在脚本内静默跳过；这种情况下 best 监测会失效但训练正常进行。

MONITOR_PID_FILE="${LOG_DIR}/best_monitor_${TIMESTAMP}.pid"
BEST_LOG="${BEST_DIR}/best_log.txt"
echo "[init] $(date '+%F %T') best_loss=inf, monitoring ${OUTPUT_DIR}" > "${BEST_LOG}"

# 抽出 metrics 解析为独立 python，避免 heredoc 在子 shell 嵌 heredoc 的 bash 兼容问题
PARSE_SCRIPT="${LOG_DIR}/parse_eval_loss_${TIMESTAMP}.py"
cat > "${PARSE_SCRIPT}" <<'PYEOF'
import json, sys
try:
    with open(sys.argv[1]) as f:
        d = json.load(f)
except Exception:
    print(""); sys.exit(0)
candidates = []
for k in ["eval_epoch_loss", "eval_loss", "val_loss", "val_epoch_loss", "val_epoch_perplexity"]:
    if isinstance(d.get(k), (int, float)):
        candidates.append(d[k])
    elif isinstance(d.get(k), list) and d[k]:
        candidates.append(min(v for v in d[k] if isinstance(v, (int, float))))
if not candidates:
    print(""); sys.exit(0)
print(min(candidates))
PYEOF

(
    set +e
    LAST_BEST_LOSS="inf"
    LAST_BEST_SRC_TS=""
    while true; do
        sleep 30
        LATEST_METRICS=$(ls -t "${OUTPUT_DIR}"/metrics_data_*.json 2>/dev/null | head -1 || true)
        if [ -z "${LATEST_METRICS}" ]; then
            continue
        fi
        EVAL_LOSS=$(python3 "${PARSE_SCRIPT}" "${LATEST_METRICS}" 2>/dev/null || echo "")
        if [ -z "${EVAL_LOSS}" ] || [ "${EVAL_LOSS}" = "None" ]; then
            continue
        fi
        IS_BETTER=$(awk -v cur="${EVAL_LOSS}" -v last="${LAST_BEST_LOSS}" 'BEGIN { print (cur+0 < last+0) ? 1 : 0 }')
        if [ "${IS_BETTER}" = "1" ]; then
            echo "[$(date '+%F %T')] (new-best) eval_loss=${EVAL_LOSS}  (was ${LAST_BEST_LOSS})  snapshot=$(basename "${LATEST_METRICS}")" >> "${BEST_LOG}"
            cp -f "${OUTPUT_DIR}/adapter_model.safetensors" "${BEST_DIR}/adapter_model.safetensors" 2>/dev/null || true
            cp -f "${OUTPUT_DIR}/adapter_config.json" "${BEST_DIR}/adapter_config.json" 2>/dev/null || true
            cp -f "${LATEST_METRICS}" "${BEST_DIR}/metrics_snapshot.json" 2>/dev/null || true
            cat > "${BEST_DIR}/best_meta.json" <<META
{
  "best_eval_loss": ${EVAL_LOSS},
  "epoch_metrics_file": "$(basename "${LATEST_METRICS}")",
  "snapshot_at": "$(date '+%F %T')",
  "training_log": "${LOG_DIR}/run_${TIMESTAMP}.log"
}
META
            LAST_BEST_LOSS="${EVAL_LOSS}"
            LAST_BEST_SRC_TS="${TIMESTAMP}"
        fi
    done
) &
MONITOR_PID=$!
echo "${MONITOR_PID}" > "${MONITOR_PID_FILE}"
echo "[best-monitor] started PID=${MONITOR_PID}, polling ${OUTPUT_DIR}/metrics_data_*.json every 30s"

cleanup() {
    if kill -0 "${MONITOR_PID}" 2>/dev/null; then
        echo "[cleanup] stopping best-monitor PID=${MONITOR_PID}"
        kill "${MONITOR_PID}" 2>/dev/null || true
        wait "${MONITOR_PID}" 2>/dev/null || true
    fi
    rm -f "${MONITOR_PID_FILE}"
}
trap cleanup EXIT

# ---------------- 训练 (与原版完全一致) ----------------
python recipes/quickstart/finetuning/finetuning.py \
    --use_peft \
    --peft_method lora \
    --model_name "${MODEL_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --batch_size_training 1 \
    --batching_strategy "padding" \
    --num_epochs 10 \
    --dataset biotriplex_qakshot_dataset \
    --data_path "${DATA_PATH}" \
    --num_of_shots 0 \
    --context_length 10000 \
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
    --save_metrics True \
    --weight_decay 0.2 \
    --gradient_clipping True \
    --gradient_clipping_threshold 1.0 \
    --seed 42 \
    2>&1 | tee "${LOG_DIR}/run_${TIMESTAMP}.log"
EXIT_CODE=${PIPESTATUS[0]}

# best-monitor 仍在跑，清理交由 trap 处理
if [ ${EXIT_CODE} -ne 0 ]; then
    echo "ERROR: Classification task exited with code ${EXIT_CODE}"
    exit ${EXIT_CODE}
fi

echo "=========================================="
echo "Classification GenRel QA fine-tuning finished"
echo "End time: $(date)"
echo "=========================================="
echo ""
echo "[summary] latest adapter (epoch ${NUM_EPOCHS:-10}): ${OUTPUT_DIR}/adapter_model.safetensors"
echo "[summary] best-checkpoint (by val_loss):"
if [ -f "${BEST_DIR}/best_meta.json" ]; then
    cat "${BEST_DIR}/best_meta.json"
    echo ""
    echo "[summary] best adapter saved at: ${BEST_DIR}/adapter_model.safetensors"
    echo "[summary] corresponding F1 evaluation:"
    echo "  python scripts/evaluate_f1.py --adapter_dir ${BEST_DIR} --split val"
else
    echo "  (none — 监控未触发，详见 ${BEST_LOG})"
fi
