#!/usr/bin/env bash
# wait_then_run_f1.sh
# 用法：bash wait_then_run_f1.sh
# 行为：探测 PID 423417 与父 bash 423402 都消失 → 等待 60s 显存回收 → 执行 evaluate_f1.py

set -e
set -o pipefail

NER_PID="423417"
NER_PARENT_PID="423402"
WORK_DIR="/root/autodl-tmp/_work"
LOG="${WORK_DIR}/wait_then_run_f1.log"
mkdir -p "${WORK_DIR}"

cd /root/autodl-tmp/SLG-HE-PIR/baseline/classification_genrel
EVAL_DIR="/root/autodl-tmp/SLG-HE-PIR/baseline/classification_genrel"
EVAL_LOG="${WORK_DIR}/evaluate_f1.log"
F1_OUT_JSON="${EVAL_DIR}/checkpoints/metrics_f1_val.json"

{
    echo "==========================================="
    echo "[wait_then_run_f1] start: $(date '+%F %T')"
    echo "[wait_then_run_f1] waiting for NER PID ${NER_PID} + parent ${NER_PARENT_PID} to exit"
} | tee -a "${LOG}"

# Phase A: 每 60s 检查一次，直到进程消失；最长等 4 小时
DEADLINE=$((SECONDS + 14400))
while [ $SECONDS -lt $DEADLINE ]; do
    # kill -0 检测进程存在（不会真发信号）
    if ! kill -0 "${NER_PID}" 2>/dev/null; then
        echo "[wait_then_run_f1] $(date '+%F %T') NER python PID ${NER_PID} 已消失" | tee -a "${LOG}"
        break
    fi
    # 每 5 分钟把进展写到 log 一次
    if [ $((SECONDS % 300)) -lt 60 ]; then
        ELAPSED=$(ps -p "${NER_PID}" -o etime= 2>/dev/null | tr -d ' ' || echo "?")
        echo "[wait_then_run_f1] $(date '+%F %T') NER still running, elapsed=${ELAPSED}" >> "${LOG}"
    fi
    sleep 60
done

if [ $SECONDS -ge $DEADLINE ]; then
    echo "[wait_then_run_f1] $(date '+%F %T') TIMEOUT reached (4h), 仍执行 F1" | tee -a "${LOG}"
fi

# 检查父进程（bash run_finetune.sh）是否也退出
for i in 1 2 3 4 5; do
    if ! kill -0 "${NER_PARENT_PID}" 2>/dev/null; then
        echo "[wait_then_run_f1] $(date '+%F %T') NER parent bash PID ${NER_PARENT_PID} 已退出" | tee -a "${LOG}"
        break
    fi
    echo "[wait_then_run_f1] $(date '+%F %T') parent bash still alive (try $i/5), 继续等" | tee -a "${LOG}"
    sleep 30
done

# Phase B: 等 GPU 显存释放
echo "[wait_then_run_f1] $(date '+%F %T') GPU 显存回收等待中（90s 渐降）..." | tee -a "${LOG}"
for i in 1 2 3 4 5 6; do
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
    USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    echo "[wait_then_run_f1] $(date '+%F %T') GPU free=${FREE}MiB used=${USED}MiB" | tee -a "${LOG}"
    if [ "${USED:-0}" -lt 2000 ]; then
        echo "[wait_then_run_f1] $(date '+%F %T') GPU 几乎空闲（used=${USED}MiB），可以执行 F1" | tee -a "${LOG}"
        break
    fi
    sleep 15
done

# Phase C: 执行 F1 评测脚本（仅 val，方便先看结果；test 后续追加）
echo "[wait_then_run_f1] $(date '+%F %T') >>> 开始执行 evaluate_f1.py --split val" | tee -a "${LOG}"
cd "${EVAL_DIR}"
python3 scripts/evaluate_f1.py \
    --split val \
    --max_new_tokens 20 \
    --temperature 0.0 \
    --repetition_penalty 2.0 \
    > "${EVAL_LOG}" 2>&1
EVAL_EXIT=$?
echo "[wait_then_run_f1] $(date '+%F %T') evaluate_f1.py exit_code=${EVAL_EXIT}" | tee -a "${LOG}"

if [ "${EVAL_EXIT}" -ne 0 ]; then
    echo "[wait_then_run_f1] $(date '+%F %T') F1 评测失败，查看 ${EVAL_LOG} 末尾" | tee -a "${LOG}"
    tail -50 "${EVAL_LOG}" | tee -a "${LOG}"
    exit ${EVAL_EXIT}
fi

# 显示出关键指标
if [ -f "${F1_OUT_JSON}" ]; then
    echo "===========================================" | tee -a "${LOG}"
    echo "[wait_then_run_f1] F1 结果摘要（val, 来自 ${F1_OUT_JSON}）" | tee -a "${LOG}"
    python3 -c "
import json
with open('${F1_OUT_JSON}') as f:
    d = json.load(f)
m = d['metrics']
print(f'micro-F1={m[\"micro_f1\"]:.4f}  macro-F1={m[\"macro_f1\"]:.4f}')
print(f'subset_acc={m[\"subset_accuracy\"]:.4f}  hamming_loss={m[\"hamming_loss\"]:.4f}')
print('per-class F1:')
for c in m['per_class']:
    print(f\"  {c['letter']}) {c['name']:<25s} P={c['precision']:.3f} R={c['recall']:.3f} F1={c['f1']:.3f} support={c['support']}\")
" | tee -a "${LOG}"
fi

echo "[wait_then_run_f1] $(date '+%F %T') COMPLETED" | tee -a "${LOG}"
