#!/bin/bash
# BioTriplex 1B long-running monitor (defensive — never dies).
LOG="/root/autodl-tmp/CipherForgeCode/SLG-HE-PIR/test-data/BioTriplex1BTestData/runs/baseline/B-T/seed_42/logs/train.log"
RUNNER_LOG="/root/autodl-tmp/CipherForgeCode/SLG-HE-PIR/test-data/BioTriplex1BTestData/runs/baseline/_runner_logs/runner.log"
MONITOR_LOG="/root/autodl-tmp/CipherForgeCode/SLG-HE-PIR/test-data/BioTriplex1BTestData/runs/baseline/_runner_logs/monitor.log"
ALERT_LOG="/root/autodl-tmp/CipherForgeCode/SLG-HE-PIR/test-data/BioTriplex1BTestData/runs/baseline/_runner_logs/alerts.log"

mkdir -p "$(dirname "${MONITOR_LOG}")"
touch "${MONITOR_LOG}" "${ALERT_LOG}"

emit() { printf '%s\n' "$*"; }

while :; do
    TS=$(date '+%H:%M:%S')
    PID=$(pgrep -f "bio_baseline_trainer" 2>/dev/null | head -1)
    PID="${PID:-DEAD}"

    GPU_LINE=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || echo "0, 0")
    GPU_MEM=$(echo "${GPU_LINE}" | cut -d',' -f1 | tr -d ' ')
    GPU_UTIL=$(echo "${GPU_LINE}" | cut -d',' -f2 | tr -d ' ')
    GPU_UTIL="${GPU_UTIL:-0}"

    LAST_3=$(grep -oE "loss=[0-9.]+" "${LOG}" 2>/dev/null | tail -3 | tr '\n' ' ')
    LAST_EPOCH=$(grep -E "Epoch [0-9]+ metrics" "${LOG}" 2>/dev/null | tail -1)
    LAST_NAN=$(grep -c "skipping batch" "${LOG}" 2>/dev/null)
    RUNNER_LAST=$(tail -1 "${RUNNER_LOG}" 2>/dev/null)

    LINE="${TS}  PID=${PID}  GPU=${GPU_MEM}MB/${GPU_UTIL}%  loss_last3=[${LAST_3}]  nan_skips=${LAST_NAN}  last_epoch=${LAST_EPOCH:-none}  runner=${RUNNER_LAST}"
    emit "${LINE}" | tee -a "${MONITOR_LOG}"

    # Alerts (non-fatal)
    if [ "${PID}" = "DEAD" ]; then
        echo "${TS} ALERT: bio_baseline_trainer not running" >> "${ALERT_LOG}"
    elif [ "${GPU_UTIL}" -lt 50 ] 2>/dev/null; then
        echo "${TS} ALERT: GPU util=${GPU_UTIL}% (stalled?)" >> "${ALERT_LOG}"
    fi

    sleep 300
done
