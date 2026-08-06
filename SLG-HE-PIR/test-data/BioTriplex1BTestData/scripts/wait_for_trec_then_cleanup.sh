#!/usr/bin/env bash
# wait_for_trec_then_cleanup.sh — block until TREC phase 1 finishes, then clean.
#
# What it does:
#   1. Polls ${TREC_ROOT}/runs/baseline for new DONE.flag files every 30s.
#   2. Once DONE.count == 27, kills the trec runner (PID file or pgrep).
#   3. Removes temp/checkpoint artifacts from trec baseline output dirs
#      to free disk space for the BioTriplex 1B run.
#   4. Logs progress to ${BIO_ROOT}/runs/baseline/_runner_logs/wait.log.
#
# Usage:
#   bash wait_for_trec_then_cleanup.sh [trec_done_target]
#     trec_done_target: how many DONE.flag to wait for (default 27)

set -euo pipefail

TREC_ROOT="${TREC_ROOT:-/root/autodl-tmp/CipherForgeCode/SLG-HE-PIR/test-data/TrecAATestData}"
BIO_ROOT="${BIO_ROOT:-/root/autodl-tmp/CipherForgeCode/SLG-HE-PIR/test-data/BioTriplex1BTestData}"
TREC_DONE_TARGET="${1:-27}"

LOG_ROOT="${BIO_ROOT}/runs/baseline/_runner_logs"
mkdir -p "${LOG_ROOT}"

ts_now() { date '+%Y-%m-%d %H:%M:%S'; }
log_line() {
    echo "[$(ts_now)] $*" | tee -a "${LOG_ROOT}/wait.log"
}

log_line "==============================================================="
log_line "wait_for_trec_then_cleanup STARTED"
log_line "TREC_ROOT=${TREC_ROOT} BIO_ROOT=${BIO_ROOT} target=${TREC_DONE_TARGET}"
log_line "==============================================================="

while true; do
    DONE=$(find "${TREC_ROOT}/runs/baseline" -name DONE.flag 2>/dev/null | wc -l)
    log_line "TREC progress: ${DONE}/${TREC_DONE_TARGET}"
    if [ "${DONE}" -ge "${TREC_DONE_TARGET}" ]; then
        log_line "TREC phase 1 reached target — proceeding to cleanup"
        break
    fi
    sleep 30
done

# Kill the trec runner if still alive
RUNNER_PID=$(cat "${TREC_ROOT}/runs/baseline/_runner_logs/runner.pid" 2>/dev/null || echo "")
if [ -n "${RUNNER_PID}" ] && kill -0 "${RUNNER_PID}" 2>/dev/null; then
    log_line "Killing TREC runner PID=${RUNNER_PID}"
    kill "${RUNNER_PID}" || true
    sleep 5
    kill -9 "${RUNNER_PID}" 2>/dev/null || true
fi
# Defensive pgrep fallback
PIDS=$(pgrep -f "_trec_internal_runner.sh" || true)
if [ -n "${PIDS}" ]; then
    log_line "Killing leftover trec runner pids: ${PIDS}"
    pkill -9 -f "_trec_internal_runner.sh" || true
fi

# Cleanup: remove PEFT adapter weights from trec baseline runs (keep logs+metrics)
log_line "Cleaning trec baseline PEFT adapters (preserving logs/metrics)..."
for OUT_DIR in $(find "${TREC_ROOT}/runs/baseline" -name adapter -type d 2>/dev/null); do
    log_line "  rm -rf ${OUT_DIR}"
    rm -rf "${OUT_DIR}"
done
for CKPT in $(find "${TREC_ROOT}/runs/baseline" -name checkpoint-* -type d 2>/dev/null); do
    log_line "  rm -rf ${CKPT}"
    rm -rf "${CKPT}"
done
log_line "TREC cleanup DONE"

# Show GPU + disk
echo "--- GPU ---"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null
echo "--- Disk ---"
df -h /root/autodl-tmp 2>/dev/null | head -3

log_line "wait_for_trec_then_cleanup DONE"