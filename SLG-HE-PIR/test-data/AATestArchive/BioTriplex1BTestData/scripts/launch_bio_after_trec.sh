#!/usr/bin/env bash
# launch_bio_after_trec.sh — orchestrate the full BioTriplex 1B migration.
#
#   1. Wait for TREC phase 1 to complete (default 27 DONE flags).
#   2. Kill the TREC runner and clean up its artifacts.
#   3. Launch the BioTriplex 1B internal runner in background.
#
# Idempotent: if TREC is already done, skips step 1-2 and goes straight to 3.
#
# Usage:
#   bash launch_bio_after_trec.sh [trec_done_target]
#     trec_done_target: how many DONE.flag to wait for (default 27)

set -euo pipefail

TREC_ROOT="${TREC_ROOT:-/root/autodl-tmp/CipherForgeCode/SLG-HE-PIR/test-data/TrecAATestData}"
BIO_ROOT="${BIO_ROOT:-/root/autodl-tmp/CipherForgeCode/SLG-HE-PIR/test-data/BioTriplex1BTestData}"
TREC_DONE_TARGET="${1:-27}"

LOG_ROOT="${BIO_ROOT}/runs/baseline/_runner_logs"
mkdir -p "${LOG_ROOT}"

ts_now() { date '+%Y-%m-%d %H:%M:%S'; }
log_line() {
    echo "[$(ts_now)] $*" | tee -a "${LOG_ROOT}/launch.log"
}

# Step 1-2: wait for TREC + cleanup (uses the existing wait_for_trec script)
log_line "===== Step 1+2: wait for TREC + cleanup ====="
bash "${BIO_ROOT}/scripts/wait_for_trec_then_cleanup.sh" "${TREC_DONE_TARGET}"

# Step 3: launch BioTriplex 1B runner
log_line "===== Step 3: launch BioTriplex 1B runner ====="
cd /root/autodl-tmp/CipherForgeCode/SLG-HE-PIR
nohup setsid bash "${BIO_ROOT}/scripts/_bio_internal_runner.sh" \
    > "${LOG_ROOT}/bio_nohup.log" 2>&1 < /dev/null &
RUNNER_PID=$!
echo "${RUNNER_PID}" > "${LOG_ROOT}/bio_runner.pid"
log_line "Launched _bio_internal_runner.sh as PID=${RUNNER_PID}"
log_line "Live log: ${LOG_ROOT}/runner.log"

# Show GPU + summary
echo "--- GPU ---"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null

log_line "launch_bio_after_trec.sh DONE"