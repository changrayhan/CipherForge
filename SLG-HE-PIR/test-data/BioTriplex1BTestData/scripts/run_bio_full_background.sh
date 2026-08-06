#!/usr/bin/env bash
# run_bio_full_background.sh — launch the BioTriplex 1B accuracy-ablation
# runner in the background (detached via setsid + nohup).
#
# Usage:
#   bash run_bio_full_background.sh
#
# Env overrides:
#   PHASES=1,4      (default; both)
#   SEEDS=42,123,2025
#   PHASE1_EPOCHS=8
#   PHASE4_EPOCHS=10

set -euo pipefail

BIO_ROOT="${BIO_ROOT:-/root/autodl-tmp/CipherForgeCode/SLG-HE-PIR/test-data/BioTriplex1BTestData}"
LOG_ROOT="${BIO_ROOT}/runs/baseline/_runner_logs"
mkdir -p "${LOG_ROOT}"

cd /root/autodl-tmp/CipherForgeCode/SLG-HE-PIR

# Launch detached. setsid + nohup so the runner survives the parent shell exit.
nohup setsid bash "${BIO_ROOT}/scripts/_bio_internal_runner.sh" \
    > "${LOG_ROOT}/nohup.log" 2>&1 < /dev/null &
RUNNER_PID=$!
echo "Launched _bio_internal_runner.sh as PID=${RUNNER_PID}"
echo "Log: ${LOG_ROOT}/runner.log"
echo "Stdout/err: ${LOG_ROOT}/nohup.log"

# Save PID for later teardown
echo "${RUNNER_PID}" > "${LOG_ROOT}/runner.pid"