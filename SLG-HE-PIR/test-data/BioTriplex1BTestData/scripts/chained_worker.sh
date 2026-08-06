#!/usr/bin/env bash
# worker.sh — single worker for chained_runner pool.
#
# Each worker is launched by the main chained_runner.sh and given a unique
# slot_id via env. The worker pulls phases from PHASE_QUEUE in order,
# skipping ones already done. flock mutex on the completed file makes
# pick atomic across workers.
#
# This script is invoked as:
#   env CHAINED_WORKER=$i CHAINED_POOL_DIR=$pool_dir bash scripts/worker.sh
# and runs `worker_loop` from chained_runner.sh via source.

set -uo pipefail

BIO_ROOT="${BIO_ROOT:-/root/autodl-tmp/CipherForgeCode/SLG-HE-PIR/test-data/BioTriplex1BTestData}"
SCRIPT_DIR="${BIO_ROOT}/scripts"

# Source the orchestrator to get all functions and PHASE_QUEUE.
# We disable the dispatch block at the end (main call) by overriding
# CHAINED_WORKER check via overriding dispatch with a custom one.
# Actually simpler: just source, but CHAINED_WORKER is already set in our
# env so the dispatch at file end will run worker_loop, not main.

source "${SCRIPT_DIR}/chained_runner.sh"
