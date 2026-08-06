#!/usr/bin/env bash
# run_trec_slg.sh — entry point to start TREC-QC SLG-fixed experiments.
#
# Usage:
#   ./run_trec_slg.sh start     # start in background
#   ./run_trec_slg.sh status
#   ./run_trec_slg.sh stop
#   ./run_trec_slg.sh tail
#
# This is a thin wrapper that delegates to run_trec_full_background.sh with
# PHASES=4 (SLG-fixed only).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run_trec_full_background.sh" "$@"