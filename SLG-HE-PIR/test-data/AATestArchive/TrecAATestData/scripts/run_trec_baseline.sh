#!/usr/bin/env bash
# run_trec_baseline.sh — entry point to start TREC-QC baseline experiments.
#
# Usage:
#   ./run_trec_baseline.sh start     # start in background
#   ./run_trec_baseline.sh status    # check progress
#   ./run_trec_baseline.sh stop      # kill runner
#   ./run_trec_baseline.sh tail      # follow log
#
# This is a thin wrapper that delegates to run_trec_full_background.sh with
# PHASES=1 (baseline only).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run_trec_full_background.sh" "$@"