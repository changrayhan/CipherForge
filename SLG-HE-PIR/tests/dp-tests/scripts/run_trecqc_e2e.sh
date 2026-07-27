#!/usr/bin/env bash
# End-to-end TREC-QC + Llama-3.2-1B (u_layers=2) + d_χ privacy integration.
#
# Note: this runs the lightweight forward path (U shard → privatiser); a
# full HeteroProtocol 10-step run would also require the BFV cache and
# the S/M shards, which is out of scope for the DP test suite.
set -euo pipefail
GPU=${1:-0}
OUTPUT_DIR=${2:-"./dp_test_output"}

cd "$(dirname "$0")/.."
mkdir -p "${OUTPUT_DIR}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/../../src:$(pwd)/.."

python -m pytest \
    test_trecqc_e2e.py \
    -v --tb=short \
    -o cache_dir="${OUTPUT_DIR}"

echo "[PASS] TREC-QC / Llama-3.2-1B end-to-end test"
