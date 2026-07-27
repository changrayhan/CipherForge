#!/usr/bin/env bash
# Smoke test: real Llama-3.2-1B U shard + d_χ privatiser, no full protocol.
set -euo pipefail
GPU=${1:-0}
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/../../src:$(pwd)/.."

python -m pytest test_trecqc_e2e.py -v --tb=short

echo "[PASS] TREC-QC / Llama-3.2-1B smoke test"
