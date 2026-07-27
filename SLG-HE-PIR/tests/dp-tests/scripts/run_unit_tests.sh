#!/usr/bin/env bash
# Unit tests for the d_χ privacy module (CPU + numpy).
set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/../../src:$(pwd)/.."

python -m pytest \
    test_dchi_sampler.py \
    test_cti_label_based.py \
    test_calibrator.py \
    test_h15_privatizer.py \
    test_party_u_integration.py \
    test_protocol_smoke.py \
    -v --tb=short -q

echo "[PASS] All unit tests"
