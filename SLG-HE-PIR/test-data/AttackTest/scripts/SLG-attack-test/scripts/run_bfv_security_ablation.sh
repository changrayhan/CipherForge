#!/usr/bin/env bash
# BFV 安全参数对照测试（poly_degree=2048 vs 4096）
#
# 预计 GPU 时间（Llama-3.2-1B, RTX 5090, batch=4, 50 步/次）:
#   - N=2048 (128-bit): ~25 min
#   - N=4096 (192-bit): ~50 min（密文乘法慢约 2-4x）
#   - 总计: ~75 min

set -euo pipefail
cd /root/autodl-tmp/SLG-HE-PIR-code/SLG-HE-PIR
exec python SLG-attack-test/scripts/run_bfv_security_ablation.py "$@"