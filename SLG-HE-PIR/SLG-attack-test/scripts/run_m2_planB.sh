#!/usr/bin/env bash
# 方案 B：M-2 Dummy Forward Pre-LoRA Baseline — 一键启动脚本
#
# 用法:
#   bash SLG-attack-test/scripts/run_m2_planB.sh           # 跑方案 B + 大样本对照
#   bash SLG-attack-test/scripts/run_m2_planB.sh --legacy  # 仅重跑 legacy warmup
#   bash SLG-attack-test/scripts/run_m2_planB.sh --compare # 跑全部 3 个实验（含 legacy）
#   bash SLG-attack-test/scripts/run_m2_planB.sh --only A  # 仅重跑 legacy
#
# 预计 GPU 时间（Llama-3.2-1B, RTX 5090, batch=4, 80 步/次）:
#   - Baseline-A (legacy): ~25 min
#   - 方案 B (K=40):       ~25 min
#   - 方案 B (K=80):       ~25 min
#   - 总计（--compare）:    ~75 min

set -euo pipefail
cd /root/autodl-tmp/SLG-HE-PIR-code/SLG-HE-PIR
exec python SLG-attack-test/scripts/run_m2_planB.py "$@"