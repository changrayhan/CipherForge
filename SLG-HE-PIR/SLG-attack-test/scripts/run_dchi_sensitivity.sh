#!/usr/bin/env bash
# dχ 参数敏感性分析 — 一键启动脚本
# 跑完所有 21 个实验点（alpha/beta/cal/grid 共 4 组扫描）。
#
# 用法:
#   bash SLG-attack-test/scripts/run_dchi_sensitivity.sh              # 跑全部 21 个点
#   bash SLG-attack-test/scripts/run_dchi_sensitivity.sh --axis alpha # 仅 alpha 6 个点
#   bash SLG-attack-test/scripts/run_dchi_sensitivity.sh --dry-run    # 仅打印命令
#
# 预计 GPU 时间（Llama-3.2-1B, RTX 5090, batch=4, 50 步/次）:
#   - alpha (6 点): ~60 min
#   - beta  (5 点): ~50 min
#   - cal   (5 点): ~50 min
#   - grid  (5 点): ~50 min
#   - 总计: ~3.5 h

set -euo pipefail
cd /root/autodl-tmp/SLG-HE-PIR-code/SLG-HE-PIR
exec python SLG-attack-test/scripts/run_dchi_sensitivity.py "$@"