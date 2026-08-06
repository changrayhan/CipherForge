#!/usr/bin/env python3
"""extract_slg_params.py — 从 SLG checkpoint 提取 BFV + LoRA 配置参数。

Usage:
    python scripts/extract_slg_params.py \
        --ckpt_path /path/to/SLG/checkpoint_epoch_001.pt \
        --output configs/slg_extracted.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

# 把 accuracy_ablation 子包加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from accuracy_ablation.slg_param_extractor import extract_slg_params


def main():
    p = argparse.ArgumentParser(description="Extract SLG BFV/LoRA config to yaml")
    p.add_argument(
        "--ckpt_path",
        required=True,
        help="SLG .pt checkpoint 路径",
    )
    p.add_argument(
        "--output",
        required=True,
        help="输出 yaml 路径",
    )
    p.add_argument(
        "--verbose", "-v", action="store_true",
    )
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    cfg = extract_slg_params(args.ckpt_path)
    cfg.to_yaml(args.output)

    print(f"[extract_slg_params] Wrote config to {args.output}")
    print(f"  source: {cfg.source}")
    print(f"  scale: {cfg.scale}")
    print(f"  plain_bits: {cfg.plain_bits}")
    print(f"  hidden_dim: {cfg.hidden_dim}")
    print(f"  vocab_size: {cfg.vocab_size}")
    print(f"  lora_rank: {cfg.lora_rank}")
    print(f"  lora_alpha: {cfg.lora_alpha}")
    print(f"  target_modules ({len(cfg.target_modules)}): {cfg.target_modules}")


if __name__ == "__main__":
    main()