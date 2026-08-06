#!/usr/bin/env python3
"""run_variant.py — 对单个量化变体在所有 epoch × seed 上跑评估。

Usage:
    python scripts/run_variant.py \
        --variant Q1 \
        --config configs/slg_extracted.yaml \
        --baseline_infer_dir /path/to/baseline/logs/ \
        --gold_path /path/to/test_gold_general_qa.txt \
        --output_dir outputs/q1_v_quant
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from accuracy_ablation.quant_config import QuantConfig
from accuracy_ablation.eval_replay import replay_variant


VARIANT_DIRS = {
    "Q0":   "q0_7target",
    "Q0'":  "q0p_2target",
    "Q1":   "q1_v_quant",
    "Q2'":  "q2p_full_token",
    "Q2":   "q2_g_h_quant",
    "Q3":   "q3_full_slg_sim",
}


def main():
    p = argparse.ArgumentParser(description="Run single quantization variant")
    p.add_argument("--variant", required=True, choices=list(VARIANT_DIRS.keys()))
    p.add_argument("--config", default="configs/slg_extracted.yaml")
    p.add_argument(
        "--baseline_infer_dir",
        default="/root/autodl-tmp/CipherForgeCode/SLG-HE-PIR/test-data/baseline-test-data/new-cls-baseline-test-data/logs/",
    )
    p.add_argument(
        "--gold_path",
        default="/root/autodl-tmp/CipherForgeCode/SLG-HE-PIR/datasets/botriplex/Preprocessed BioTriplex/test_gold_general_qa.txt",
    )
    p.add_argument(
        "--output_dir",
        default=None,
        help="默认 outputs/{VARIANT_DIRS[variant]}",
    )
    p.add_argument("--seeds", default="42,123,456")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger("run_variant")

    config = QuantConfig.from_yaml(args.config)
    seeds = [int(s) for s in args.seeds.split(",")]
    output_dir = args.output_dir or f"outputs/{VARIANT_DIRS[args.variant]}"

    logger.info(
        "Running variant=%s output=%s seeds=%s epochs=%d",
        args.variant, output_dir, seeds, args.epochs,
    )

    results = replay_variant(
        variant=args.variant,
        config=config,
        baseline_infer_dir=args.baseline_infer_dir,
        gold_path=args.gold_path,
        output_dir=output_dir,
        seeds=seeds,
        epochs=args.epochs,
    )

    # 输出每个 seed 的 best epoch
    print(f"\n[run_variant] ====== Variant={args.variant} 汇总 ======")
    for seed, epochs_data in results.items():
        if not epochs_data:
            continue
        best_epoch = max(
            epochs_data.items(),
            key=lambda kv: kv[1].get("macro_f1", 0) or 0,
        )
        print(f"  seed={seed}: best_epoch={best_epoch[0]} "
              f"macro_f1={best_epoch[1].get('macro_f1', 0):.4f} "
              f"macro_auc={best_epoch[1].get('macro_auc_ovr', 0) or 0:.4f}")

    logger.info("Done. Summary: %s/summary.json", output_dir)


if __name__ == "__main__":
    main()