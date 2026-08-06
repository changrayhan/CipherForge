#!/usr/bin/env python3
"""generate_report.py — 读取 6 个变体的 summary.json 生成最终学术报告。

Usage:
    python scripts/generate_report.py \
        --outputs_root outputs \
        --baseline_jsonl /path/to/baseline/epoch_metrics.jsonl \
        --slg_jsonl /path/to/SLG/epoch_metrics.jsonl \
        --report_md outputs/QUANT_ABLATION_REPORT.md \
        --report_json outputs/quant_ablation_data.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from accuracy_ablation.quant_config import QuantConfig
from accuracy_ablation.report_generator import (
    PRIMARY_METRICS,
    VariantResult,
    generate_report,
)


VARIANT_DIRS = {
    "Q0":   "q0_7target",
    "Q0'":  "q0p_2target",
    "Q1":   "q1_v_quant",
    "Q2'":  "q2p_full_token",
    "Q2":   "q2_g_h_quant",
    "Q3":   "q3_full_slg_sim",
}

VARIANT_DESCRIPTIONS = {
    "Q0":   "无量化, 7-target (LoRA 配置对齐)",
    "Q0'":  "无量化, 2-target (Baseline 一致)",
    "Q1":   "V 量化 + H_M 量化 (fixed-point 量化税)",
    "Q2'":  "Q1 + 全 token g_H 量化 (无协议约束对照)",
    "Q2":   "Q1 + gold-token-only 全 token g_H 量化 (协议约束 + g_H 税)",
    "Q3":   "Q2 + g_H bf16 转换 (bf16 转换税)",
}


def main():
    p = argparse.ArgumentParser(description="Generate final quant ablation report")
    p.add_argument("--outputs_root", default="outputs")
    p.add_argument("--baseline_jsonl", default=None)
    p.add_argument("--slg_jsonl", default=None)
    p.add_argument("--report_md", default="outputs/QUANT_ABLATION_REPORT.md")
    p.add_argument("--report_json", default="outputs/quant_ablation_data.json")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger("generate_report")

    outputs_root = Path(args.outputs_root)
    variant_results = {}

    for variant, dirname in VARIANT_DIRS.items():
        summary_path = outputs_root / dirname / "summary.json"
        if not summary_path.exists():
            logger.warning(
                "[generate_report] Missing summary for variant=%s at %s",
                variant, summary_path,
            )
            continue
        with open(summary_path, "r") as f:
            summary = json.load(f)
        if not summary:
            logger.warning("[generate_report] Empty summary for variant=%s", variant)
            continue
        vr = VariantResult.from_summary(
            variant=variant,
            description=VARIANT_DESCRIPTIONS[variant],
            summary=summary,
        )
        variant_results[variant] = vr
        logger.info(
            "[generate_report] Loaded %s: n_seeds=%d, macro_f1 mean=%.4f ± %.4f",
            variant, len(summary),
            vr.stats["macro_f1"]["mean"] or 0,
            vr.stats["macro_f1"]["std"] or 0,
        )

    if not variant_results:
        logger.error("[generate_report] No variant results found!")
        sys.exit(1)

    report = generate_report(
        variant_results=variant_results,
        baseline_jsonl_path=args.baseline_jsonl,
        slg_jsonl_path=args.slg_jsonl,
        output_md_path=args.report_md,
        output_json_path=args.report_json,
    )

    print(f"\n[generate_report] ====== Report Summary ======")
    for v, vr in variant_results.items():
        f1 = vr.stats["macro_f1"]["mean"]
        std = vr.stats["macro_f1"]["std"]
        print(f"  {v}: macro_f1 = {f1:.4f} ± {std:.4f}")
    print(f"\nReport: {args.report_md}")
    print(f"Data:   {args.report_json}")


if __name__ == "__main__":
    main()