#!/usr/bin/env python3
"""Summarize TREC-QC accuracy-ablation results into a final report.

Reads all ``runs/baseline/<exp>/seed<seed>/epoch_metrics.jsonl`` and (if present)
``runs/slg/<exp>/seed<seed>/logs/epoch_metrics.jsonl`` to produce a unified table.

Output:
    test-data/TrecAATestData/docs/trec_summary.md     (markdown table)
    test-data/TrecAATestData/docs/trec_summary.json   (machine-readable)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


def load_baseline(exp_dir: Path) -> List[Dict[str, Any]]:
    """Load epoch_metrics.jsonl from one baseline experiment directory."""
    metrics_path = exp_dir / "epoch_metrics.jsonl"
    if not metrics_path.exists():
        return []
    rows = []
    with open(metrics_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_slg(exp_dir: Path) -> List[Dict[str, Any]]:
    """Load stage-1 metrics from SLG runs (logs/epoch_metrics.jsonl)."""
    metrics_path = exp_dir / "logs" / "epoch_metrics.jsonl"
    if not metrics_path.exists():
        return []
    rows = []
    with open(metrics_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def aggregate(rows: List[Dict]) -> Dict[str, Any]:
    """Aggregate per-seed metrics into mean ± std over seeds."""
    if not rows:
        return {"n_epochs": 0}
    last_epoch = rows[-1]
    return {
        "last_epoch": last_epoch.get("epoch", 0),
        "last_train_loss": last_epoch.get("train_loss", 0.0),
        "last_accuracy": last_epoch.get("accuracy", 0.0),
        "last_macro_f1": last_epoch.get("macro_f1", 0.0),
        "last_macro_auc": last_epoch.get("macro_auc", 0.0),
        "n_epochs_recorded": len(rows),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", required=True,
                    help="TrecAATestData/runs directory")
    ap.add_argument("--output_dir", required=True,
                    help="Where to write summary.json/md")
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {"baseline": {}, "slg": {}}

    # --- Baseline ---
    base_root = runs_root / "baseline"
    if base_root.is_dir():
        for exp_dir in sorted(base_root.iterdir()):
            if not exp_dir.is_dir():
                continue
            name = exp_dir.name
            seeds_data: Dict[str, Any] = {}
            for seed_dir in sorted(exp_dir.iterdir()):
                if not seed_dir.is_dir() or not seed_dir.name.startswith("seed"):
                    continue
                rows = load_baseline(seed_dir)
                if rows:
                    seeds_data[seed_dir.name] = aggregate(rows)
            if seeds_data:
                summary["baseline"][name] = seeds_data

    # --- SLG ---
    slg_root = runs_root / "slg"
    if slg_root.is_dir():
        for exp_dir in sorted(slg_root.iterdir()):
            if not exp_dir.is_dir():
                continue
            name = exp_dir.name
            seeds_data: Dict[str, Any] = {}
            for seed_dir in sorted(exp_dir.iterdir()):
                if not seed_dir.is_dir() or not seed_dir.name.startswith("seed"):
                    continue
                rows = load_slg(seed_dir)
                if rows:
                    seeds_data[seed_dir.name] = aggregate(rows)
            if seeds_data:
                summary["slg"][name] = seeds_data

    # --- Write JSON ---
    json_path = out_dir / "trec_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[ok] summary.json → {json_path}")

    # --- Write markdown ---
    md_lines = ["# TREC-QC Accuracy Ablation Summary\n\n"]
    md_lines.append("## Baseline (9 experiments × 3 seeds × 5 epochs)\n\n")
    md_lines.append("| Experiment | seed=42 macro_F1 | seed=42 acc | seed=123 macro_F1 | seed=2025 macro_F1 |\n")
    md_lines.append("|---|---|---|---|---|\n")
    for exp_name, seeds in summary["baseline"].items():
        row = [f"`{exp_name}`"]
        for s in ["seed42", "seed123", "seed2025"]:
            sd = seeds.get(s, {})
            row.append(f"{sd.get('last_macro_f1', 0.0):.4f}" if sd else "—")
            row.append(f"{sd.get('last_accuracy', 0.0):.4f}" if sd else "—")
        md_lines.append("| " + " | ".join(row) + " |\n")
    if summary["slg"]:
        md_lines.append("\n## SLG-fixed (1 experiment × 3 seeds × 10 epochs)\n\n")
        md_lines.append("| Experiment | seed | macro_F1 | acc | macro_AUC |\n")
        md_lines.append("|---|---|---|---|---|\n")
        for exp_name, seeds in summary["slg"].items():
            for s, sd in sorted(seeds.items()):
                md_lines.append(
                    f"| `{exp_name}` | {s} | "
                    f"{sd.get('last_macro_f1', 0.0):.4f} | "
                    f"{sd.get('last_accuracy', 0.0):.4f} | "
                    f"{sd.get('last_macro_auc', 0.0):.4f} |\n"
                )

    md_path = out_dir / "trec_summary.md"
    with open(md_path, "w") as f:
        f.writelines(md_lines)
    print(f"[ok] summary.md → {md_path}")


if __name__ == "__main__":
    main()