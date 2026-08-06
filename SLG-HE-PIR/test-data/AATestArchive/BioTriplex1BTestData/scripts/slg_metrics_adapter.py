#!/usr/bin/env python3
"""slg_metrics_adapter.py — Convert SLG trainer epoch_metrics.jsonl into the
baseline-style metrics_history.json consumed by bio_summarize.py.

Baseline schema (bio_evaluator.py:74-291) writes metrics with TOP-LEVEL fields:
    macro_f1, micro_f1, accuracy, macro_auc, macro_precision, macro_recall,
    n_samples, per_class, confusion_matrix, ...

SLG trainer schema (trainer.py:813) writes metrics with NESTED fields:
    val_bt_macro_f1, val_bt_micro_f1, val_bt_macro_roc_auc, val_macro_f1,
    val_micro_precision, val_micro_recall, val_micro_accuracy, val_samples, ...

This adapter reads the SLG epoch_metrics.jsonl (and the
``genrel_<TS>_evaluate_metrics.json`` from stage 2 if available) and emits
``metrics_history.json`` whose top-level fields match the baseline schema.

Usage:
    python3 slg_metrics_adapter.py \\
        --epoch_jsonl <SLG>/runs/slg/<exp>/seed_<N>/logs/epoch_metrics.jsonl \\
        --stage2_json <SLG>/runs/slg/<exp>/seed_<N>/logs/genrel_*_evaluate_metrics.json \\
        --output <SLG>/runs/slg/<exp>/seed_<N>/logs/metrics_history.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List


def _flatten_slg_metrics(epoch_record: Dict[str, Any]) -> Dict[str, Any]:
    """Map SLG field names to baseline schema (top-level keys)."""
    # Prefer BioTriplex-specific metrics when present (val_bt_*).
    # Fall back to generic val_* fields.
    out: Dict[str, Any] = {
        "epoch": epoch_record.get("epoch", 0),
        "macro_f1": epoch_record.get("val_bt_macro_f1", epoch_record.get("val_macro_f1", 0.0)),
        "micro_f1": epoch_record.get("val_bt_micro_f1", epoch_record.get("val_entity_micro_f1", 0.0)),
        "accuracy": epoch_record.get("val_micro_accuracy", 0.0),
        "macro_auc": epoch_record.get("val_bt_macro_roc_auc", 0.0),
        "macro_precision": epoch_record.get("val_micro_precision", 0.0),
        "macro_recall": epoch_record.get("val_micro_recall", 0.0),
        "n_samples": epoch_record.get("val_samples", 0),
    }
    # Carry through anything else in baseline schema (per_class etc.) if present.
    for k, v in epoch_record.items():
        if k in ("per_class", "confusion_matrix", "y_pred_distribution",
                 "y_true_distribution", "n_parse_failures"):
            out[k] = v
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--epoch_jsonl", required=True,
                   help="SLG trainer epoch_metrics.jsonl (one record per line)")
    p.add_argument("--stage2_json", default=None,
                   help="Optional: stage 2 genrel_<TS>_evaluate_metrics.json "
                        "to enrich with per_class metrics.")
    p.add_argument("--output", required=True,
                   help="Path to write baseline-schema metrics_history.json")
    args = p.parse_args()

    if not os.path.exists(args.epoch_jsonl):
        print(f"ERROR: {args.epoch_jsonl} not found", file=sys.stderr)
        return 1

    # 1. Read epoch_metrics.jsonl
    records: List[Dict[str, Any]] = []
    with open(args.epoch_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    if not records:
        print("ERROR: epoch_metrics.jsonl is empty", file=sys.stderr)
        return 1

    # 2. Optionally enrich with stage 2 per-class metrics.
    stage2_metrics: Dict[str, Any] = {}
    if args.stage2_json:
        candidates = sorted(glob.glob(args.stage2_json))
        if candidates:
            with open(candidates[-1]) as f:
                stage2_metrics = json.load(f)
    # Some evaluator outputs wrap fields under a "metrics" sub-dict; check both layouts.
    s2_inner = stage2_metrics.get("metrics", stage2_metrics) if stage2_metrics else {}
    # Map per-class field name: evaluate_biotriplex writes "per_class_metrics",
    # baseline writes "per_class".
    s2_per_class = stage2_metrics.get("per_class") or stage2_metrics.get("per_class_metrics")

    # 3. Flatten each epoch record into baseline schema.
    flat: List[Dict[str, Any]] = []
    for r in records:
        flat_r = _flatten_slg_metrics(r)
        # If stage 2 metrics carry per_class info, attach to last epoch only.
        if stage2_metrics and r is records[-1]:
            if s2_per_class:
                flat_r["per_class"] = s2_per_class
            for k in ("confusion_matrix", "y_pred_distribution",
                      "y_true_distribution"):
                v = stage2_metrics.get(k)
                if v is not None:
                    flat_r[k] = v
            # Stage 2 evaluates on test (not val), so n_samples can differ.
            flat_r["n_test_samples"] = stage2_metrics.get(
                "n_samples", s2_inner.get("n_samples", 0))
            flat_r["n_parse_failures"] = stage2_metrics.get(
                "n_parse_failures", s2_inner.get("n_parse_failures", 0))
        flat.append(flat_r)

    # 4. Write metrics_history.json
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(flat, f, indent=2)

    print(f"Wrote {len(flat)} epoch records → {args.output}")
    if flat:
        last = flat[-1]
        print(f"  last epoch ({last.get('epoch')}): macro_f1={last['macro_f1']:.4f}, "
              f"micro_f1={last['micro_f1']:.4f}, accuracy={last['accuracy']:.4f}, "
              f"macro_auc={last['macro_auc']:.4f}, n_samples={last['n_samples']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())