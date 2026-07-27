"""Offline audit summariser for the dχ-privacy mechanism.

Usage::

    python -m src.audit.lia_h15_audit \
        --log_dir /path/to/log_dir \
        --output /path/to/lia_report.json

Reads ``log_dir/dp_audit.jsonl`` (one JSON record per training step) and
emits:

* ``<output>`` — a JSON file with per-step summary statistics.
* ``<output-without-ext>.md`` — a Markdown report next to the JSON.

The audit records are produced by :class:`H15Privatizer` via
``PrivatizerAudit.as_dict()``.  The expected keys are::

    activated, eta_used_context, eta_used_answer, noise_l2_context,
    noise_l2_answer, calibration_updated, alpha, step
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


_KEYS = (
    "eta_used_context",
    "eta_used_answer",
    "noise_l2_context",
    "noise_l2_answer",
    "alpha",
)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"n": 0, "mean": 0.0, "stdev": 0.0, "min": 0.0, "max": 0.0}
    return {
        "n": len(values),
        "mean": float(statistics.fmean(values)),
        "stdev": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _summarise(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    activated = sum(1 for r in rows if r.get("activated"))
    n_calib = sum(1 for r in rows if r.get("calibration_updated"))
    out: Dict[str, Any] = {
        "total_records": len(rows),
        "activated_records": activated,
        "activation_rate": (activated / len(rows)) if rows else 0.0,
        "calibration_updated_records": n_calib,
        "fields": {},
    }
    for key in _KEYS:
        vals = [float(r.get(key, 0.0) or 0.0) for r in rows if r.get(key) is not None]
        out["fields"][key] = _stats(vals)
    return out


def _render_markdown(summary: Dict[str, Any], log_dir: str) -> str:
    lines = [
        "# dχ-privacy offline audit report",
        "",
        f"Source: `{log_dir}/dp_audit.jsonl`",
        "",
        f"Total records: **{summary['total_records']}**",
        f"Activated records: **{summary['activated_records']}** "
        f"({summary['activation_rate'] * 100:.1f}%)",
        f"Calibration-updated records: **{summary['calibration_updated_records']}**",
        "",
        "| field | n | mean | stdev | min | max |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, s in summary["fields"].items():
        lines.append(
            f"| `{key}` | {s['n']} | {s['mean']:.4f} | {s['stdev']:.4f} | "
            f"{s['min']:.4f} | {s['max']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate dχ-privacy audit JSONL into a Markdown + JSON report.",
    )
    parser.add_argument("--log_dir", required=True,
                        help="Directory containing dp_audit.jsonl.")
    parser.add_argument("--output", required=True,
                        help="Path to the JSON summary (Markdown written next to it).")
    args = parser.parse_args(argv)

    log_dir = Path(args.log_dir).expanduser().resolve()
    in_path = log_dir / "dp_audit.jsonl"
    rows = _read_jsonl(in_path)
    summary = _summarise(rows)
    summary["source_log_dir"] = str(log_dir)
    summary["source_jsonl"] = str(in_path)

    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    md_path = out_path.with_suffix(".md")
    with md_path.open("w", encoding="utf-8") as fh:
        fh.write(_render_markdown(summary, str(log_dir)))

    print(f"[lia_h15_audit] wrote {out_path}")
    print(f"[lia_h15_audit] wrote {md_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
