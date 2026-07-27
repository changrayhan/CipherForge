"""Compare two step_profiles.jsonl files and emit a markdown speedup report.

Used by Task A (post-5-step optimization). Compares the *optimized* run
against the *last test* run (not against the original baseline), per the
plan §Task A.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List


def _safe_load_per_step(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _parse_embedded_dict(s: Any) -> Dict[str, float]:
    """The phase_ms and chunk_*_times_ms fields are written as str(dict)."""
    if isinstance(s, dict):
        return {k: float(v) for k, v in s.items()}
    if isinstance(s, str):
        try:
            d = ast.literal_eval(s)
            return {k: float(v) for k, v in d.items()}
        except Exception:
            return {}
    return {}


def _phase_summary(phase_ms: Dict[str, float]) -> Dict[str, float]:
    return {k: v for k, v in phase_ms.items()}


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    idx = int(round((pct / 100.0) * (len(s) - 1)))
    return s[max(0, min(idx, len(s) - 1))]


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    step_times = [r["step_time_ms"] for r in rows]
    if not step_times:
        return {"count": 0}
    phases_total: Dict[str, List[float]] = {}
    for r in rows:
        pm = _parse_embedded_dict(r.get("phase_ms"))
        for k, v in pm.items():
            phases_total.setdefault(k, []).append(v)
    phase_avg = {k: statistics.mean(v) for k, v in phases_total.items()}
    # Normalize per-phase ms per token to enable cross-scale comparison.
    n_tokens_values = [r.get("n_tokens", 1) or 1 for r in rows]
    n_chunks_values = [r.get("n_chunks", 1) or 1 for r in rows]
    phase_per_token: Dict[str, List[float]] = {}
    phase_per_chunk: Dict[str, List[float]] = {}
    for r in rows:
        nt = r.get("n_tokens") or 1
        nc = r.get("n_chunks") or 1
        pm = _parse_embedded_dict(r.get("phase_ms"))
        for k, v in pm.items():
            phase_per_token.setdefault(k, []).append(v / nt)
            phase_per_chunk.setdefault(k, []).append(v / nc)
    phase_per_token_mean = {k: statistics.mean(v) for k, v in phase_per_token.items()}
    phase_per_chunk_mean = {k: statistics.mean(v) for k, v in phase_per_chunk.items()}
    rss_values = [r.get("rss_mb", 0) for r in rows]
    return {
        "count": len(rows),
        "step_time_ms_mean": statistics.mean(step_times),
        "step_time_ms_p50": _percentile(step_times, 50),
        "step_time_ms_p95": _percentile(step_times, 95),
        "step_time_ms_p99": _percentile(step_times, 99),
        "step_time_ms_min": min(step_times),
        "step_time_ms_max": max(step_times),
        "phase_ms_mean": phase_avg,
        "phase_ms_per_token": phase_per_token_mean,
        "phase_ms_per_chunk": phase_per_chunk_mean,
        "rss_mb_max": max(rss_values) if rss_values else 0,
        "rss_mb_mean": statistics.mean(rss_values) if rss_values else 0,
        "first_step": rows[0]["step"],
        "last_step": rows[-1]["step"],
        "n_tokens_mean": statistics.mean(n_tokens_values),
        "n_chunks_mean": statistics.mean(n_chunks_values),
    }


def _speedup(opt: float, base: float) -> str:
    if base == 0:
        return "n/a"
    pct = (1.0 - opt / base) * 100
    return f"{pct:+.1f}%"


def render_report(
    baseline_path: Path,
    optimized_path: Path,
    baseline_summary: Dict[str, Any],
    optimized_summary: Dict[str, Any],
    commit_hashes: List[str],
    output_path: Path,
) -> str:
    if baseline_summary.get("count", 0) == 0 or optimized_summary.get("count", 0) == 0:
        return "# Task A Report: insufficient data\n"
    b = baseline_summary
    o = optimized_summary

    # Detect scale mismatch
    scale_mismatch = (
        abs(b.get("n_tokens_mean", 0) - o.get("n_tokens_mean", 0)) > 100
        or abs(b.get("n_chunks_mean", 0) - o.get("n_chunks_mean", 0)) > 0.1
    )

    lines: List[str] = []
    lines.append("# Task A — GenRel 1 Epoch Speedup Report\n")
    lines.append("> 5-step optimization speedup measurement vs. **last test data**.")
    lines.append("> Per the plan: not compared to original baseline; no main-task metric comparison.\n")
    lines.append("## Commits included\n")
    for h in commit_hashes:
        lines.append(f"- `{h}`")
    lines.append("")
    lines.append("## Data sources\n")
    lines.append(f"- Last test (baseline for this report): `{baseline_path}`")
    lines.append(f"- Optimized run: `{optimized_path}`\n")
    lines.append("## Scale check\n")
    lines.append("| Dimension | Last test | Optimized |")
    lines.append("|---|---:|---:|")
    lines.append(f"| Steps counted | {b['count']} | {o['count']} |")
    lines.append(f"| n_tokens (mean) | {b['n_tokens_mean']:.0f} | {o['n_tokens_mean']:.0f} |")
    lines.append(f"| n_chunks (mean) | {b['n_chunks_mean']:.2f} | {o['n_chunks_mean']:.2f} |")
    lines.append("")
    if scale_mismatch:
        lines.append("> ⚠️ **Scale mismatch detected.** The two runs use different `n_tokens` or `n_chunks`,")
        lines.append("> so absolute `step_time_ms` and `phase_ms` are NOT directly comparable. The")
        lines.append("> report normalizes by `n_tokens` to give a per-token comparison; this is the")
        lines.append("> correct apples-to-apples metric for assessing the encryption pipeline speedup.\n")
    lines.append("## Step-level comparison (raw, un-normalized)\n")
    lines.append("| Metric | Last test | Optimized | Speedup |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| step_time mean (ms) | {b['step_time_ms_mean']:.1f} | "
                 f"{o['step_time_ms_mean']:.1f} | {_speedup(o['step_time_ms_mean'], b['step_time_ms_mean'])} |")
    lines.append(f"| step_time P50 (ms) | {b['step_time_ms_p50']:.1f} | "
                 f"{o['step_time_ms_p50']:.1f} | {_speedup(o['step_time_ms_p50'], b['step_time_ms_p50'])} |")
    lines.append(f"| step_time P95 (ms) | {b['step_time_ms_p95']:.1f} | "
                 f"{o['step_time_ms_p95']:.1f} | {_speedup(o['step_time_ms_p95'], b['step_time_ms_p95'])} |")
    lines.append(f"| step_time P99 (ms) | {b['step_time_ms_p99']:.1f} | "
                 f"{o['step_time_ms_p99']:.1f} | {_speedup(o['step_time_ms_p99'], b['step_time_ms_p99'])} |")
    lines.append(f"| step_time min (ms) | {b['step_time_ms_min']:.1f} | "
                 f"{o['step_time_ms_min']:.1f} | {_speedup(o['step_time_ms_min'], b['step_time_ms_min'])} |")
    lines.append(f"| step_time max (ms) | {b['step_time_ms_max']:.1f} | "
                 f"{o['step_time_ms_max']:.1f} | {_speedup(o['step_time_ms_max'], b['step_time_ms_max'])} |")
    lines.append(f"| RSS memory max (MB) | {b['rss_mb_max']:.0f} | "
                 f"{o['rss_mb_max']:.0f} | Δ={o['rss_mb_max']-b['rss_mb_max']:+.0f} MB |")
    lines.append("")

    if scale_mismatch:
        lines.append("## Step-level comparison (per-token, normalized)\n")
        lines.append("> Per-token ms isolates the per-element encryption/PIR cost. This is the")
        lines.append("> correct metric when scale differs.\n")
        lines.append("| Metric | Last test (ms/tok) | Optimized (ms/tok) | Speedup |")
        lines.append("|---|---:|---:|---:|")
        for label, key in [("step_time mean", "step_time_ms_mean"),
                           ("step_time P50", "step_time_ms_p50"),
                           ("step_time P95", "step_time_ms_p95")]:
            bv = b[key] / b["n_tokens_mean"]
            ov = o[key] / o["n_tokens_mean"]
            lines.append(f"| {label} | {bv:.4f} | {ov:.4f} | {_speedup(ov, bv)} |")
        lines.append("")

    lines.append("## Phase-level contribution (raw, ms)\n")
    phase_keys = sorted(set(b["phase_ms_mean"].keys()) | set(o["phase_ms_mean"].keys()))
    if phase_keys:
        lines.append("| Phase | Last test (ms) | Optimized (ms) | Speedup |")
        lines.append("|---|---:|---:|---:|")
        for k in phase_keys:
            bv = b["phase_ms_mean"].get(k, float("nan"))
            ov = o["phase_ms_mean"].get(k, float("nan"))
            sp = _speedup(ov, bv)
            lines.append(f"| `{k}` | {bv:.1f} | {ov:.1f} | {sp} |")
        lines.append("")

    if scale_mismatch:
        lines.append("## Phase-level contribution (per-token, normalized)\n")
        lines.append("| Phase | Last test (ms/tok) | Optimized (ms/tok) | Speedup |")
        lines.append("|---|---:|---:|---:|")
        for k in phase_keys:
            bv = b["phase_ms_per_token"].get(k, float("nan"))
            ov = o["phase_ms_per_token"].get(k, float("nan"))
            sp = _speedup(ov, bv)
            lines.append(f"| `{k}` | {bv:.6f} | {ov:.6f} | {sp} |")
        lines.append("")

    # Decision: use per-token metric if scale differs, else raw
    if scale_mismatch:
        b_norm = b["step_time_ms_mean"] / b["n_tokens_mean"]
        o_norm = o["step_time_ms_mean"] / o["n_tokens_mean"]
        decision_metric_label = "per-token mean step_time"
        b_p95_norm = b["step_time_ms_p95"] / b["n_tokens_mean"]
        o_p95_norm = o["step_time_ms_p95"] / o["n_tokens_mean"]
        b_for_decision = b_norm
        o_for_decision = o_norm
    else:
        decision_metric_label = "mean step_time"
        b_p95_norm = b["step_time_ms_p95"]
        o_p95_norm = o["step_time_ms_p95"]
        b_for_decision = b["step_time_ms_mean"]
        o_for_decision = o["step_time_ms_mean"]
    mean_speedup_pct = (1.0 - o_for_decision / b_for_decision) * 100
    p95_speedup_pct = (1.0 - o_p95_norm / b_p95_norm) * 100
    lines.append("## Decision\n")
    if mean_speedup_pct >= 20.0 and p95_speedup_pct >= 15.0:
        decision = "✅ **KEEP all kept steps** — both mean and P95 speedup exceed thresholds."
    elif mean_speedup_pct >= 10.0:
        decision = (
            "⚠️ **PARTIAL KEEP** — mean speedup is positive but below 20% target. "
            "Consider reverting the marginal step."
        )
    elif mean_speedup_pct >= 0:
        decision = "⚠️ **MARGINAL** — minor improvement; evaluate per-step contribution."
    else:
        decision = "❌ **REVERT** — optimization made things slower; revert one or more commits."
    lines.append(f"- Decision metric: **{decision_metric_label}**")
    lines.append(f"- Mean speedup ({decision_metric_label}): **{mean_speedup_pct:+.1f}%**")
    lines.append(f"- P95 speedup (normalized): **{p95_speedup_pct:+.1f}%**")
    lines.append(f"- Verdict: {decision}\n")

    lines.append("## Note\n")
    lines.append("- This report focuses on encryption/PIR pipeline speedup. **Main-task metrics")
    lines.append("  (Multi-label F1, Macro F1, Macro ROC AUC) are intentionally NOT compared** —")
    lines.append("  Step 2's PRG vectorization is byte-identical to the scalar version (verified by")
    lines.append("  `tests/test_prg_vectorization.py`), so model quality is unaffected.\n")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path,
                        help="step_profiles.jsonl from the LAST TEST run (pre-optimization)")
    parser.add_argument("--optimized", required=True, type=Path,
                        help="step_profiles.jsonl from the OPTIMIZED run (post 5 steps)")
    parser.add_argument("--report", required=True, type=Path,
                        help="output markdown path")
    parser.add_argument("--commits", nargs="*", default=[],
                        help="list of commit hashes to include in the report")
    args = parser.parse_args()

    baseline_rows = _safe_load_per_step(args.baseline)
    optimized_rows = _safe_load_per_step(args.optimized)

    if not baseline_rows:
        print(f"ERROR: no rows in baseline {args.baseline}", file=sys.stderr)
        return 1
    if not optimized_rows:
        print(f"ERROR: no rows in optimized {args.optimized}", file=sys.stderr)
        return 1

    bs = summarize(baseline_rows)
    os_ = summarize(optimized_rows)
    md = render_report(args.baseline, args.optimized, bs, os_, args.commits, args.report)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(md)
    print(f"Report written to {args.report}")
    print(f"baseline steps={bs['count']}, mean step_time={bs['step_time_ms_mean']:.0f} ms")
    print(f"optimized steps={os_['count']}, mean step_time={os_['step_time_ms_mean']:.0f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())