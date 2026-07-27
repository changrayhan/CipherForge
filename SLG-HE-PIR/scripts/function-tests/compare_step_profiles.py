#!/usr/bin/env python3
"""
Side-by-side compare two StepProfiler JSONL runs.

Reads ``--flat`` and ``--chunked`` JSONL files and produces a 2-panel:
  - bar chart: mean phase time, both modes stacked or grouped
  - box plot: chunk_U/chunk_M per-chunk distributions

Plus a Markdown table comparing:
  - per-phase mean time
  - total wall time
  - RSS delta
  - chunk jitter (CV)

Use case: A/B comparison of "step_train vs step_train_chunked" on identical
training data.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List

import plot_step_profiles as psp  # type: ignore
from plot_step_profiles import StepRecord  # type: ignore

ROOT = Path("/root/autodl-tmp/SLG-HE-PIR")


def load(path: Path) -> List[StepRecord]:
    return psp.load_records(path)


def compare(
    flat: List[StepRecord],
    chunked: List[StepRecord],
) -> str:
    """Return a Markdown string comparing the two runs."""
    lines: List[str] = ["# Flat vs Chunked comparison\n"]
    flat_t = [r.step_time_ms for r in flat]
    chunked_t = [r.step_time_ms for r in chunked]

    # 1. Total.
    lines.append("## Step total time\n")
    lines.append("| mode | mean | p50 | p95 | max |")
    lines.append("|---|---:|---:|---:|---:|")
    if flat_t:
        lines.append(
            f"| flat | {statistics.mean(flat_t):.0f} | {psp._pct(flat_t, 50):.0f} | "
            f"{psp._pct(flat_t, 95):.0f} | {max(flat_t):.0f} |"
        )
    if chunked_t:
        lines.append(
            f"| chunked | {statistics.mean(chunked_t):.0f} | {psp._pct(chunked_t, 50):.0f} | "
            f"{psp._pct(chunked_t, 95):.0f} | {max(chunked_t):.0f} |"
        )
        if flat_t and statistics.mean(flat_t):
            speedup = statistics.mean(flat_t) / statistics.mean(chunked_t)
            lines.append(f"\n**Chunked speedup vs flat: {speedup:.2f}×** (mean step time)\n")

    # 2. Per-phase.
    def phase_table(records: List[StepRecord], label: str) -> None:
        phase_totals: Dict[str, List[float]] = {}
        for r in records:
            for n, v in r.phase_ms.items():
                phase_totals.setdefault(n, []).append(v)
        lines.append(f"\n## Phases — {label}\n")
        lines.append("| phase | mean | p50 | p95 | share |")
        lines.append("|---|---:|---:|---:|---:|")
        grand = statistics.mean([r.step_time_ms for r in records]) if records else 1.0
        for n in sorted(phase_totals, key=lambda x: -statistics.mean(phase_totals[x])):
            v = phase_totals[n]
            share = statistics.mean(v) / grand * 100 if grand else 0
            lines.append(
                f"| {n} | {statistics.mean(v):.0f} | {psp._pct(v, 50):.0f} | "
                f"{psp._pct(v, 95):.0f} | {share:.1f}% |"
            )

    if flat:
        phase_table(flat, "flat")
    if chunked:
        phase_table(chunked, "chunked")

    # 3. Chunk jitter.
    if chunked:
        chunk_u = [v for r in chunked for v in r.chunk_u_times_ms]
        if chunk_u:
            mean_v = statistics.mean(chunk_u)
            stddev = statistics.pstdev(chunk_u)
            cv = stddev / mean_v if mean_v else 0
            lines.append(f"\n## Chunk jitter (chunked path)\n")
            lines.append(
                f"U per-chunk: mean = {mean_v:.0f} ms, stddev = {stddev:.0f} ms, "
                f"CV = {cv:.3f} ({'⚠️ high' if cv > 0.25 else 'OK'})"
            )

    # 4. RSS.
    if flat and chunked:
        f_rss = [r.rss_mb for r in flat]
        c_rss = [r.rss_mb for r in chunked]
        lines.append(f"\n## RSS growth\n")
        lines.append(f"- flat Δ: {f_rss[-1] - f_rss[0]:+.0f} MB")
        lines.append(f"- chunked Δ: {c_rss[-1] - c_rss[0]:+.0f} MB")

    return "\n".join(lines) + "\n"


def plot_side_by_side(
    flat: List[StepRecord],
    chunked: List[StepRecord],
    out_path: Path,
) -> None:
    """Bar chart: per-phase mean time, flat vs chunked."""
    if not psp._has_mpl():
        return
    import matplotlib.pyplot as plt  # type: ignore

    def phase_means(records: List[StepRecord]) -> Dict[str, float]:
        out: Dict[str, List[float]] = {}
        for r in records:
            for n, v in r.phase_ms.items():
                out.setdefault(n, []).append(v)
        return {n: statistics.mean(v) for n, v in out.items()}

    flat_p = phase_means(flat) if flat else {}
    chunked_p = phase_means(chunked) if chunked else {}

    phases = sorted(set(flat_p) | set(chunked_p))
    if not phases:
        return

    x = list(range(len(phases)))
    width = 0.4

    fig, ax = plt.subplots(figsize=(10, 5))
    flat_vals = [flat_p.get(p, 0) for p in phases]
    chunked_vals = [chunked_p.get(p, 0) for p in phases]
    ax.bar([i - width / 2 for i in x], flat_vals, width=width,
           color="#4e79a7", label="flat")
    ax.bar([i + width / 2 for i in x], chunked_vals, width=width,
           color="#e15759", label="chunked")
    ax.set_xticks(x)
    ax.set_xticklabels(phases, rotation=20)
    ax.set_ylabel("Mean phase time (ms)")
    ax.set_title("Per-phase mean time — flat vs chunked")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--flat", type=Path,
                   default=ROOT / "logs/profiles/demo_flat.jsonl")
    p.add_argument("--chunked", type=Path,
                   default=ROOT / "logs/profiles/demo_chunked.jsonl")
    p.add_argument("--out-dir", type=Path,
                   default=ROOT / "logs/profiles/compare")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    flat = load(args.flat)
    chunked = load(args.chunked)
    print(f"Flat: {len(flat)} records;  Chunked: {len(chunked)} records")

    md = compare(flat, chunked)
    out_md = args.out_dir / "compare.md"
    out_md.write_text(md)
    print(f"✓ Markdown: {out_md}")

    plot_side_by_side(flat, chunked, args.out_dir / "compare_phases.png")
    print(f"✓ Bar chart: {args.out_dir / 'compare_phases.png'}")

    print("\n" + md)
    return 0


if __name__ == "__main__":
    sys.exit(main())