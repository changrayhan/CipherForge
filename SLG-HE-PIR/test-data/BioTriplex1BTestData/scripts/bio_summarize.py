"""Summarize BioTriplex 1B accuracy ablation results across all (exp, seed, epoch).

Reads the ``metrics_history.json`` from each ``runs/<phase>/<exp>/seed_<N>/logs/``
directory and produces per-phase Markdown / CSV summaries, plus a cross-phase
overview.

Default behavior (no CLI args): scan every subdirectory of ``runs/`` that
contains at least one (exp, seed) pair, except:
  - directories starting with ``_`` (logs, summary, backups)
  - ``_backup*`` siblings produced by Phase 1.5 backup sweeps

Output per phase:
    ${BIO_ROOT}/runs/<phase>/_summary/all_metrics.md
    ${BIO_ROOT}/runs/<phase>/_summary/all_metrics.csv

Output cross-phase:
    ${BIO_ROOT}/runs/_summary/all_phases.md
    ${BIO_ROOT}/runs/_summary/all_phases.csv

Phase order in the cross-phase report follows the order given on the CLI (or
the order of the runs/ directory scan if no phases are specified).

Usage:
    # Summarize every phase found under runs/
    python3 scripts/bio_summarize.py

    # Summarize specific phases (order matters for the cross-phase table)
    python3 scripts/bio_summarize.py baseline quant_dp15 cumulative fullstack_baseline slg

    # Same as above using a comma-separated list
    python3 scripts/bio_summarize.py baseline,quant_dp15,cumulative,fullstack_baseline,slg
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

BIO_ROOT = Path("/root/autodl-tmp/CipherForgeCode/SLG-HE-PIR/test-data/BioTriplex1BTestData")
RUNS_ROOT = BIO_ROOT / "runs"

# Subdirectories under runs/ that we always skip.
SKIP_PREFIXES = ("_",)

# Phases shown in the cross-phase report when --phases is omitted. The order
# here is intentional — it matches the natural narrative flow of the report
# (baseline → quant → DP×quant → cumulative → fullstack → SLG).
DEFAULT_PHASE_ORDER = [
    "baseline",
    "quant",
    "quant_dp15",
    "cumulative",
    "baseline_extra_seeds",
    "dp_alpha_scan",
    "fullstack_baseline",
    "slg",
]


def discover_phases(runs_root: Path, requested: Optional[List[str]]) -> List[str]:
    """Resolve which phases to summarize.

    - If ``requested`` is None, scan every subdir of ``runs/`` that has at
      least one (exp, seed) pair (excluding underscored bookkeeping dirs).
    - If ``requested`` is given, use that order verbatim and skip phases that
      don't exist (with a stderr warning).
    """
    if requested:
        phases = list(requested)
        for p in phases:
            if not (runs_root / p).is_dir():
                print(f"[warn] requested phase '{p}' not found under {runs_root}", file=sys.stderr)
        return phases

    # Auto-discovery: every immediate subdir with at least one exp/seed pair.
    found = []
    for entry in sorted(runs_root.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name.startswith(SKIP_PREFIXES):
            continue
        if any_has_seed_pair(entry):
            found.append(entry.name)
    return found


def any_has_seed_pair(phase_dir: Path) -> bool:
    """True if phase_dir/<exp>/seed_N exists for some exp, N."""
    for exp_dir in phase_dir.iterdir():
        if not exp_dir.is_dir() or exp_dir.name.startswith(SKIP_PREFIXES):
            continue
        for seed_dir in exp_dir.iterdir():
            if seed_dir.is_dir() and seed_dir.name.startswith("seed_"):
                return True
    return False


def load_history(json_path: Path) -> List[Dict[str, Any]]:
    if not json_path.exists():
        return []
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def aggregate(records: List[Dict[str, Any]], key: str) -> Dict[str, float]:
    """Compute mean / std / min / max over seeds for a given metric at the best epoch."""
    vals = [r.get(key, 0.0) for r in records]
    if not vals:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n": 0}
    import numpy as np
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n": int(arr.size),
    }


def collect_rows(phase_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    """Walk one phase directory and return (flat rows, by_exp group)."""
    rows: List[Dict[str, Any]] = []
    by_exp: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    if not phase_dir.is_dir():
        return rows, by_exp

    for exp_dir in sorted(phase_dir.iterdir()):
        if not exp_dir.is_dir() or exp_dir.name.startswith(SKIP_PREFIXES):
            continue
        exp_name = exp_dir.name
        for seed_dir in sorted(exp_dir.iterdir()):
            if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                continue
            try:
                seed = int(seed_dir.name.replace("seed_", ""))
            except ValueError:
                continue
            history = load_history(seed_dir / "logs" / "metrics_history.json")
            if not history:
                continue
            # Pick best epoch by macro_f1 (matches original semantics).
            best = max(history, key=lambda e: e.get("macro_f1", 0.0))
            row = {
                "exp": exp_name,
                "seed": seed,
                "epoch": best.get("epoch", 0),
                "macro_f1": best.get("macro_f1", 0.0),
                "micro_f1": best.get("micro_f1", 0.0),
                "accuracy": best.get("accuracy", 0.0),
                "macro_auc": best.get("macro_auc", 0.0),
                "n_samples": best.get("n_samples", 0),
                "done": (seed_dir / "DONE.flag").exists(),
            }
            rows.append(row)
            by_exp[exp_name].append(row)
    return rows, by_exp


def write_phase_summary(phase: str, rows: List[Dict[str, Any]],
                        by_exp: Dict[str, List[Dict[str, Any]]],
                        summary_dir: Path) -> None:
    """Write per-phase CSV + Markdown."""
    summary_dir.mkdir(parents=True, exist_ok=True)
    csv_path = summary_dir / "all_metrics.csv"
    md_path = summary_dir / "all_metrics.md"

    # --- CSV ---
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["exp", "seed", "epoch", "macro_f1", "micro_f1",
                        "accuracy", "macro_auc", "n_samples", "done"],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"[{phase}] CSV:  {csv_path}")

    # --- Markdown ---
    n_total = len(rows)
    n_done = sum(1 for r in rows if r["done"])
    with open(md_path, "w") as f:
        f.write(f"# BioTriplex 1B — phase `{phase}` summary\n\n")
        f.write(f"- Total (exp, seed) rows with metrics: **{n_total}**\n")
        f.write(f"- Of which DONE.flag present: **{n_done}**\n\n")

        f.write("## Per-experiment aggregates (mean ± std across seeds)\n\n")
        f.write("| exp | n_seeds | macro_F1 (mean±std) | accuracy (mean±std) | macro_AUC |\n")
        f.write("|-----|---------|---------------------|---------------------|----------|\n")
        for exp_name in sorted(by_exp.keys()):
            rs = by_exp[exp_name]
            mf1 = aggregate(rs, "macro_f1")
            acc = aggregate(rs, "accuracy")
            auc = aggregate(rs, "macro_auc")
            f.write(
                f"| {exp_name} | {mf1['n']} | "
                f"{mf1['mean']:.4f} ± {mf1['std']:.4f} | "
                f"{acc['mean']:.4f} ± {acc['std']:.4f} | "
                f"{auc['mean']:.4f} |\n"
            )

        f.write("\n## Per-(exp, seed) detail\n\n")
        f.write("| exp | seed | epoch | macro_F1 | accuracy | macro_AUC | DONE |\n")
        f.write("|-----|------|-------|----------|----------|-----------|------|\n")
        for r in rows:
            f.write(
                f"| {r['exp']} | {r['seed']} | {r['epoch']} | "
                f"{r['macro_f1']:.4f} | {r['accuracy']:.4f} | {r['macro_auc']:.4f} | "
                f"{'✓' if r['done'] else '…'} |\n"
            )
    print(f"[{phase}] MD:   {md_path}")


def write_cross_phase(phases_data: List[Tuple[str, Path, List[Dict[str, Any]],
                                              Dict[str, List[Dict[str, Any]]]]],
                      out_dir: Path) -> None:
    """Write the cross-phase overview.

    phases_data is a list of (phase, phase_dir, rows, by_exp) tuples in the
    order they should appear in the report.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "all_phases.csv"
    md_path = out_dir / "all_phases.md"

    # --- CSV ---
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["phase", "exp", "seed", "epoch", "macro_f1",
                        "micro_f1", "accuracy", "macro_auc", "n_samples", "done"],
        )
        writer.writeheader()
        for phase, _dir, rows, _by in phases_data:
            for r in rows:
                writer.writerow({"phase": phase, **r})
    print(f"[cross] CSV:  {csv_path}")

    # --- Markdown ---
    with open(md_path, "w") as f:
        f.write("# BioTriplex 1B — cross-phase summary\n\n")
        f.write(f"Phases (in order): {', '.join(p for p, _, _, _ in phases_data)}\n\n")
        f.write("## Per-phase totals\n\n")
        f.write("| phase | n_rows | n_done | n_distinct_exp |\n")
        f.write("|-------|--------|--------|----------------|\n")
        for phase, _dir, rows, by_exp in phases_data:
            n_rows = len(rows)
            n_done = sum(1 for r in rows if r["done"])
            n_exp = len(by_exp)
            f.write(f"| {phase} | {n_rows} | {n_done} | {n_exp} |\n")

        f.write("\n## Per-(phase, exp) aggregates\n\n")
        f.write("| phase | exp | n_seeds | macro_F1 (mean±std) | accuracy (mean±std) | macro_AUC |\n")
        f.write("|-------|-----|---------|---------------------|---------------------|----------|\n")
        for phase, _dir, _rows, by_exp in phases_data:
            for exp_name in sorted(by_exp.keys()):
                rs = by_exp[exp_name]
                mf1 = aggregate(rs, "macro_f1")
                acc = aggregate(rs, "accuracy")
                auc = aggregate(rs, "macro_auc")
                f.write(
                    f"| {phase} | {exp_name} | {mf1['n']} | "
                    f"{mf1['mean']:.4f} ± {mf1['std']:.4f} | "
                    f"{acc['mean']:.4f} ± {acc['std']:.4f} | "
                    f"{auc['mean']:.4f} |\n"
                )
    print(f"[cross] MD:   {md_path}")


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize BioTriplex 1B results across one or more phases."
    )
    parser.add_argument(
        "phases",
        nargs="*",
        help="Phase names to summarize (subdirectories of runs/). If omitted, "
             "every phase that has at least one (exp, seed) row is summarized.",
    )
    parser.add_argument(
        "--auto-order",
        action="store_true",
        help="When phases is omitted, sort discovered phases in DEFAULT_PHASE_ORDER "
             "instead of alphabetical.",
    )
    parser.add_argument(
        "--bio-root",
        type=Path,
        default=BIO_ROOT,
        help=f"Override BIO_ROOT (default: {BIO_ROOT}).",
    )
    parser.add_argument(
        "--cross-only",
        action="store_true",
        help="Only write the cross-phase overview (skip per-phase files).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    bio_root: Path = args.bio_root
    runs_root = bio_root / "runs"
    if not runs_root.is_dir():
        print(f"[error] {runs_root} does not exist", file=sys.stderr)
        return 1

    # Resolve phase list. CLI may pass either ["a","b","c"] or "a,b,c" via
    # shells; we always normalize to a flat list.
    if args.phases:
        flat: List[str] = []
        for tok in args.phases:
            flat.extend(p for p in tok.split(",") if p)
        phases = discover_phases(runs_root, flat)
    else:
        if args.auto_order:
            # Use default order, intersected with what's actually on disk.
            existing = {p.name for p in runs_root.iterdir() if p.is_dir()}
            phases = [p for p in DEFAULT_PHASE_ORDER if p in existing]
            # Then append any newly-discovered phases alphabetically.
            extras = sorted(existing - set(DEFAULT_PHASE_ORDER) - {p for p in existing if p.startswith(SKIP_PREFIXES)})
            phases.extend(extras)
            # Filter to only those that have data.
            phases = [p for p in phases if any_has_seed_pair(runs_root / p)]
        else:
            phases = discover_phases(runs_root, None)

    if not phases:
        print("[error] no phases with metrics found", file=sys.stderr)
        return 1

    print(f"Phases to summarize ({len(phases)}): {phases}")

    cross_data: List[Tuple[str, Path, List[Dict[str, Any]],
                           Dict[str, List[Dict[str, Any]]]]] = []
    for phase in phases:
        phase_dir = runs_root / phase
        rows, by_exp = collect_rows(phase_dir)
        if not args.cross_only:
            summary_dir = phase_dir / "_summary"
            write_phase_summary(phase, rows, by_exp, summary_dir)
        cross_data.append((phase, phase_dir, rows, by_exp))
        n_done = sum(1 for r in rows if r["done"])
        print(f"  [{phase}] rows={len(rows)} done={n_done} "
              f"exps={len(by_exp)}")

    write_cross_phase(cross_data, runs_root / "_summary")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
