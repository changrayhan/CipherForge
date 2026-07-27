"""Re-render ``m1_verdict_bar.png``.

Reads ``<run>/attack_results.json`` and renders a horizontal bar for
each remaining M-1 sub-attack.  Sub-attacks that have been retired
(default confidence_distribution) are dropped automatically.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "LEAK_DETECTED":      "#d62728",
    "PRIVACY_PRESERVED":  "#2ca02c",
    "INCONCLUSIVE":       "#7f7f7f",
}


# Order in which the M-1 sub-attacks appear, mirroring the original figure.
# `confidence_distribution` is no longer produced by the suite (see
# `attacks/M1_logits_distillation.py`); if a stale `attack_results.json`
# still contains it, it is auto-filtered out below.
SUB_ORDER = [
    "prediction_consistency",
    "distillation_convergence",
    "surrogate_model",
    "information_leakage",
]


PRETTY = {
    "prediction_consistency":   "Prediction confidence variance",
    "distillation_convergence":  "Distillation convergence",
    "surrogate_model":           "Surrogate model accuracy",
    "information_leakage":       "6-bucket prediction diversity",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", required=True, help="attack_results.json path")
    p.add_argument("--out", required=True, help="Output PNG path")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    with open(args.json) as f:
        results = json.load(f)["attack_results"]

    by_sub = {r["sub_attack"]: r for r in results}

    rows = [r for r in SUB_ORDER if r in by_sub]
    pretty = [PRETTY.get(r, r) for r in rows]
    verdicts = [by_sub[r]["verdict"] for r in rows]
    values = [by_sub[r]["value"] for r in rows]

    fig, ax = plt.subplots(figsize=(10, 4.2))
    y = np.arange(len(rows))[::-1]
    ax.barh(
        y,
        [1.0] * len(rows),
        color=[COLORS[v] for v in verdicts],
        height=0.65,
        edgecolor="black",
        linewidth=0.5,
    )

    for idx, (sub, value, verdict) in enumerate(zip(rows, values, verdicts)):
        yi = y[idx]
        text = f"{PRETTY.get(sub, sub)}\nvalue={value:.4g}  verdict={verdict}"
        ax.text(1.02, yi, text, va="center", ha="left", fontsize=8,
                family="monospace", color="black")

    ax.set_yticks(y)
    ax.set_yticklabels([""] * len(rows))
    ax.set_xticks([])
    ax.set_xlim(0, 1.7)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_title(
        f"M-1 Per-subattack Verdict (red = LEAK, green = PRIVACY, gray = INCONCLUSIVE) — "
        f"{len(rows)} sub-attacks"
    )
    for spine in ("top", "right", "bottom"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#888888")

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")

    counts = {"LEAK_DETECTED": 0, "PRIVACY_PRESERVED": 0, "INCONCLUSIVE": 0}
    for v in verdicts:
        counts[v] = counts.get(v, 0) + 1
    print(f"verdict counts: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())