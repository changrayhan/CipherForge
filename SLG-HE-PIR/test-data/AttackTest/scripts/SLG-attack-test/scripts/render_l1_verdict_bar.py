"""Re-render ``l1_verdict_bar.png``.

The figure shows the 7 L-1 sub-attacks and their verdicts on a single
horizontal bar.  Each bar is coloured by the verdict tag (red = LEAK,
green = PRIVACY_PRESERVED, gray = INCONCLUSIVE).

Input data:
  --json : ``attack_results.json`` produced by ``run_attack_suite.py``;
           each entry has ``sub_attack``, ``value``, and ``verdict`` keys.
  --out  : output PNG path.
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


# Order in which the seven L-1 sub-attacks appear, mirroring the original
# figure: forward-phase H_U items on top, backward-phase g_accum items below.
SUB_ORDER = [
    "h_u_mean_anova",
    "h_u_norm_anova",
    "kmeans_ari",
    "nn_agreement",
    "cosine_auc",
    "permutation_test",
    "magnitude_anova",
]


PRETTY = {
    "h_u_mean_anova":     "H_U class-mean ANOVA",
    "h_u_norm_anova":     "H_U L2-norm ANOVA",
    "kmeans_ari":         "K-Means ARI",
    "nn_agreement":       "1-NN agreement",
    "cosine_auc":         "Cosine AUC",
    "permutation_test":   "1-NN permutation p",
    "magnitude_anova":    "Gradient magnitude ANOVA",
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

    fig, ax = plt.subplots(figsize=(11.55, 5.31))
    y = np.arange(len(rows))[::-1]  # top to bottom
    bars = ax.barh(
        y,
        [1.0] * len(rows),
        color=[COLORS[v] for v in verdicts],
        height=0.7,
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
    ax.set_xlim(0, 1.55)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_title("L-1 Per-subattack Verdict (red = LEAK, green = PRIVACY, gray = INCONCLUSIVE)")
    for spine in ("top", "right", "bottom"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#888888")

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")

    # Also print the summary for the audit script.
    counts = {"LEAK_DETECTED": 0, "PRIVACY_PRESERVED": 0, "INCONCLUSIVE": 0}
    for v in verdicts:
        counts[v] = counts.get(v, 0) + 1
    print(f"verdict counts: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())