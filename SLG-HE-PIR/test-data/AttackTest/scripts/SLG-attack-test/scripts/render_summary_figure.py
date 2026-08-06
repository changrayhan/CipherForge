"""Re-generate summary_verdict_counts.png from the canonical run directories.

The previous figure was generated when different L2 data was in place; the
current L2 run has 0 LEAK_DETECTED, but the legacy summary still shows 1.
This script reads the four canonical run directories and produces a fresh
stacked bar chart.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ATTACK_DATA = Path("/root/autodl-tmp/SLG-HE-PIR-code/SLG-HE-PIR/test-data/attack-test-data")
OUT = Path("/root/autodl-tmp/SLG-HE-PIR-code/SLG-HE-PIR/test-data/figures/summary_verdict_counts.png")

RUNS = [
    ("L1", "run_20260726_140508_L1_with_dp"),
    ("L2", "run_20260725_202031"),
    ("M1", "run_20260725_202421"),
    ("M2", "run_20260726_213344"),
]

COLORS = {
    "LEAK_DETECTED":      "#d62728",
    "PRIVACY_PRESERVED":  "#2ca02c",
    "INCONCLUSIVE":       "#7f7f7f",
}


def main() -> int:
    counts = {}
    for label, run in RUNS:
        with open(ATTACK_DATA / run / "attack_results.json") as f:
            d = json.load(f)
        cnt = {"LEAK_DETECTED": 0, "PRIVACY_PRESERVED": 0, "INCONCLUSIVE": 0}
        for v in d["attack_results"]:
            cnt[v["verdict"]] += 1
        counts[label] = cnt

    labels = [r[0] for r in RUNS]
    leak = np.array([counts[l]["LEAK_DETECTED"] for l in labels])
    priv = np.array([counts[l]["PRIVACY_PRESERVED"] for l in labels])
    inc = np.array([counts[l]["INCONCLUSIVE"] for l in labels])

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    p1 = ax.bar(x, leak, color=COLORS["LEAK_DETECTED"], label="LEAK_DETECTED")
    p2 = ax.bar(x, priv, bottom=leak, color=COLORS["PRIVACY_PRESERVED"], label="PRIVACY_PRESERVED")
    p3 = ax.bar(x, inc, bottom=leak + priv, color=COLORS["INCONCLUSIVE"], label="INCONCLUSIVE")

    for i, lbl in enumerate(labels):
        c = counts[lbl]
        ax.text(i, c["LEAK_DETECTED"] / 2, str(c["LEAK_DETECTED"]),
                ha="center", va="center", color="white", fontweight="bold")
        ax.text(i, c["LEAK_DETECTED"] + c["PRIVACY_PRESERVED"] / 2,
                str(c["PRIVACY_PRESERVED"]),
                ha="center", va="center", color="white", fontweight="bold")
        if c["INCONCLUSIVE"] > 0:
            ax.text(i, c["LEAK_DETECTED"] + c["PRIVACY_PRESERVED"] + c["INCONCLUSIVE"] / 2,
                    str(c["INCONCLUSIVE"]),
                    ha="center", va="center", color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Number of sub-attacks")
    ax.set_title("Per-attack Sub-attack Verdict Counts")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"saved {OUT}")
    print("counts:", counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
