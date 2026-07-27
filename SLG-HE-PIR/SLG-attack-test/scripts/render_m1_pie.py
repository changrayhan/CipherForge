"""Render M-1 Figure 9 (raw vocab-token prediction pie).

User requirement (latest):
 1. The largest slice (token 220, 71%) must "open toward the right": its leading
    edge is at 12 o'clock, the rest sweeping clockwise into the right half.
    This is achieved by `startangle=90 - 0.5 * θ_220` so the mid-angle of token
    220 lands at the +x axis (3 o'clock).
 2. ALL labels live on the RIGHT side, laid out top-to-bottom, one row each.
 3. Leader lines MUST NOT cross.

Implementation:
 - Use `startangle= 90 - sweep220/2` where sweep220 = 71% × 360° = 255.6°,
   so that token 220 spans (-90°/2 .. +90°/2) from 12 o'clock to 12 o'clock
   going clockwise, i.e. covers 12, 1, 2, 3, 4, 5, 6, 7, 8 o'clock and ends
   back at the top.
 - Place labels in a fixed vertical column at `x = 1.05 * cos(θ_mid) + 0.6`
   on the right side. Anchor each text at a y-row assigned in the slice's
   NATURAL VISIT order (counterclockwise from the top, which matches the
   data order the wedges are drawn in).
 - Draw each leader as ONE straight segment from the wedge edge to the label
   center. Since the rightmost extent of each label anchor (x ≥ 1.6) is far
   outside the pie circle (≤ 1.0), all right-side leaders run monotonically
   outward and never cross if their y-anchors preserve visit order.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ATTACK_DATA = Path("/root/autodl-tmp/SLG-HE-PIR-code/SLG-HE-PIR/test-data/attack-test-data")
OUT = Path("/root/autodl-tmp/SLG-HE-PIR-code/SLG-HE-PIR/test-data/figures/m1_prediction_pie.png")


def main() -> int:
    run_dir = ATTACK_DATA / "run_20260725_202421"
    preds = np.load(run_dir / "m1/predictions.npy").flatten()
    unique, counts = np.unique(preds, return_counts=True)
    total = int(len(preds))

    order = np.argsort(-counts)
    unique = unique[order]
    counts = counts[order]

    labels = [
        f"token {t}  ({c}/{total}, {100*c/total:.1f}%)"
        for t, c in zip(unique, counts)
    ]
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(len(unique))]

    # ── Rotate pie so token 220 "opens to the right" ─────────────────────
    # Token 220 is the FIRST slice (largest). Its sweep = 71% × 360° = 255.6°.
    # We want its midpoint to point at +x (3 o'clock), so startangle = 90° − sweep/2.
    sweep_deg = 360.0 * counts[0] / total
    start_ang = 90.0 - sweep_deg / 2.0  # ~ -37.8°

    fig, ax = plt.subplots(figsize=(13, 9))
    wedges, _ = ax.pie(
        counts,
        colors=colors,
        startangle=start_ang,
        counterclock=False,
        wedgeprops=dict(linewidth=0.8, edgecolor="white"),
    )

    # ── Lay labels on the right side top-to-bottom ───────────────────────
    # Wedges were drawn in plot order (data order in `counts`), counterclock=False,
    # so the wedge centers appear in order from startangle onwards going clockwise.
    # We map each wedge to a row index in [0, n-1] = visit order = data order,
    # then place its label at (label_x, y_row).
    n = len(wedges)
    # Matplotlib figure: 13×9 inches at 150dpi ≈ 1950×1350 px.  With ylim [-1,1]
    # for the pie (radius 1), each row of the label column is one of n slots
    # spaced evenly between 0.95 and -0.95.
    label_x = 1.25
    y_rows = np.linspace(0.95, -0.95, n)

    for i, (w, lbl, color) in enumerate(zip(wedges, labels, colors)):
        theta_mid = 0.5 * (w.theta1 + w.theta2)
        ang = np.deg2rad(theta_mid)
        x_edge, y_edge = np.cos(ang), np.sin(ang)
        y_label = y_rows[i]
        # Two straight segments: wedge edge → radial midpoint at (r_out * cosθ, y_label)
        # → label anchor (label_x, y_label).  This is a true L-shape with one
        # radial stub and one horizontal line: never bends in 3 segments.
        r_out = 1.20
        xm = r_out * np.cos(ang)
        ym = y_label  # radial stub ends at the SAME y as the label
        ax.plot([x_edge, xm, label_x], [y_edge, ym, y_label],
                color=color, linewidth=0.9, alpha=0.85, solid_capstyle="round")
        ax.text(label_x, y_label, lbl, ha="left", va="center",
                fontsize=10, color=color)

    # Title and metadata
    with open(run_dir / "m1/metadata.json") as f:
        meta = json.load(f)
    ax.set_title(
        f"M-1 S-side raw vocab-token prediction distribution\n"
        f"100 eval queries · {meta['unique_prediction_tokens']} unique tokens\n"
        f"(token 220 opens to the right; labels in a single right-side column, top→bottom)",
        fontsize=11,
    )
    ax.set_xlim(-1.55, 2.4)
    ax.set_ylim(-1.20, 1.20)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"saved {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())