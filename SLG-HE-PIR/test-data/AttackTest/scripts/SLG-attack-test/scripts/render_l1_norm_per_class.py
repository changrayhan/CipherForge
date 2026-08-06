"""Re-render ``l1_norm_per_class.png``.

The figure shows two side-by-side boxplots of L2 norms grouped by the
6 coarse TREC-QC classes:

  * Left  panel — L2 norms of ``g_accum`` (M-side decrypted gradient).
  * Right panel — L2 norms of ``H_U`` (U→M channel smashed data).  When
    ``--dp_enable`` is on, the right panel reflects the noisy ``H̃_U``
    that M would actually observe in production.

Input data:
  --g_accum : (N, hidden_dim) float32 numpy array
  --h_u     : (N, hidden_dim) float32 numpy array
  --labels  : (N,) integer label array
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# TREC-QC coarse class names (LOC/ABBR/DESC/HUM/NUM/ENTY) ordered by
# label index 0..5.  The audit script also assumes this mapping.
COARSE_LABELS = ["LOC", "ABBR", "DESC", "HUM", "NUM", "ENTY"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--g_accum", required=True, help="Path to gradient_matrix.npy")
    p.add_argument("--h_u", required=True, help="Path to h_u_matrix.npy")
    p.add_argument("--labels", required=True, help="Path to label_array.npy")
    p.add_argument("--out", required=True, help="Output PNG path")
    p.add_argument(
        "--h_u_title",
        default=r"$H_U$",
        help="Title for the right panel (use r'$H̃_U$' when dχ is enabled).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    G = np.asarray(np.load(args.g_accum)).astype(np.float64)
    H = np.asarray(np.load(args.h_u)).astype(np.float64)
    y = np.asarray(np.load(args.labels)).astype(int)

    if G.ndim != 2 or H.ndim != 2:
        raise ValueError(
            f"Expected 2D arrays, got g_accum.shape={G.shape}, h_u.shape={H.shape}"
        )
    if G.shape[0] != H.shape[0] or G.shape[0] != y.shape[0]:
        raise ValueError(
            f"Row mismatch: G={G.shape[0]}, H={H.shape[0]}, labels={y.shape[0]}"
        )
    if G.shape[1] != H.shape[1]:
        raise ValueError(
            f"hidden_dim mismatch: G={G.shape[1]}, H={H.shape[1]}"
        )

    g_norms = np.linalg.norm(G, axis=1)
    h_norms = np.linalg.norm(H, axis=1)

    classes = sorted(set(y.tolist()))
    class_names = [COARSE_LABELS[c] if c < len(COARSE_LABELS) else str(c) for c in classes]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.1))

    # ── Left: g_accum L2 norms ────────────────────────────────────────────
    ax = axes[0]
    g_by_class = [g_norms[y == c] for c in classes]
    bp = ax.boxplot(
        g_by_class,
        tick_labels=class_names,
        patch_artist=True,
        widths=0.6,
        medianprops=dict(color="black"),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("#1f77b4")
        patch.set_alpha(0.7)
    ax.set_title(r"$g_{\mathrm{accum}}$ L2 norm per coarse class")
    ax.set_xlabel("Coarse class")
    ax.set_ylabel("L2 norm")
    ax.grid(axis="y", alpha=0.3)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    # ── Right: H_U L2 norms ──────────────────────────────────────────────
    ax = axes[1]
    h_by_class = [h_norms[y == c] for c in classes]
    bp = ax.boxplot(
        h_by_class,
        tick_labels=class_names,
        patch_artist=True,
        widths=0.6,
        medianprops=dict(color="black"),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("#ff7f0e")
        patch.set_alpha(0.7)
    ax.set_title(f"{args.h_u_title} L2 norm per coarse class")
    ax.set_xlabel("Coarse class")
    ax.set_ylabel("L2 norm")
    ax.grid(axis="y", alpha=0.3)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")

    # Print compact stats so the audit script can verify medians.
    print("\n--- L-1 L2-norm per class medians ---")
    for c, name in zip(classes, class_names):
        g_med = float(np.median(g_norms[y == c]))
        h_med = float(np.median(h_norms[y == c]))
        print(f"  class={name} ({c}): g_accum={g_med:.4g}, h_u={h_med:.4g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())