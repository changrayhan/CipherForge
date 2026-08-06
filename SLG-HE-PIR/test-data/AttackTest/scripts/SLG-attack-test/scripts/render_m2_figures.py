"""Render M-2 figures from a single run's saved artefacts.

Reads:
  <run_dir>/m2/activation_matrix.npy
  <run_dir>/m2/activation_matrix_pre.npy  (optional)
  <run_dir>/m2/delta_activation_matrix.npy (optional)
  <run_dir>/m2/metadata.json

Produces (in <out_dir>):
  m2_spectrum.png        — log-scale singular values of Δa_t (post − pre)
                            with Marchenko-Pastur bulk edge highlighted
  m2_perm_null.png       — observed ρ(r) vs the permutation-null distribution
  m2_verdict_bar.png     — sub-attack verdict bar chart
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np


def _marcenko_pastur_edge(S: np.ndarray, N: int, D: int) -> float:
    if len(S) < 4:
        return float(S[-1]) if len(S) else 0.0
    bulk = float(np.median(S[len(S) // 2:]))
    edge = bulk * (1.0 + np.sqrt(min(N, D) / max(N, D)))
    return edge


def _compute_observed_delta(A_post: np.ndarray, A_pre: np.ndarray, r: int, k_max: int, seed: int):
    n_pair = min(len(A_pre), len(A_post))
    A_post_p = A_post[:n_pair]
    A_pre_p = A_pre[:n_pair]
    Delt = A_post_p - A_pre_p
    Delta = Delt - Delt.mean(axis=0)
    # Match the attack pipeline's k_max / sigma_mp formula exactly so that
    # σ_mp and n_above agree with the values stamped into the run's
    # metadata notes.  The attack uses N_post = A_post.shape[0] in the
    # Marchenko-Pastur √(N/D) factor (NOT n_pair).
    N_post, D = A_post.shape
    if k_max is None:
        k_max_eff = max(8, min(4 * r, N_post - 1, D))
    else:
        k_max_eff = min(k_max, N_post - 1)
    try:
        from sklearn.decomposition import TruncatedSVD
        tsvd = TruncatedSVD(n_components=k_max_eff, random_state=seed)
        tsvd.fit(Delta)
        S = tsvd.singular_values_
    except Exception:
        _, S, _ = np.linalg.svd(Delta, full_matrices=False)
        S = S[:k_max_eff]
    sigma_mp = _marcenko_pastur_edge(S, N_post, D)
    head = float(np.mean(S[:r])) if r < len(S) else 0.0
    tail = float(np.mean(S[r:])) if r < len(S) else 1e-12
    rho = head / (tail + 1e-12)
    return S, sigma_mp, rho, n_pair


def _compute_perm_null(A_post: np.ndarray, A_pre: np.ndarray, r: int, k_max: int,
                       n_perm: int, seed: int) -> np.ndarray:
    n_pair = min(len(A_pre), len(A_post))
    A_post_p = A_post[:n_pair]
    A_pre_p = A_pre[:n_pair]
    rng = np.random.default_rng(seed)
    perm_null = np.zeros(n_perm, dtype=np.float64)
    rows = np.arange(n_pair)
    for k in range(n_perm):
        perm = rng.permutation(rows)
        Delt_null = A_post_p[perm] - A_pre_p
        Delt_null -= Delt_null.mean(axis=0)
        try:
            from sklearn.decomposition import TruncatedSVD as _TSVD
            tsvd0 = _TSVD(n_components=min(k_max, Delt_null.shape[0] - 1),
                           random_state=seed)
            tsvd0.fit(Delt_null)
            S0 = tsvd0.singular_values_
        except Exception:
            _, S0, _ = np.linalg.svd(Delt_null, full_matrices=False)
            S0 = S0[:k_max]
        head0 = float(np.mean(S0[:r])) if r < len(S0) else 0.0
        tail0 = float(np.mean(S0[r:])) if r < len(S0) else 1e-12
        perm_null[k] = head0 / (tail0 + 1e-12)
    return perm_null


def _verdict_colours(verdicts: list[str]) -> list[str]:
    palette = {
        "LEAK_DETECTED": "#d62728",        # red
        "PRIVACY_PRESERVED": "#2ca02c",    # green
        "INCONCLUSIVE": "#7f7f7f",         # grey
    }
    return [palette.get(v, "#1f77b4") for v in verdicts]


def render(run_dir: Path, out_dir: Path, lora_rank: int = 8, n_perm: int = 999,
           seed: int = 42) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)

    artefacts = run_dir / "m2"
    A_post = np.load(artefacts / "activation_matrix.npy")
    A_pre_path = artefacts / "activation_matrix_pre.npy"
    A_pre = np.load(A_pre_path) if A_pre_path.exists() else None
    meta = json.loads((artefacts / "metadata.json").read_text())

    # Figure 1: spectrum of Δa_t with Marchenko-Pastur bulk edge.
    fig1 = out_dir / "m2_spectrum.png"
    if A_pre is not None and len(A_pre) >= 8:
        S, sigma_mp, rho, n_pair = _compute_observed_delta(
            A_post.astype(np.float64), A_pre.astype(np.float64),
            lora_rank, k_max=4 * lora_rank, seed=seed,
        )
        ks = np.arange(1, len(S) + 1)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.semilogy(ks, S, "o-", color="#1f77b4", markersize=4, label="σ(Δa_t)")
        ax.axhline(sigma_mp, color="#ff7f0e", linestyle="--",
                   label=f"Marchenko-Pastur bulk edge σ_mp={sigma_mp:.3e}")
        ax.axhline(sigma_mp * 1.5, color="#d62728", linestyle=":",
                   label=f"1.5·σ_mp = {sigma_mp * 1.5:.3e}")
        ax.axvline(lora_rank, color="#9467bd", linestyle="-.", alpha=0.7,
                   label=f"LoRA rank r={lora_rank}")
        ax.set_xlabel("index k")
        ax.set_ylabel("singular value (log scale)")
        ax.set_title(
            f"M-2 — Δa_t spectrum (n={n_pair}, D={A_post.shape[1]})\n"
            f"ρ(r={lora_rank}) = {rho:.3f}"
        )
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig1, dpi=140)
        plt.close(fig)
    else:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5,
                "No a_t_pre available — Δa_t spectrum skipped (weak baseline).",
                ha="center", va="center", fontsize=12)
        ax.set_axis_off()
        fig.savefig(fig1, dpi=140)
        plt.close(fig)

    # Figure 2: permutation null distribution + observed ρ.
    fig2 = out_dir / "m2_perm_null.png"
    if A_pre is not None and len(A_pre) >= 8:
        perm_null = _compute_perm_null(
            A_post.astype(np.float64), A_pre.astype(np.float64),
            lora_rank, k_max=4 * lora_rank, n_perm=n_perm, seed=seed,
        )
        S, sigma_mp, rho, _ = _compute_observed_delta(
            A_post.astype(np.float64), A_pre.astype(np.float64),
            lora_rank, k_max=4 * lora_rank, seed=seed,
        )
        null_thr = float(np.quantile(perm_null, 1.0 - 0.05))
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(perm_null, bins=30, color="#1f77b4", alpha=0.7,
                label=f"permutation null (n={len(perm_null)})")
        ax.axvline(rho, color="#d62728", linewidth=2,
                   label=f"observed ρ(r={lora_rank}) = {rho:.3f}")
        ax.axvline(null_thr, color="#ff7f0e", linestyle="--",
                   label=f"null 95% quantile = {null_thr:.3f}")
        ax.set_xlabel("ρ(r)")
        ax.set_ylabel("count")
        ax.set_title(
            "M-2 — Rank fingerprint: observed ρ vs permutation null\n"
            f"p = {float(np.mean(perm_null >= rho)):.3f}  "
            f"(α=0.05, mode={'weak_baseline' if meta.get('n_pre') == meta.get('n_post') else 'paired'})"
        )
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig2, dpi=140)
        plt.close(fig)
    else:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, "No a_t_pre available — permutation null skipped.",
                ha="center", va="center", fontsize=12)
        ax.set_axis_off()
        fig.savefig(fig2, dpi=140)
        plt.close(fig)

    # Figure 3: per-sub-attack verdict bar.
    # 本版本移除 retired 子指标：energy_fingerprint、baseline_control
    # （说明见 §3.1.4 / §2.1.4）。m2_aggregate 保留为可视化的 meta-verdict。
    RETIRED_M2_SUB_ATTACKS = {
        "energy_fingerprint", "baseline_control",
    }
    fig3 = out_dir / "m2_verdict_bar.png"
    verdicts_all = meta.get("verdicts", [])
    verdicts = [
        v for v in verdicts_all
        if v.get("sub_attack") not in RETIRED_M2_SUB_ATTACKS
    ]
    if verdicts:
        labels = [v["sub_attack"] for v in verdicts]
        outcomes = [v["verdict"] for v in verdicts]
        colours = _verdict_colours(outcomes)
        fig, ax = plt.subplots(figsize=(8, 4.0))
        ax.barh(labels, [1.0] * len(labels), color=colours,
                edgecolor="black", linewidth=0.5)
        for i, (lbl, out) in enumerate(zip(labels, outcomes)):
            ax.text(0.02, i, f"  {out}", va="center", ha="left",
                    fontsize=9, color="white" if out == "LEAK_DETECTED" else "black")
        ax.set_xticks([])
        ax.set_xlim(0, 1)
        ax.set_xlabel("verdict")
        ax.set_title(
            f"M-2 — core sub-attack verdicts "
            f"({len(verdicts)} retained after removing "
            f"{len(verdicts_all) - len(verdicts)} retired sub-attacks)\n"
            f"n_post={meta.get('n_post')}, n_pre={meta.get('n_pre')}, "
            f"lora_rank={meta.get('lora_rank')}"
        )
        ax.invert_yaxis()
        fig.tight_layout()
        fig.savefig(fig3, dpi=140)
        plt.close(fig)

    return [str(fig1), str(fig2), str(fig3)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True,
                        help="path to run_*/ containing m2/")
    parser.add_argument("--out_dir", required=True,
                        help="directory to write figures into")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--n_perm", type=int, default=999)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    figs = render(Path(args.run_dir), Path(args.out_dir),
                   lora_rank=args.lora_rank,
                   n_perm=args.n_perm, seed=args.seed)
    for f in figs:
        print(f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
