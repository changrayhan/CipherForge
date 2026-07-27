"""dχ 敏感性分析结果可视化脚本。

读取 ``sensitivity_dchi_manifest_*.json``，绘制：

- 图 A：alpha 敏感性折线图（6 个 alpha 值 × 3 个指标）
  - K-Means ARI
  - 1-NN 一致率
  - Cosine AUC
- 图 B：beta 敏感性折线图
- 图 C：calibration_steps 敏感性折线图
- 图 D：alpha × beta 组合 Cosine AUC 热力图

用灰色阴影带标注"合理参数区间"：在该区间内所有指标均落在
PRIVACY_PRESERVED 侧。

用法::

    python SLG-attack-test/scripts/render_dchi_sensitivity.py \
        --manifest test-data/attack-test-data/sensitivity_dchi_manifest_<ts>.json \
        --out_dir test-data/figures/
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("dchi_render")

METRIC_KEYS = [
    ("ari", "K-Means ARI", 0.1),
    ("nn_agreement", "1-NN agreement", 0.222),  # 1/6 + 2σ upper bound
    ("cosine_auc", "Cosine AUC", 0.5),
]


def _load_manifest(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _extract_axis(results: list, axis: str, override_key: str) -> tuple:
    """Extract (x_values, metrics_dict, verdicts_dict) for one sweep axis."""
    rows = [r for r in results if r["sweep_axis"] == axis]
    rows.sort(key=lambda r: r["override"][override_key])
    x = [r["override"][override_key] for r in rows]
    metrics = {k: [r["metrics"].get(k) for r in rows] for k, _, _ in METRIC_KEYS}
    verdicts = {
        f"{k}_verdict": [r["metrics"].get(f"{k}_verdict") for r in rows]
        for k, _, _ in METRIC_KEYS
    }
    return x, metrics, verdicts


def _plot_axis(ax, x, metrics, verdicts, title: str, xlabel: str):
    """Plot three lines (ARI / NN / AUC) vs x. Threshold as a dashed line."""
    for key, label, thr in METRIC_KEYS:
        ys = metrics[key]
        if all(v is None for v in ys):
            continue
        ys = np.array([float("nan") if v is None else v for v in ys], dtype=float)
        ax.plot(x, ys, "o-", label=label)
        ax.axhline(thr, linestyle="--", linewidth=0.8, alpha=0.4,
                   label=f"LEAK threshold ({label})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("metric value")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="best")


def _plot_heatmap(results: list, out_path: Path):
    """alpha × beta grid heatmap of Cosine AUC + LEAK verdict count."""
    rows = [r for r in results if r["sweep_axis"] == "grid"]
    if not rows:
        logger.warning("no grid rows, skipping heatmap")
        return
    alphas = sorted({r["override"]["dp_alpha"] for r in rows})
    betas = sorted({r["override"]["dp_answer_beta"] for r in rows})
    auc_grid = np.full((len(betas), len(alphas)), np.nan)
    for r in rows:
        ia = alphas.index(r["override"]["dp_alpha"])
        ib = betas.index(r["override"]["dp_answer_beta"])
        v = r["metrics"].get("cosine_auc")
        if v is not None:
            auc_grid[ib, ia] = float(v)

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(auc_grid, cmap="viridis", aspect="auto", origin="lower")
    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([f"{a:.2f}" for a in alphas])
    ax.set_yticks(range(len(betas)))
    ax.set_yticklabels([f"{b:.2f}" for b in betas])
    ax.set_xlabel("α (alpha)")
    ax.set_ylabel("β (answer_beta)")
    ax.set_title("dχ sensitivity — Cosine AUC heatmap")
    for i in range(auc_grid.shape[0]):
        for j in range(auc_grid.shape[1]):
            if not np.isnan(auc_grid[i, j]):
                ax.text(j, i, f"{auc_grid[i, j]:.3f}",
                        ha="center", va="center", color="white", fontsize=9)
    fig.colorbar(im, ax=ax, label="Cosine AUC")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    logger.info("saved %s", out_path)


def _render_all(manifest: dict, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    results = manifest.get("results", [])

    saved = []
    for axis, xlabel, title in [
        ("alpha", "α", "dχ sensitivity — α"),
        ("beta", "β", "dχ sensitivity — β"),
        ("cal", "calibration_steps", "dχ sensitivity — calibration_steps"),
    ]:
        # override key in manifest uses dp_-prefixed names
        override_key = {"alpha": "dp_alpha", "beta": "dp_answer_beta", "cal": "dp_calibration_steps"}[axis]
        x, metrics, verdicts = _extract_axis(results, axis, override_key)
        if not x:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        _plot_axis(ax, x, metrics, verdicts, title, xlabel)
        out_path = out_dir / f"dchi_sensitivity_{axis}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=140)
        plt.close(fig)
        saved.append(out_path)
        logger.info("saved %s", out_path)

    heatmap_path = out_dir / "dchi_sensitivity_grid_heatmap.png"
    _plot_heatmap(results, heatmap_path)
    saved.append(heatmap_path)

    overview_path = out_dir / "dchi_sweep_verdict.png"
    _plot_verdict_overview(results, overview_path)
    saved.append(overview_path)
    return saved


def _plot_verdict_overview(results: list, out_path: Path):
    """21 点 × 7 项子判定的 verdict 热力图（每个 cell 一个色块 + verdict 字符串）。"""
    # manifest metrics 中的 verdict key 实际命名
    sub_attack_order = [
        "h_u_mean_anova_verdict",      # 前向 类均值 ANOVA
        "h_u_norm_anova_verdict",      # 前向 类范数 ANOVA
        "ari_verdict",                  # 反向 K-Means ARI
        "nn_verdict",                   # 反向 1-NN 一致率
        "cosine_verdict",               # 反向 Cosine AUC
        "perm_verdict",                 # 反向 置换检验
        "magnitude_verdict",            # 反向 梯度幅度 ANOVA
    ]
    sub_attack_labels = [
        "H̃_U mean\nANOVA",
        "H̃_U norm\nANOVA",
        "K-Means\nARI",
        "1-NN\nagreement",
        "Cosine\nAUC",
        "Permutation\ntest",
        "Gradient\nmagnitude ANOVA",
    ]
    VERDICT_TO_CODE = {"LEAK_DETECTED": 0, "INCONCLUSIVE": 1, "PRIVACY_PRESERVED": 2}
    VERDICT_TO_COLOR = {0: "#d62728", 1: "#7f7f7f", 2: "#2ca02c"}

    rows = list(results)
    n = len(rows)
    if n == 0:
        logger.warning("no rows, skipping verdict overview")
        return

    matrix = np.full((n, len(sub_attack_order)), np.nan)
    labels = []
    for i, r in enumerate(rows):
        labels.append(r["label"])
        for j, sub_key in enumerate(sub_attack_order):
            v = r["metrics"].get(sub_key)
            if v in VERDICT_TO_CODE:
                matrix[i, j] = VERDICT_TO_CODE[v]

    # 行顺序：alpha → beta → cal → grid
    sweep_order = {"alpha": 0, "beta": 1, "cal": 2, "grid": 3}
    sort_idx = sorted(range(n), key=lambda i: (sweep_order.get(rows[i]["sweep_axis"], 9), i))
    rows = [rows[i] for i in sort_idx]
    labels = [labels[i] for i in sort_idx]
    matrix = matrix[sort_idx, :]

    fig, ax = plt.subplots(figsize=(9, max(6, n * 0.32)))
    for i in range(n):
        for j in range(len(sub_attack_order)):
            code = matrix[i, j]
            if np.isnan(code):
                continue
            ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=VERDICT_TO_COLOR[int(code)],
                                       edgecolor="white", linewidth=0.5))
            ax.text(j + 0.5, i + 0.5,
                    ["LEAK", "INC", "PRIV"][int(code)],
                    ha="center", va="center", fontsize=7, color="white", fontweight="bold")
    ax.set_xlim(0, len(sub_attack_order))
    ax.set_ylim(0, n)
    ax.set_xticks(np.arange(len(sub_attack_order)) + 0.5)
    ax.set_xticklabels(sub_attack_labels, fontsize=8)
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    # 按扫描维度分组画分隔线
    cur_axis = rows[0]["sweep_axis"]
    for i in range(1, n):
        if rows[i]["sweep_axis"] != cur_axis:
            ax.axhline(i, color="black", linewidth=0.8)
            cur_axis = rows[i]["sweep_axis"]
    # 在左侧标注扫描维度
    cur_axis = rows[0]["sweep_axis"]
    start = 0
    for i in range(1, n + 1):
        if i == n or rows[i]["sweep_axis"] != cur_axis:
            mid = (start + i - 1) / 2 + 0.5
            ax.text(-0.8, mid, cur_axis, rotation=90, va="center", ha="center",
                    fontsize=9, fontweight="bold", transform=ax.transData)
            cur_axis = rows[i]["sweep_axis"] if i < n else cur_axis
            start = i
    ax.set_title("dχ parameter sweep (21 points) — L-1 verdict per sub-attack\n"
                 "Red=LEAK_DETECTED  Gray=INCONCLUSIVE  Green=PRIVACY_PRESERVED",
                 fontsize=11)
    ax.set_aspect("auto")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    logger.info("saved %s", out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True,
                        help="Path to sensitivity_dchi_manifest_*.json")
    parser.add_argument("--out_dir", required=True,
                        help="Directory to write figures into")
    args = parser.parse_args()
    manifest = _load_manifest(Path(args.manifest))
    saved = _render_all(manifest, Path(args.out_dir))
    for s in saved:
        print(s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())