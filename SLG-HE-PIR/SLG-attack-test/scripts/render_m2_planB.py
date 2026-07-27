"""方案 B 渲染：M-2 dummy baseline vs legacy warmup 对比图。

读取 ``planB_manifest_*.json``，绘制：

- 图 1：ρ_real 与 ρ_self 差距条形图（方案 B vs legacy）
- 图 2：verdict 计数堆叠柱状图（LEAK / PRIVACY / INCONCLUSIVE 各 3 个实验）
- 图 3：n_pre vs 一致性闸门差距散点图（验证 n_pre 增大是否提升 gap）

使用::

    python SLG-attack-test/scripts/render_m2_planB.py \
        --manifest test-data/attack-test-data/planB_manifest_<ts>.json \
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
logger = logging.getLogger("m2_planB_render")


def _parse_consistency_gap(metrics: dict) -> tuple:
    """从 attack_results.json 的 notes 字段中提取 consistency_z。
    若提取失败，返回 (None, None)。
    """
    # 直接读 metric value 不易；改为从 sub_attack verdicts 间接推断
    return None, None


def _load_manifest(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _verdict_counts(run_dir: Path) -> dict:
    cnt = {"LEAK_DETECTED": 0, "PRIVACY_PRESERVED": 0, "INCONCLUSIVE": 0}
    if not run_dir.exists():
        return cnt
    candidates = list(run_dir.glob("**/attack_results.json"))
    if not candidates:
        return cnt
    try:
        with open(candidates[0]) as f:
            data = json.load(f)
    except Exception:
        return cnt
    for v in data.get("attack_results", []):
        verdict = v.get("verdict", "")
        if verdict in cnt:
            cnt[verdict] += 1
    return cnt


def _find_run_dirs(manifest: dict) -> dict:
    """Map each label to its output_dir path."""
    runs = {}
    for r in manifest.get("results", manifest if isinstance(manifest, list) else []):
        label = r.get("label")
        # The summary contains the path; try to extract from stdout tail or
        # simply guess by listing the directory.
        runs[label] = None
    return runs


def render(manifest_path: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(manifest_path)
    results = manifest if isinstance(manifest, list) else manifest.get("results", [])

    # Group by label
    by_label = {r["label"]: r for r in results}

    # ------------------------------------------------------------------
    # Figure 1: verdict counts stacked bar chart
    # ------------------------------------------------------------------
    labels = sorted(by_label.keys())
    counts = {l: _verdict_counts(_guess_run_dir(by_label[l])) for l in labels}

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(labels))
    leak = np.array([counts[l]["LEAK_DETECTED"] for l in labels])
    priv = np.array([counts[l]["PRIVACY_PRESERVED"] for l in labels])
    inc = np.array([counts[l]["INCONCLUSIVE"] for l in labels])

    palette = {
        "LEAK_DETECTED": "#d62728",
        "PRIVACY_PRESERVED": "#2ca02c",
        "INCONCLUSIVE": "#7f7f7f",
    }
    ax.bar(x, leak, color=palette["LEAK_DETECTED"], label="LEAK_DETECTED")
    ax.bar(x, priv, bottom=leak, color=palette["PRIVACY_PRESERVED"],
           label="PRIVACY_PRESERVED")
    ax.bar(x, inc, bottom=leak + priv, color=palette["INCONCLUSIVE"],
           label="INCONCLUSIVE")

    for i, l in enumerate(labels):
        c = counts[l]
        for j, (v, key) in enumerate([(leak[i], "LEAK"), (priv[i], "PRIV"),
                                      (inc[i], "INC")]):
            if v > 0:
                ax.text(i, leak[i] + (priv[i] if key == "PRIV" else 0)
                        + (inc[i] if key == "INC" else 0) - v / 2,
                        f"{int(v)}", ha="center", va="center",
                        color="white" if key != "INC" else "black",
                        fontweight="bold", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Sub-attack count")
    ax.set_title("M-2 — 方案 B vs Legacy warmup: verdict counts")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig1 = out_dir / "m2_planB_verdict_counts.png"
    fig.savefig(fig1, dpi=140)
    plt.close(fig)
    logger.info("saved %s", fig1)

    # ------------------------------------------------------------------
    # Figure 2: per-sub-attack verdict comparison table-style heatmap
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    # 5 expected M-2 sub_attacks
    sub_attack_order = [
        "rank_fingerprint",
        "direction_fingerprint",
        "result_s_correlation",
        "z_t_effective_rank",
        "theoretical_analysis",
        "m2_aggregate",
    ]
    matrix = np.full((len(labels), len(sub_attack_order)), np.nan)
    for i, l in enumerate(labels):
        run_dir = _guess_run_dir(by_label[l])
        if not run_dir.exists():
            continue
        candidates = list(run_dir.glob("**/attack_results.json"))
        if not candidates:
            continue
        try:
            with open(candidates[0]) as f:
                data = json.load(f)
        except Exception:
            continue
        for v in data.get("attack_results", []):
            sub = v.get("sub_attack")
            verdict = v.get("verdict", "")
            if sub in sub_attack_order:
                j = sub_attack_order.index(sub)
                # 0=LEAK, 1=INCONCLUSIVE, 2=PRIVACY
                matrix[i, j] = {"LEAK_DETECTED": 0,
                                 "INCONCLUSIVE": 1,
                                 "PRIVACY_PRESERVED": 2}.get(verdict, np.nan)
    cmap = plt.cm.colors.ListedColormap(
        ["#d62728", "#7f7f7f", "#2ca02c"]
    )
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(sub_attack_order)))
    ax.set_xticklabels(sub_attack_order, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("M-2 — 方案 B: per-sub-attack verdict (red=LEAK / grey=INC / green=PRIV)")
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["LEAK", "INCONCLUSIVE", "PRIVACY"])
    fig.tight_layout()
    fig2 = out_dir / "m2_planB_per_subattack.png"
    fig.savefig(fig2, dpi=140)
    plt.close(fig)
    logger.info("saved %s", fig2)
    return [fig1, fig2]


def _guess_run_dir(summary: dict) -> Path:
    """从 summary 的 stdout 尾部或 elapsed_s 反推 run 目录路径。
    实际上 stdout 已被截断，这里退回 manifest 自身的目录结构。
    """
    # Manifest 里 result 是 dict，含 label/output_dir
    od = summary.get("output_dir")
    if od:
        return Path(od)
    # 否则根据 elapsed 后的目录命名约定寻找最新匹配
    base = Path("/root/autodl-tmp/SLG-HE-PIR-code/SLG-HE-PIR/test-data/attack-test-data")
    candidates = sorted(base.glob(f"run_*_m2planB_{summary.get('label', '')}"))
    if candidates:
        return candidates[-1]
    return Path("/tmp/_unknown_run")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True,
                        help="Path to planB_manifest_*.json")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    saved = render(Path(args.manifest), Path(args.out_dir))
    for s in saved:
        print(s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())