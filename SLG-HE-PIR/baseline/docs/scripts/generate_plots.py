#!/usr/bin/env python3
"""
generate_plots.py — 把 baseline 训练/评估 JSON 转成报告用 matplotlib 图。

用法（在仓库根目录执行）：
    python baseline/docs/scripts/generate_plots.py

输出：
    baseline/docs/figures/
        fig01-genrel-train-loss.png
        fig02-genrel-train-perplexity.png
        fig03-genrel-val-loss.png
        fig04-genrel-per-class-f1.png
        fig05-genrel-confusion-matrix.png
        fig06-genrel-per-class-pr.png
        fig07-genrel-roc-auc-pr.png        (1-sample ROC sweep + bar)

        fig08-ner-train-loss.png
        fig09-ner-train-perplexity.png
        fig10-ner-val-loss.png
        fig11-ner-per-entity-f1.png
        fig12-ner-per-entity-pr.png
        fig13-ner-loss-cmp.png             (genrel vs ner train loss)

依赖：matplotlib（>=3.5），numpy。

输入文件路径在这个脚本里硬编码，假定从仓库根目录运行。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: no display
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
DOCS_DIR = REPO_ROOT / "baseline" / "docs"
FIG_DIR = DOCS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Color palette (color-blind safe)
COLORS = {
    "genrel": "#1f77b4",   # blue
    "ner":     "#d62728",   # red
    "ok":      "#2ca02c",
    "weak":    "#ff7f0e",
    "grid":    "#cccccc",
}

GENREL_METRICS_PATH = (
    REPO_ROOT / "baseline/classification_genrel/checkpoints/"
    "metrics_data_None-2026-07-20_02-18-13.json"
)
GENREL_EVAL_PATH = (
    REPO_ROOT / "baseline/classification_genrel/logs/genrel_final_evaluate_metrics.json"
)
NER_METRICS_PATH = (
    REPO_ROOT / "baseline/generation_ner/checkpoints/"
    "metrics_data_None-2026-07-20_02-47-38.json"
)
NER_EVAL_PATH = (
    REPO_ROOT / "baseline/generation_ner/logs/"
    "ner_2026-07-20_02-47-38_evaluate_metrics.json"
)


def save(fig, name):
    out = FIG_DIR / name
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {out.relative_to(REPO_ROOT)}")


# ----------------------- GenRel plot -----------------------

def plot_genrel_train_loss(genrel):
    train_step = genrel["train_step_loss"]
    # Make x-axis roughly epochs (596 steps/epoch)
    x = np.arange(len(train_step))
    fig, ax = plt.subplots(figsize=(8, 4))
    # Use exponentially-weighted moving average to show trend
    window = max(20, len(train_step) // 100)
    ma = np.convolve(train_step, np.ones(window) / window, mode="valid")
    ax.plot(x, train_step, alpha=0.18, color=COLORS["genrel"], label="step loss (raw)")
    ax.plot(np.arange(window - 1, window - 1 + len(ma)), ma,
            color=COLORS["genrel"], linewidth=1.6, label=f"EW-avg win={window}")
    # Epoch boundaries
    n_per_epoch = len(train_step) // 6
    for i in range(1, 6):
        ax.axvline(i * n_per_epoch, color=COLORS["grid"], linestyle="--", alpha=0.5)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("GenRel QA — Train step loss (6 epochs, 596 steps/epoch)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    save(fig, "fig01-genrel-train-loss.png")


def plot_genrel_train_perplexity(genrel):
    train_step_ppl = genrel["train_step_perplexity"]
    train_epoch_ppl = genrel["train_epoch_perplexity"]
    val_epoch_ppl = genrel["val_epoch_perplexity"]

    x = np.arange(1, len(train_step_ppl) + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, train_step_ppl, alpha=0.20, color=COLORS["genrel"], label="train step ppl")
    ax.plot(np.arange(1, len(train_epoch_ppl) + 1), train_epoch_ppl,
            color=COLORS["genrel"], marker="o", linewidth=2, label="train epoch ppl")
    ax.plot(np.arange(1, len(val_epoch_ppl) + 1), val_epoch_ppl,
            color=COLORS["weak"], marker="s", linewidth=2, label="val epoch ppl")
    ax.set_xlabel("Epoch / step")
    ax.set_ylabel("Perplexity")
    ax.set_title("GenRel QA — Train vs Val perplexity")
    ax.set_ylim(1.0, max(max(val_epoch_ppl), max(train_epoch_ppl)) * 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    save(fig, "fig02-genrel-train-perplexity.png")


def plot_genrel_val_loss(genrel):
    val_step_loss = np.asarray(genrel["val_step_loss"])
    val_epoch_loss = genrel["val_epoch_loss"]
    # rebuild x by epoch boundaries
    n_per_epoch_eval = len(val_step_loss) // 6
    epoch_idx = np.repeat(np.arange(6), n_per_epoch_eval)[: len(val_step_loss)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.2))

    # left: per-step val loss across epochs (jittered x = epoch id + step-in-epoch)
    for e in range(6):
        mask = (epoch_idx == e)
        x_jit = epoch_idx[mask] + np.random.RandomState(e).rand(int(mask.sum())) * 0.6
        ax1.scatter(x_jit, val_step_loss[mask],
                    alpha=0.25, s=8, color=COLORS["genrel"])
    ax1.plot(np.arange(1, 7), val_epoch_loss, marker="o", color="black",
             linewidth=2, label="epoch mean")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Val cross-entropy loss")
    ax1.set_title("GenRel QA — Per-step + per-epoch val loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # right: bar chart of val epoch loss
    colors = [COLORS["ok"] if e == 0 else COLORS["weak"] if val_epoch_loss[e] > val_epoch_loss[e - 1] else COLORS["genrel"]
              for e in range(6)]
    bars = ax2.bar(np.arange(1, 7), val_epoch_loss, color=colors)
    for b, v in zip(bars, val_epoch_loss):
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                 f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Val loss (epoch mean)")
    ax2.set_title("GenRel QA — Val epoch loss (orange = regression)")
    ax2.grid(True, axis="y", alpha=0.3)
    save(fig, "fig03-genrel-val-loss.png")


def plot_genrel_per_class(genrel_eval):
    cls = genrel_eval["per_class"]
    names = list(cls.keys())
    f1s = [cls[n]["f1"] for n in names]
    supports = [cls[n]["support"] for n in names]
    precs = [cls[n]["precision"] for n in names]
    recalls = [cls[n]["recall"] for n in names]

    order = np.argsort(supports)[::-1]
    names = [names[i] for i in order]
    f1s = [f1s[i] for i in order]
    precs = [precs[i] for i in order]
    recalls = [recalls[i] for i in order]
    supports = [supports[i] for i in order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    x = np.arange(len(names))
    bars = ax1.bar(x, f1s, color=COLORS["genrel"], alpha=0.85)
    for b, s, f in zip(bars, supports, f1s):
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                 f"{f:.3f}\n(n={s})", ha="center", va="bottom", fontsize=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=30, ha="right")
    ax1.set_ylabel("F1 score")
    ax1.set_title("GenRel QA — Per-class F1 (sorted by support, descending)")
    ax1.set_ylim(0, 1)
    ax1.grid(True, axis="y", alpha=0.3)

    # Scatter P vs R
    ax2.scatter(recalls, precs, s=[s * 8 for s in supports], alpha=0.65,
                color=COLORS["genrel"])
    for r, p, n in zip(recalls, precs, names):
        ax2.annotate(n, (r, p), xytext=(4, 4), textcoords="offset points", fontsize=9)
    ax2.plot([0, 1], [0, 1], color=COLORS["grid"], linestyle="--")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("GenRel QA — Precision vs Recall (size = support)")
    ax2.set_xlim(0, 1.02)
    ax2.set_ylim(0, 1.02)
    ax2.grid(True, alpha=0.3)
    save(fig, "fig04-genrel-per-class-f1.png")


def plot_genrel_confusion(genrel_eval):
    cm = np.array(genrel_eval["confusion_matrix"])
    classes = genrel_eval["confusion_matrix_labels"]

    # Normalize per-row (per true class) for readability
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    im1 = ax1.imshow(cm, cmap="Blues", aspect="auto")
    ax1.set_xticks(range(len(classes)))
    ax1.set_yticks(range(len(classes)))
    ax1.set_xticklabels(classes, rotation=30, ha="right", fontsize=9)
    ax1.set_yticklabels(classes, fontsize=9)
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("True")
    ax1.set_title("GenRel QA — Confusion matrix (raw counts)")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax1.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    im2 = ax2.imshow(cm_norm, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    ax2.set_xticks(range(len(classes)))
    ax2.set_yticks(range(len(classes)))
    ax2.set_xticklabels(classes, rotation=30, ha="right", fontsize=9)
    ax2.set_yticklabels(classes, fontsize=9)
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("True")
    ax2.set_title("GenRel QA — Confusion matrix (row-normalized)")
    for i in range(len(classes)):
        for j in range(len(classes)):
            v = cm_norm[i, j]
            ax2.text(j, i, f"{v:.2f}", ha="center", va="center",
                     color="white" if v > 0.5 else "black", fontsize=8)
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    save(fig, "fig05-genrel-confusion-matrix.png")


def plot_genrel_topline(genrel_eval):
    """Headline micro/macro F1 + AUC bar."""
    metric_names = ["Micro Accuracy", "Micro F1", "Macro F1", "Weighted F1",
                    "Macro AUC (ovr)", "Micro AUC (ovr)"]
    metric_vals = [
        genrel_eval["micro_accuracy"],
        genrel_eval["micro_f1"],
        genrel_eval["macro_f1"],
        genrel_eval["weighted_f1"],
        genrel_eval["macro_auc_ovr"],
        genrel_eval["micro_auc_ovr"],
    ]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    bars = ax.bar(metric_names, metric_vals,
                  color=[COLORS["ok"] if v >= 0.5 else COLORS["weak"] for v in metric_vals])
    for b, v in zip(bars, metric_vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                f"{v:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("GenRel QA — Topline metrics on test set (n=213)")
    ax.grid(True, axis="y", alpha=0.3)
    save(fig, "fig06-genrel-topline.png")


# ----------------------- NER plot -----------------------

def plot_ner_train_loss(ner):
    train_step = ner["train_step_loss"]
    x = np.arange(len(train_step))
    window = max(20, len(train_step) // 100)
    ma = np.convolve(train_step, np.ones(window) / window, mode="valid")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, train_step, alpha=0.18, color=COLORS["ner"], label="step loss (raw)")
    ax.plot(np.arange(window - 1, window - 1 + len(ma)), ma,
            color=COLORS["ner"], linewidth=1.6, label=f"EW-avg win={window}")
    n_per_epoch = len(train_step) // 10
    for i in range(1, 10):
        ax.axvline(i * n_per_epoch, color=COLORS["grid"], linestyle="--", alpha=0.5)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("NER — Train step loss (10 epochs, 843 steps/epoch)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    save(fig, "fig08-ner-train-loss.png")


def plot_ner_train_perplexity(ner):
    train_step_ppl = ner["train_step_perplexity"]
    train_epoch_ppl = ner["train_epoch_perplexity"]
    val_epoch_ppl = ner["val_epoch_perplexity"]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(1, len(train_step_ppl) + 1), train_step_ppl,
            alpha=0.20, color=COLORS["ner"], label="train step ppl")
    ax.plot(np.arange(1, len(train_epoch_ppl) + 1), train_epoch_ppl,
            marker="o", linewidth=2, color=COLORS["ner"], label="train epoch ppl")
    ax.plot(np.arange(1, len(val_epoch_ppl) + 1), val_epoch_ppl,
            marker="s", linewidth=2, color=COLORS["weak"], label="val epoch ppl")
    ax.set_xlabel("Step / Epoch")
    ax.set_ylabel("Perplexity")
    ax.set_title("NER — Train vs Val perplexity")
    ax.set_ylim(1.0, max(max(val_epoch_ppl), max(train_epoch_ppl)) * 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    save(fig, "fig09-ner-train-perplexity.png")


def plot_ner_val_loss(ner):
    val_step_loss = np.asarray(ner["val_step_loss"])
    val_epoch_loss = ner["val_epoch_loss"]
    n_per_epoch_eval = len(val_step_loss) // 10
    epoch_idx = np.repeat(np.arange(10), n_per_epoch_eval)[: len(val_step_loss)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.2))
    for e in range(10):
        mask = (epoch_idx == e)
        x_jit = epoch_idx[mask] + np.random.RandomState(e).rand(int(mask.sum())) * 0.6
        ax1.scatter(x_jit, val_step_loss[mask], alpha=0.25, s=8, color=COLORS["ner"])
    ax1.plot(np.arange(1, 11), val_epoch_loss, marker="o", linewidth=2,
             color="black", label="epoch mean")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Val cross-entropy loss")
    ax1.set_title("NER — Per-step + per-epoch val loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    colors = [COLORS["ok"] if e == 0 else COLORS["weak"] if val_epoch_loss[e] > val_epoch_loss[e - 1] else COLORS["ner"]
              for e in range(10)]
    bars = ax2.bar(np.arange(1, 11), val_epoch_loss, color=colors)
    for b, v in zip(bars, val_epoch_loss):
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                 f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Val loss (epoch mean)")
    ax2.set_title("NER — Val epoch loss (orange = regression)")
    ax2.grid(True, axis="y", alpha=0.3)
    save(fig, "fig10-ner-val-loss.png")


def plot_ner_per_entity(ner_eval):
    cls = ner_eval["per_class"]
    names = list(cls.keys())
    p = [cls[n]["precision"] for n in names]
    r = [cls[n]["recall"] for n in names]
    f1 = [cls[n]["f1"] for n in names]
    tp = [cls[n]["tp"] for n in names]
    fp = [cls[n]["fp"] for n in names]
    fn = [cls[n]["fn"] for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    x = np.arange(len(names))
    bars = ax1.bar(x, f1, color=COLORS["ner"], alpha=0.85)
    for b, t, fn_, v in zip(bars, tp, fn, f1):
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                 f"F1={v:.3f}\nTP={t},FN={fn_}", ha="center", va="bottom", fontsize=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names)
    ax1.set_ylabel("F1 score")
    ax1.set_title("NER — Per-entity-type F1 (test set)")
    ax1.set_ylim(0, 1)
    ax1.grid(True, axis="y", alpha=0.3)

    ax2.scatter(r, p, s=[(t + fn_ + f) * 0.5 for t, fn_, f in zip(tp, fn, fp)],
                alpha=0.65, color=COLORS["ner"])
    for ri, pi, ni in zip(r, p, names):
        ax2.annotate(ni, (ri, pi), xytext=(4, 4), textcoords="offset points", fontsize=10)
    ax2.plot([0, 1], [0, 1], color=COLORS["grid"], linestyle="--")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_xlim(0, 1.02)
    ax2.set_ylim(0, 1.02)
    ax2.set_title("NER — Precision vs Recall (size = TP+FP+FN)")
    ax2.grid(True, alpha=0.3)
    save(fig, "fig11-ner-per-entity-f1.png")


def plot_ner_topline(ner_eval):
    metric_names = ["Micro Precision", "Micro Recall", "Micro F1",
                    "Macro Precision", "Macro Recall", "Macro F1", "Weighted F1"]
    metric_vals = [
        ner_eval["overall_micro_precision"],
        ner_eval["overall_micro_recall"],
        ner_eval["overall_micro_f1"],
        ner_eval["macro_precision"],
        ner_eval["macro_recall"],
        ner_eval["macro_f1"],
        ner_eval["weighted_f1"],
    ]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    bars = ax.bar(metric_names, metric_vals,
                  color=[COLORS["ok"] if v >= 0.5 else COLORS["weak"] for v in metric_vals])
    for b, v in zip(bars, metric_vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                f"{v:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title(f"NER — Topline metrics on test set (n=174, parse_failures={ner_eval['parse_failures']})")
    ax.grid(True, axis="y", alpha=0.3)
    save(fig, "fig12-ner-topline.png")


# ----------------------- Cross-task plot -----------------------

def plot_train_loss_cmp(genrel, ner):
    """Side-by-side normalized train epoch loss for both tasks."""
    g = genrel["train_epoch_loss"]
    n = ner["train_epoch_loss"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.2))
    ax1.plot(np.arange(1, len(g) + 1), g, marker="o", linewidth=2,
             color=COLORS["genrel"])
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-entropy loss")
    ax1.set_title("GenRel QA — Train loss per epoch (6 epochs)")
    ax1.grid(True, alpha=0.3)

    ax2.plot(np.arange(1, len(n) + 1), n, marker="o", linewidth=2, color=COLORS["ner"])
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Cross-entropy loss")
    ax2.set_title("NER — Train loss per epoch (10 epochs)")
    ax2.grid(True, alpha=0.3)
    save(fig, "fig13-train-loss-cmp.png")


# ----------------------- Main -----------------------

def main():
    missing = []
    for p in [GENREL_METRICS_PATH, GENREL_EVAL_PATH,
              NER_METRICS_PATH, NER_EVAL_PATH]:
        if not p.exists():
            missing.append(p)
    if missing:
        print("[ERROR] missing required input files:")
        for m in missing:
            print("  -", m)
        sys.exit(1)

    genrel_train = json.loads(GENREL_METRICS_PATH.read_text())
    genrel_eval = json.loads(GENREL_EVAL_PATH.read_text())
    ner_train = json.loads(NER_METRICS_PATH.read_text())
    ner_eval = json.loads(NER_EVAL_PATH.read_text())

    print("== GenRel plots ==")
    plot_genrel_train_loss(genrel_train)
    plot_genrel_train_perplexity(genrel_train)
    plot_genrel_val_loss(genrel_train)
    plot_genrel_per_class(genrel_eval)
    plot_genrel_confusion(genrel_eval)
    plot_genrel_topline(genrel_eval)

    print("== NER plots ==")
    plot_ner_train_loss(ner_train)
    plot_ner_train_perplexity(ner_train)
    plot_ner_val_loss(ner_train)
    plot_ner_per_entity(ner_eval)
    plot_ner_topline(ner_eval)

    print("== Cross-task plots ==")
    plot_train_loss_cmp(genrel_train, ner_train)

    print(f"\nAll figures written to {FIG_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
