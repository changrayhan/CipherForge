#!/usr/bin/env python3
"""Generate derived comparison artifacts from Baseline + SLG epoch_metrics.jsonl.

Outputs into derived/:
  - best_epoch_comparison.json  : best per-method (by macro_f1)
  - avg_comparison.json         : 5-epoch averages
  - per_class_comparison.json   : per-class P/R/F1 at best epoch
  - precision_gradient.md       : Baseline -> SLG precision gradient table
  - eval_caliber_diff.md        : explains metric-computation differences
"""
import json
import sys
from pathlib import Path

ROOT = Path("/root/autodl-tmp/CipherForgeCode/SLG-HE-PIR/test-data/AccuracyAblationTestData")
BASELINE_JSONL = ROOT / "baseline" / "epoch_metrics.jsonl"
SLG_JSONL = ROOT / "slg" / "epoch_metrics.jsonl"
DERIVED = ROOT / "derived"
DERIVED.mkdir(parents=True, exist_ok=True)


# Metrics we want to compare across all variants
KEY_METRICS = [
    ("micro_f1",         "val_bt_micro_f1"),
    ("macro_f1",         "val_bt_macro_f1"),
    ("weighted_f1",      "val_bt_weighted_f1"),
    ("micro_accuracy",   "val_micro_accuracy"),
    ("macro_precision",  "val_macro_precision"),
    ("macro_recall",     "val_macro_recall"),
    ("micro_auc",        "val_bt_micro_roc_auc"),
    ("macro_auc",        "val_bt_macro_roc_auc"),
    ("parse_failures",   "val_bt_n_parse_failures"),
    ("n_samples",        "val_samples"),
    ("train_steps",      "train_steps"),
]


def load_jsonl(path: Path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def best_epoch(rows, key="val_bt_macro_f1"):
    return max(rows, key=lambda r: r.get(key) or -1)


def average(rows, key):
    vals = [r.get(key) for r in rows if r.get(key) is not None]
    return sum(vals) / len(vals) if vals else None


def fmt(v, decimals=4):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    return str(v)


def gen_best_epoch_comparison(baseline_rows, slg_rows):
    bb = best_epoch(baseline_rows)
    sb = best_epoch(slg_rows)
    result = {
        "baseline": {"label": "Baseline (Plaintext, 2-target LoRA)", "best_epoch": bb["epoch"], "metrics": {}},
        "slg":      {"label": "SLG-HE-PIR (3-party HE-PIR, 7-target LoRA)", "best_epoch": sb["epoch"], "metrics": {}},
        "delta":    {"label": "Baseline − SLG", "metrics": {}},
    }
    for label, key in KEY_METRICS:
        b = bb.get(key)
        s = sb.get(key)
        result["baseline"]["metrics"][label] = b
        result["slg"]["metrics"][label] = s
        if isinstance(b, (int, float)) and isinstance(s, (int, float)):
            result["delta"]["metrics"][label] = round(b - s, 6)
        else:
            result["delta"]["metrics"][label] = None
    return result


def gen_avg_comparison(baseline_rows, slg_rows):
    result = {
        "baseline_5epoch_avg": {},
        "slg_5epoch_avg": {},
        "delta": {},
    }
    for label, key in KEY_METRICS:
        ba = average(baseline_rows, key)
        sa = average(slg_rows, key)
        result["baseline_5epoch_avg"][label] = ba
        result["slg_5epoch_avg"][label] = sa
        if isinstance(ba, (int, float)) and isinstance(sa, (int, float)):
            result["delta"][label] = round(ba - sa, 6)
        else:
            result["delta"][label] = None
    return result


def gen_per_class_comparison(baseline_rows, slg_rows):
    bb = best_epoch(baseline_rows)
    sb = best_epoch(slg_rows)
    classes = ["pathological", "modulatory", "expression change",
               "diagnosis", "therapy", "no relation", "relation undefined"]
    result = {"baseline_best_epoch": bb["epoch"], "slg_best_epoch": sb["epoch"], "per_class": {}}
    for cls in classes:
        bcls = (bb.get("per_class") or {}).get(cls, {})
        scls = (sb.get("per_class") or {}).get(cls, {})
        result["per_class"][cls] = {
            "baseline": bcls,
            "slg":      scls,
            "delta_f1": (
                round(bcls.get("f1", 0) - scls.get("f1", 0), 6)
                if bcls.get("f1") is not None and scls.get("f1") is not None
                else None
            ),
        }
    return result


def gen_precision_gradient_md(best_cmp, avg_cmp):
    md = []
    md.append("# Precision Gradient: Baseline → SLG (Task A, BioTriplex 7-class)\n")
    md.append("> **⚠️ Caliber warning**: Baseline uses `evaluate_metrics.py` (sklearn, "
              "keep parse_idx=-1). SLG uses `compute_classification_metrics` (replace "
              "missing predictions with `relation undefined`, custom macro-auc). See "
              "`eval_caliber_diff.md`. Numbers below are *raw, as-reported* — they are "
              "**not** a clean encryption-tax estimate.\n")
    md.append("## Best-epoch comparison\n")
    md.append("| metric | Baseline (best) | SLG (best) | Δ (Baseline−SLG) |")
    md.append("|--------|----------------:|-----------:|------------------:|")
    for label, _ in KEY_METRICS:
        b = best_cmp["baseline"]["metrics"].get(label)
        s = best_cmp["slg"]["metrics"].get(label)
        d = best_cmp["delta"]["metrics"].get(label)
        md.append(f"| {label} | {fmt(b)} | {fmt(s)} | {fmt(d)} |")
    md.append("")
    md.append(f"Baseline best epoch = {best_cmp['baseline']['best_epoch']}, "
              f"SLG best epoch = {best_cmp['slg']['best_epoch']}\n")

    md.append("## 5-epoch average comparison\n")
    md.append("| metric | Baseline avg | SLG avg | Δ (Baseline−SLG) |")
    md.append("|--------|-------------:|--------:|------------------:|")
    for label, _ in KEY_METRICS:
        b = avg_cmp["baseline_5epoch_avg"].get(label)
        s = avg_cmp["slg_5epoch_avg"].get(label)
        d = avg_cmp["delta"].get(label)
        md.append(f"| {label} | {fmt(b)} | {fmt(s)} | {fmt(d)} |")
    md.append("")

    md.append("## Per-class F1 (best epoch)\n")
    md.append("| class | Baseline F1 | SLG F1 | Δ F1 |")
    md.append("|-------|------------:|-------:|-----:|")
    pc = gen_per_class_comparison(load_jsonl(BASELINE_JSONL), load_jsonl(SLG_JSONL))
    for cls, data in pc["per_class"].items():
        bf = (data.get("baseline") or {}).get("f1")
        sf = (data.get("slg") or {}).get("f1")
        df = data.get("delta_f1")
        md.append(f"| {cls} | {fmt(bf)} | {fmt(sf)} | {fmt(df)} |")
    md.append("")

    md.append("## Data caliber caveats\n")
    md.append("- **Sample count**: Baseline 213 vs SLG 203 (-10). Likely root cause: "
              "different dataset class implementations (`biotriplex_qakshot_dataset.py` "
              "vs `src/data/biotriplex_dataset.py`) yielding different doc_key sets.\n"
              "- **Train steps**: Baseline 596 vs SLG 734 (+138). Same root cause.\n"
              "- **Eval code paths differ**: see `eval_caliber_diff.md`.\n")
    md.append("## Interpretation\n")
    md.append("As-reported gap: ≈ 9-11 pp on macro_f1, ≈ 9-10 pp on micro_f1, ≈ 9-10 pp on macro_auc.\n"
              "After accounting for target_modules (2→7) and quantization (Q0/Q1/Q2/Q3), "
              "the residual gap is the **BFV-encryption tax** — to be quantified by the "
              "AccuracyAblationTest sub-package.\n")
    return "\n".join(md) + "\n"


def gen_eval_caliber_diff_md():
    return """# Evaluation-caliber differences: Baseline vs SLG

Baseline uses `baseline/classification_genrel/scripts/evaluate_metrics.py` (sklearn-backed).
SLG uses `src/training/biotriplex_metrics.py::compute_classification_metrics`.

| 维度 | Baseline (`evaluate_metrics.py`) | SLG (`biotriplex_metrics.py`) |
|------|----------------------------------|--------------------------------|
| 缺失预测处理 | 保留 `pred_idx=-1`，最后 y_pred_safe 替换为 0 | 替换为 "relation undefined" (idx=6) |
| Multilabel F1 输入 | 把 single-label 转成 binary 7-vec | 解析 multi-letter 字符串 `j),o)` |
| macro_roc_auc_ovr | sklearn 标准 ovr（包含无 support 的类） | 仅在 y_true 含该类时计算，取有效类均值 |
| ROC AUC 的 y_score | softmax over 7-class logits | softmax over 7-class logits（相同路径）|
| Per-class P/R/F1 | 自实现的 TP/FP/FN + zero_division | sklearn `precision_score` + `f1_score` |
| Confusion matrix | sklearn 完整 | sklearn 完整 |

**结论**:
1. parse_failures 都为 0 时 (本次数据)，缺失预测处理差异**不会影响最终指标**（因为没有失败样本）
2. macro_auc 计算路径**不同**：SLG 排除 y_true 中没有的类——本次测试集中 "no relation" 和 "relation undefined" 在 SLG 全部 support=0，所以 SLG 实际只在 5 个类上算 macro_auc，而 Baseline 在 7 个类上算（其中两个为 0）→ **SLG 的 macro_auc 比 Baseline 的更"乐观"**
3. multilabel F1 的解析路径**相同**（因为生成为单字母）

## 建议

为了获得**统一口径**的 Baseline vs SLG 对比，新会话的 `report_generator.py` 必须：

```python
from src.training.biotriplex_metrics import compute_classification_metrics
# 从 baseline raw_inference/infer_outputs_epoch_XXX.json 抽 answer + logits
# 重新计算，统一口径
```

或者在 QUANT_ABLATION_REPORT.md 中**明确标注** "本报告中的 Baseline 指标来自 sklearn 口径，SLG 指标来自自定义口径，两者不可直接相减"。
"""


def main():
    baseline_rows = load_jsonl(BASELINE_JSONL)
    slg_rows = load_jsonl(SLG_JSONL)
    print(f"loaded baseline rows: {len(baseline_rows)}, slg rows: {len(slg_rows)}")

    best = gen_best_epoch_comparison(baseline_rows, slg_rows)
    avg = gen_avg_comparison(baseline_rows, slg_rows)
    pc = gen_per_class_comparison(baseline_rows, slg_rows)

    (DERIVED / "best_epoch_comparison.json").write_text(
        json.dumps(best, indent=2, ensure_ascii=False) + "\n"
    )
    print(f"wrote {DERIVED / 'best_epoch_comparison.json'}")

    (DERIVED / "avg_comparison.json").write_text(
        json.dumps(avg, indent=2, ensure_ascii=False) + "\n"
    )
    print(f"wrote {DERIVED / 'avg_comparison.json'}")

    (DERIVED / "per_class_comparison.json").write_text(
        json.dumps(pc, indent=2, ensure_ascii=False) + "\n"
    )
    print(f"wrote {DERIVED / 'per_class_comparison.json'}")

    (DERIVED / "precision_gradient.md").write_text(
        gen_precision_gradient_md(best, avg)
    )
    print(f"wrote {DERIVED / 'precision_gradient.md'}")

    (DERIVED / "eval_caliber_diff.md").write_text(gen_eval_caliber_diff_md())
    print(f"wrote {DERIVED / 'eval_caliber_diff.md'}")


if __name__ == "__main__":
    main()