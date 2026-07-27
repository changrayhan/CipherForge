#!/usr/bin/env python3
"""
evaluate_metrics.py — BioTriplex GenRel QA 分类任务评估脚本

输入：
  - inference.py 生成的 outputs JSON（包含 generation 文本 + 7 类 logits）
  - 训练期间生成的 gold JSONL：{data_path}/{split}_gold_general_qa.txt

输出：
  - 控制台打印指标 + 写到 results/evaluate_metrics.json

指标（按用户最终决策）：
  1. 多标签 F1（每个 relation 一个二分类问题）
  2. Macro F1（7 类 macro average）
  3. Macro ROC AUC（one-vs-rest, multi_class='ovr'）
"""

import argparse
import json
import os
import re
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


GENERAL_RELATIONS = [
    "pathological",
    "modulatory",
    "expression change",
    "diagnosis",
    "therapy",
    "no relation",
    "relation undefined",
]

# 在 prompt 中出现的 7 个选项标签字符
OPTION_LETTERS = ["a", "b", "c", "d", "e", "f", "g"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--outputs_json", required=True,
                   help="inference.py 写出的 outputs JSON 路径")
    p.add_argument("--gold_jsonl", required=True,
                   help="gold 文件路径：{split}_gold_general_qa.txt")
    p.add_argument("--results_dir", default=None,
                   help="保存指标 JSON 的目录（默认与 outputs_json 同目录）")
    p.add_argument("--save_prefix", default="",
                   help="结果 JSON 文件名前缀")
    return p.parse_args()


def letter_to_relation(text):
    """从生成的 'a)' / 'b)' 等字符串解析 relation。失败返回 None。"""
    if text is None:
        return None
    t = text.strip().lower()
    if not t:
        return None
    head = t[0]
    if head in OPTION_LETTERS:
        idx = OPTION_LETTERS.index(head)
        return GENERAL_RELATIONS[idx]
    return None


def extract_logits_from_entry(entry):
    """从 outputs JSON 单条 entry 抽取 logits 向量（list of 7 floats）。

    期望 entry 形如 {"answer": "a)", "logits": [0.1, 0.2, ...]}；
    若缺失，返回 None。
    """
    if isinstance(entry, dict) and "logits" in entry:
        lg = entry["logits"]
        if isinstance(lg, list) and len(lg) == 7:
            return [float(x) for x in lg]
    return None


def load_gold(gold_path):
    """读取 {split}_gold_general_qa.txt，返回 {doc_key: relation_label}。

    Gold JSON 每行形如:
        {doc_key, output='c)', input=..., relation={gene, disease, relation=fine}, ...}

    'output' 是细粒度 relation 经过 triplet_to_answer 转成的字母（a..g），
    对应的 coarse-general relation 来自 item['relation']['relation']。
    本函数返回的是 *coarse-general label* (pathological / modulatory / ...)。
    """
    gold = {}
    # Coarse mapping table mirroring GENERAL_REL in the QA dataset.
    fine_to_general = {
        # pathological
        "pathological role": "pathological",
        "causative activation": "pathological",
        "causative inhibition": "pathological",
        "causative mutation": "pathological",
        "associated mutation": "pathological",
        # modulatory
        "modulator decrease disease": "modulatory",
        "modulator increase disease": "modulatory",
        "genetic susceptibility": "modulatory",
        # expression change
        "increased expression": "expression change",
        "decreased expression": "expression change",
        "dysregulation": "expression change",
        # diagnosis
        "biomarker": "diagnosis",
        "diagnostic tool": "diagnosis",
        "epigenetic marker": "diagnosis",
        "prognostic indicator": "diagnosis",
        "positive prognostic marker": "diagnosis",
        "negative prognostic marker": "diagnosis",
        # therapy
        "therapy resistance": "therapy",
        "therapeutic target": "therapy",
        # no relation / undefined (passed through)
        "no relation": "no relation",
        "relation undefined": "relation undefined",
    }
    with open(gold_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            fine_rel = (item.get("relation") or {}).get("relation", "").lower().strip()
            if fine_rel in fine_to_general:
                gold[item["doc_key"]] = fine_to_general[fine_rel]
            # If no mapping found we skip; main() will count it as parse failure
    return gold


def load_outputs(outputs_path):
    """读取 inference.py 写出的 outputs JSON。

    支持两种格式：
      1) {"doc_key": "a)"} — 仅有 answer 文本
      2) {"doc_key": {"answer": "a)", "logits": [...]}} — 带 logits
    """
    with open(outputs_path, "r") as f:
        raw = json.load(f)
    outputs = {}
    for doc_key, val in raw.items():
        if isinstance(val, str):
            outputs[doc_key] = {"answer": val, "logits": None}
        elif isinstance(val, dict):
            outputs[doc_key] = {
                "answer": val.get("answer", val.get("text", "")),
                "logits": val.get("logits"),
            }
        else:
            outputs[doc_key] = {"answer": str(val), "logits": None}
    return outputs


def softmax(x):
    x = np.array(x, dtype=np.float64)
    x = x - x.max()
    e = np.exp(x)
    return (e / e.sum()).tolist()


def main():
    args = parse_args()

    outputs_path = Path(args.outputs_json)
    gold_path = Path(args.gold_jsonl)

    if not outputs_path.exists():
        sys.exit(f"[ERROR] outputs JSON not found: {outputs_path}")
    if not gold_path.exists():
        sys.exit(f"[ERROR] gold JSONL not found: {gold_path}")

    print(f"[INFO] Loading outputs: {outputs_path}")
    outputs = load_outputs(outputs_path)
    print(f"[INFO] Loading gold:    {gold_path}")
    gold = load_gold(gold_path)

    print(f"[INFO] outputs count: {len(outputs)}, gold count: {len(gold)}")

    # 1) 仅统计两者交集
    common_keys = sorted(set(outputs.keys()) & set(gold.keys()))
    print(f"[INFO] common doc_keys: {len(common_keys)}")

    # 2) 构造 y_true / y_pred（label index）
    y_true, y_pred = [], []
    logits_per_sample = []  # 用于 AUC
    relation_misclass = {r: {r2: 0 for r2 in GENERAL_RELATIONS} for r in GENERAL_RELATIONS}
    per_class_tp = {r: 0 for r in GENERAL_RELATIONS}
    per_class_fp = {r: 0 for r in GENERAL_RELATIONS}
    per_class_fn = {r: 0 for r in GENERAL_RELATIONS}
    per_class_total = {r: 0 for r in GENERAL_RELATIONS}
    parse_fail = 0

    for k in common_keys:
        gold_label = gold[k]
        if isinstance(gold_label, str) and gold_label in GENERAL_RELATIONS:
            gold_idx = GENERAL_RELATIONS.index(gold_label)
        else:
            parse_fail += 1
            continue

        pred_answer = outputs[k]["answer"]
        pred_label = letter_to_relation(pred_answer)
        if pred_label is None:
            parse_fail += 1
            pred_idx = -1
        else:
            pred_idx = GENERAL_RELATIONS.index(pred_label)

        y_true.append(gold_idx)
        y_pred.append(pred_idx)
        per_class_total[gold_label] += 1

        # logits
        lg = outputs[k]["logits"]
        if lg is not None:
            logits_per_sample.append(softmax(lg))
        else:
            # 没有 logits 就用 one-hot 作为退化值（AUC 会很弱）
            oh = [0.0] * 7
            if pred_idx >= 0:
                oh[pred_idx] = 1.0
            logits_per_sample.append(oh)

        if pred_idx == gold_idx:
            per_class_tp[gold_label] += 1
        else:
            if 0 <= pred_idx < 7:
                relation_misclass[gold_label][pred_label] += 1
                per_class_fp[pred_label] += 1
            per_class_fn[gold_label] += 1

    n = len(y_true)
    print(f"[INFO] evaluated samples: {n}, parse failures: {parse_fail}")

    # 3) Macro F1（sklearn macro average）
    macro_f1 = f1_score(y_true, y_pred, labels=range(7), average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=range(7), average="weighted", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, labels=range(7), average="micro", zero_division=0)
    macro_precision = precision_score(y_true, y_pred, labels=range(7), average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, labels=range(7), average="macro", zero_division=0)
    micro_accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / max(n, 1)

    # 4) 多标签 F1：把 single-label 转化为 7 位 binary 向量，每个类一个二分类问题
    #    average='samples' 计算每个样本 7 类 F1 后取平均（论文 multi-label 解读）
    y_true_bin = np.zeros((n, 7), dtype=np.int64)
    y_pred_bin = np.zeros((n, 7), dtype=np.int64)
    for i, (t, p) in enumerate(zip(y_true, y_pred)):
        if t >= 0:
            y_true_bin[i, t] = 1
        if p >= 0:
            y_pred_bin[i, p] = 1
    multilabel_f1_samples = f1_score(y_true_bin, y_pred_bin, average="samples", zero_division=0)
    multilabel_f1_macro = f1_score(y_true_bin, y_pred_bin, average="macro", zero_division=0)
    multilabel_f1_micro = f1_score(y_true_bin, y_pred_bin, average="micro", zero_division=0)

    # 5) Macro ROC AUC（one-vs-rest）
    y_score = np.array(logits_per_sample, dtype=np.float64)
    try:
        macro_auc_ovr = roc_auc_score(
            y_true_bin, y_score, average="macro", multi_class="ovr"
        )
    except ValueError as e:
        warnings.warn(f"AUC macro-ovr failed: {e}; falling back to None")
        macro_auc_ovr = None
    try:
        micro_auc_ovr = roc_auc_score(
            y_true_bin, y_score, average="micro", multi_class="ovr"
        )
    except ValueError as e:
        micro_auc_ovr = None

    # 6) 每类 P/R/F1
    per_class_metrics = {}
    for r in GENERAL_RELATIONS:
        tp = per_class_tp[r]
        fp = per_class_fp[r]
        fn = per_class_fn[r]
        support = per_class_total[r]
        prec = tp / (tp + fp) if tp + fp > 0 else 0.0
        rec = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1c = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
        per_class_metrics[r] = {
            "precision": prec,
            "recall": rec,
            "f1": f1c,
            "support": support,
            "tp": tp, "fp": fp, "fn": fn,
        }

    # 7) 打印 sklearn 的 classification_report（参考用）
    target_names = GENERAL_RELATIONS
    y_pred_safe = [p if p >= 0 else 0 for p in y_pred]  # 报告里不能有 -1
    cls_report = classification_report(
        y_true, y_pred_safe, labels=range(7),
        target_names=target_names, digits=4, zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred_safe, labels=range(7))

    # 8) 输出
    print("\n" + "=" * 60)
    print("GenRel QA — Classification Metrics")
    print("=" * 60)
    print(f"Samples evaluated   : {n}")
    print(f"Parse failures      : {parse_fail}")
    print(f"Micro Accuracy      : {micro_accuracy:.4f}")
    print(f"Macro Precision     : {macro_precision:.4f}")
    print(f"Macro Recall        : {macro_recall:.4f}")
    print(f"Macro F1            : {macro_f1:.4f}")
    print(f"Weighted F1         : {weighted_f1:.4f}")
    print(f"Micro F1            : {micro_f1:.4f}")
    print(f"Multi-label F1 (samples avg): {multilabel_f1_samples:.4f}")
    print(f"Multi-label F1 (macro)      : {multilabel_f1_macro:.4f}")
    print(f"Multi-label F1 (micro)      : {multilabel_f1_micro:.4f}")
    print(f"Macro ROC AUC (ovr) : {macro_auc_ovr if macro_auc_ovr is None else f'{macro_auc_ovr:.4f}'}")
    print(f"Micro ROC AUC (ovr) : {micro_auc_ovr if micro_auc_ovr is None else f'{micro_auc_ovr:.4f}'}")
    print("\n--- Per-class ---")
    for r, m in per_class_metrics.items():
        print(f"  {r:>20s}  P={m['precision']:.4f}  R={m['recall']:.4f}  F1={m['f1']:.4f}  support={m['support']}")
    print("\n--- Confusion Matrix ---")
    cm_header = " " * 20 + "  ".join(f"{r[:4]:>4s}" for r in GENERAL_RELATIONS)
    print(cm_header)
    for i, r in enumerate(GENERAL_RELATIONS):
        row = " " * 20 + "  ".join(f"{cm[i, j]:>4d}" for j in range(7))
        print(f"{r:>20s}{row}")
    print("\n--- sklearn classification_report ---")
    print(cls_report)

    results = {
        "task": "classification_genrel_qa",
        "n_samples": n,
        "parse_failures": parse_fail,
        "micro_accuracy": micro_accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "micro_f1": micro_f1,
        "multilabel_f1_samples": multilabel_f1_samples,
        "multilabel_f1_macro": multilabel_f1_macro,
        "multilabel_f1_micro": multilabel_f1_micro,
        "macro_auc_ovr": macro_auc_ovr,
        "micro_auc_ovr": micro_auc_ovr,
        "per_class": per_class_metrics,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": GENERAL_RELATIONS,
    }

    if args.results_dir is None:
        results_dir = outputs_path.parent
    else:
        results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_json = results_dir / f"{args.save_prefix}evaluate_metrics.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[INFO] results saved to: {out_json}")


if __name__ == "__main__":
    main()