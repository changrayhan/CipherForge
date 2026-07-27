#!/usr/bin/env python3
"""
evaluate_metrics.py — BioTriplex NER 生成任务评估脚本

输入：
  - inference.py 生成的 outputs JSON: {doc_key: "..."}
  - 训练期间生成的 gold JSONL：{data_path}/{split}_gold_ner.txt

输出：
  - 控制台打印指标 + 写到 results/evaluate_metrics.json

指标（按用户最终决策）：
  1. span-level F1（exact match，按 entity_type 分别 + 整体 macro/weighted）
  注：用户明确取消 ROUGE-L，故不计算。
"""

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


ENTITY_TYPES = ["GENE", "DISEASE", "RELATION"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--outputs_json", required=True,
                   help="inference.py 写出的 outputs JSON 路径")
    p.add_argument("--gold_jsonl", required=True,
                   help="gold 文件路径：{split}_gold_ner.txt")
    p.add_argument("--results_dir", default=None,
                   help="保存指标 JSON 的目录（默认与 outputs_json 同目录）")
    p.add_argument("--save_prefix", default="",
                   help="结果 JSON 文件名前缀")
    return p.parse_args()


def load_gold(gold_path):
    """读取 {split}_gold_ner.txt，返回 {doc_key: {"GENE": set(span), "DISEASE": ..., "RELATION": ...}}。

    gold 文件中每个 item 形如:
      {
        "doc_key": "...",
        "input": "<原句>",
        "output": "[{...}]",   # 模型要预测的 JSON 字符串
        "entities": [[start, end, entity_type], ...]
      }
    span = input[start:end]
    """
    gold = {}
    with open(gold_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            sent = item.get("input", "")
            bucket = {"GENE": set(), "DISEASE": set(), "RELATION": set()}
            for ent in item.get("entities", []):
                # ent = [start, end, entity_type]
                try:
                    start, end, et = ent[0], ent[1], ent[2]
                    span = sent[start:end]
                except (IndexError, TypeError):
                    continue
                et = et.upper()
                if et in bucket:
                    bucket[et].add(span)
            gold[item["doc_key"]] = bucket
    return gold


def load_outputs(outputs_path):
    """读取 outputs JSON，doc_key -> raw text（"### Response:\n[...]" 之后的部分）。"""
    with open(outputs_path, "r") as f:
        return json.load(f)


def extract_entities_from_text(output_text):
    """从模型输出的文本中提取 entity 列表。

    兼容：
      - shot 模式（output 含 'assistant\\n\\n'）
      - 非 shot 模式（output 含 '### Response:\\n'）
    """
    if not isinstance(output_text, str):
        return []
    s = output_text
    try:
        if "assistant\n\n" in s:
            s = s.split("assistant\n\n")[-1]
        elif "### Response:\n" in s:
            s = s.split("### Response:\n")[1]
    except IndexError:
        return []
    s = s.strip()
    if not s or s == "[]":
        return []
    try:
        arr = json.loads(s)
    except json.JSONDecodeError:
        # 容错：尝试截取首个 [..]
        try:
            start = s.index("[")
            end = s.rindex("]") + 1
            arr = json.loads(s[start:end])
        except Exception:
            return []
    if not isinstance(arr, list):
        return []
    out = []
    for item in arr:
        if isinstance(item, dict):
            span = item.get("span", "")
            et = item.get("entity_type", "").upper()
            if et and span:
                out.append((span, et))
    return out


def main():
    args = parse_args()

    outputs_path = Path(args.outputs_json)
    gold_path = Path(args.gold_jsonl)

    if not outputs_path.exists():
        sys.exit(f"[ERROR] outputs JSON not found: {outputs_path}")
    if not gold_path.exists():
        sys.exit(f"[ERROR] gold JSONL not found: {gold_path}")

    print(f"[INFO] Loading outputs: {outputs_path}")
    raw_outputs = load_outputs(outputs_path)
    print(f"[INFO] Loading gold:    {gold_path}")
    gold = load_gold(gold_path)
    print(f"[INFO] outputs count: {len(raw_outputs)}, gold count: {len(gold)}")

    common_keys = sorted(set(raw_outputs.keys()) & set(gold.keys()))
    print(f"[INFO] common doc_keys: {len(common_keys)}")

    # 1) 严格 exact-match：每个 entity type 单独统计 TP/FP/FN
    tp_strict = {et: 0 for et in ENTITY_TYPES}
    fp_strict = {et: 0 for et in ENTITY_TYPES}
    fn_strict = {et: 0 for et in ENTITY_TYPES}

    # 2) 把每个 doc_key 的预测/真实按 entity_type 收集，便于整体 P/R/F1
    y_true_types, y_pred_types = [], []
    parse_fail = 0

    for k in common_keys:
        gold_bucket = gold[k]
        pred_entities = extract_entities_from_text(raw_outputs[k])
        if raw_outputs[k] not in (None, "", "[]") and not pred_entities and gold_bucket_any_nonempty(gold_bucket):
            parse_fail += 1

        pred_bucket = {"GENE": set(), "DISEASE": set(), "RELATION": set()}
        for span, et in pred_entities:
            if et in pred_bucket:
                pred_bucket[et].add(span)

        for et in ENTITY_TYPES:
            gset = gold_bucket[et]
            pset = pred_bucket[et]
            for span in gset:
                if span in pset:
                    tp_strict[et] += 1
                else:
                    fn_strict[et] += 1
            for span in pset:
                if span not in gset:
                    fp_strict[et] += 1

            # 收集每个 span 作为样本（用于 sklearn classification_report）
            for span in gset:
                y_true_types.append(et)
                y_pred_types.append(et if span in pset else f"NOT_{et}")
            for span in pset:
                if span not in gset:
                    y_true_types.append(f"NOT_{et}")
                    y_pred_types.append(et)

    # 3) 计算每类 P/R/F1（exact match）
    per_class_metrics = {}
    sum_p, sum_r, sum_f1 = 0.0, 0.0, 0.0
    sum_wf1, sum_wf1_weight = 0.0, 0
    overall_tp = sum(tp_strict.values())
    overall_fp = sum(fp_strict.values())
    overall_fn = sum(fn_strict.values())

    for et in ENTITY_TYPES:
        tp, fp, fn = tp_strict[et], fp_strict[et], fn_strict[et]
        p = tp / (tp + fp) if tp + fp > 0 else 0.0
        r = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
        per_class_metrics[et] = {
            "precision": p,
            "recall": r,
            "f1": f1,
            "tp": tp, "fp": fp, "fn": fn,
        }
        sum_p += p
        sum_r += r
        sum_f1 += f1
        sum_wf1 += f1 * (tp + fn)
        sum_wf1_weight += (tp + fn)

    macro_p = sum_p / len(ENTITY_TYPES)
    macro_r = sum_r / len(ENTITY_TYPES)
    macro_f1 = sum_f1 / len(ENTITY_TYPES)
    weighted_f1 = sum_wf1 / sum_wf1_weight if sum_wf1_weight > 0 else 0.0
    overall_p = overall_tp / (overall_tp + overall_fp) if overall_tp + overall_fp > 0 else 0.0
    overall_r = overall_tp / (overall_tp + overall_fn) if overall_tp + overall_fn > 0 else 0.0
    overall_f1 = 2 * overall_p * overall_r / (overall_p + overall_r) if overall_p + overall_r > 0 else 0.0

    print("\n" + "=" * 60)
    print("NER — Span-level Exact-Match Metrics")
    print("=" * 60)
    print(f"Common doc_keys     : {len(common_keys)}")
    print(f"Parse failures      : {parse_fail}")
    print(f"\nPer entity type (exact match):")
    for et in ENTITY_TYPES:
        m = per_class_metrics[et]
        print(f"  {et:>8s}  P={m['precision']:.4f}  R={m['recall']:.4f}  F1={m['f1']:.4f}  "
              f"TP={m['tp']}  FP={m['fp']}  FN={m['fn']}")
    print(f"\nMacro P / R / F1    : {macro_p:.4f} / {macro_r:.4f} / {macro_f1:.4f}")
    print(f"Weighted F1         : {weighted_f1:.4f}")
    print(f"Overall (micro)     : P={overall_p:.4f}  R={overall_r:.4f}  F1={overall_f1:.4f}")

    results = {
        "task": "generation_ner_span_f1",
        "metric_set": ["span-level exact-match P/R/F1 per entity type",
                       "Macro P/R/F1", "Weighted F1", "Overall micro F1"],
        "n_doc_keys_common": len(common_keys),
        "parse_failures": parse_fail,
        "per_class": per_class_metrics,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "overall_micro_precision": overall_p,
        "overall_micro_recall": overall_r,
        "overall_micro_f1": overall_f1,
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


def gold_bucket_any_nonempty(bucket):
    return any(len(s) > 0 for s in bucket.values())


if __name__ == "__main__":
    main()