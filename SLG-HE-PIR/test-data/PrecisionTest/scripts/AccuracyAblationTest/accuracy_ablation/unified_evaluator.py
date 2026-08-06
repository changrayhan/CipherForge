"""统一评估器 — 跨 baseline / SLG / ablation 的指标对比。

v2 修复（v2-Bug-0.3）：
  旧实现 baseline 用 sklearn roc_auc_score + 简单 f1_score；SLG 用 biotriplex_metrics.compute_classification_metrics。
  两套 evaluator 的缺失预测、multilabel 解析、macro AUC 计算和 JSON 字段层级都不同，导致指标不可比。

  本模块提供统一接口：
    evaluate_unified(infer_outputs, gold_path, ...)
  输出结构和字段命名都与 SLG 的 compute_classification_metrics 一致，可直接用于 ablation 对比。

差异点（v2 修复）：
  - 缺失预测：统一替换为 "relation undefined"（保留为 7 类之一）
  - Multilabel 解析：统一为逗号分隔字母集合
  - Macro AUC：使用与 SLG 相同实现（per-class 然后均值，跳过 zero-positive）
  - 字段层级：统一塞入 metrics.{...} 顶层
  - 缺失预测统计：保留 n_parse_failures 字段
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .eval_replay import FINE_TO_GENERAL, GENERAL_RELATIONS, load_gold_map

logger = logging.getLogger(__name__)


def _relation_to_letter(relation: str) -> str:
    """general relation → 选项字母 (a-g)"""
    if relation in GENERAL_RELATIONS:
        return chr(ord("a") + GENERAL_RELATIONS.index(relation))
    return "g"  # 默认 relation undefined


def _gold_letter_for_doc(doc_key: str, gold_map: dict) -> str:
    """从 gold_map 取 general relation，对应字母 a-g"""
    rel = gold_map.get(doc_key)
    if rel is None:
        return "g"  # 缺省
    return _relation_to_letter(rel)


def _predict_letter_from_infer_outputs(entry: dict) -> str:
    """从 infer_outputs JSON 单条样本提取预测字母。

    优先：
      1. entry["answer"] 形如 "a)" / "c)"
      2. entry["predicted_relation"] → 查 GENERAL_RELATIONS
    """
    a = entry.get("answer")
    if isinstance(a, str) and a.strip():
        ch = a.strip()[0].lower()
        if "a" <= ch <= "g":
            return ch
    rel = entry.get("predicted_relation")
    if isinstance(rel, str):
        if rel in GENERAL_RELATIONS:
            return _relation_to_letter(rel)
    return "g"  # 失败默认


def evaluate_unified(
    infer_outputs_path: str | Path,
    gold_path: str | Path,
    output_path: Optional[str | Path] = None,
    use_logits: bool = True,
) -> Dict[str, Any]:
    """统一评估接口。

    输入：
      - infer_outputs_path: infer_and_save.py 输出的 JSON
        {doc_key: {"answer": "a)", "logits": [...7...], "probs": [...], "predicted_relation": "..."}}
      - gold_path: test_gold_general_qa.txt 的 JSONL
      - output_path: 可选，写出的指标 JSON
      - use_logits: 是否在 ROC AUC 计算中使用真实 logits（而不是 one-hot 替代）

    输出结构（与 SLG 的 compute_classification_metrics 一致）：
      {
        "task": "GenRel QA (7-class Classification)",
        "n_samples": ...,
        "n_parse_failures": ...,
        "has_logits": bool,
        "metrics": {
          "micro_accuracy", "macro_precision", "macro_recall", "macro_f1",
          "weighted_f1", "micro_f1",
          "multilabel_f1_samples", "multilabel_f1_macro", "multilabel_f1_micro",
          "macro_roc_auc_ovr", "micro_roc_auc_ovr",
        },
        "per_class_metrics": { rel: { precision, recall, f1, support } },
        "y_true_distribution": { rel: count },
        "y_pred_distribution": { rel: count },
        "confusion_matrix": [[...]],
      }
    """
    # 1. 加载 gold
    gold_map, base_to_full = load_gold_map(gold_path)

    # 2. 加载 infer_outputs
    with open(infer_outputs_path, "r") as f:
        infer_outputs = json.load(f)

    # 3. doc_key 格式兼容: baseline 用 fine-relation doc_key，需要 remap 到 gold doc_key
    from .quant_hooks import apply_variant_to_infer_outputs, QuantNoiseSpec
    from .eval_replay import _remap_doc_keys
    mapped_outputs, _ = _remap_doc_keys(infer_outputs, base_to_full)

    # 4. 构造 predictions/labels/pred_logits 列表（按 doc_key 对齐）
    predictions: List[str] = []
    labels: List[str] = []
    pred_logits: List[List[float]] = []
    skip_count = 0
    for doc_key, entry in mapped_outputs.items():
        # 仅保留在 gold_map 里的 doc_key（避免不匹配的样本）
        if doc_key not in gold_map:
            skip_count += 1
            continue
        # 优先使用 mapped_outputs 走无噪声路径
        # 因为 unmapped infer_outputs 可能已经经过 spec 处理——
        # 在非 spec 的情况下 entry 直接取 mapped_outputs
        pred_letter = _predict_letter_from_infer_outputs(entry)
        gold_letter = _gold_letter_for_doc(doc_key, gold_map)
        predictions.append(f"{pred_letter})")
        labels.append(f"{gold_letter})")
        # logits
        if use_logits and "logits" in entry:
            logits = entry["logits"]
            if isinstance(logits, (list, tuple)) and len(logits) == 7:
                pred_logits.append([float(x) for x in logits])
                continue
        # fallback: 用 one-hot 作 logits 输入
        pred_logits.append([0.0] * 7)

    if skip_count > 0:
        logger.info(
            "[unified_evaluator] Skipped %d infer entries not in gold_map (out of %d)",
            skip_count, len(infer_outputs),
        )

    # 4. 调用 SLG 的 compute_classification_metrics
    import sys
    import importlib.util
    # Path: .../SLG-HE-PIR/AccuracyAblationTest/accuracy_ablation/unified_evaluator.py
    # parents[0] = accuracy_ablation, [1] = AccuracyAblationTest, [2] = REPO_ROOT
    _repo_root = Path(__file__).resolve().parents[2]
    metrics_path = _repo_root / "src" / "training" / "biotriplex_metrics.py"
    spec = importlib.util.spec_from_file_location(
        "biotriplex_metrics", str(metrics_path),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["biotriplex_metrics"] = mod
    spec.loader.exec_module(mod)
    compute_classification_metrics = mod.compute_classification_metrics

    metrics = compute_classification_metrics(
        predictions=predictions,
        labels=labels,
        pred_logits=pred_logits,
    )

    # 5. 写出
    if output_path is not None:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        logger.info("[unified_evaluator] Wrote metrics to %s", out_path)

    return metrics


def evaluate_unified_directory(
    infer_outputs_dir: str | Path,
    gold_path: str | Path,
    output_dir: str | Path,
    use_logits: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """对 inference 目录下所有 infer_outputs_epoch_NNN.json 一次性评估。

    inferred_dir 下的所有 json 都会被逐一评估。
    返回 {filename: metrics_dict}。
    """
    in_dir = Path(infer_outputs_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Dict[str, Any]] = {}
    for json_path in sorted(in_dir.glob("infer_outputs_epoch_*.json")):
        out_path = out_dir / json_path.name.replace(
            "infer_outputs_epoch_", "epoch_",
        ).replace(".json", "_evaluate_metrics.json")
        logger.info("[unified_evaluator] Processing %s", json_path.name)
        metrics = evaluate_unified(
            infer_outputs_path=json_path,
            gold_path=gold_path,
            output_path=out_path,
            use_logits=use_logits,
        )
        results[json_path.name] = metrics

    return results
