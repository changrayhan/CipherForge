"""BioTriplex 7-class GenRel unified evaluator.

Mirrors the BioTriplex ``accuracy_ablation.unified_evaluator.evaluate_unified``
API but uses the 7 coarse classes defined in
``src.data.biotriplex_dataset.GENERAL_RELATIONS``:

    0  a  pathological
    1  b  modulatory
    2  c  expression change
    3  d  diagnosis
    4  e  therapy
    5  f  no relation
    6  g  relation undefined

Input (matches the BioTriplex infer_outputs JSON schema):
    {doc_key: {"answer": "a", "logits": [..7..], "probs": [..7..],
               "predicted_relation": "pathological", "label_idx": 0}}

Output JSON has the same field naming as the BioTriplex unified evaluator:
    {
        "macro_f1": float,
        "micro_f1": float,
        "accuracy": float,
        "macro_auc": float,
        "macro_precision": float,
        "macro_recall": float,
        "per_class": {label_text: {"precision", "recall", "f1", "support"}},
        "confusion_matrix": [[...7...]],
        "n_samples": int,
        "n_parse_failures": int,
        "metric_definitions": {...},
    }
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

REPO_ROOT = Path("/root/autodl-tmp/CipherForgeCode/SLG-HE-PIR")
sys.path.insert(0, str(REPO_ROOT))

from src.data.biotriplex_dataset import GENERAL_RELATIONS  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  IO helpers
# ---------------------------------------------------------------------------
def load_gold_map(gold_path: str) -> Dict[str, Dict[str, Any]]:
    """Load BioTriplex test gold → {doc_key: {label_idx, coarse_relation, ...}}."""
    gold_map: Dict[str, Dict[str, Any]] = {}
    with open(gold_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            gold_map[item["doc_key"]] = {
                "label_idx": int(item.get("label_idx", -1)),
                "coarse_relation": item.get("coarse_relation", ""),
                "input": item.get("input", ""),
                "relation": item.get("relation", {}),
            }
    return gold_map


def load_infer_outputs(infer_path: str) -> Dict[str, Dict[str, Any]]:
    """Load inference outputs JSON → {doc_key: {answer, logits, probs, ...}}."""
    with open(infer_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _predict_idx(entry: Dict[str, Any], n_classes: int = 7) -> Optional[int]:
    """Extract 0-based predicted class index from an infer_outputs entry.

    Priority:
      1. entry["answer"] letter a..g  → 0..6
      2. entry["predicted_relation"] (text) → lookup in GENERAL_RELATIONS
      3. argmax over entry["probs"]   → 0..6
      4. argmax over entry["logits"]  → 0..6
    Returns None if unparseable.
    """
    a = entry.get("answer")
    if isinstance(a, str) and a.strip():
        ch = a.strip()[0].lower()
        if "a" <= ch <= chr(ord("a") + n_classes - 1):
            return ord(ch) - ord("a")
    rel = entry.get("predicted_relation")
    if isinstance(rel, str) and rel in GENERAL_RELATIONS:
        return GENERAL_RELATIONS.index(rel)
    probs = entry.get("probs")
    if isinstance(probs, list) and probs:
        try:
            arr = np.asarray(probs, dtype=np.float32)
            if arr.ndim == 1 and arr.size <= n_classes:
                return int(arr.argmax())
        except Exception:
            pass
    logits = entry.get("logits")
    if isinstance(logits, list) and logits:
        try:
            arr = np.asarray(logits, dtype=np.float32)
            if arr.ndim == 1 and arr.size <= n_classes:
                return int(arr.argmax())
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
#  Metrics
# ---------------------------------------------------------------------------
def _per_class_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int
) -> Dict[str, Dict[str, float]]:
    """Compute per-class precision/recall/F1/support (one-vs-rest)."""
    out: Dict[str, Dict[str, float]] = {}
    for k in range(n_classes):
        tp = int(((y_pred == k) & (y_true == k)).sum())
        fp = int(((y_pred == k) & (y_true != k)).sum())
        fn = int(((y_pred != k) & (y_true == k)).sum())
        support = int((y_true == k).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        out[GENERAL_RELATIONS[k]] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    return out


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> List[List[int]]:
    """Return confusion matrix as list-of-lists, rows=true, cols=pred."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm.tolist()


def _macro_auc(y_true: np.ndarray, probs: np.ndarray, n_classes: int) -> float:
    """Macro-averaged one-vs-rest AUC. Returns mean over classes with positive support."""
    aucs: List[float] = []
    for k in range(n_classes):
        pos = y_true == k
        neg = y_true != k
        if pos.sum() == 0 or neg.sum() == 0:
            continue
        scores = probs[:, k] if probs.ndim == 2 else probs
        order = scores.argsort()
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, len(scores) + 1)
        n = len(scores)
        n_pos = int(pos.sum())
        n_neg = int(neg.sum())
        if n_pos == 0 or n_neg == 0:
            continue
        sum_ranks_pos = ranks[pos].sum()
        u = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
        auc = u / (n_pos * n_neg)
        aucs.append(auc)
    return float(np.mean(aucs)) if aucs else 0.0


# ---------------------------------------------------------------------------
#  Public API — mirrors BioTriplex evaluate_unified()
# ---------------------------------------------------------------------------
def evaluate_unified_bio(
    infer_outputs_path: str,
    gold_path: str,
    output_path: Optional[str] = None,
    use_logits: bool = True,
    experiment_name: str = "bio",
    seed: int = 42,
) -> Dict[str, Any]:
    """Unified BioTriplex 7-class evaluator.

    Reads inference JSON and gold JSONL, computes the full metric set, and
    optionally writes to ``output_path``. Returns the dict.
    """
    gold_map = load_gold_map(gold_path)
    infer_outputs = load_infer_outputs(infer_outputs_path)

    n_classes = len(GENERAL_RELATIONS)
    y_true: List[int] = []
    y_pred: List[int] = []
    prob_rows: List[List[float]] = []
    parse_failures = 0

    for doc_key, gold_info in gold_map.items():
        gold_idx = gold_info["label_idx"]
        entry = infer_outputs.get(doc_key, {})
        pred_idx = _predict_idx(entry, n_classes=n_classes)
        if pred_idx is None:
            pred_idx = 0  # default to pathological (most common in train)
            parse_failures += 1
        y_true.append(gold_idx)
        y_pred.append(pred_idx)
        if use_logits:
            probs = entry.get("probs")
            if not isinstance(probs, list):
                logits = entry.get("logits")
                if isinstance(logits, list):
                    arr = np.asarray(logits, dtype=np.float32)
                    e = np.exp(arr - arr.max())
                    probs = (e / e.sum()).tolist()
            if isinstance(probs, list) and len(probs) >= n_classes:
                prob_rows.append(probs[:n_classes])
            else:
                onehot = [0.0] * n_classes
                onehot[pred_idx] = 1.0
                prob_rows.append(onehot)
        else:
            onehot = [0.0] * n_classes
            onehot[pred_idx] = 1.0
            prob_rows.append(onehot)

    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_pred_arr = np.asarray(y_pred, dtype=np.int64)
    probs_arr = np.asarray(prob_rows, dtype=np.float32)

    # Macro/micro F1
    per_class = _per_class_metrics(y_true_arr, y_pred_arr, n_classes)
    macro_f1 = float(np.mean([v["f1"] for v in per_class.values()]))
    macro_precision = float(np.mean([v["precision"] for v in per_class.values()]))
    macro_recall = float(np.mean([v["recall"] for v in per_class.values()]))

    # Micro F1 (accuracy when classes are predicted one-vs-rest)
    tp_total = sum(int(((y_pred_arr == k) & (y_true_arr == k)).sum()) for k in range(n_classes))
    fp_total = sum(int(((y_pred_arr == k) & (y_true_arr != k)).sum()) for k in range(n_classes))
    fn_total = sum(int(((y_pred_arr != k) & (y_true_arr == k)).sum()) for k in range(n_classes))
    micro_precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    micro_recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    micro_f1 = (2 * micro_precision * micro_recall / (micro_precision + micro_recall)
                if (micro_precision + micro_recall) > 0 else 0.0)

    accuracy = float((y_pred_arr == y_true_arr).mean())
    macro_auc = _macro_auc(y_true_arr, probs_arr, n_classes)
    conf_mat = _confusion_matrix(y_true_arr, y_pred_arr, n_classes)

    metrics = {
        "experiment_name": experiment_name,
        "seed": seed,
        "n_samples": int(len(y_true)),
        "n_classes": n_classes,
        "n_parse_failures": int(parse_failures),
        "macro_f1": macro_f1,
        "micro_f1": float(micro_f1),
        "accuracy": accuracy,
        "macro_auc": macro_auc,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "per_class": per_class,
        "confusion_matrix": conf_mat,
        "metric_definitions": {
            "macro_f1": "mean F1 across all 7 coarse classes",
            "micro_f1": "F1 over all (true,pred) pairs (== accuracy for one-vs-rest)",
            "accuracy": "fraction of samples where pred class == true class",
            "macro_auc": "mean one-vs-rest AUC across classes with positive support",
            "macro_precision": "mean per-class precision",
            "macro_recall": "mean per-class recall",
            "per_class": "one-vs-rest precision/recall/F1/support for each of the 7 classes",
            "confusion_matrix": "rows=true class, cols=pred class",
        },
    }

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        logger.info("Wrote metrics → %s", output_path)

    # One-line summary
    logger.info(
        "BIO | n=%d | macro_F1=%.4f | micro_F1=%.4f | acc=%.4f | macro_AUC=%.4f | parse_fail=%d",
        metrics["n_samples"], macro_f1, micro_f1, accuracy, macro_auc, parse_failures,
    )
    return metrics


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="BioTriplex 7-class unified evaluator")
    p.add_argument("--infer_path", required=True)
    p.add_argument("--gold_path", required=True)
    p.add_argument("--output_path", default=None)
    p.add_argument("--experiment_name", default="bio")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = parse_args()
    metrics = evaluate_unified_bio(
        infer_outputs_path=args.infer_path,
        gold_path=args.gold_path,
        output_path=args.output_path,
        experiment_name=args.experiment_name,
        seed=args.seed,
    )
    # Always print the JSON so the trainer subprocess can capture it
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()