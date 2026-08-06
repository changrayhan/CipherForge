"""TREC-QC 6-class unified evaluator.

Mirrors the BioTriplex ``accuracy_ablation.unified_evaluator.evaluate_unified`` API
but uses TREC-QC's 6 coarse classes:

    DESC (a), ENTY (b), ABBR (c), HUM (d), NUM (e), LOC (f)

Input (matches the BioTriplex infer_outputs JSON schema, but with a..f instead of a..g):

    {doc_key: {"answer": "a)", "logits": [..6..], "probs": [..6..],
               "predicted_relation": "description and abstract concepts"}}

Output JSON has the same field naming as BioTriplex's evaluator:
    {
        "macro_f1": float,
        "micro_f1": float,
        "accuracy": float,
        "macro_auc": float,
        "macro_precision": float,
        "macro_recall": float,
        "per_class": {label_text: {"precision", "recall", "f1", "support"}},
        "confusion_matrix": [[...6...]],
        "n_samples": int,
        "n_parse_failures": int,
        "metric_definitions": {...},
    }

Also writes ``per_epoch/metrics.json`` and prints a one-line summary.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  TREC-QC coarse class definitions (mirrors src.data.biotriplex_dataset)
# ---------------------------------------------------------------------------
TREC_QC_COARSE_CLASSES: List[str] = [
    "description and abstract concepts",  # 0  a  DESC
    "entities",                            # 1  b  ENTY
    "abbreviation",                        # 2  c  ABBR
    "human beings",                        # 3  d  HUM
    "numeric values",                      # 4  e  NUM
    "locations",                           # 5  f  LOC
]
TREC_QC_LETTERS = ["a", "b", "c", "d", "e", "f"]


# ---------------------------------------------------------------------------
#  IO helpers
# ---------------------------------------------------------------------------
def load_gold_map(gold_path: str) -> Dict[str, Dict[str, Any]]:
    """Load TREC-QC test gold → {doc_key: {label_idx, coarse_text, ...}}."""
    gold_map: Dict[str, Dict[str, Any]] = {}
    with open(gold_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            gold_map[item["doc_key"]] = {
                "label_idx": int(item.get("label_idx", -1)),
                "coarse_text": item.get("coarse_relation", ""),
                "input": item.get("input", ""),
            }
    return gold_map


def load_infer_outputs(infer_path: str) -> Dict[str, Dict[str, Any]]:
    """Load inference outputs JSON → {doc_key: {answer, logits, probs, predicted_relation}}."""
    with open(infer_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _predict_idx(entry: Dict[str, Any]) -> Optional[int]:
    """Extract 0-based predicted class index from an infer_outputs entry.

    Priority:
      1. entry["answer"] letter a..f  → 0..5
      2. entry["predicted_relation"] (text) → lookup in TREC_QC_COARSE_CLASSES
      3. argmax over entry["probs"]   → 0..5
    Returns None if unparseable.
    """
    a = entry.get("answer")
    if isinstance(a, str) and a.strip():
        ch = a.strip()[0].lower()
        if "a" <= ch <= "f":
            return ord(ch) - ord("a")
    rel = entry.get("predicted_relation")
    if isinstance(rel, str) and rel in TREC_QC_COARSE_CLASSES:
        return TREC_QC_COARSE_CLASSES.index(rel)
    probs = entry.get("probs")
    if isinstance(probs, list) and probs:
        try:
            arr = np.asarray(probs, dtype=np.float32)
            if arr.ndim == 1 and arr.size <= 6:
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
        out[TREC_QC_COARSE_CLASSES[k]] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    return out


def _macro_auc(y_true: np.ndarray, probs: np.ndarray, n_classes: int) -> float:
    """Macro-averaged one-vs-rest AUC. Returns mean over classes with positive support."""
    aucs: List[float] = []
    for k in range(n_classes):
        pos = y_true == k
        neg = y_true != k
        if pos.sum() == 0 or neg.sum() == 0:
            continue
        # rank-based AUC (no sklearn dependency)
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
        # Mann-Whitney U formula
        u = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
        auc = u / (n_pos * n_neg)
        aucs.append(auc)
    return float(np.mean(aucs)) if aucs else 0.0


# ---------------------------------------------------------------------------
#  Public API — mirrors BioTriplex evaluate_unified()
# ---------------------------------------------------------------------------
def evaluate_unified_trec(
    infer_outputs_path: str,
    gold_path: str,
    output_path: Optional[str] = None,
    use_logits: bool = True,
    experiment_name: str = "trec",
    seed: int = 42,
) -> Dict[str, Any]:
    """Unified TREC-QC evaluator. Mirrors BioTriplex API."""
    gold_map = load_gold_map(gold_path)
    infer_outputs = load_infer_outputs(infer_outputs_path)

    n_classes = len(TREC_QC_COARSE_CLASSES)
    y_true: List[int] = []
    y_pred: List[int] = []
    prob_rows: List[List[float]] = []
    parse_failures = 0

    for doc_key, gold_info in gold_map.items():
        gold_idx = gold_info["label_idx"]
        entry = infer_outputs.get(doc_key, {})
        pred_idx = _predict_idx(entry)
        if pred_idx is None:
            pred_idx = 0  # default to DESC
            parse_failures += 1
        y_true.append(gold_idx)
        y_pred.append(pred_idx)
        # Try to collect probs for AUC
        if use_logits:
            probs = entry.get("probs")
            if not isinstance(probs, list):
                # Fallback: softmax(logits)
                logits = entry.get("logits")
                if isinstance(logits, list):
                    arr = np.asarray(logits, dtype=np.float32)
                    e = np.exp(arr - arr.max())
                    probs = (e / e.sum()).tolist()
            if isinstance(probs, list) and len(probs) >= n_classes:
                prob_rows.append(probs[:n_classes])
            else:
                prob_rows.append([0.0] * n_classes)
                prob_rows[-1][pred_idx] = 1.0
        else:
            onehot = [0.0] * n_classes
            onehot[pred_idx] = 1.0
            prob_rows.append(onehot)

    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_pred_arr = np.asarray(y_pred, dtype=np.int64)
    probs_arr = np.asarray(prob_rows, dtype=np.float32)

    # Confusion matrix
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true_arr, y_pred_arr):
        cm[int(t), int(p)] += 1

    # Aggregate metrics
    n = len(y_true_arr)
    accuracy = float((y_true_arr == y_pred_arr).mean())
    micro_f1 = float(_f1_micro(y_true_arr, y_pred_arr, n_classes))
    macro_f1 = float(_f1_macro(y_true_arr, y_pred_arr, n_classes))
    macro_p, macro_r = _precision_recall_macro(y_true_arr, y_pred_arr, n_classes)
    macro_auc = _macro_auc(y_true_arr, probs_arr, n_classes)
    per_class = _per_class_metrics(y_true_arr, y_pred_arr, n_classes)

    metrics: Dict[str, Any] = {
        "experiment_name": experiment_name,
        "seed": seed,
        "dataset": "trec-qc",
        "n_classes": n_classes,
        "class_names": TREC_QC_COARSE_CLASSES,
        "n_samples": int(n),
        "n_parse_failures": int(parse_failures),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_auc": macro_auc,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "metric_definitions": {
            "accuracy": "fraction of correctly classified samples",
            "macro_f1": "per-class F1, then unweighted mean",
            "micro_f1": "global TP / (TP + 0.5*(FP+FN)) across all classes",
            "macro_precision": "per-class precision, then mean",
            "macro_recall": "per-class recall, then mean",
            "macro_auc": "one-vs-rest AUC, mean over classes with positive support",
            "per_class": "one-vs-rest precision/recall/F1/support per class",
        },
    }

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

    return metrics


def _f1_micro(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    tp = fp = fn = 0
    for k in range(n_classes):
        tp += int(((y_pred == k) & (y_true == k)).sum())
        fp += int(((y_pred == k) & (y_true != k)).sum())
        fn += int(((y_pred != k) & (y_true == k)).sum())
    if tp == 0:
        return 0.0
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    return 2 * p * r / (p + r)


def _f1_macro(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    f1s = []
    for k in range(n_classes):
        tp = int(((y_pred == k) & (y_true == k)).sum())
        fp = int(((y_pred == k) & (y_true != k)).sum())
        fn = int(((y_pred != k) & (y_true == k)).sum())
        if (tp + fp) == 0 or (tp + fn) == 0:
            f1s.append(0.0)
            continue
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        f1s.append(2 * p * r / (p + r) if (p + r) > 0 else 0.0)
    return float(np.mean(f1s))


def _precision_recall_macro(y_true: np.ndarray, y_pred: np.ndarray,
                            n_classes: int):
    ps, rs = [], []
    for k in range(n_classes):
        tp = int(((y_pred == k) & (y_true == k)).sum())
        fp = int(((y_pred == k) & (y_true != k)).sum())
        fn = int(((y_pred != k) & (y_true == k)).sum())
        ps.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
        rs.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
    return float(np.mean(ps)), float(np.mean(rs))


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infer_outputs", required=True,
                    help="Path to infer_outputs JSON")
    ap.add_argument("--gold_path", required=True,
                    help="Path to test_gold_general_qa.txt")
    ap.add_argument("--output_path", default=None,
                    help="Where to write the metrics JSON")
    ap.add_argument("--experiment_name", default="trec")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    metrics = evaluate_unified_trec(
        infer_outputs_path=args.infer_outputs,
        gold_path=args.gold_path,
        output_path=args.output_path,
        experiment_name=args.experiment_name,
        seed=args.seed,
    )
    # One-line summary
    print(
        f"[trec_evaluator] n={metrics['n_samples']} "
        f"acc={metrics['accuracy']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f} "
        f"micro_f1={metrics['micro_f1']:.4f} "
        f"macro_auc={metrics['macro_auc']:.4f} "
        f"parse_fails={metrics['n_parse_failures']}"
    )


if __name__ == "__main__":
    main()