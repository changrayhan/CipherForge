"""TREC-QC dataset adapter for the SLG-HE-PIR attack test suite.

TREC-QC (TREC Question Classification) is a question classification task
with two levels of granularity. This module adapts TREC-QC JSONL files
into the same dict shape expected by BioTriplexQADataset, enabling reuse
of the existing protocol infrastructure.

Coarse-grained labels (6 classes, used for attack tests):
    DESC  — Description / Definition
    ENTY  — Entity
    ABBR  — Abbreviation
    HUM   — Human / Person
    NUM   — Numeric (number, date, etc.)
    LOC   — Location
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# TREC-QC coarse-grained label map
COARSE_LABELS = ["DESC", "ENTY", "ABBR", "HUM", "NUM", "LOC"]
COARSE_LABEL_TO_IDX = {label: idx for idx, label in enumerate(COARSE_LABELS)}
COARSE_IDX_TO_LABEL = {idx: label for idx, label in enumerate(COARSE_LABELS)}

# Fine-grained labels (50 classes) — for reference only, not used in attack tests
FINE_LABELS = [
    "DESC:def", "DESC:desc", "DESC:manner", "DESC:reason",
    "ENTY:animal", "ENTY:body", "ENTY:color", "ENTY:creative",
    "ENTY:currency", "ENTY:disease", "ENTY:event", "ENTY:food",
    "ENTY:instrument", "ENTY:language", "ENTY:law", "ENTY:letter",
    "ENTY:other", "ENTY:plant", "ENTY:product", "ENTY:religion",
    "ENTY:sport", "ENTY:substance", "ENTY:symbol", "ENTY:techmet",
    "ENTY:veh", "ENTY:word",
    "ABBR:abb", "ABBR:exp",
    "HUM:desc", "HUM:gr", "HUM:ind", "HUM:title",
    "NUM:code", "NUM:count", "NUM:date", "NUM:dist", "NUM:money",
    "NUM:ord", "NUM:other", "NUM:period", "NUM:percent", "NUM:speed",
    "NUM:temp", "NUM:size", "NUM:weight",
    "LOC:city", "LOC:country", "LOC:mountain", "LOC:other", "LOC:state",
]
FINE_LABEL_TO_IDX = {label: idx for idx, label in enumerate(FINE_LABELS)}


def load_trecqc_dataset(
    data_dir: str,
    train_ratio: float = 0.85,
    seed: int = 42,
    coarse_only: bool = True,
) -> tuple:
    """Load TREC-QC dataset from JSONL files.

    Args:
        data_dir: Path to directory containing train.jsonl and val.jsonl
        train_ratio: Fraction for training split (rest goes to val)
        seed: Random seed
        coarse_only: If True, remap fine labels to coarse labels

    Returns:
        (train_samples, val_samples) each as list of dicts with keys:
            id, question, coarse_label, fine_label, coarse_idx
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    train_path = data_dir / "train.jsonl"
    val_path = data_dir / "val.jsonl"

    if train_path.exists():
        all_train = _load_jsonl(train_path)
        logger.info("Loaded %d samples from %s", len(all_train), train_path)
    else:
        raise FileNotFoundError(f"train.jsonl not found in {data_dir}")

    if val_path.exists():
        val_samples = _load_jsonl(val_path)
        logger.info("Loaded %d samples from %s", len(val_samples), val_path)
    else:
        random.seed(seed)
        shuffled = all_train.copy()
        random.shuffle(shuffled)
        split_i = int(len(shuffled) * train_ratio)
        all_train = shuffled[split_i:]
        val_samples = shuffled[:split_i]
        logger.info("Split dataset: train=%d, val=%d", len(all_train), len(val_samples))

    # Normalize labels
    train_samples = [_normalize_sample(s, coarse_only) for s in all_train]
    val_samples = [_normalize_sample(s, coarse_only) for s in val_samples]

    return train_samples, val_samples


def _normalize_sample(sample: Dict[str, Any], coarse_only: bool) -> Dict[str, Any]:
    """Normalize a TREC-QC sample dict to attack-test format.

    The on-disk TREC-QC JSONL has the following fields (per TEST_REPORT.md §1.3.2):
        text, label, label_text, label_original, label_coarse,
        label_coarse_text, label_coarse_original.

    This function adapts them to the canonical sample dict expected by
    ``TRECQCDataset`` and the protocol:
        {id, question, coarse_label, fine_label, coarse_idx, fine_idx,
         output_text}, where ``output_text`` is the coarse label name (e.g. "DESC")
        so that the trained model is a 6-way coarse classifier.
    """
    # Use the actual JSONL field name `text` (it is the question body).
    question = sample.get("text") or sample.get("question", "")

    # Fine label: prefer the human-readable ``label_original`` (e.g. "DESC:manner")
    # because it preserves coarse:fine structure. Fall back to numeric ``label``.
    fine_label = (
        sample.get("label_original")
        or sample.get("label_fine")
        or sample.get("label_text")
        or sample.get("label", "")
    )
    if not isinstance(fine_label, str):
        fine_label = str(fine_label)

    # Coarse label: prefer the explicit coarse field; fall back to deriving
    # the coarse prefix from the original ``DESC:def`` style string.
    if coarse_only:
        coarse_label = sample.get("label_coarse_original") or sample.get("label_coarse_text")
        if not coarse_label and isinstance(fine_label, str) and ":" in fine_label:
            coarse_label = fine_label.split(":")[0]
        coarse_idx = sample.get("label_coarse")
        if coarse_idx is None:
            coarse_idx = COARSE_LABEL_TO_IDX.get(coarse_label, -1)
        else:
            coarse_idx = int(coarse_idx)
    else:
        coarse_label = ""
        coarse_idx = -1

    # Fine id (numeric 0-49) for future fine-grained analysis.
    fine_idx = sample.get("label")
    if not isinstance(fine_idx, int):
        try:
            fine_idx = int(fine_idx)
        except Exception:
            fine_idx = -1

    return {
        "id": str(sample.get("id", "")),
        "question": question,
        "coarse_label": coarse_label or "",
        "fine_label": fine_label,
        "coarse_idx": int(coarse_idx) if coarse_idx is not None else -1,
        "fine_idx": int(fine_idx) if fine_idx is not None else -1,
        "output_text": coarse_label or "",  # the supervised label is the coarse class
    }


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples


class TRECQCDataset(Dataset):
    """PyTorch Dataset for TREC-QC question classification.

    Maps TREC-QC JSONL samples to the same dict shape as BioTriplexQADataset:
        {
            "input_ids": Tensor[seq_len],
            "attention_mask": Tensor[seq_len],
            "labels": Tensor[seq_len],
            "prompt": str,
            "input_text": str,
            "output_text": str,          # coarse label name, e.g. "DESC"
            "id": str,
            "coarse_idx": int,           # 0-5
            "coarse_label": str,
        }
    """

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        tokenizer: Any,
        max_length: int = 64,
        task: str = "train",
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task = task

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        # The pre-normalized samples expose the question text under ``question``.
        question = sample.get("question", "")
        if not question:
            question = sample.get("text", "")
        output_text = sample.get("output_text", "")
        coarse_idx = sample.get("coarse_idx", -1)
        if coarse_idx is None or coarse_idx == -1:
            # Fall back to mapping the coarse label string to a 0-5 index.
            coarse_label = sample.get("coarse_label", "")
            coarse_idx = COARSE_LABEL_TO_IDX.get(coarse_label, -1)
        sample_id = sample.get("id", str(idx))

        # Format: "Question: {question}"
        full_text = f"Question: {question}\nAnswer:"

        # Tokenize
        encoded = self.tokenizer.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Tokenize the coarse label as the output token (e.g. "DESC")
        label_encoded = self.tokenizer.tokenizer(
            output_text if output_text else "UNK",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": label_encoded["input_ids"].squeeze(0),
            "prompt": question,
            "input_text": full_text,
            "output_text": output_text,
            "id": sample_id,
            "coarse_idx": coarse_idx,
            "coarse_label": sample.get("coarse_label", output_text),
        }
        return item


def trecqc_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function handling mixed tensor/string fields."""
    import collections.abc

    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])

    result = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "batch_size": len(batch),
    }

    # String fields — keep as list
    for key in ("prompt", "input_text", "output_text", "id", "coarse_label"):
        result[key] = [b[key] for b in batch]

    # Numeric
    result["coarse_idx"] = [b["coarse_idx"] for b in batch]

    return result
