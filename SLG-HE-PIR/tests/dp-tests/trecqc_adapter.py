"""Lightweight TREC-QC adapter for the DP test suite.

Bridges the attack-test dataset module into the DP test pipeline so we can
reuse the canonical TREC-QC loader/collate without duplication.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, List

# Make ``SLG-attack-test/data`` importable when running tests directly.
_THIS = Path(__file__).resolve()
_REPO_ROOT = _THIS.parents[2]
_SRC = _REPO_ROOT / "SLG-HE-PIR" / "src"
_ATTACK = _REPO_ROOT / "SLG-HE-PIR" / "SLG-attack-test"
for p in (str(_REPO_ROOT), str(_SRC), str(_ATTACK)):
    if p not in sys.path:
        sys.path.insert(0, p)

from SLG_attack_test.data.trecqc_dataset import (  # noqa: E402  (path munging above)
    COARSE_LABELS,
    TRECQCDataset,
    load_trecqc_dataset,
    trecqc_collate_fn,
)

__all__ = [
    "COARSE_LABELS",
    "TRECQCDataset",
    "load_trecqc_dataset",
    "trecqc_collate_fn",
    "build_trecqc_dp_dataset",
    "build_trecqc_loader",
]


def build_trecqc_dp_dataset(
    data_dir: str,
    tokenizer: Any,
    split: str = "train",
    max_length: int = 64,
):
    """Construct a :class:`TRECQCDataset` for the DP test pipeline.

    Args:
        data_dir: directory holding ``train.jsonl`` / ``val.jsonl``.
        tokenizer: an object exposing a ``.tokenizer`` callable (matches
            :class:`src.training.trainer.Trainer`).
        split: ``"train"`` or ``"val"``.
        max_length: tokenisation cap (TREC-QC sentences are short).
    """
    train_samples, val_samples = load_trecqc_dataset(data_dir, coarse_only=True)
    samples = train_samples if split == "train" else val_samples
    return TRECQCDataset(samples, tokenizer, max_length=max_length, task=split)


def build_trecqc_loader(
    dataset: Any,
    batch_size: int = 2,
    shuffle: bool = True,
):
    """Wrap a TREC-QC dataset in a torch DataLoader using the canonical collate."""
    import torch.utils.data as tud

    return tud.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=trecqc_collate_fn,
    )


def coarse_label_to_idx(label: str) -> int:
    """Helper: ``"DESC"`` → 0, etc."""
    return COARSE_LABELS.index(label) if label in COARSE_LABELS else -1


def derive_batch_class_idx(batch: List[Any]) -> List[int]:
    """Extract the coarse class index from a list of dataset items.

    Used to fit the LabelBasedCTI from a mini-batch.
    """
    return [int(item.get("coarse_idx", -1)) if hasattr(item, "get") else int(item["coarse_idx"]) for item in batch]
