"""Unit tests for the label-conditioned CTI."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_THIS = Path(__file__).resolve().parent
for p in (str(_THIS), str(_THIS.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.core.dchi_privacy import LabelBasedCTI  # noqa: E402


def _fit_tiny_cti(num_classes=3, vocab_size=64):
    cti = LabelBasedCTI(vocab_size=vocab_size, num_classes=num_classes, device="cpu")
    # Class 0 favours tokens 0-9, class 1 favours 10-19, class 2 favours 20-29.
    cls = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    ids = torch.tensor([
        [0, 1, 2, 3, 4, 30],
        [5, 6, 7, 8, 9, 31],
        [10, 11, 12, 13, 14, 32],
        [15, 16, 17, 18, 19, 33],
        [20, 21, 22, 23, 24, 34],
        [25, 26, 27, 28, 29, 35],
    ], dtype=torch.long)
    mask = torch.ones_like(ids)
    cti.update(cls, ids, mask)
    return cti


# --------------------------------------------------------------------------- #
# 1. UI shape matches (B, S)
# --------------------------------------------------------------------------- #
def test_ui_shapes():
    cti = _fit_tiny_cti()
    input_ids = torch.randint(0, 64, (3, 16))
    labels = torch.full((3, 16), -100, dtype=torch.long)
    labels[:, -1] = torch.tensor([0, 1, 2])  # per-sample coarse label
    mask = torch.ones_like(input_ids)
    ui = cti.compute_ui(input_ids, labels, mask)
    assert ui.shape == (3, 16)


# --------------------------------------------------------------------------- #
# 2. answer_mask flag suppresses UI at the answer positions
# --------------------------------------------------------------------------- #
def test_answer_mask_suppresses_ui():
    """When ``answer_mask`` is passed, answer-position UI is set to a
    large negative value so that the answer β < 1 has the expected effect."""
    cti = _fit_tiny_cti()
    input_ids = torch.tensor([[0, 1, 2, 30]], dtype=torch.long)
    labels = torch.tensor([[-100, -100, -100, 0]], dtype=torch.long)
    mask = torch.ones_like(input_ids)
    answer_mask = torch.tensor([[False, False, False, True]])
    ui = cti.compute_ui(input_ids, labels, mask, answer_mask=answer_mask)
    assert ui[0, -1].item() < ui[0, 0].item()
    assert ui[0, -1].item() < -1e2


# --------------------------------------------------------------------------- #
# 3. Returns zero UI when unfitted
# --------------------------------------------------------------------------- #
def test_unfitted_returns_zeros():
    cti = LabelBasedCTI(vocab_size=64, num_classes=3, device="cpu")
    input_ids = torch.randint(0, 64, (2, 8))
    labels = torch.tensor([[-100, 0, -100, -100, -100, -100, -100, -100],
                           [-100, -100, -100, -100, 1, -100, -100, -100]])
    mask = torch.ones_like(input_ids)
    ui = cti.compute_ui(input_ids, labels, mask)
    assert torch.all(ui == 0)


# --------------------------------------------------------------------------- #
# 4. Update rejects malformed class tensor
# --------------------------------------------------------------------------- #
def test_update_rejects_malformed_class():
    cti = LabelBasedCTI(vocab_size=64, num_classes=3)
    with pytest.raises(ValueError):
        cti.update(torch.zeros(2, 2), torch.zeros(2, 5), torch.ones(2, 5))
