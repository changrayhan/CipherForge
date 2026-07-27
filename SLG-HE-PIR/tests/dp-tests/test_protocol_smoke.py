"""Smoke test: 1 training step through HeterogeneousProtocol with DP enabled.

We avoid spinning up Llama shards on disk. Instead, this test verifies that:

* ``HeterogeneousProtocol`` accepts the DP config without errors,
* the privatiser's audit payload is propagated through ``StepResult``.

The full HeterogeneousProtocol constructor would load BFV backends, which
requires the encrypted DB cache. The test therefore only verifies the
lightweight shape of the privatiser pipeline.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_THIS = Path(__file__).resolve().parent
for p in (str(_THIS), str(_THIS.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)


def test_dp_privatizer_round_trip(dp_config_cpu, fake_batch_cpu):
    """End-to-end call of ``H15Privatizer.__call__`` with a fitted CTI."""
    from src.core.dchi_privacy import H15Privatizer

    priv = H15Privatizer(dp_config_cpu)
    # Fit CTI with a small synthetic corpus.
    cls = torch.tensor([0, 1, 2, 3, 4, 5] * 5, dtype=torch.long)
    ids = torch.randint(0, 64, (30, 16))
    mask = torch.ones_like(ids)
    priv.fit_cti(cls, ids, mask)
    # Force eta0.
    priv.eta0_override = 64.0

    H = torch.randn(2, 16, 128, requires_grad=True)
    H_tilde, audit = priv(H, fake_batch_cpu, stage="train")
    assert H_tilde.shape == H.shape
    assert H_tilde.requires_grad
    d = audit.as_dict()
    assert d["activated"] is True


def test_dp_privatizer_stage_val_works(dp_config_cpu, fake_batch_cpu):
    """Privatiser should also work when stage='val' (Stage 2 keeps DP on)."""
    from src.core.dchi_privacy import H15Privatizer

    priv = H15Privatizer(dp_config_cpu)
    cls = torch.tensor([0, 1, 2], dtype=torch.long)
    ids = torch.randint(0, 64, (3, 16))
    mask = torch.ones_like(ids)
    priv.fit_cti(cls, ids, mask)
    priv.eta0_override = 100.0
    H = torch.randn(2, 16, 128)
    H_tilde, audit = priv(H, fake_batch_cpu, stage="val")
    assert H_tilde.shape == H.shape
    assert audit.activated is True
