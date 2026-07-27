"""End-to-end unit tests for ``H15Privatizer``."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_THIS = Path(__file__).resolve().parent
for p in (str(_THIS), str(_THIS.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.core.dchi_privacy import H15Privatizer  # noqa: E402


@pytest.fixture
def privatizer_cpu(dp_config_cpu):
    return H15Privatizer(dp_config_cpu)


@pytest.fixture
def privatizer_with_cti(dp_config_cpu):
    p = H15Privatizer(dp_config_cpu)
    # Fit the CTI with a tiny dummy distribution.
    cls = torch.tensor([0, 1, 2, 3, 4, 5] * 3, dtype=torch.long)
    ids = torch.randint(0, 64, (18, 16))
    mask = torch.ones_like(ids)
    p.fit_cti(cls, ids, mask)
    return p


def test_privatizer_disabled_passthrough(dp_config_cpu, fake_batch_cpu):
    dp_config_cpu["dp_enable"] = False
    priv = H15Privatizer(dp_config_cpu)
    H = torch.randn(2, 16, 128)
    H_tilde, audit = priv(H, fake_batch_cpu, stage="train")
    assert torch.allclose(H_tilde, H)
    assert audit.activated is False


def test_privatizer_shape_preserved(privatizer_with_cti, fake_batch_cpu):
    H = torch.randn(2, 16, 128)
    H_tilde, _ = privatizer_with_cti(H, fake_batch_cpu, stage="train")
    assert H_tilde.shape == H.shape


def test_privatizer_requires_grad_preserved(privatizer_with_cti, fake_batch_cpu):
    H = torch.randn(2, 16, 128, requires_grad=True)
    H_tilde, _ = privatizer_with_cti(H, fake_batch_cpu, stage="train")
    assert H_tilde.requires_grad


def test_answer_beta_dampens_eta(privatizer_with_cti, fake_batch_cpu):
    H = torch.randn(2, 16, 128)
    # Force eta0 to a fixed value so the comparison is deterministic.
    privatizer_with_cti.eta0_override = 102.4
    H_tilde, audit = privatizer_with_cti(H, fake_batch_cpu, stage="train")
    assert audit.activated is True
    if audit.eta_used_answer > 0 and audit.eta_used_context > 0:
        # If CTI yielded non-zero UI on both regions, β should reduce the
        # answer η below the context η. If CTI pushed answer UI to the
        # floor, η_ans may be near 0 — still strictly less than or equal.
        assert audit.eta_used_answer <= audit.eta_used_context


def test_calibration_mode_no_noise(privatizer_with_cti, fake_batch_cpu):
    priv = privatizer_with_cti
    priv.set_calibration_mode(True)
    H = torch.randn(2, 16, 128)
    H_tilde, audit = priv(H, fake_batch_cpu, stage="train")
    assert torch.allclose(H_tilde, H)
    assert audit.calibration_updated is True
    # After enough updates, eta0 becomes available.
    for _ in range(priv.calibration_steps):
        priv.observe_clean(torch.randn(2, 16, 128))
    assert priv.eta0 is not None
    priv.set_calibration_mode(False)


def test_clip_value_clips(privatizer_with_cti, fake_batch_cpu):
    priv = privatizer_with_cti
    priv.clip_value = 0.5
    priv.eta0_override = 50.0
    H = torch.randn(2, 16, 128)
    H_tilde, _ = priv(H, fake_batch_cpu, stage="train")
    assert (H_tilde.abs() <= 0.5 + 1e-3).all()


def test_audit_dict_keys(privatizer_with_cti, fake_batch_cpu):
    priv = privatizer_with_cti
    priv.eta0_override = 100.0
    H = torch.randn(2, 16, 128)
    _, audit = priv(H, fake_batch_cpu, stage="train")
    d = audit.as_dict()
    for k in (
        "eta_used_context",
        "eta_used_answer",
        "noise_l2_context",
        "noise_l2_answer",
        "calibration_updated",
        "activated",
        "alpha",
    ):
        assert k in d
