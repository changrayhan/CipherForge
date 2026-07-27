"""Unit tests for ``ActivationNormCalibrator``."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_THIS = Path(__file__).resolve().parent
for p in (str(_THIS), str(_THIS.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.core.dchi_privacy import ActivationNormCalibrator  # noqa: E402


# --------------------------------------------------------------------------- #
# 1. Convergence on a synthetic stream
# --------------------------------------------------------------------------- #
def test_estimator_convergence():
    """EMA estimate should converge to the empirical mean of synthetic norms."""
    d = 256
    cal = ActivationNormCalibrator(target_relative_alpha=0.15, hidden_dim=d)
    # Each row's L2 norm has mean ~ 50 (50 = sqrt(d) * scale)
    fake = torch.randn(64, d) * 5 + 50
    cal.update(fake)
    A, eta0 = cal.finalize()
    # Empirical mean norm:
    empirical = fake.float().norm(dim=-1).mean().item()
    assert abs(A - empirical) / empirical < 0.20, (
        f"calibrated A={A:.2f}, empirical={empirical:.2f}"
    )
    # eta0 formula: d / (α * A)
    expected_eta0 = d / (0.15 * A)
    assert abs(eta0 - expected_eta0) < 1e-3, (eta0, expected_eta0)


# --------------------------------------------------------------------------- #
# 2. finalize() before update() raises
# --------------------------------------------------------------------------- #
def test_finalize_without_update_raises():
    cal = ActivationNormCalibrator(target_relative_alpha=0.15, hidden_dim=128)
    with pytest.raises(RuntimeError):
        cal.finalize()


# --------------------------------------------------------------------------- #
# 3. eta0 formula correctness
# --------------------------------------------------------------------------- #
def test_eta0_formula():
    d, alpha = 2048, 0.20
    cal = ActivationNormCalibrator(target_relative_alpha=alpha, hidden_dim=d)
    # Inject a single deterministic sample with known norm 50.
    H = torch.zeros(1, d)
    H[0, :50] = 1.0  # ||H||_2 == sqrt(50) ~ 7.07
    cal.update(H)
    A, eta0 = cal.finalize()
    expected = d / (alpha * A)
    assert abs(eta0 - expected) < 1e-2


# --------------------------------------------------------------------------- #
# 4. Idempotent finalize
# --------------------------------------------------------------------------- #
def test_finalize_idempotent():
    cal = ActivationNormCalibrator(target_relative_alpha=0.1, hidden_dim=128)
    cal.update(torch.randn(8, 128))
    A1, e1 = cal.finalize()
    A2, e2 = cal.finalize()
    assert A1 == A2 and e1 == e2
