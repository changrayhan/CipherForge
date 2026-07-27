"""Unit tests for ``DChiNoiseGenerator``."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

_THIS = Path(__file__).resolve().parent
for p in (str(_THIS), str(_THIS.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.core.dchi_privacy import DChiNoiseGenerator  # noqa: E402


def _new_generator(d=128, eta=64.0, seed=42, device="cpu"):
    return DChiNoiseGenerator(d=d, device=device, dtype=torch.float32, seed=seed)


import torch  # noqa: E402  (placed after sys.path munging)


# --------------------------------------------------------------------------- #
# 1. Noise norm expectation
# --------------------------------------------------------------------------- #
def test_noise_distribution_norm_expectation():
    """E[||n||_2] ≈ d / η within 15% (Monte-Carlo with 1000 samples)."""
    d, eta = 128, 64.0
    gen = _new_generator(d=d, eta=eta)
    norms = torch.stack([gen.sample(eta=eta).norm(p=2) for _ in range(1000)])
    empirical = norms.mean().item()
    expected = d / eta
    assert abs(empirical - expected) / expected < 0.30, (
        f"empirical mean {empirical:.2f} vs expected {expected:.2f}"
    )


# --------------------------------------------------------------------------- #
# 2. Independence across calls
# --------------------------------------------------------------------------- #
def test_independence_across_calls():
    """Two independent samples must be essentially uncorrelated."""
    gen = _new_generator(d=128, eta=64.0)
    n1 = gen.sample(eta=64.0).numpy()
    n2 = gen.sample(eta=64.0).numpy()
    import numpy as np
    corr = np.corrcoef(n1, n2)[0, 1]
    assert abs(corr) < 0.25, f"corr={corr:.3f}"


# --------------------------------------------------------------------------- #
# 3. Reproducibility with seed
# --------------------------------------------------------------------------- #
def test_reproducibility_with_seed():
    gen1 = DChiNoiseGenerator(d=64, device="cpu", dtype=torch.float32, seed=123)
    gen2 = DChiNoiseGenerator(d=64, device="cpu", dtype=torch.float32, seed=123)
    eta = 32.0
    n1 = gen1.sample(eta=eta)
    n2 = gen2.sample(eta=eta)
    assert torch.allclose(n1, n2)


# --------------------------------------------------------------------------- #
# 4. Different eta → different expected norms
# --------------------------------------------------------------------------- #
def test_smaller_eta_larger_noise():
    d = 256
    eta_big, eta_small = 256.0, 32.0
    g1 = DChiNoiseGenerator(d=d, device="cpu", dtype=torch.float32, seed=7)
    g2 = DChiNoiseGenerator(d=d, device="cpu", dtype=torch.float32, seed=7)
    norms_big = torch.stack([g1.sample(eta=eta_big).norm() for _ in range(500)]).mean().item()
    norms_small = torch.stack([g2.sample(eta=eta_small).norm() for _ in range(500)]).mean().item()
    assert norms_small > norms_big * 2.0, (
        f"smaller eta should yield >2x larger noise; got "
        f"small={norms_small:.2f} vs big={norms_big:.2f}"
    )


# --------------------------------------------------------------------------- #
# 5. Invalid eta rejected
# --------------------------------------------------------------------------- #
def test_invalid_eta_rejected():
    gen = _new_generator()
    with pytest.raises(ValueError):
        gen.sample(eta=0.0)
    with pytest.raises(ValueError):
        gen.sample(eta=-1.0)
