"""pytest fixtures for the d_χ privacy test suite."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

# Ensure src/ is importable.
_THIS = Path(__file__).resolve()
_REPO_ROOT = _THIS.parents[2]
_SRC = _REPO_ROOT / "src"
_TESTS = _REPO_ROOT / "tests"
for p in (str(_REPO_ROOT), str(_SRC), str(_TESTS)):
    if p not in sys.path:
        sys.path.insert(0, p)


# --------------------------------------------------------------------------- #
#  Project paths
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def repo_root() -> Path:
    return _REPO_ROOT


@pytest.fixture(scope="session")
def trecqc_data_dir() -> Path:
    return _REPO_ROOT / "SLG-HE-PIR" / "datasets" / "trec-qc"


@pytest.fixture(scope="session")
def llama32_1b_local_dir() -> Path:
    """Path to the local Llama-3.2-1B snapshot directory.

    Resolved to the actual ``snapshots/<hash>`` subdirectory when present
    so HF can find ``config.json`` directly.
    """
    base = Path(
        "/root/autodl-tmp/SLG-HE-PIR-code/hf_cache/models--unsloth--Llama-3.2-1B"
    )
    if not base.exists():
        # Fall back to the snapshot directory if the base exists.
        return base
    snapshots = base / "snapshots"
    if snapshots.exists():
        children = sorted(snapshots.iterdir())
        if children:
            return children[0]
    return base


# --------------------------------------------------------------------------- #
#  DP configurations
# --------------------------------------------------------------------------- #
def _make_dp_config(**overrides: Any) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {
        "dp_enable": True,
        "dp_alpha": 0.15,
        "dp_answer_beta": 0.5,
        "dp_calibration_steps": 1,
        "dp_num_classes": 6,  # TREC-QC coarse
        "hidden_dim": 2048,
        "vocab_size": 128256,
        "seed": 42,
    }
    cfg.update(overrides)
    return cfg


@pytest.fixture
def dp_config_cpu() -> Dict[str, Any]:
    """DP config with a CPU device — for fast unit tests."""
    cfg = _make_dp_config(dp_device="cpu")
    return cfg


@pytest.fixture
def dp_config_cuda() -> Dict[str, Any]:
    """DP config with CUDA when available, CPU otherwise."""
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = _make_dp_config(dp_device=device)
    return cfg


# --------------------------------------------------------------------------- #
#  Fake batches
# --------------------------------------------------------------------------- #
@pytest.fixture
def fake_batch_cpu() -> Dict[str, Any]:
    """A tiny fake batch on CPU for unit tests.

    * B=2, S=64
    * labels != -100 only at the last 4 positions (simulating answer span).
    """
    import torch
    input_ids = torch.randint(low=10, high=128_256, size=(2, 64), dtype=torch.long)
    attention_mask = torch.ones((2, 64), dtype=torch.long)
    labels = torch.full((2, 64), -100, dtype=torch.long)
    # Mark the last 4 positions as answer for each sample.
    labels[:, -4:] = input_ids[:, -4:]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


@pytest.fixture
def fake_batch_cuda() -> Dict[str, Any]:
    """A tiny fake batch on the active device."""
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch = {
        "input_ids": torch.randint(low=10, high=128_256, size=(2, 64), dtype=torch.long, device=device),
        "attention_mask": torch.ones((2, 64), dtype=torch.long, device=device),
        "labels": torch.full((2, 64), -100, dtype=torch.long, device=device),
    }
    batch["labels"][:, -4:] = batch["input_ids"][:, -4:]
    return batch
