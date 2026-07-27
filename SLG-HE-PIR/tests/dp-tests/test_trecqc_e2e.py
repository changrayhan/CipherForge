"""End-to-end TREC-QC test with the d_χ privatiser attached.

These tests load the actual U shard from the local Llama-3.2-1B snapshot
(``u_layers=2`` so the bottom model is small). They are skipped
automatically when the model is unavailable or CUDA is not present.

Run via::

    bash tests/dp-tests/scripts/run_trecqc_e2e.sh 0 ./dp_test_output
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest
import torch

_THIS = Path(__file__).resolve().parent
for p in (str(_THIS), str(_THIS.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)


LLAMA_LOCAL = Path(
    "/root/autodl-tmp/SLG-HE-PIR-code/hf_cache/models--unsloth--Llama-3.2-1B"
)
TRECQC_DIR = Path("/root/autodl-tmp/SLG-HE-PIR-code/SLG-HE-PIR/datasets/trec-qc")


def _snapshot_dir() -> Path | None:
    if not LLAMA_LOCAL.exists():
        return None
    snapshots = LLAMA_LOCAL / "snapshots"
    if snapshots.exists():
        children = sorted(snapshots.iterdir())
        if children:
            return children[0]
    return LLAMA_LOCAL


# --------------------------------------------------------------------------- #
#  Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def llama_path() -> str:
    p = _snapshot_dir()
    if p is None:
        pytest.skip(f"Llama-3.2-1B not found at {LLAMA_LOCAL}")
    return str(p)


@pytest.fixture(scope="module")
def tokenizer(llama_path):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(llama_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


# --------------------------------------------------------------------------- #
#  Lightweight E2E: H_15 + DP with the real Llama-3.2-1B U shard
# --------------------------------------------------------------------------- #
def test_h15_dchi_with_real_llama(llama_path, tokenizer, tmp_path, fake_batch_cpu):
    """U-shard forward with DP attached returns a 3-D privatised tensor."""
    from src.model.model_splitting import detect_model_spec, load_u_submodel
    from src.core.dchi_privacy import H15Privatizer

    spec = detect_model_spec(llama_path, u_layers=2)
    assert spec.num_layers in (16, 24), f"unexpected layer count: {spec.num_layers}"
    assert spec.hidden_size == 2048

    model = load_u_submodel(
        spec=spec,
        model_path=llama_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_flash_attention=False,
        use_sage_attention=False,
        gradient_checkpointing_style="reentrant",
        all_weights=None,
    )
    model.eval()
    dp_cfg = {
        "dp_enable": True,
        "dp_alpha": 0.15,
        "dp_answer_beta": 0.5,
        "dp_calibration_steps": 1,
        "dp_num_classes": 6,
        "hidden_dim": spec.hidden_size,
        "vocab_size": spec.vocab_size,
        "seed": 42,
        "dp_device": next(model.parameters()).device,
    }
    priv = H15Privatizer(dp_cfg)
    # Force a fixed eta0 so the test is deterministic.
    priv.eta0_override = 1024.0

    input_ids = torch.randint(10, 128_256, (2, 32), dtype=torch.long, device=next(model.parameters()).device)
    with torch.no_grad():
        H = model(input_ids)
    assert H.shape == (2, 32, spec.hidden_size)

    # Build a minimal fake_batch so the privatiser has labels + attention_mask.
    fake_batch = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": torch.full((2, 32), -100, dtype=torch.long, device=input_ids.device),
    }
    fake_batch["labels"][:, -2:] = input_ids[:, -2:]

    H_tilde, audit = priv(H, fake_batch, stage="train")
    assert H_tilde.shape == H.shape
    assert audit.activated is True


# --------------------------------------------------------------------------- #
#  Sanity: forward_with_attention returns attention list
# --------------------------------------------------------------------------- #
def test_u_shard_forward_with_attention(llama_path):
    from src.model.model_splitting import detect_model_spec, load_u_submodel

    spec = detect_model_spec(llama_path, u_layers=2)
    model = load_u_submodel(
        spec=spec,
        model_path=llama_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_flash_attention=False,
        use_sage_attention=False,
        gradient_checkpointing_style="reentrant",
        all_weights=None,
    )
    model.eval()
    input_ids = torch.randint(10, 128_256, (2, 16), dtype=torch.long, device=next(model.parameters()).device)
    with torch.no_grad():
        H, attns = model.forward_with_attention(input_ids)
    assert H.shape == (2, 16, spec.hidden_size)
    assert isinstance(attns, list)
    assert len(attns) == spec.u_layers
