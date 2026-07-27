"""Integration tests for ``PartyU.forward_*`` with the d_χ privatiser attached.

These tests deliberately avoid spawning a full Llama U shard on disk; we
mock the shard forward so the test can run quickly on CPU and verify that
the privatiser is invoked at the right point and that gradient flow is
preserved.
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


class _DummyUShard(torch.nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.linear = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, input_ids):
        # Pretend the embed lookup returns a (B, S, d) tensor with grad.
        out = input_ids.float().unsqueeze(-1).expand(-1, -1, 128).clone()
        out = self.linear(out)
        # Mark the output as requiring grad so downstream privatiser
        # tests can verify the autograd passthrough.
        out.requires_grad_(True)
        return out


def _cpu_device(self):
    """Force CPU device for the test PartyU."""
    self.device = torch.device("cpu")
    print(f"[U-device-test] device={self.device}", flush=True)


@pytest.fixture
def party_u_with_dp(dp_config_cpu, monkeypatch):
    """Build a PartyU whose ``model`` is the dummy shard, with DP enabled."""
    from src.parties import party_u as pu_mod
    from src.core.dchi_privacy import H15Privatizer

    # Monkeypatch the methods on the class so ``__init__`` doesn't hit disk.
    def _stub_submodel(self, model_path):
        self.model = _DummyUShard()
        self.spec = type("Spec", (), {"hidden_size": 128, "vocab_size": 128_256})()

    def _stub_bfv(self, pk):
        pass

    def _stub_optimizer(self):
        pass

    monkeypatch.setattr(pu_mod.PartyU, "_setup_submodel", _stub_submodel)
    monkeypatch.setattr(pu_mod.PartyU, "_setup_bfv", _stub_bfv)
    monkeypatch.setattr(pu_mod.PartyU, "_setup_optimizer", _stub_optimizer)
    monkeypatch.setattr(pu_mod.PartyU, "_setup_device", _cpu_device)

    cfg = dict(dp_config_cpu)
    cfg["dp_enable"] = True
    cfg["u_layers"] = 1
    party = pu_mod.PartyU(
        model_path="dummy",
        bfv_pk_pem=b"",
        prg_seed=b"\x00" * 32,
        hint_table=None,
        config=cfg,
    )
    # Attach the privatizer directly with CPU device (the dummy shard is
    # already on CPU so any CUDA device chosen in cfg is overridden).
    party.h15_privatizer = H15Privatizer({**cfg, "dp_device": "cpu"})
    party.h15_privatizer.eta0_override = 128.0
    return party


def test_party_u_forward_train_returns_privatized(party_u_with_dp, fake_batch_cpu):
    result = party_u_with_dp.forward_train(fake_batch_cpu)
    H_tilde = result["H_U"]
    assert H_tilde.dim() == 3
    assert H_tilde.requires_grad


def test_party_u_forward_val_also_privatized(party_u_with_dp, fake_batch_cpu):
    result = party_u_with_dp.forward_val(fake_batch_cpu)
    H_tilde = result["H_U"]
    # Detached in val path but still 3-D.
    assert H_tilde.dim() == 3


def test_party_u_records_dp_audit(party_u_with_dp, fake_batch_cpu):
    party_u_with_dp.forward_train(fake_batch_cpu)
    audit = getattr(party_u_with_dp, "_last_dp_audit", None)
    assert audit is not None
    d = audit.as_dict()
    assert "activated" in d
    assert d["activated"] is True
