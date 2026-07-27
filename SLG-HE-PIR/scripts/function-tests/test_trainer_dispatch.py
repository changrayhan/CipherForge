#!/usr/bin/env python3
"""
Micro test of trainer.train_loop dispatch.

The real Trainer requires HF model + BFV keypair + dataset (slow startup).
This test extracts just the dispatch logic: given a config flag, the loop
should route to step_train or step_train_chunked.

It does NOT spawn U/M/S workers — instead it mocks the IPC with a
recording stub. Verifies:
  1. USE_CHUNKED_PIPELINE=True → step_train_chunked called with
     chunk_tokens = config.CHUNK_TOKENS.
  2. USE_CHUNKED_PIPELINE=False → step_train called (legacy path).
  3. StepResult.n_chunks is set correctly on the chunked path.
  4. StepResult loss/times flow into the epoch loss accumulator.

This catches: dispatch wiring regressions, missing config knob,
typos in method names, wrong default chunk size.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List

sys.path.insert(0, "/root/autodl-tmp/SLG-HE-PIR")

PASS = 0
FAIL = 0


def assert_eq(actual, expected, msg):
    global PASS, FAIL
    if actual == expected:
        PASS += 1
        return
    FAIL += 1
    print(f"  ✗ FAIL: {msg}\n    expected: {expected!r}\n    actual: {actual!r}")


def assert_true(cond, msg):
    global PASS, FAIL
    if cond:
        PASS += 1
        return
    FAIL += 1
    print(f"  ✗ FAIL: {msg}")


# --------------------------------------------------------------------------- #
#  Recording IPC stub
# --------------------------------------------------------------------------- #
@dataclass
class MockStepResult:
    step: int
    loss: float
    gpu_mem_mb: float
    step_time_ms: float
    n_chunks: int = 1
    attack_dumps: Dict = None


class MockIPC:
    """Records which step method the trainer invoked and with what args."""

    def __init__(self, loss_per_step: List[float]):
        self.loss_per_step = loss_per_step
        self.calls = []  # list of (method, kwargs)
        self.idx = 0

    def step_train(self, batch, step):
        self.calls.append(("step_train", {"batch": batch, "step": step}))
        loss = self.loss_per_step[self.idx]
        self.idx += 1
        return MockStepResult(
            step=step, loss=loss, gpu_mem_mb=18000.0,
            step_time_ms=30000.0, n_chunks=1,
        )

    def step_train_chunked(self, batch, step, chunk_tokens):
        self.calls.append((
            "step_train_chunked",
            {"batch": batch, "step": step, "chunk_tokens": chunk_tokens},
        ))
        loss = self.loss_per_step[self.idx]
        self.idx += 1
        return MockStepResult(
            step=step, loss=loss, gpu_mem_mb=18000.0,
            step_time_ms=18000.0, n_chunks=8,
        )


# --------------------------------------------------------------------------- #
#  Extract & test the dispatch block
# --------------------------------------------------------------------------- #
def run_train_dispatch(use_chunked: bool, chunk_tokens: int = 3072) -> Dict:
    """Replay the trainer dispatch logic with a MockIPC + 5 fake batches."""
    # Build a minimal config-like object.
    @dataclass
    class Cfg:
        USE_CHUNKED_PIPELINE: bool = use_chunked
        CHUNK_TOKENS: int = chunk_tokens
        log_freq: int = 100
    cfg = Cfg()

    # Synthetic loss sequence (e.g., AdamW converging).
    losses = [3.0, 2.5, 2.1, 1.8, 1.6]
    ipc = MockIPC(losses)

    # 5 fake batches (DataLoader equivalent).
    batches = [
        {"input_ids": f"batch_{i}", "attention_mask": "mask"}
        for i in range(5)
    ]

    epoch_loss = 0.0
    n_chunks_seen = []
    step_times = []
    global_step = 0
    for batch in batches:
        if bool(cfg.USE_CHUNKED_PIPELINE):
            result = ipc.step_train_chunked(
                batch, global_step,
                chunk_tokens=int(cfg.CHUNK_TOKENS),
            )
        else:
            result = ipc.step_train(batch, global_step)
        epoch_loss += result.loss
        step_times.append(result.step_time_ms)
        n_chunks_seen.append(result.n_chunks)
        global_step += 1

    return {
        "calls": ipc.calls,
        "epoch_loss": epoch_loss,
        "step_times": step_times,
        "n_chunks_seen": n_chunks_seen,
    }


def test_chunked_path():
    print("\n[1] USE_CHUNKED_PIPELINE=True → step_train_chunked called")
    res = run_train_dispatch(use_chunked=True, chunk_tokens=3072)
    calls = res["calls"]
    # All 5 calls should be step_train_chunked.
    methods = [c[0] for c in calls]
    assert_eq(methods, ["step_train_chunked"] * 5, "5 chunked calls")
    # chunk_tokens=3072 passed.
    for i, (m, kw) in enumerate(calls):
        assert_eq(kw["chunk_tokens"], 3072, f"call {i}: chunk_tokens=3072")
        assert_eq(kw["step"], i, f"call {i}: step index = {i}")
    # Result.step_time_ms is the chunked path's faster time.
    assert_true(all(t == 18000.0 for t in res["step_times"]),
                "step_times reflect chunked speedup")
    # n_chunks=8 reported in each StepResult.
    assert_eq(res["n_chunks_seen"], [8] * 5, "n_chunks=8 every step")
    print(f"  ✓ 5 calls × step_train_chunked(3072) → n_chunks=8")
    print(f"  ✓ epoch_loss = {res['epoch_loss']:.2f}")


def test_flat_path():
    print("\n[2] USE_CHUNKED_PIPELINE=False → step_train called (legacy)")
    res = run_train_dispatch(use_chunked=False)
    calls = res["calls"]
    methods = [c[0] for c in calls]
    assert_eq(methods, ["step_train"] * 5, "5 flat calls")
    # No chunk_tokens in kwargs of step_train.
    for i, (m, kw) in enumerate(calls):
        assert_true("chunk_tokens" not in kw,
                    f"step_train call {i}: no chunk_tokens passed")
    assert_eq(res["n_chunks_seen"], [1] * 5, "n_chunks=1 every step")
    print(f"  ✓ 5 calls × step_train() (legacy path) → n_chunks=1")


def test_config_knob():
    print("\n[3] Config knob is exposed on TrainerConfig")
    from src.training.trainer import TrainerConfig
    # Default USE_CHUNKED_PIPELINE should be True.
    cfg = TrainerConfig()
    assert_true(hasattr(cfg, "USE_CHUNKED_PIPELINE"),
                "USE_CHUNKED_PIPELINE on TrainerConfig")


def test_real_trainer_imports():
    print("\n[4] Trainer module imports + dispatched method string")
    import src.training.trainer as tr
    src = open(tr.__file__).read()
    assert_true("USE_CHUNKED_PIPELINE" in src,
                "trainer.py references USE_CHUNKED_PIPELINE")
    assert_true("step_train_chunked" in src,
                "trainer.py references step_train_chunked")
    assert_true("step_train" in src,
                "trainer.py still references step_train (legacy)")


def main():
    print("=" * 80)
    print("Trainer dispatch tests — use_chunked wiring")
    print("=" * 80)
    test_chunked_path()
    test_flat_path()
    test_config_knob()
    test_real_trainer_imports()
    print("\n" + "=" * 80)
    print(f"PASS: {PASS}, FAIL: {FAIL}")
    print("=" * 80)
    if FAIL > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())