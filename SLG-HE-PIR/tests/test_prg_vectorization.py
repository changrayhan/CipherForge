"""Unit test for Step 2: PRG numpy vectorization numerical equivalence.

Verifies that the vectorized _prf_block produces byte-identical output to a
reference scalar implementation. This is the gate before Step 2's commit.
"""
import hashlib
import os
import struct
import sys

import numpy as np

# Make src importable
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "..", "src"))

from core.bfv_privselect_v2_adapter import PRGShareProtocolBFV


def reference_prf_block(seed: bytes, step: int, t_flat: int, start_i: int, n: int,
                        plain_modulus: int) -> np.ndarray:
    """Scalar reference matching the original implementation, byte-for-byte."""
    half = plain_modulus // 2
    out = np.empty(n, dtype=np.int64)
    for i in range(n):
        h = hashlib.sha256(seed + struct.pack("!QQQ", step, t_flat, start_i + i)).digest()
        val = int.from_bytes(h[:8], "big")
        val = (val % plain_modulus) - half
        out[i] = int(val)
    return out


def test_prf_byte_identical():
    seed = os.urandom(32)
    plain_modulus = 1 << 30  # 2^30
    poly_degree = 4096
    prg = PRGShareProtocolBFV(seed=seed, poly_degree=poly_degree, plain_bits=30)

    # Sweep a variety of (step, t_flat, start_i, n) shapes
    cases = [
        (0, 0, 0, 1),       # single
        (0, 0, 0, 4096),    # full poly_degree
        (1, 7, 0, 100),     # partial
        (42, 1234, 5, 500), # partial offset
        (9999, 0, 4090, 10),# tail
        (0, 0, 0, 2),
    ]
    for step, t_flat, start_i, n in cases:
        ref = reference_prf_block(seed, step, t_flat, start_i, n, plain_modulus)
        got = prg._prf_block(step, t_flat, start_i, n)
        assert np.array_equal(ref, got), (
            f"Mismatch for step={step} t_flat={t_flat} start_i={start_i} n={n}\n"
            f"ref[:8]={ref[:8]}\ngot[:8]={got[:8]}"
        )
        assert got.dtype == np.int64
        half = plain_modulus // 2
        assert np.all(got >= -half) and np.all(got < half), "out of centred range"

    print("test_prf_byte_identical OK")


def test_prf_speed():
    """Smoke test: the vectorized path should not regress vs scalar for n=4096."""
    import time
    seed = os.urandom(32)
    plain_modulus = 1 << 30
    poly_degree = 4096
    prg = PRGShareProtocolBFV(seed=seed, poly_degree=poly_degree, plain_bits=30)

    n_iter = 10
    t0 = time.perf_counter()
    for i in range(n_iter):
        _ = prg._prf_block(0, 0, 0, poly_degree)
    elapsed = time.perf_counter() - t0
    per_call_ms = elapsed / n_iter * 1000
    print(f"PRG per-call latency for n={poly_degree}: {per_call_ms:.3f} ms")
    # Generous upper bound — 5 ms/call is plenty for the 0.8 ms target
    assert per_call_ms < 5.0, f"PRG too slow: {per_call_ms:.3f} ms/call"


if __name__ == "__main__":
    test_prf_byte_identical()
    test_prf_speed()
    print("ALL PRG TESTS PASSED")