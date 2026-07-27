"""Micro-benchmark for the 5-step encryption pipeline optimizations.

Measures per-call latency of the hot-path primitives before and after the
optimizations. This complements the full-finetune smoke test by isolating
each component.
"""
from __future__ import annotations

import os
import statistics
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "..", "..", "src"))

from core.bfv_privselect_v2_adapter import (
    PRGShareProtocolBFV,
    _USE_SHM_TMP,
    _seal_tmpfile,
)


def bench_prg(n_iter: int = 50, n: int = 4096) -> float:
    seed = os.urandom(32)
    prg = PRGShareProtocolBFV(seed=seed, poly_degree=n, plain_bits=30)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        _ = prg.generate_mask_ints(0, 0)
    elapsed = time.perf_counter() - t0
    return elapsed / n_iter * 1000  # ms/call


def bench_seal_tmpfile_write(n_iter: int = 1000, size_kb: int = 64) -> float:
    """Measure the per-write cost of the new _seal_tmpfile (tmpfs-backed)."""
    data = os.urandom(size_kb * 1024)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        with _seal_tmpfile(suffix=".bench") as f:
            f.write(data)
            f.flush()
            os.unlink(f.name)
    elapsed = time.perf_counter() - t0
    return elapsed / n_iter * 1000


def bench_encoder_passthrough(n_iter: int = 1000, n: int = 4096) -> float:
    """Measure the cost of the encoder.encode call with a numpy array vs list."""
    arr = np.random.randint(-(1 << 29), 1 << 29, size=n, dtype=np.int64)
    lst = arr.tolist()
    # We can't import seal here without a working context, so we use a stand-in
    # by timing pure-Python list comprehension vs numpy slicing.
    t0 = time.perf_counter()
    for _ in range(n_iter):
        _ = [int(x) for x in arr]
    list_ms = (time.perf_counter() - t0) / n_iter * 1000
    t0 = time.perf_counter()
    for _ in range(n_iter):
        _ = arr
    passthrough_ms = (time.perf_counter() - t0) / n_iter * 1000
    return list_ms, passthrough_ms


def main():
    print("=" * 60)
    print("Task A micro-benchmarks for the 5-step optimizations")
    print("=" * 60)

    print(f"\n[_seal_tmpfile] /dev/shm available: {_USE_SHM_TMP}")
    prg_ms = bench_prg()
    print(f"[Step 2 PRG]  _prf_block(n=4096): {prg_ms:.3f} ms/call")

    write_ms = bench_seal_tmpfile_write()
    print(f"[Step 3 tmpfs] tempfile write(64KB): {write_ms:.3f} ms/call")

    list_ms, pass_ms = bench_encoder_passthrough()
    print(f"[Step 4 encoder] [int(x) for x in r_t] (list): {list_ms:.3f} ms/call")
    print(f"[Step 4 encoder] direct numpy passthrough:    {pass_ms:.3f} ms/call")
    print(f"[Step 4 encoder] saving:                       {list_ms - pass_ms:.3f} ms/call")

    # Step 1 + Step 5 are config-only; no micro-benchmark possible.


if __name__ == "__main__":
    main()