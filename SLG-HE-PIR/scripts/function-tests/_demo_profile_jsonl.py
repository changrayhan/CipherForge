#!/usr/bin/env python3
"""
Generate fake but realistic StepProfiler JSONL for dashboard testing.

Simulates:
  - 100 steps of "flat" mode (no chunking)
  - 100 steps of "chunked" mode (8 chunks/step)

Phase timings follow Llama-3.1-8B / RTX 5090 / 25-core profile:
  - forward_U: 80 ms ± 5
  - forward_M: 8500 ms ± 50 (M-side holds GPU heavy lift)
  - s_logits: 12 ms ± 0.5 (mmap zero-copy)
  - priv_U (flat): 16 300 ms ± 200
  - priv_U (chunked): sum of K chunks of ~2 050 ms each
  - backward_M (flat): 13 800 ms ± 150
  - backward_M (chunked): same (M-side is one-shot at end)

Run mode JSONL goes to logs/profiles/demo_flat.jsonl + demo_chunked.jsonl.
"""
from __future__ import annotations

import json
import math
import os
import random
import statistics
import time
from pathlib import Path

ROOT = Path("/root/autodl-tmp/SLG-HE-PIR")
OUT = ROOT / "logs" / "profiles"
OUT.mkdir(parents=True, exist_ok=True)


def gen_record(step: int, mode: str, n_tokens: int, n_chunks: int,
               rss_base: float) -> dict:
    """Generate one realistic step record."""
    # Mild linear RSS growth + noise.
    rss_mb = rss_base + step * 0.5 + random.gauss(0, 1.5)

    forward_u = random.gauss(80, 5)
    forward_m = random.gauss(8500, 50)
    s_logits = random.gauss(12, 0.5)

    if mode == "chunked" and n_chunks > 1:
        # Each chunk has its own time. Wall sum = sum of chunks (since chunked
        # pool reuses workers). Within-step spread ±5%.
        priv_u_per_chunk = random.gauss(2050, 30)
        chunk_u_times = [priv_u_per_chunk * (1 + random.gauss(0, 0.03))
                         for _ in range(n_chunks)]
        priv_u = sum(chunk_u_times)
        chunk_m_times = []  # M is one-shot in chunked path
    else:
        priv_u = random.gauss(16300, 200)
        chunk_u_times = []
        chunk_m_times = []

    backward_m = random.gauss(13800, 150)
    step_time_ms = forward_u + forward_m + s_logits + priv_u + backward_m

    return {
        "step": step,
        "n_tokens": n_tokens,
        "n_chunks": n_chunks,
        "step_time_ms": step_time_ms,
        "phase_ms": {
            "forward_U": forward_u,
            "forward_M": forward_m,
            "s_logits":  s_logits,
            "priv_U":    priv_u,
            "backward_M": backward_m,
        },
        "phase_order": ["forward_U", "forward_M", "s_logits", "priv_U", "backward_M"],
        "chunk_u_times_ms": chunk_u_times,
        "chunk_m_times_ms": chunk_m_times,
        "rss_mb": rss_mb,
        "ts": time.time(),
        "extra": {"mode": mode},
    }


def write_jsonl(path: Path, records: list) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def main():
    random.seed(42)
    N_TOKENS = 24576
    K = 8 if N_TOKENS % 3072 == 0 else 1
    CHUNK = N_TOKENS // K

    # Flat: 100 steps, K=1 chunk each.
    flat_recs = [gen_record(i, "flat", N_TOKENS, 1, rss_base=450)
                 for i in range(100)]
    write_jsonl(OUT / "demo_flat.jsonl", flat_recs)
    print(f"Wrote {len(flat_recs)} flat records → {OUT/'demo_flat.jsonl'}")

    # Chunked: 100 steps, K=8 chunks each.
    chunked_recs = [gen_record(i, "chunked", N_TOKENS, K, rss_base=470)
                    for i in range(100)]
    write_jsonl(OUT / "demo_chunked.jsonl", chunked_recs)
    print(f"Wrote {len(chunked_recs)} chunked records → {OUT/'demo_chunked.jsonl'}")

    # Print brief stats to confirm.
    flat_t = [r["step_time_ms"] for r in flat_recs]
    chunked_t = [r["step_time_ms"] for r in chunked_recs]
    # Note: in this synthetic data the chunked path uses *same* per-chunk time
    # * N_chunks (because each chunk still goes through add_mask → same total
    # work). Real-world chunked path on overlapping processes would be lower,
    # but for dashboard structure, both shapes are useful.
    print()
    print(f"Flat:    mean={statistics.mean(flat_t):.0f}ms  "
          f"p95={sorted(flat_t)[int(len(flat_t)*0.95)]:.0f}ms  "
          f"max={max(flat_t):.0f}ms")
    print(f"Chunked: mean={statistics.mean(chunked_t):.0f}ms  "
          f"p95={sorted(chunked_t)[int(len(chunked_t)*0.95)]:.0f}ms  "
          f"max={max(chunked_t):.0f}ms")


if __name__ == "__main__":
    main()