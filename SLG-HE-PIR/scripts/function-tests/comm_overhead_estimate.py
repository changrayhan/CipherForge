#!/usr/bin/env python3
"""Estimate communication overhead from BFV ciphertext sizes.

The SLG-HE-PIR protocol sends BFV ciphertexts at three channels during each
training step. The ciphertext size in BFV with poly_degree N and L coefficient
modulus primes is:

    |ct| = N * L * 8 bytes (when stored in 64-bit uint form per slot)

Per the protocol design (per TEST_REPORT.md §2.2.1):
  - U -> M (forward): smashed-data BFV ciphertext per token per chunk
  - M -> S (forward): PIR query / Hint table index
  - S -> U (forward): PIR response (BFV ciphertext, returned)
  - U -> M (backward): mask-added BFV ciphertext (same size as forward)
  - M -> S (backward): gradient share (plaintext, ~hidden_dim ints)
  - S -> U (backward): PRG share payload (plaintext, ~hidden_dim ints)

Per the test design (Llama-3.1-8B-Instruct, hidden_dim=4096, poly_degree=4096):
  - Forward U->M:  N=4096, L=3 (coeff primes 36+36+37), 8 bytes/slot
                  per-token ct size = 4096 * 3 * 8 = 98304 bytes ≈ 96 KB
  - Forward S->U: same size (PIR returns the same ciphertext after S fetch)
  - Backward U->M: same size (mask addition does not change ct size)
  - M -> S, S -> U (shares): hidden_dim * 8 = 4096 * 8 = 32768 bytes ≈ 32 KB

Total per training step (production scale: B=1, S=512 → 512 tokens per chunk,
K≈8 chunks per step → 4096 tokens per step):

  Forward  U -> M : 4096 tokens × 96 KB ≈ 384 MB / step
  Forward  M -> S : small (PIR query)
  Forward  S -> U : 4096 tokens × 96 KB ≈ 384 MB / step
  Backward U -> M : 4096 tokens × 96 KB ≈ 384 MB / step
  Backward M -> S : 4096 tokens × 32 KB ≈ 128 MB / step
  Backward S -> U : 4096 tokens × 32 KB ≈ 128 MB / step

  TOTAL ≈ 1.4 GB / step (forward + backward)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

POLY_DEGREE = 4096
COEFF_PRIMES = 3  # [36, 36, 37]
SLOT_BYTES = 8
HIDDEN_DIM = 4096

PER_TOKEN_CT_BYTES = POLY_DEGREE * COEFF_PRIMES * SLOT_BYTES
PER_TOKEN_PLAIN_BYTES = HIDDEN_DIM * SLOT_BYTES


def estimate_step_comm(tokens_per_step: int = 4096) -> dict:
    """Return per-channel bytes for one training step."""
    fwd_u_to_m = tokens_per_step * PER_TOKEN_CT_BYTES
    fwd_m_to_s = 1024  # small PIR query (placeholder, ~1 KB)
    fwd_s_to_u = tokens_per_step * PER_TOKEN_CT_BYTES
    bwd_u_to_m = tokens_per_step * PER_TOKEN_CT_BYTES
    bwd_m_to_s = tokens_per_step * PER_TOKEN_PLAIN_BYTES
    bwd_s_to_u = tokens_per_step * PER_TOKEN_PLAIN_BYTES
    return {
        "tokens_per_step": tokens_per_step,
        "per_token_ct_bytes": PER_TOKEN_CT_BYTES,
        "per_token_plain_bytes": PER_TOKEN_PLAIN_BYTES,
        "forward_u_to_m_bytes": fwd_u_to_m,
        "forward_m_to_s_bytes": fwd_m_to_s,
        "forward_s_to_u_bytes": fwd_s_to_u,
        "backward_u_to_m_bytes": bwd_u_to_m,
        "backward_m_to_s_bytes": bwd_m_to_s,
        "backward_s_to_u_bytes": bwd_s_to_u,
        "total_forward_bytes": fwd_u_to_m + fwd_m_to_s + fwd_s_to_u,
        "total_backward_bytes": bwd_u_to_m + bwd_m_to_s + bwd_s_to_u,
        "total_bytes_per_step": (
            fwd_u_to_m + fwd_m_to_s + fwd_s_to_u
            + bwd_u_to_m + bwd_m_to_s + bwd_s_to_u
        ),
    }


def main():
    # Llama-3.1-8B-Instruct: batch=1, seq_len=512, K=8 chunks of ~512
    # chunk_tokens=3072 in classification, 1536 in NER
    # production tokens per step ≈ 8 chunks × 384 = 3072 (cls)
    #                              14 chunks × 256 = 3584 (NER)
    for tokens in [3072, 3584, 4096]:
        d = estimate_step_comm(tokens)
        print(f"--- tokens_per_step={tokens} ---")
        for k, v in d.items():
            if k.startswith("total") or "u_to_m" in k or "to_s" in k or "s_to_u" in k:
                if isinstance(v, int):
                    if v >= 1024 * 1024:
                        print(f"  {k:30s}: {v/1024/1024:.2f} MB ({v} bytes)")
                    elif v >= 1024:
                        print(f"  {k:30s}: {v/1024:.2f} KB")
                    else:
                        print(f"  {k:30s}: {v} bytes")
                else:
                    print(f"  {k:30s}: {v}")
        print()

    # Save the JSON to test-data
    out_dir = Path("/root/autodl-tmp/SLG-HE-PIR/test-data/perf-test-data")
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "communication_overhead.json"
    with open(out_file, "w") as f:
        json.dump(
            {
                "config": {
                    "poly_degree": POLY_DEGREE,
                    "coeff_primes": COEFF_PRIMES,
                    "slot_bytes": SLOT_BYTES,
                    "hidden_dim": HIDDEN_DIM,
                    "per_token_ct_bytes": PER_TOKEN_CT_BYTES,
                    "per_token_plain_bytes": PER_TOKEN_PLAIN_BYTES,
                },
                "estimates": {
                    "cls_task_3072_tokens": estimate_step_comm(3072),
                    "ner_task_3584_tokens": estimate_step_comm(3584),
                },
                "source": "scripts/function-tests/comm_overhead_estimate.py",
            },
            f,
            indent=2,
        )
    print(f"Saved communication overhead estimate to {out_file}")


if __name__ == "__main__":
    main()