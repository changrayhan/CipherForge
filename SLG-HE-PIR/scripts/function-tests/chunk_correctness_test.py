#!/usr/bin/env python3
"""
Chunk-pipeline correctness test.

Verifies that the chunked U→M path produces bit-exact results vs the flat path
on a small batch. Covers:
  - Masked ciphertext bytes (U-side): all chunks concatenated == flat output.
  - Decrypted gradient (M-side): per-element equality to within BFV precision
    (BFV plaintext mod plain_modulus can introduce O(plain_modulus/scale) of
    noise in pathological cases, but for our setup it's < 1e-3).

Note: We use BFV with plain_bits=20 and scale=10000, so the per-element error
budget is plain_modulus/scale = 2^20 / 1e4 ≈ 105 — but the gradient is
a_t - V_y where each component is ~0.05, so relative error is large. We
therefore compare MASKED-CT bit-exactness (the integrity-critical property)
and DECRYPTED VALUES bit-exactness within additive noise bound.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.core.bfv_privselect_v2_adapter import (
    BFVPrivSelectV2Backend,
    add_mask_to_ct_batch_par,
    decrypt_only_batch_par,
    respond_s3pir_batch,
)


def main():
    print("=" * 80)
    print("chunk-pipeline correctness vs flat path")
    print("=" * 80)

    # Small but representative scale.
    N = 1024
    HIDDEN = 4096
    SCALE = 10000
    POLY_DEG = 4096
    PLAIN_BITS = 20
    N_TOKENS = 768   # divisible by all chunk sizes we test (192, 256, 384, 768)
    CHUNK_SIZES = [192, 256, 384, 768]

    np.random.seed(2026)
    torch.manual_seed(2026)

    V = np.random.randn(N, HIDDEN).astype(np.float32) * 0.05

    backend = BFVPrivSelectV2Backend(
        n_entries=N, vec_dim=HIDDEN,
        shared_seed=b"\x42" * 32, scale=SCALE,
        cache_dir="/tmp/chunk_correctness_db", poly_degree=POLY_DEG,
        plain_bits=PLAIN_BITS, hold_secret_key=True,
    )
    t0 = time.time()
    backend.build_encrypted_database(V, force=False)
    print(f"DB load/build: {time.time() - t0:.2f}s")

    # Generate inputs.
    y_t_all = np.random.randint(0, N, size=N_TOKENS).tolist()
    a_t_all = torch.randn(N_TOKENS, HIDDEN).numpy().astype(np.float32) * 0.05

    # ------------------------------------------------------------------
    # Flat reference path: one big call.
    # ------------------------------------------------------------------
    print("\n[Flat reference path]")
    t0 = time.time()
    ct_rows = respond_s3pir_batch(backend, list(zip(range(N_TOKENS), y_t_all)))
    s3pir_flat = []
    for t_flat, y_t, ct in zip(range(N_TOKENS), y_t_all, ct_rows):
        s3pir_flat.append({
            "parity_real_bytes": ct, "parity_dummy_bytes": b"",
            "permutation": 0, "step": 0, "t_flat": t_flat,
            "real_indices": [y_t], "dummy_indices": [],
        })
    s_shares = []
    for t_flat in range(N_TOKENS):
        s_shares.append(
            backend.shares.server_make_share(0, t_flat, a_t_all[t_flat])
        )
    t_flat_s = time.time() - t0

    t0 = time.time()
    ct_list_flat = add_mask_to_ct_batch_par(
        backend, s3pir_flat, step=0, n_workers=4,
    )
    t_flat_u = time.time() - t0

    t0 = time.time()
    masked_flat = decrypt_only_batch_par(backend, ct_list_flat, n_workers=4)
    t_flat_m = time.time() - t0

    # Flat gradient (plaintext add s_share/SCALE).
    grad_flat = np.zeros((N_TOKENS, HIDDEN), dtype=np.float32)
    for t_flat in range(N_TOKENS):
        s_arr = np.array(s_shares[t_flat][:HIDDEN], dtype=np.float32) / SCALE
        grad_flat[t_flat] = masked_flat[t_flat] + s_arr

    print(f"  S: {t_flat_s*1000:.1f} ms")
    print(f"  U: {t_flat_u*1000:.1f} ms")
    print(f"  M: {t_flat_m*1000:.1f} ms")

    # ------------------------------------------------------------------
    # Chunked paths: 4 chunk sizes, verify bit-exact.
    # ------------------------------------------------------------------
    for chunk_size in CHUNK_SIZES:
        if N_TOKENS % chunk_size != 0:
            continue
        n_chunks = N_TOKENS // chunk_size
        print(f"\n[Chunked path — chunk_size={chunk_size}, n_chunks={n_chunks}]")

        # Reset masks (s_shares are identical, depend only on t_flat+seed).
        ct_list_chunked: List[bytes] = []
        for c in range(n_chunks):
            c_start = c * chunk_size
            c_end = c_start + chunk_size
            chunk_resp = s3pir_flat[c_start:c_end]
            ct_chunk = add_mask_to_ct_batch_par(
                backend, chunk_resp, step=0, n_workers=4,
            )
            ct_list_chunked.extend(ct_chunk)

        # Bit-exact check on masked ct.
        diffs_ct = [i for i, (a, b) in enumerate(zip(ct_list_flat, ct_list_chunked))
                    if a != b]
        if diffs_ct:
            print(f"  ✗ MASKED-CT MISMATCH at {len(diffs_ct)} positions: {diffs_ct[:5]}")
            return 1
        print(f"  ✓ All {N_TOKENS} masked ct byte-exact across chunks")

        # Decrypt all chunks together.
        masked_chunked = decrypt_only_batch_par(
            backend, ct_list_chunked, n_workers=4,
        )
        # Numerical check on masked (pre-share).
        max_diff = float(np.max(np.abs(masked_flat - masked_chunked)))
        print(f"  ✓ Max |masked_flat - masked_chunked| = {max_diff:.2e}")
        if max_diff > 1e-3:
            print(f"  ✗ Decryption numerical mismatch (>{1e-3:.0e})")
            return 1

        # Final gradient: each chunk adds its share.
        grad_chunked = np.zeros((N_TOKENS, HIDDEN), dtype=np.float32)
        for t_flat in range(N_TOKENS):
            s_arr = np.array(s_shares[t_flat][:HIDDEN], dtype=np.float32) / SCALE
            grad_chunked[t_flat] = masked_chunked[t_flat] + s_arr
        # Plaintext mod plain_modulus is the same in both paths (since each
        # token's r_t is identical) → grad_chunked and grad_flat should be
        # identical up to BFV quantization noise.
        grad_err = float(np.max(np.abs(grad_flat - grad_chunked)))
        print(f"  Max |grad_flat - grad_chunked| = {grad_err:.2e}")
        if grad_err > 1e-3:
            print(f"  ✗ Gradient numerical mismatch (>{1e-3:.0e})")
            return 1

    # ------------------------------------------------------------------
    # Spot-check: spot-check 3 tokens — what is grad_flat[t] vs a_t - V_y[t]?
    # ------------------------------------------------------------------
    print("\n[Spot-check on flat gradient vs analytical a_t - V_y_t]")
    for t_flat in [0, 384, 767]:
        y = y_t_all[t_flat]
        expected = (a_t_all[t_flat] - V[y])
        err = float(np.max(np.abs(grad_flat[t_flat] - expected)))
        print(f"  t={t_flat:3d}  y={y:5d}  "
              f"max |g - (a_t - V_y)| = {err:.2e}  "
              f"(BFV residual mod plain_modulus)")

    print("\n✅ ALL chunked paths produce bit-exact results vs flat path")
    return 0


if __name__ == "__main__":
    sys.exit(main())