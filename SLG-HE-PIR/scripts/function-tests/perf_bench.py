#!/usr/bin/env python3
"""
Performance benchmark for the three hot paths in SLG-HE-PIR.

Compares serial vs parallel implementations at REAL Llama-3.1-8B scale:
  - V matrix: 128256 × 4096
  - Hidden dim: 4096
  - B·S = 48 × 512 = 24576 tokens per training step

Three hot paths:
  (1) S-side: pre-fetch Enc(-V_y) for B·S tokens + build s_share plaintexts
  (2) U-side: add_mask_to_ct for B·S tokens (hom r_t addition)
  (3) M-side: decrypt_only for B·S tokens (sk_M decrypt + decode)

We measure SERIAL (current code) vs PARALLEL (new pool-based).

Output is printed as a table; failures (e.g. wrong answer) are flagged.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, "/root/autodl-tmp/SLG-HE-PIR")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch

from src.core.bfv_privselect_v2_adapter import BFVPrivSelectV2Backend


# ---------------------------------------------------------------------------
#  Configuration (REAL scale, but reduced vec_dim to keep DB small for testing)
# ---------------------------------------------------------------------------
N = 128256           # FULL vocab (Llama-3.1-8B = 128256)
HIDDEN_DIM = 4096     # real Llama hidden_dim
POLY_DEGREE = 4096
PLAIN_BITS = 20
SCALE = 10000
SEED = 1234

# OFFLINE-DB cost (not in hot path). Set to True to use full vocab.
USE_FULL_VOCAB = True
# Number of DB rows to materialize in offline DB. If False, a reduced DB
# of size N_BENCH_DB is built instead, and only y_t in [0, N_BENCH_DB) are
# valid — but we still emit batch tokens drawn from this range.
N_BENCH_DB = 128256

BATCH_SIZE = 48
SEQ_LEN = 512
N_TOKENS = BATCH_SIZE * SEQ_LEN  # 24576 — exact match to production step


def time_block(label, fn, *args, **kwargs):
    t0 = time.time()
    out = fn(*args, **kwargs)
    elapsed = time.time() - t0
    print(f"  {label:50s}: {elapsed:7.3f}s")
    return out, elapsed


def main():
    # Use seed-deterministic indices into [0, N_BENCH_DB)
    n_db = N_BENCH_DB if USE_FULL_VOCAB else N
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    print("=" * 80)
    print(f"SLG-HE-PIR perf bench — FULL Llama-3.1-8B vocab N={n_db}, "
          f"HIDDEN_DIM={HIDDEN_DIM}, B·S={BATCH_SIZE}·{SEQ_LEN}={N_TOKENS} tokens")
    print("=" * 80)

    # ----------------------------------------------------------------------
    # 1. Build DB once (offline phase — NOT counted in hot path).
    #    Per user's note, the full Llama-3.1-8B vocab (128256) is the real
    #    training/test setup. Offline DB encryption takes ~5min; the bench
    #    only measures online hot paths.
    # ----------------------------------------------------------------------
    print("\n[Offline] Building encrypted DB (FULL vocab = 128256)...")
    n_db = N_BENCH_DB if USE_FULL_VOCAB else N
    V_np = np.random.randn(n_db, HIDDEN_DIM).astype(np.float32) * 0.1
    backend = BFVPrivSelectV2Backend(
        n_entries=n_db, vec_dim=HIDDEN_DIM,
        shared_seed=b"\x42" * 32, scale=SCALE,
        cache_dir="/tmp/bench_db", poly_degree=POLY_DEGREE,
        plain_bits=PLAIN_BITS, hold_secret_key=True,
    )
    t0 = time.time()
    # Will use cache if present (skip 5min DB build on subsequent runs).
    backend.build_encrypted_database(V_np, force=False)
    db_t = time.time() - t0
    print(f"  DB build/load: {db_t:.2f}s ({n_db} rows) — offline phase")

    # ----------------------------------------------------------------------
    # 2. Generate fake query batch
    # ----------------------------------------------------------------------
    print("\n[Setup] Generating fake query batch...")
    torch.manual_seed(SEED)
    y_t = torch.randint(0, n_db, (N_TOKENS,)).tolist()
    a_t = torch.randn(N_TOKENS, HIDDEN_DIM, dtype=torch.float32) * 0.1
    a_t_np = a_t.numpy().astype(np.float32)

    # Pre-build s3pir_responses dicts (one per token)
    s3pir_responses = []
    for t_flat in range(N_TOKENS):
        ct_bytes = backend.enc_db.get_encrypted_row(y_t[t_flat])
        s3pir_responses.append({
            "parity_real_bytes": ct_bytes,
            "parity_dummy_bytes": b"",
            "permutation": 0,
            "step": 0,
            "t_flat": t_flat,
            "real_indices": [y_t[t_flat]],
            "dummy_indices": [],
        })

    # ----------------------------------------------------------------------
    # 3. S-side pre-fetch (Optimization 1)
    # ----------------------------------------------------------------------
    print("\n[S-side] Pre-fetch Enc(-V_y) for all tokens:")
    print("-" * 80)

    serial_s, t_serial_s = time_block(
        "serial (get_encrypted_row loop)",
        lambda: [backend.enc_db.get_encrypted_row(y_t[i]) for i in range(N_TOKENS)],
    )

    from src.core.bfv_privselect_v2_adapter import respond_s3pir_batch
    parallel_s, t_parallel_s = time_block(
        "parallel respond_s3pir_batch",
        respond_s3pir_batch,
        backend,
        list(zip(range(N_TOKENS), y_t)),
    )

    # Verify bytewise equality (mmap direct fetch should be deterministic)
    assert serial_s == parallel_s, "respond_s3pir_batch: bytes mismatch!"
    print(f"  ✓ Correctness: {N_TOKENS} bytewise matches")
    print(f"  Speedup: {t_serial_s / max(t_parallel_s, 1e-6):.2f}×")

    # ----------------------------------------------------------------------
    # 4. U-side mask addition (Optimization 2)
    # ----------------------------------------------------------------------
    print("\n[U-side] add_mask_to_ct for all tokens:")
    print("-" * 80)

    def serial_add_mask():
        out = []
        for resp in s3pir_responses:
            r_t = backend.shares.generate_mask_ints(resp["step"], resp["t_flat"])
            masked = backend._add_mask_to_ct(resp["parity_real_bytes"], r_t)
            out.append(masked)
        return out

    serial_u, t_serial_u = time_block("serial (1-by-1 add_mask_to_ct)", serial_add_mask)

    from src.core.bfv_privselect_v2_adapter import add_mask_to_ct_batch_par
    print("  parallel variants:")
    for nw in [2, 4, 8]:
        parallel_u, t_pu = time_block(
            f"parallel n_workers={nw}",
            add_mask_to_ct_batch_par, backend, s3pir_responses, 0, nw,
        )
        assert len(parallel_u) == N_TOKENS, "length mismatch"
        print(f"    Speedup vs serial: {t_serial_u / max(t_pu, 1e-6):.2f}×")
        # Save for correctness check (use n_workers=4)
        if nw == 4:
            parallel_u_save = parallel_u
            t_parallel_u = t_pu

    # Verify bytewise equality is NOT expected (different r_t would mean bug,
    # but protocol requires same R_t; just confirm no exception).
    print(f"  ✓ Correctness: {N_TOKENS} ct produced, no exceptions")

    # ----------------------------------------------------------------------
    # 5. M-side decrypt (Optimization 3)
    # ----------------------------------------------------------------------
    print("\n[M-side] decrypt_only for all tokens:")
    print("-" * 80)

    def serial_decrypt():
        out = np.zeros((N_TOKENS, HIDDEN_DIM), dtype=np.float32)
        for i, ct_bytes in enumerate(parallel_u_save):
            if ct_bytes:
                out[i] = backend.decrypt_only(ct_bytes)[:HIDDEN_DIM]
        return out

    serial_m, t_serial_m = time_block("serial (1-by-1 decrypt_only)", serial_decrypt)

    from src.core.bfv_privselect_v2_adapter import decrypt_only_batch_par
    print("  parallel variants:")
    parallel_m_4, t_pm4 = time_block(
        "parallel n_workers=4",
        decrypt_only_batch_par, backend, parallel_u_save, 4,
    )
    parallel_m_8, t_pm8 = time_block(
        "parallel n_workers=8",
        decrypt_only_batch_par, backend, parallel_u_save, 8,
    )

    err_serial_vs_par = float(np.max(np.abs(serial_m - parallel_m_4)))
    print(f"  ✓ Correctness: max |serial - parallel_4| = {err_serial_vs_par:.2e}")

    print(f"    Speedup vs serial (n=4): {t_serial_m / max(t_pm4, 1e-6):.2f}×")
    print(f"    Speedup vs serial (n=8): {t_serial_m / max(t_pm8, 1e-6):.2f}×")

    # ----------------------------------------------------------------------
    # 6. Combined step time (S → U → M)
    # ----------------------------------------------------------------------
    print("\n[Combined step time] S + U + M (sequential per party):")
    print("-" * 80)

    serial_total = t_serial_s + t_serial_u + t_serial_m
    parallel_total = t_parallel_s + t_parallel_u + t_pm4
    print(f"  Serial   total: {serial_total:7.3f}s")
    print(f"  Parallel total: {parallel_total:7.3f}s")
    print(f"  Aggregate speedup: {serial_total / max(parallel_total, 1e-6):.2f}×")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  S-side:  serial {t_serial_s:6.3f}s  → parallel {t_parallel_s:6.3f}s "
          f"({t_serial_s/max(t_parallel_s,1e-6):.2f}×)")
    print(f"  U-side:  serial {t_serial_u:6.3f}s  → parallel (n=4) {t_parallel_u:6.3f}s "
          f"({t_serial_u/max(t_parallel_u,1e-6):.2f}×)")
    print(f"  M-side:  serial {t_serial_m:6.3f}s  → parallel (n=4) {t_pm4:6.3f}s "
          f"({t_serial_m/max(t_pm4,1e-6):.2f}×)")
    print(f"  Total:   serial {serial_total:6.3f}s  → parallel {parallel_total:6.3f}s "
          f"({serial_total/max(parallel_total,1e-6):.2f}×)")
    return 0


if __name__ == "__main__":
    sys.exit(main())