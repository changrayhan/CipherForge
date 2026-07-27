#!/usr/bin/env python3
"""
End-to-end math verification for Design 2 protocol.

Runs the COMPLETE flow at small scale with REAL parties:
  - real BFV backend (constructed with hold_secret_key=True)
  - real S3PIR hint table
  - real BFVPrivateSelectV2Backend._respond_s3pir / _add_mask_to_ct / decrypt_only

For each token t, the true gradient is:
    g_t = softmax(logits_t) @ V  -  V_{y_t}

We check that running the protocol produces g_t within numerical precision:
  S:    a_t = softmax(logits) @ V                       # (hidden_dim,)
        parity_real = Enc_M(-V_y)
        s_share = a_t - r_t                              # plaintext (4096,)
  U:    ct_U = parity_real + r_t                        # hom add
  M:    (-V_y + r_t) + (a_t - r_t) = a_t - V_y          # decrypt + plaintext add

Asserts ||g_recovered - g_true||_∞ < tolerance.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, "/root/autodl-tmp/SLG-HE-PIR")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch

from src.core.bfv_privselect_v2_adapter import (
    BFVPrivSelectV2Backend,
    BFVQuery,
    _seal_to_bytes,
)


# ---------------------------------------------------------------------------
#  Configuration (small scale to keep test fast)
# ---------------------------------------------------------------------------
N = 64            # vocab subset
HIDDEN_DIM = 128  # hidden_dim (real Llama uses 4096; we use 128 for speed)
POLY_DEGREE = 4096  # SEAL BFV minimum for relin_keys
PLAIN_BITS = 20
SCALE = 10000
LAMBDA = 10
SEED = 1234
N_TOKENS = 4      # we test with B*S=4 tokens in one "batch"

np.random.seed(SEED)
torch.manual_seed(SEED)


def main() -> int:
    t_start = time.time()

    # ------------------------------------------------------------------
    # 1. Set up S3PIR hint table
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 1: Building S3PIR hint table...")
    from src.core.s3pir_hints import HintTable
    import tempfile
    tmp = tempfile.mkdtemp(prefix="hint_")
    hints = HintTable(
        n_entries=N, partition_size=8, lam=LAMBDA, cache_dir=tmp,
    )
    hints.compute_main_hints_skeleton()
    hints.compute_backup_hints_skeleton()
    # No compute_*_content step — Design 2 only needs the skeleton to build queries.
    print(f"  hint table: n_partitions={hints.n_partitions}, sqrt_n={hints.sqrt_n}")

    # ------------------------------------------------------------------
    # 2. Build fake V matrix (N x hidden_dim)
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 2: Building fake V matrix...")
    V_np = np.random.randn(N, HIDDEN_DIM).astype(np.float32) * 0.1
    V_t = torch.from_numpy(V_np)
    print(f"  V: shape={V_np.shape}, norm={float(np.linalg.norm(V_np)):.3f}")

    # ------------------------------------------------------------------
    # 3. Construct REAL BFV backend with hold_secret_key=True (we are M)
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 3: Constructing real BFV backend...")
    backend = BFVPrivSelectV2Backend(
        n_entries=N,
        vec_dim=HIDDEN_DIM,
        shared_seed=b"\x42" * 32,
        scale=SCALE,
        cache_dir=None,
        poly_degree=POLY_DEGREE,
        plain_bits=PLAIN_BITS,
        hold_secret_key=True,            # we are M — keep sk_M
    )
    print(f"  backend: sk_M held? {backend.secret_key is not None}, "
          f"decryptor? {backend.decryptor is not None}")

    # ------------------------------------------------------------------
    # 4. Build encrypted DB (offline, S-side)
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 4: Building encrypted DB (offline S-side)...")
    t0 = time.time()
    stats = backend.build_encrypted_database(V_np, force=True)
    print(f"  DB: {stats['n_rows']} rows in {time.time()-t0:.2f}s")

    # ------------------------------------------------------------------
    # 5. Generate fake logits and verify gradient recovery per token
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 5: Running Design 2 protocol per token...")
    max_err = 0.0
    rel_err_sum = 0.0

    for t in range(N_TOKENS):
        # Generate logits for this token; pick a random ground truth y_t
        logits = torch.randn(1, N, dtype=torch.float32)
        y_t = int(torch.randint(0, N, (1,)).item())
        true_label_onehot = torch.zeros(N, dtype=torch.float32)
        true_label_onehot[y_t] = 1.0

        # ----- S side -----
        # a_t = softmax(logits) @ V
        prob = torch.softmax(logits[0], dim=-1)            # (N,)
        a_t = (prob.unsqueeze(0) @ V_t).squeeze(0)         # (hidden_dim,)

        # Build S3PIR query via hint_table
        real_indices, dummy_indices, permutation = hints.build_query_for(y_t)
        assert len(real_indices) == 1 and real_indices[0] == y_t, \
            f"P0-1 fail at t={t}: real_indices={real_indices}, y_t={y_t}"
        assert permutation == 0, f"perm must be 0, got {permutation}"

        # S constructs BFVQuery (without a_t baked in: bake_a_t=False)
        query = BFVQuery(
            step=t, t_flat=t, y=y_t,
            real_indices=real_indices,
            dummy_indices=dummy_indices,
            permutation=0,
        )

        resp = backend._respond_s3pir(
            query=query,
            a_t_fp32=a_t.numpy().astype(np.float32),
            bake_a_t=False,
        )
        parity_real_bytes = resp[0]                          # Enc_M(-V_y)
        assert isinstance(parity_real_bytes, (bytes, bytearray))
        assert len(parity_real_bytes) > 0, "empty parity blob"

        # S produces plaintext s_share = a_t - r_t
        r_t_ints = backend.shares.generate_mask_ints(t, t)  # shared with U
        s_share = backend.shares.server_make_share(t, t, a_t.numpy().astype(np.float32))
        assert len(s_share) >= HIDDEN_DIM, f"s_share len={len(s_share)}"

        # ----- U side -----
        # U picks parity_real (perm=0), adds r_t homomorphically
        ct_U_bytes = backend._add_mask_to_ct(parity_real_bytes, r_t_ints)

        # ----- M side -----
        # 1. decrypt_only → -V_y + r_t
        masked_vec = backend.decrypt_only(ct_U_bytes)        # (poly_degree,) float
        assert masked_vec.size >= HIDDEN_DIM, \
            f"decrypted size {masked_vec.size} < hidden_dim {HIDDEN_DIM}"
        masked_vec = masked_vec[:HIDDEN_DIM]                 # truncate to hidden_dim

        # 2. plaintext-add s_share → a_t - V_y
        s_arr = np.array(s_share[:HIDDEN_DIM], dtype=np.float32) / SCALE
        g_recovered = masked_vec + s_arr                    # (HIDDEN_DIM,)

        # ----- Compare to ground truth -----
        g_true = a_t.numpy() - V_np[y_t]                     # (HIDDEN_DIM,)
        err = float(np.max(np.abs(g_recovered - g_true)))
        norm = float(np.linalg.norm(g_true))
        rel = err / max(norm, 1e-9)
        max_err = max(max_err, err)
        rel_err_sum += rel

        print(f"  t={t}: y_t={y_t:4d}, "
              f"||g_true||={norm:.3f}, "
              f"||g_recovered - g_true||_inf={err:.5f}, "
              f"rel={rel:.5f}")
        assert err < 0.5, f"t={t}: recovery error {err} too large (expected <0.5)"
        assert rel < 0.05, f"t={t}: relative error {rel:.5f} too large (expected <5%)"

    # ------------------------------------------------------------------
    # 6. Privacy boundary check
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 6: Privacy boundary (U side should NOT have sk_M)...")
    backend._drop_secret_key()
    assert backend.secret_key is None, "sk_M not dropped"
    assert backend.decryptor is None, "decryptor not dropped"

    raised = False
    try:
        backend.decrypt_only(ct_U_bytes)
    except RuntimeError as e:
        raised = True
        print(f"  decrypt_only correctly refused: {e}")
    assert raised, "decrypt_only should refuse without sk_M"

    elapsed = time.time() - t_start
    print("=" * 60)
    print(f"PASS: e2e math verified ({N_TOKENS} tokens, "
          f"max_abs_err={max_err:.5f}, mean_rel_err={rel_err_sum/N_TOKENS:.5f})")
    print(f"Total time: {elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())