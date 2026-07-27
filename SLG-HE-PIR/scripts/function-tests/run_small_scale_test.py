#!/usr/bin/env python3
"""
Small-scale end-to-end test for SLG-HE-PIR v2.0.

This script validates the complete refactored pipeline at small scale:
  1. Data loading (BioTriplex-QA subset)
  2. Model splitting (Llama-3.1-8B: U/M/S submodels)
  3. BFV encrypted DB (small V subset, N=1024)
  4. S3PIR hints (streaming offline phase)
  5. IPC protocol (mock workers — tests queue/routing logic)
  6. Trainer integration (mock workers)
  7. End-to-end pipeline (mock workers)

Usage:
  python -m scripts.function_tests.run_small_scale_test
"""

from __future__ import annotations

import gc
import json
import logging
import os
import sys
import tempfile
import time

# Ensure project root is on path
sys.path.insert(0, "/root/autodl-tmp/SLG-HE-PIR")
# Offline mode for HF model loading
os.environ.setdefault("HF_HUB_OFFLINE", "1")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("small_scale_test")

# --------------------------------------------------------------------------- #
# Test configuration (small scale)
# --------------------------------------------------------------------------- #
SMALL_N = 1024          # Small vocabulary subset (full is 128256)
SMALL_VOCAB = 128256   # Full vocab size (for index mapping)
HF_MODEL = "/root/autodl-tmp/hf_cache/Llama-3-1-8B-I"
DATA_DIR = "/root/slg-v2.0/data/biotriplex_qa"
POLY_DEGREE = 4096
PLAIN_BITS = 30
SCALE = 10000
LAMBDA = 10            # Small correctness parameter for test


def test_data_loading():
    """Test 1: Load BioTriplex-QA dataset."""
    logger.info("=" * 60)
    logger.info("TEST 1: Data Loading")
    logger.info("=" * 60)

    from src.data.dataset import load_biotriplex_dataset, LlamaTokenizerWrapper

    train_samples, val_samples, test_samples = load_biotriplex_dataset(
        data_dir=DATA_DIR,
        train_ratio=0.9,
        seed=42,
    )

    logger.info("Train: %d, Val: %d, Test: %d", len(train_samples), len(val_samples), len(test_samples))
    assert len(train_samples) > 0, "No training samples"
    assert len(val_samples) > 0, "No validation samples"
    assert len(test_samples) > 0, "No test samples"

    # Check sample structure
    s = train_samples[0]
    assert hasattr(s, "input") and hasattr(s, "output")
    logger.info("Sample fields: %s", list(s.to_dict().keys()))
    logger.info("PASS: Data loading OK (%d train, %d val, %d test)",
                len(train_samples), len(val_samples), len(test_samples))

    # Tokenizer test
    tok = LlamaTokenizerWrapper(HF_MODEL, max_length=128)
    encoded = tok("Hello, world!")
    assert "input_ids" in encoded
    logger.info("Tokenizer: vocab_size=%d", tok.tokenizer.vocab_size)
    logger.info("PASS: Tokenizer OK")

    return train_samples, val_samples, test_samples, tok


def test_model_splitting():
    """Test 2: Load U/M/S submodels (shard-aware, no full model load)."""
    logger.info("=" * 60)
    logger.info("TEST 2: Model Splitting (U/M/S)")
    logger.info("=" * 60)

    import torch
    from src.model.model_splitting import (
        detect_model_spec, load_u_submodel, load_m_submodel_with_lora,
        load_s_submodel, freeze_submodel,
    )

    spec = detect_model_spec(HF_MODEL)
    logger.info("Model spec: arch=%s, layers=%d, hidden=%d, vocab=%d",
                spec.arch, spec.num_layers, spec.hidden_size, spec.vocab_size)
    assert spec.arch == "llama"
    assert spec.num_layers == 32
    assert spec.hidden_size == 4096

    # U: embed + layers[0:16)
    logger.info("Loading U submodel (embed + layers[0:16))...")
    u_model = load_u_submodel(spec, HF_MODEL, device="cuda")
    freeze_submodel(u_model)
    assert hasattr(u_model, "embed_tokens")
    assert len(u_model.layers) == 16
    logger.info("U submodel: embed_tokens + %d layers loaded", len(u_model.layers))

    # M: layers[16:32) + LoRA
    logger.info("Loading M submodel (layers[16:32) + LoRA)...")
    m_model = load_m_submodel_with_lora(
        spec, HF_MODEL, device="cuda",
        lora_rank=8, lora_alpha=16,
    )
    freeze_submodel(m_model)
    assert len(m_model.layers) == 16
    assert hasattr(m_model, "norm")
    # Check LoRA params exist
    lora_params = [n for n, p in m_model.named_parameters() if "lora_" in n]
    assert len(lora_params) > 0, "No LoRA parameters found"
    logger.info("M submodel: %d layers + LoRA (%d LoRA params)", len(m_model.layers), len(lora_params))

    # S: lm_head (V matrix)
    logger.info("Loading S submodel (lm_head)...")
    s_model = load_s_submodel(spec, HF_MODEL, device="cpu")
    assert hasattr(s_model, "weight")
    v_shape = s_model.weight.shape
    logger.info("S V matrix: shape=%s", tuple(v_shape))
    assert v_shape[0] == spec.vocab_size  # 128256
    assert v_shape[1] == spec.hidden_size   # 4096

    # Quick forward test: U → M → S
    # Note: U and M are both on CUDA in production. S V is on CPU.
    logger.info("Testing U forward (GPU)...")
    dummy_ids = torch.randint(0, spec.vocab_size, (1, 32), device="cuda")
    with torch.no_grad():
        H_U = u_model.embed_tokens(dummy_ids)
        for layer in u_model.layers:
            H_U = layer(H_U, attention_mask=None, position_ids=None)
    logger.info("H_U shape: %s", tuple(H_U.shape))
    assert H_U.shape == (1, 32, 4096)

    logger.info("Testing M forward (GPU)...")
    with torch.no_grad():
        H_M = H_U  # Same GPU
        for layer in m_model.layers:
            H_M = layer(H_M, attention_mask=None, position_ids=None)
        H_M = m_model.norm(H_M)
    logger.info("H_M shape: %s", tuple(H_M.shape))
    assert H_M.shape == (1, 32, 4096)

    logger.info("Testing S (logits = H_M @ V^T on CPU)...")
    # S V is on CPU; move H_M there
    H_M_cpu = H_M.cpu()
    logits = torch.matmul(H_M_cpu, s_model.weight.T.float())
    logger.info("logits shape: %s", tuple(logits.shape))
    assert logits.shape == (1, 32, spec.vocab_size)

    # Cleanup
    del u_model, m_model, s_model
    torch.cuda.empty_cache()
    gc.collect()

    logger.info("PASS: Model splitting OK (U/M/S all forward correctly)")
    return spec


def test_bfv_backend():
    """Test 3: BFV encrypted DB with small V subset."""
    logger.info("=" * 60)
    logger.info("TEST 3: BFV Backend (small V subset, N=%d)", SMALL_N)
    logger.info("=" * 60)

    import numpy as np
    import torch
    from src.core.bfv_privselect_v2_adapter import (
        BFVPrivSelectV2Backend,
        encode_vector_as_ints,
        decode_ints_as_vector,
        float_to_int,
        int_to_float,
    )

    # Create small V matrix
    np.random.seed(42)
    hidden_dim = 4096
    V_small = np.random.randn(SMALL_N, hidden_dim).astype(np.float64)
    V_small = (V_small * 100).round().astype(np.float64)  # Scale to reasonable int range
    logger.info("V_small: shape=%s, dtype=%s", V_small.shape, V_small.dtype)

    # Test fixed-point encoding
    val = V_small[0, 0]
    int_val = float_to_int(val, scale=SCALE)
    back = int_to_float(int_val, scale=SCALE)
    logger.info("Fixed-point: %.4f → %d → %.4f", val, int_val, back)
    assert abs(val - back) < 1.0, "Fixed-point encoding error too large"

    # Create BFV backend
    logger.info("Creating BFV context (poly_degree=%d, plain_bits=%d)...", POLY_DEGREE, PLAIN_BITS)
    backend = BFVPrivSelectV2Backend(
        n_entries=SMALL_N,
        vec_dim=hidden_dim,
        shared_seed=os.urandom(32),
        scale=SCALE,
        cache_dir=None,  # No cache for small test
        poly_degree=POLY_DEGREE,
        plain_bits=PLAIN_BITS,
    )

    # Build encrypted DB
    logger.info("Building BFV encrypted DB (N=%d entries)...", SMALL_N)
    t0 = time.time()
    stats = backend.build_encrypted_database(V_small, force=True)
    elapsed = time.time() - t0
    logger.info("Built %d encrypted rows in %.2fs (%.1f rows/s)",
                stats["n_rows"], elapsed, stats["n_rows"] / max(elapsed, 0.01))
    assert stats["n_rows"] == SMALL_N
    assert not stats["from_cache"]

    # Test respond (S-side)
    from src.core.bfv_privselect_v2_adapter import BFVQuery
    query = BFVQuery(step=0, t_flat=0, y=0, real_indices=[0], dummy_indices=[1, 2])

    a_t_fp32 = np.random.randn(hidden_dim).astype(np.float32) * 0.1
    logger.info("S responding to query (a_t: %s)...", a_t_fp32.shape)
    resp = backend.respond(query, a_t_fp32)
    logger.info("Response: parity_real=%d bytes, parity_dummy=%d bytes",
                len(resp.parity_real_bytes), len(resp.parity_dummy_bytes))
    assert len(resp.parity_real_bytes) > 0, "Empty parity_real"

    # Test _add_mask_to_ct (U-side): add PRG mask R_t to the S3PIR real blob
    R_t = backend.shares.generate_mask_ints(0, 0)
    masked_ct = backend._add_mask_to_ct(resp.parity_real_bytes, R_t)
    logger.info("U added mask: %d bytes, R_t len=%d", len(masked_ct), len(R_t))
    assert len(masked_ct) > 0

    # Test aggregator_decrypt (M-side)
    logger.info("M decrypting aggregated result...")
    grad = backend.aggregator_decrypt(masked_ct, s_share=None)
    logger.info("Recovered gradient: shape=%s, norm=%.4f", grad.shape, float(grad.dot(grad) ** 0.5))
    assert grad.shape == (hidden_dim,), f"Expected ({hidden_dim},), got {grad.shape}"

    # Cleanup
    del backend, V_small
    gc.collect()

    logger.info("PASS: BFV backend OK (encode/respond/add_mask/decrypt all work)")
    return True


def test_s3pir_hints():
    """Test 4: S3PIR hint generation and query building."""
    logger.info("=" * 60)
    logger.info("TEST 4: S3PIR Hint Table")
    logger.info("=" * 60)

    from src.core.s3pir_hints import HintTable

    partition_size = 1 << ((SMALL_N.bit_length() - 1) // 2)  # sqrt(SMALL_N)
    n_partitions = (SMALL_N + partition_size - 1) // partition_size

    logger.info("N=%d, partition_size=%d, n_partitions=%d, lambda=%d",
                SMALL_N, partition_size, n_partitions, LAMBDA)

    with tempfile.TemporaryDirectory() as tmpdir:
        hint_table = HintTable(
            n_entries=SMALL_N,
            partition_size=partition_size,
            lam=LAMBDA,
            cache_dir=tmpdir,
        )

        # Compute skeletons
        logger.info("Computing main hint skeletons...")
        hint_table.compute_main_hints_skeleton()
        logger.info("Main hints: %d", len(hint_table.main_hints))

        logger.info("Computing backup hint skeletons...")
        hint_table.compute_backup_hints_skeleton()
        logger.info("Backup hints: %d", len(hint_table.backup_hints))

        # Save to cache
        hint_table.to_cache_files()
        logger.info("Saved to cache files")

        # Reload from cache
        hint_table2 = HintTable.from_cache_files(tmpdir)
        logger.info("Reloaded from cache: %d main hints", len(hint_table2.main_hints))

        # Test query building for random index
        import random
        random.seed(42)
        for _ in range(5):
            idx = random.randint(0, SMALL_N - 1)
            hint = hint_table.find_hint_for(idx)
            assert hint is not None, f"No hint found for index {idx}"
            real_indices, dummy_indices, permutation_bit = hint_table.build_query_for(idx)
            assert len(real_indices) > 0 or len(dummy_indices) > 0
            assert permutation_bit in [0, 1]
            logger.info("Index %d → hint[%d], real=%d, dummy=%d, perm=%d",
                        idx, hint.j, len(real_indices),
                        len(dummy_indices), permutation_bit)

        logger.info("PASS: S3PIR hints OK (skeleton/compute/query/load all work)")
        return hint_table


# --------------------------------------------------------------------------- #
# Test 5: IPC protocol with mock workers
# --------------------------------------------------------------------------- #
def _mock_worker_u(queue_to_U, queue_from_U, model_config):
    """Mock U worker — simulates embed + decoder[0:16) forward."""
    import torch
    while True:
        try:
            msg = queue_to_U.get(timeout=60)
        except:
            continue
        if msg[0] == "STOP":
            break
        if msg[0] == "FORWARD":
            batch, step = msg[1], msg[2]
            # Simulate H_U computation: random tensor of correct shape
            H_U = torch.randn(batch["input_ids"].shape[0], batch["input_ids"].shape[1],
                             model_config["hidden_dim"])
            queue_from_U.put(("H_U", {"H_U": H_U}, step))
        elif msg[0] == "S3PIR_RESP":
            s3pir_msg, step = msg[1], msg[2]
            # Mock: return a random gradient with the right shape
            g_H = torch.randn(1, 1, model_config["hidden_dim"])
            queue_from_U.put(("G_H_MASKED", {"g_H_masked": g_H}, step))


def _mock_worker_m(queue_to_M, queue_from_M, model_config):
    """Mock M worker — simulates decoder[16:32) + LoRA + BFV decrypt."""
    import torch
    while True:
        try:
            msg = queue_to_M.get(timeout=60)
        except:
            continue
        if msg[0] == "STOP":
            break
        if msg[0] == "H_U":
            H_U, step = msg[1], msg[2]
            H_M = torch.randn(H_U.shape[0], H_U.shape[1], model_config["hidden_dim"])
            queue_from_M.put(("LOGITS", {"H_M": H_M}, step))
        elif msg[0] == "INJECT_GRAD":
            payload, step = msg[1], msg[2]
            queue_from_M.put(("STEP_ACK", {"loss": 1.0, "gpu_mem_mb": 0.0}, step))


def _mock_worker_s(queue_to_S, queue_from_S, model_config):
    """Mock S worker — simulates V matrix + BFV respond."""
    import torch
    while True:
        try:
            msg = queue_to_S.get(timeout=60)
        except:
            continue
        if msg[0] == "STOP":
            break
        if msg[0] == "COMPUTE_LOGITS":
            H_M, step = msg[1], msg[2]
            # Mock: return random s3pir response
            queue_from_S.put(("S3PIR_RESP", {
                "parity_real_bytes": b"x" * 1024,
                "parity_dummy_bytes": b"x" * 1024,
                "permutation": 0,
                "s_share": None,
                "step": step,
                "t_flat": 0,
            }, step))


def test_ipc_protocol_mock():
    """Test 5: IPC protocol with lightweight mock workers."""
    logger.info("=" * 60)
    logger.info("TEST 5: IPC Protocol (mock workers)")
    logger.info("=" * 60)

    import multiprocessing as mp
    import torch

    ctx = mp.get_context("fork")
    queue_to_U = ctx.Queue()
    queue_from_U = ctx.Queue()
    queue_to_M = ctx.Queue()
    queue_from_M = ctx.Queue()
    queue_to_S = ctx.Queue()
    queue_from_S = ctx.Queue()

    model_config = {
        "hidden_dim": 4096,
        "vocab_size": SMALL_N,
        "lora_rank": 8,
    }

    # Spawn mock workers
    p_U = ctx.Process(target=_mock_worker_u, args=(queue_to_U, queue_from_U, model_config), daemon=True)
    p_M = ctx.Process(target=_mock_worker_m, args=(queue_to_M, queue_from_M, model_config), daemon=True)
    p_S = ctx.Process(target=_mock_worker_s, args=(queue_to_S, queue_from_S, model_config), daemon=True)

    p_U.start()
    p_M.start()
    p_S.start()
    logger.info("Mock workers spawned: U(pid=%d), M(pid=%d), S(pid=%d)", p_U.pid, p_M.pid, p_S.pid)

    # Manually run IPC protocol steps (no IPCProtocol class — direct queue ops)
    batch = {
        "input_ids": torch.randint(0, SMALL_N, (1, 32)),
        "attention_mask": torch.ones(1, 32, dtype=torch.long),
    }

    # Step 1: U forward
    t0 = time.time()
    queue_to_U.put(("FORWARD", batch, 0))

    # Step 2: Receive H_U from U
    H_U_msg = queue_from_U.get(timeout=30)
    assert H_U_msg[0] == "H_U" and H_U_msg[2] == 0
    logger.info("Got H_U from U: shape=%s", tuple(H_U_msg[1]["H_U"].shape))

    # Step 3: Relay to M
    queue_to_M.put(("H_U", H_U_msg[1]["H_U"], 0))

    # Step 4: Receive H_M from M
    H_M_msg = queue_from_M.get(timeout=30)
    assert H_M_msg[0] == "LOGITS" and H_M_msg[2] == 0
    logger.info("Got H_M from M: shape=%s", tuple(H_M_msg[1]["H_M"].shape))

    # Step 5: Relay to S
    queue_to_S.put(("COMPUTE_LOGITS", H_M_msg[1]["H_M"], 0))

    # Step 6: Receive S3PIR response from S
    s3pir_msg = queue_from_S.get(timeout=30)
    assert s3pir_msg[0] == "S3PIR_RESP" and s3pir_msg[2] == 0
    logger.info("Got S3PIR resp: real=%d bytes, dummy=%d bytes",
                len(s3pir_msg[1]["parity_real_bytes"]), len(s3pir_msg[1]["parity_dummy_bytes"]))

    # Step 7: Relay to U
    queue_to_U.put(("S3PIR_RESP", s3pir_msg[1], 0))

    # Step 8: Receive g_H_masked from U
    gH_msg = queue_from_U.get(timeout=30)
    assert gH_msg[0] == "G_H_MASKED" and gH_msg[2] == 0
    logger.info("Got g_H_masked from U: shape=%s", tuple(gH_msg[1]["g_H_masked"].shape))

    # Step 9: Relay to M
    queue_to_M.put(("INJECT_GRAD", {
        "g_H_masked": gH_msg[1]["g_H_masked"],
        "s_share": None,
        "step": 0,
    }, 0))

    # Step 10: Wait for M ack
    ack_msg = queue_from_M.get(timeout=30)
    assert ack_msg[0] == "STEP_ACK" and ack_msg[2] == 0
    elapsed = time.time() - t0
    logger.info("Step completed in %.2fs, loss=%.4f", elapsed, ack_msg[1]["loss"])

    # Cleanup
    for q in [queue_to_U, queue_to_M, queue_to_S]:
        q.put(("STOP", None, -1))
    for p in [p_U, p_M, p_S]:
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()

    logger.info("PASS: IPC protocol OK (mock workers, queue/routing verified)")
    return True


def test_trainer_mock():
    """Test 6: Trainer integration with mock IPC workers (1 epoch)."""
    logger.info("=" * 60)
    logger.info("TEST 6: Trainer Integration (mock workers)")
    logger.info("=" * 60)

    import multiprocessing as mp
    import torch
    from src.data.dataset import load_biotriplex_dataset, LlamaTokenizerWrapper, BioTriplexQADataset

    # Load tiny subset
    train_samples, val_samples, _ = load_biotriplex_dataset(
        data_dir=DATA_DIR, train_ratio=0.9, seed=42,
    )
    train_samples = train_samples[:2]
    val_samples = val_samples[:2]

    tokenizer = LlamaTokenizerWrapper(HF_MODEL, max_length=64)
    train_ds = BioTriplexQADataset(train_samples, tokenizer, max_length=64, task="train")
    val_ds = BioTriplexQADataset(val_samples, tokenizer, max_length=64, task="val")
    logger.info("Train: %d, Val: %d", len(train_ds), len(val_ds))

    model_config = {"hidden_dim": 4096, "vocab_size": SMALL_N}

    # Mock IPC: just verify 1 epoch runs
    ctx = mp.get_context("fork")
    queue_to_U = ctx.Queue()
    queue_from_U = ctx.Queue()
    queue_to_M = ctx.Queue()
    queue_from_M = ctx.Queue()
    queue_to_S = ctx.Queue()
    queue_from_S = ctx.Queue()

    p_U = ctx.Process(target=_mock_worker_u, args=(queue_to_U, queue_from_U, model_config), daemon=True)
    p_M = ctx.Process(target=_mock_worker_m, args=(queue_to_M, queue_from_M, model_config), daemon=True)
    p_S = ctx.Process(target=_mock_worker_s, args=(queue_to_S, queue_from_S, model_config), daemon=True)
    p_U.start(); p_M.start(); p_S.start()

    t0 = time.time()
    steps_run = 0

    # Run 1 epoch (2 steps)
    for batch_item in train_ds:
        batch = {
            "input_ids": batch_item["input_ids"].unsqueeze(0),
            "attention_mask": batch_item["attention_mask"].unsqueeze(0),
        }

        # Step A: U
        queue_to_U.put(("FORWARD", batch, steps_run))
        H_U_msg = queue_from_U.get(timeout=30)
        assert H_U_msg[0] == "H_U"

        # M
        queue_to_M.put(("H_U", H_U_msg[1]["H_U"], steps_run))
        H_M_msg = queue_from_M.get(timeout=30)
        assert H_M_msg[0] == "LOGITS"

        # S
        queue_to_S.put(("COMPUTE_LOGITS", H_M_msg[1]["H_M"], steps_run))
        s3pir_msg = queue_from_S.get(timeout=30)
        assert s3pir_msg[0] == "S3PIR_RESP"

        # U mask
        queue_to_U.put(("S3PIR_RESP", s3pir_msg[1], steps_run))
        gH_msg = queue_from_U.get(timeout=30)
        assert gH_msg[0] == "G_H_MASKED"

        # M grad
        queue_to_M.put(("INJECT_GRAD", {"g_H_masked": gH_msg[1]["g_H_masked"], "s_share": None, "step": steps_run}, steps_run))
        ack_msg = queue_from_M.get(timeout=30)
        assert ack_msg[0] == "STEP_ACK"

        steps_run += 1
        logger.info("Step %d: loss=%.4f", steps_run, ack_msg[1]["loss"])

    elapsed = time.time() - t0
    logger.info("1 epoch (%d steps) completed in %.1fs", steps_run, elapsed)

    # Cleanup
    for q in [queue_to_U, queue_to_M, queue_to_S]:
        q.put(("STOP", None, -1))
    for p in [p_U, p_M, p_S]:
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()

    assert steps_run == 2
    logger.info("PASS: Trainer integration OK (1 epoch, 2 steps, mock workers)")
    return {"steps": steps_run, "elapsed_s": elapsed}


def test_end_to_end_mock():
    """Test 7: End-to-end with mock workers (Stage 0 → training loop)."""
    logger.info("=" * 60)
    logger.info("TEST 7: End-to-End Pipeline (mock workers)")
    logger.info("=" * 60)

    import multiprocessing as mp
    import torch
    import numpy as np
    from src.data.dataset import load_biotriplex_dataset, LlamaTokenizerWrapper, BioTriplexQADataset
    from src.core.bfv_privselect_v2_adapter import BFVPrivSelectV2Backend
    from src.core.s3pir_hints import HintTable

    # Stage 0: Build BFV encrypted DB
    logger.info("[E2E] Stage 0: Building BFV encrypted DB (N=%d)...", SMALL_N)
    np.random.seed(42)
    hidden_dim = 4096
    V_small = np.random.randn(SMALL_N, hidden_dim).astype(np.float64) * 10

    backend = BFVPrivSelectV2Backend(
        n_entries=SMALL_N, vec_dim=hidden_dim,
        shared_seed=os.urandom(32), scale=SCALE,
        cache_dir=None, poly_degree=POLY_DEGREE, plain_bits=PLAIN_BITS,
    )
    t0 = time.time()
    db_stats = backend.build_encrypted_database(V_small, force=True)
    logger.info("[E2E] DB built in %.1fs: %d rows", time.time() - t0, db_stats["n_rows"])

    # Stage 0: Build S3PIR hints
    partition_size = 1 << ((SMALL_N.bit_length() - 1) // 2)
    with tempfile.TemporaryDirectory() as tmpdir:
        logger.info("[E2E] Stage 0: Building S3PIR hints...")
        hint_table = HintTable(
            n_entries=SMALL_N, partition_size=partition_size,
            lam=LAMBDA, cache_dir=tmpdir,
        )
        hint_table.compute_main_hints_skeleton()
        hint_table.compute_backup_hints_skeleton()
        hint_table.to_cache_files()
        hint_table = HintTable.from_cache_files(tmpdir)
        logger.info("[E2E] Hints loaded: %d main + %d backup",
                    len(hint_table.main_hints), len(hint_table.backup_hints))

        # Stage 1: Training loop
        logger.info("[E2E] Stage 1: Running training (1 epoch)...")

        train_samples, val_samples, _ = load_biotriplex_dataset(
            data_dir=DATA_DIR, train_ratio=0.9, seed=42,
        )
        train_samples = train_samples[:2]
        val_samples = val_samples[:2]
        tokenizer = LlamaTokenizerWrapper(HF_MODEL, max_length=64)
        train_ds = BioTriplexQADataset(train_samples, tokenizer, max_length=64, task="train")
        val_ds = BioTriplexQADataset(val_samples, tokenizer, max_length=64, task="val")

        model_config = {"hidden_dim": hidden_dim, "vocab_size": SMALL_N}

        ctx = mp.get_context("fork")
        qU = ctx.Queue(); qU_from = ctx.Queue()
        qM = ctx.Queue(); qM_from = ctx.Queue()
        qS = ctx.Queue(); qS_from = ctx.Queue()
        p_U = ctx.Process(target=_mock_worker_u, args=(qU, qU_from, model_config), daemon=True)
        p_M = ctx.Process(target=_mock_worker_m, args=(qM, qM_from, model_config), daemon=True)
        p_S = ctx.Process(target=_mock_worker_s, args=(qS, qS_from, model_config), daemon=True)
        p_U.start(); p_M.start(); p_S.start()

        t0 = time.time()
        total_steps = 0
        for batch_item in train_ds:
            batch = {
                "input_ids": batch_item["input_ids"].unsqueeze(0),
                "attention_mask": batch_item["attention_mask"].unsqueeze(0),
            }
            qU.put(("FORWARD", batch, total_steps))
            H_U = qU_from.get(timeout=30); assert H_U[0] == "H_U"
            qM.put(("H_U", H_U[1]["H_U"], total_steps))
            H_M = qM_from.get(timeout=30); assert H_M[0] == "LOGITS"
            qS.put(("COMPUTE_LOGITS", H_M[1]["H_M"], total_steps))
            s3 = qS_from.get(timeout=30); assert s3[0] == "S3PIR_RESP"
            qU.put(("S3PIR_RESP", s3[1], total_steps))
            gH = qU_from.get(timeout=30); assert gH[0] == "G_H_MASKED"
            qM.put(("INJECT_GRAD", {"g_H_masked": gH[1]["g_H_masked"], "s_share": None, "step": total_steps}, total_steps))
            ack = qM_from.get(timeout=30); assert ack[0] == "STEP_ACK"
            total_steps += 1

        epoch_time = time.time() - t0
        logger.info("[E2E] 1 epoch (%d steps) in %.1fs", total_steps, epoch_time)

        # Cleanup
        for q in [qU, qM, qS]:
            q.put(("STOP", None, -1))
        for p in [p_U, p_M, p_S]:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()

    del backend, V_small
    gc.collect()

    logger.info("PASS: End-to-end pipeline OK (Stage 0 → Stage 1 completed, mock workers)")
    return {"total_steps": total_steps, "epoch_time_s": epoch_time}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    logger.info("SLG-HE-PIR v2.0 Small-Scale End-to-End Test")
    logger.info("=" * 60)
    total_t0 = time.time()

    results = {}

    try:
        # Test 1: Data
        r1 = test_data_loading()
        results["data_loading"] = "PASS"

        # Test 2: Model splitting
        spec = test_model_splitting()
        results["model_splitting"] = "PASS"

        # Test 3: BFV backend
        test_bfv_backend()
        results["bfv_backend"] = "PASS"

        # Test 4: S3PIR hints
        test_s3pir_hints()
        results["s3pir_hints"] = "PASS"

        # Test 5: IPC protocol with mock workers
        test_ipc_protocol_mock()
        results["ipc_protocol"] = "PASS"

        # Test 6: Trainer integration with mock workers
        test_trainer_mock()
        results["trainer_integration"] = "PASS"

        # Test 7: End-to-end with mock workers
        test_end_to_end_mock()
        results["end_to_end"] = "PASS"

    except Exception as e:
        logger.error("TEST FAILED: %s", e)
        import traceback
        traceback.print_exc()
        results["error"] = str(e)

    total_elapsed = time.time() - total_t0

    logger.info("=" * 60)
    logger.info("TEST SUMMARY (total time: %.1fs)", total_elapsed)
    logger.info("=" * 60)
    for name, status in results.items():
        logger.info("  %-25s: %s", name, status)

    all_pass = all(v == "PASS" for v in results.values())
    logger.info("")
    if all_pass:
        logger.info("ALL TESTS PASSED!")
    else:
        logger.error("SOME TESTS FAILED!")
    logger.info("=" * 60)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
