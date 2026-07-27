#!/usr/bin/env python3
"""
Quick 10-step smoke test for SLG-HE-PIR.

Reuses two_epoch_test.py initialization but only runs the first 10 training
steps of epoch 0 and then reports peak GPU memory, so we can verify the
bf16 + reentrant-checkpoint + GPU-V-matrix optimizations don't OOM
without waiting ~47 minutes for the full 2-epoch run.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("quick_smoke")


def main() -> int:
    import torch

    # ---- Run full initialization (same as two_epoch_test.py) ----
    from scripts.function_tests.two_epoch_test import TwoEpochConfig, run_two_epoch_test

    cfg = TwoEpochConfig()
    cfg.max_epochs = 1  # only need 1 epoch, we'll stop early

    # Manually re-implement initialization but stop before training
    from src.data.dataset import load_biotriplex_dataset, LlamaTokenizerWrapper, BioTriplexQADataset
    from src.core.bfv_privselect_v2_adapter import BFVPrivSelectV2Backend
    from src.core.s3pir_hints import HintTable
    from src.parties.heterogeneous_protocol import HeterogeneousProtocol
    from src.training.trainer import Trainer, TrainerConfig
    import pickle as _pickle
    from src.core.bfv_privselect_v2_adapter import _seal_to_bytes
    from safetensors.torch import load_file
    import json
    import numpy as np

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
        torch.cuda.reset_peak_memory_stats()

    logger.info("=" * 60)
    logger.info("10-Step Smoke Test (bf16 + GPU-V)")
    logger.info("=" * 60)
    logger.info("Config: batch_size=%d, max_seq_length=%d", cfg.batch_size, cfg.max_seq_length)

    if torch.cuda.is_available():
        logger.info("GPU: %s", torch.cuda.get_device_name(0))
        logger.info("GPU Memory: %.1f GB total, %.1f GB available",
                    torch.cuda.get_device_properties(0).total_memory / 1e9,
                    torch.cuda.mem_get_info()[0] / 1e9)

    logger.info("[Step 1] Loading datasets...")
    train_samples, val_samples, test_samples = load_biotriplex_dataset(
        data_dir=cfg.data_dir,
        train_ratio=cfg.train_ratio,
        seed=cfg.seed,
    )
    tokenizer = LlamaTokenizerWrapper(cfg.hf_model, max_length=cfg.max_seq_length)
    train_ds = BioTriplexQADataset(train_samples, tokenizer, max_length=cfg.max_seq_length, task="train")
    val_ds = BioTriplexQADataset(val_samples, tokenizer, max_length=cfg.max_seq_length, task="val")
    test_ds = BioTriplexQADataset(test_samples, tokenizer, max_length=cfg.max_seq_length, task="test")
    logger.info("Datasets: train=%d, val=%d, test=%d", len(train_ds), len(val_ds), len(test_ds))

    logger.info("[Step 2] Initializing BFV backend...")
    bfv_backend = BFVPrivSelectV2Backend(
        n_entries=cfg.vocab_size,
        vec_dim=cfg.hidden_dim,
        shared_seed=os.urandom(32),
        cache_dir=cfg.bfv_cache_dir,
        poly_degree=cfg.poly_degree,
        plain_bits=cfg.plain_bits,
        scale=float(cfg.scale),
    )

    logger.info("[Step 3] Building encrypted DB (from cache if present)...")
    snap_path = cfg.hf_model
    index_path = Path(snap_path) / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        wm = index["weight_map"]
        lm_head_files = sorted({
            str(Path(snap_path) / fn) for k, fn in wm.items() if "lm_head" in k
        })
    else:
        lm_head_files = sorted(Path(snap_path).glob("*.safetensors"))

    V = None
    for sf in lm_head_files:
        sd = load_file(str(sf), device="cpu")
        for k, v in sd.items():
            if "lm_head" in k and "weight" in k:
                v_fp = v.float().numpy() if v.dtype != torch.float32 else v.numpy()
                V = v_fp.astype(np.float64) if V is None else np.concatenate(
                    [V, v_fp.astype(np.float64)], axis=0
                )
        del sd

    db_result = bfv_backend.build_encrypted_database(V, force=True)
    logger.info("Encrypted DB: %d rows, from_cache=%s, time=%.1fs",
                db_result["n_rows"], db_result["from_cache"], db_result.get("build_time_s", 0))

    logger.info("[Step 4] Extracting keys...")
    sk_pem = _seal_to_bytes(bfv_backend._secret_key)
    pk_pem = _pickle.dumps({"pk_bytes": bfv_backend.public_key_bytes})
    bfv_backend._drop_secret_key()
    prg_seed = os.urandom(32)

    logger.info("[Step 5] Loading hint table...")
    hints_dir = os.path.join(cfg.bfv_cache_dir, "s3pir_hints")
    partition_size = 1 << ((cfg.vocab_size.bit_length() - 1) // 2)
    if os.path.exists(os.path.join(hints_dir, "hint_table.json")):
        hint_table = HintTable.from_cache_files(hints_dir)
        logger.info("Hint table loaded from cache")
    else:
        hint_table = HintTable(
            n_entries=cfg.vocab_size,
            partition_size=partition_size,
            lam=cfg.lam,
            cache_dir=hints_dir,
        )
        hint_table.compute_main_hints_skeleton()
        hint_table.compute_backup_hints_skeleton()
        hint_table.to_cache_files()
        logger.info("Hint table computed and saved")

    logger.info("[Step 6] Building HeterogeneousProtocol...")
    worker_config = {
        "vocab_size": cfg.vocab_size,
        "hidden_dim": cfg.hidden_dim,
        "poly_degree": cfg.poly_degree,
        "plain_bits": cfg.plain_bits,
        "scale": cfg.scale,
        "bfv_cache_dir": cfg.bfv_cache_dir,
        "lam": cfg.lam,
        "lora_rank": cfg.lora_rank,
        "lora_alpha": cfg.lora_alpha,
        "learning_rate": cfg.learning_rate,
        "weight_decay": cfg.weight_decay,
        "warmup_steps": cfg.warmup_steps,
        "batch_size": cfg.batch_size,
        "max_epochs": cfg.max_epochs,
        "n_train_samples": len(train_ds),
        "hf_model_path": cfg.hf_model,
        "u_layers": cfg.u_layers,
        "m_layers": cfg.m_layers,
        "N_CRYPTO_U_WORKERS": cfg.N_CRYPTO_U_WORKERS,
        "N_CRYPTO_M_WORKERS": cfg.N_CRYPTO_M_WORKERS,
        "N_CRYPTO_S_WORKERS": cfg.N_CRYPTO_S_WORKERS,
        "USE_CHUNKED_PIPELINE": cfg.USE_CHUNKED_PIPELINE,
        "CHUNK_TOKENS": cfg.CHUNK_TOKENS,
    }

    protocol = HeterogeneousProtocol(
        u_submodel_path=cfg.hf_model,
        m_submodel_path=cfg.hf_model,
        s_lm_head_path=cfg.hf_model,
        bfv_backend=bfv_backend,
        hint_table=hint_table,
        bfv_sk_pem=sk_pem,
        bfv_pk_pem=pk_pem,
        prg_seed=prg_seed,
        config=worker_config,
    )
    logger.info("Protocol initialized")

    # ---- Verify V matrix is on GPU ----
    party_s = protocol.party_s if hasattr(protocol, "party_s") else None
    if party_s is not None:
        try:
            v_dev = party_s.V.weight.device
            v_dtype = party_s.V.weight.dtype
            logger.info("✓ PartyS.V matrix device=%s, dtype=%s, shape=%s",
                        v_dev, v_dtype, tuple(party_s.V.weight.shape))
        except Exception as e:
            logger.warning("Could not inspect PartyS.V: %s", e)

    # ---- Setup trainer with custom stop hook ----
    logger.info("[Step 7] Running 10 training steps...")
    trainer_cfg = TrainerConfig(
        max_epochs=1,
        patience=999,
        batch_size=cfg.batch_size,
        max_seq_length=cfg.max_seq_length,
        USE_CHUNKED_PIPELINE=cfg.USE_CHUNKED_PIPELINE,
        CHUNK_TOKENS=cfg.CHUNK_TOKENS,
        save_freq=999,  # don't save
        log_freq=1,     # log every step
    )

    trainer = Trainer(
        config=trainer_cfg,
        ipc_protocol=protocol,
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        tokenizer=tokenizer,
    )

    # Monkey-patch _run_epoch to stop after 10 steps
    MAX_STEPS = 10
    orig_run_epoch = trainer._run_epoch

    def patched_run_epoch(epoch: int):
        from torch.utils.data import DataLoader
        train_loader = DataLoader(
            trainer.train_ds,
            batch_size=trainer.config.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )
        total_steps = 0
        epoch_loss = 0.0
        gpu_mem_samples = []
        step_times = []

        t_epoch = time.time()
        for batch in train_loader:
            if bool(trainer.config.USE_CHUNKED_PIPELINE):
                chunk_tokens = int(trainer.config.CHUNK_TOKENS)
                result = trainer.ipc.step_train_chunked(
                    batch, trainer.global_step,
                    chunk_tokens=chunk_tokens,
                )
            else:
                result = trainer.ipc.step_train(batch, trainer.global_step)
            epoch_loss += result.loss
            gpu_mem_samples.append(result.gpu_mem_mb)
            step_times.append(result.step_time_ms)
            trainer.global_step += 1
            total_steps += 1

            # Per-step GPU memory snapshot
            if torch.cuda.is_available():
                peak_now = torch.cuda.max_memory_allocated() / 1024**2
                cur_now = torch.cuda.memory_allocated() / 1024**2
                logger.info(
                    "Step %2d/%d: loss=%.4f, time=%.1fms, "
                    "step_reported_mem=%.0fMB, peak_mem=%.0fMB, current_mem=%.0fMB",
                    total_steps, MAX_STEPS, result.loss, result.step_time_ms,
                    result.gpu_mem_mb, peak_now, cur_now,
                )

            if total_steps >= MAX_STEPS:
                logger.info("Reached MAX_STEPS=%d, stopping early", MAX_STEPS)
                break

        elapsed = time.time() - t_epoch
        peak_mem_mb = max(gpu_mem_samples) if gpu_mem_samples else 0
        avg_mem_mb = sum(gpu_mem_samples) / max(len(gpu_mem_samples), 1)
        avg_step_time = sum(step_times) / max(len(step_times), 1)

        logger.info("=" * 60)
        logger.info("10-Step Smoke Test Result")
        logger.info("=" * 60)
        logger.info("Total steps run: %d", total_steps)
        logger.info("Elapsed: %.1fs (avg %.1fs/step)", elapsed, avg_step_time / 1000)
        logger.info("Peak GPU memory (per-step reported): %.0f MB", peak_mem_mb)
        if torch.cuda.is_available():
            logger.info("Peak GPU memory (cuda.max_memory_allocated): %.0f MB",
                        torch.cuda.max_memory_allocated() / 1024**2)
            logger.info("Total GPU memory: %.0f MB",
                        torch.cuda.get_device_properties(0).total_memory / 1024**2)
            free, total = torch.cuda.mem_get_info()
            logger.info("GPU mem after test: %.0f MB used, %.0f MB free",
                        (total - free) / 1024**2, free / 1024**2)

        return {
            "train_loss": epoch_loss / max(total_steps, 1),
            "train_steps": total_steps,
            "avg_step_time_ms": avg_step_time,
            "avg_gpu_mem_mb": avg_mem_mb,
            "peak_gpu_mem_mb": peak_mem_mb,
            "elapsed_s": elapsed,
        }

    trainer._run_epoch = patched_run_epoch

    try:
        result = patched_run_epoch(epoch=0)
        logger.info("Smoke test PASSED: loss=%.4f, peak_mem=%.0f MB",
                    result["train_loss"], result["peak_gpu_mem_mb"])
    except Exception as e:
        logger.error("Smoke test FAILED: %s", e, exc_info=True)
        protocol.shutdown()
        return 1

    protocol.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())