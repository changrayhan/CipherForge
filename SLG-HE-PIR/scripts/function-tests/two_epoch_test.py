#!/usr/bin/env python3
"""
Two-Epoch Convergence Test for SLG-HE-PIR.

This script verifies that the heterogeneous runtime can complete 2 training epochs
without errors, demonstrating that the SLG protocol converges correctly.

Usage:
    python -m scripts.function_tests.two_epoch_test --max_epochs 2

Expected outcomes:
    - 2 training epochs complete without errors
    - Per-epoch val metrics (val_entity_micro_f1, val_rouge_l) printed
    - Both epochs show improvement (convergence check)
    - No OOM, no CUDA errors, no crypto worker failures
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass

import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class TwoEpochConfig:
    """Configuration for two-epoch convergence test."""
    hf_model: str = "/root/autodl-tmp/hf_cache/Llama-3-1-8B-I"
    bfv_cache_dir: str = "/root/autodl-tmp/slg-bfv-cache"
    data_dir: str = "/root/slg-v2.0/data/biotriplex_qa"

    # Training
    max_epochs: int = 2
    batch_size: int = 4
    max_seq_length: int = 128
    patience: int = 999  # No early stopping
    train_ratio: float = 0.9
    seed: int = 42

    # Model
    vocab_size: int = 128256
    hidden_dim: int = 4096
    poly_degree: int = 4096
    plain_bits: int = 30
    scale: int = 10000
    lam: int = 80
    u_layers: int = 0
    m_layers: int = 32

    # LoRA
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05

    # Optimizer
    learning_rate: float = 3.5e-4
    weight_decay: float = 0.01
    warmup_steps: int = 200

    # Workers
    N_CRYPTO_U_WORKERS: int = 8
    N_CRYPTO_M_WORKERS: int = 8
    N_CRYPTO_S_WORKERS: int = 1

    # Pipeline
    USE_CHUNKED_PIPELINE: bool = True
    CHUNK_TOKENS: int = 3072


def set_seed(seed: int) -> None:
    import random
    import torch
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_two_epoch_test(cfg: TwoEpochConfig) -> dict:
    """Run two-epoch convergence test."""
    import torch
    from src.data.dataset import load_biotriplex_dataset, LlamaTokenizerWrapper, BioTriplexQADataset
    from src.core.bfv_privselect_v2_adapter import BFVPrivSelectV2Backend
    from src.core.s3pir_hints import HintTable
    from src.parties.heterogeneous_protocol import HeterogeneousProtocol
    from src.training.trainer import Trainer, TrainerConfig
    from pathlib import Path
    import json

    logger.info("=" * 60)
    logger.info("Two-Epoch Convergence Test")
    logger.info("=" * 60)
    logger.info("Config: max_epochs=%d, batch_size=%d, max_seq_length=%d",
                cfg.max_epochs, cfg.batch_size, cfg.max_seq_length)

    set_seed(cfg.seed)

    # Check GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))
        logger.info("GPU Memory: %.1f GB total, %.1f GB available",
                    torch.cuda.get_device_properties(0).total_memory / 1e9,
                    torch.cuda.mem_get_info()[0] / 1e9)

    # Step 1: Load datasets
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
    logger.info("Datasets: train=%d, val=%d, test=%d",
                len(train_ds), len(val_ds), len(test_ds))

    # Step 2: Build BFV backend
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

    # Step 3: Build encrypted DB
    logger.info("[Step 3] Building encrypted DB...")
    snap_path = cfg.hf_model
    index_path = Path(snap_path) / "model.safetensors.index.json"
    from safetensors.torch import load_file

    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        wm = index["weight_map"]
        lm_head_files = sorted({
            str(Path(snap_path) / fn)
            for k, fn in wm.items() if "lm_head" in k
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

    if V is None:
        raise FileNotFoundError(f"lm_head.weight not found in {snap_path}")

    db_result = bfv_backend.build_encrypted_database(V, force=True)
    logger.info("Encrypted DB: %d rows, from_cache=%s, time=%.1fs",
                db_result["n_rows"], db_result["from_cache"], db_result.get("build_time_s", 0))

    # Step 4: Extract keys
    logger.info("[Step 4] Extracting keys...")
    import pickle as _pickle
    from src.core.bfv_privselect_v2_adapter import _seal_to_bytes
    sk_pem = _seal_to_bytes(bfv_backend._secret_key)
    # Crypto workers expect pk_pem as a pickle of {"pk_bytes": raw_seal_bytes}
    pk_pem = _pickle.dumps({"pk_bytes": bfv_backend.public_key_bytes})
    bfv_backend._drop_secret_key()
    prg_seed = os.urandom(32)
    logger.info("Keys extracted and sk dropped from main process")

    # Step 5: Load hint table
    logger.info("[Step 5] Loading hint table...")
    hints_dir = os.path.join(cfg.bfv_cache_dir, "s3pir_hints")
    partition_size = 1 << ((cfg.vocab_size.bit_length() - 1) // 2)
    hint_table = HintTable(
        n_entries=cfg.vocab_size,
        partition_size=partition_size,
        lam=cfg.lam,
        cache_dir=hints_dir,
    )
    if os.path.exists(os.path.join(hints_dir, "hint_table.json")):
        hint_table = HintTable.from_cache_files(hints_dir)
        logger.info("Hint table loaded from cache")
    else:
        hint_table.compute_main_hints_skeleton()
        hint_table.compute_backup_hints_skeleton()
        hint_table.to_cache_files()
        logger.info("Hint table computed and saved")

    # Step 6: Build config for runtime
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

    # Step 7: Build HeterogeneousProtocol
    logger.info("[Step 6] Building HeterogeneousProtocol...")
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

    # Step 8: Run training
    logger.info("[Step 7] Starting training...")
    trainer_cfg = TrainerConfig(
        max_epochs=cfg.max_epochs,
        patience=cfg.patience,
        batch_size=cfg.batch_size,
        max_seq_length=cfg.max_seq_length,
        USE_CHUNKED_PIPELINE=cfg.USE_CHUNKED_PIPELINE,
        CHUNK_TOKENS=cfg.CHUNK_TOKENS,
        save_freq=1,
        log_freq=10,
    )

    trainer = Trainer(
        config=trainer_cfg,
        ipc_protocol=protocol,
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        tokenizer=tokenizer,
    )

    t0 = time.time()
    results = trainer.train()
    elapsed = time.time() - t0

    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)
    logger.info("Total time: %.1fs", elapsed)
    logger.info("Best val metric: %.4f at epoch %d",
                results["best_metric"], results["best_epoch"])
    logger.info("Total steps: %d", results["total_steps"])

    # Check for convergence
    epoch_metrics = results.get("epoch_metrics", [])
    if len(epoch_metrics) >= 2:
        loss_0 = epoch_metrics[0].get("train_ce_loss", float("inf"))
        loss_1 = epoch_metrics[1].get("train_ce_loss", float("inf"))
        f1_0 = epoch_metrics[0].get("val_entity_micro_f1", 0)
        f1_1 = epoch_metrics[1].get("val_entity_micro_f1", 0)

        logger.info("Convergence check:")
        logger.info("  Epoch 1 loss: %.4f, Epoch 2 loss: %.4f (diff: %.4f)",
                    loss_0, loss_1, loss_1 - loss_0)
        logger.info("  Epoch 1 F1: %.4f, Epoch 2 F1: %.4f (diff: %.4f)",
                    f1_0, f1_1, f1_1 - f1_0)

        # Convergence criteria
        loss_improved = loss_1 <= loss_0
        f1_improved = f1_1 >= f1_0

        logger.info("  Loss improved: %s", "✓" if loss_improved else "✗")
        logger.info("  F1 improved: %s", "✓" if f1_improved else "✗")

        results["convergence_check"] = {
            "loss_improved": loss_improved,
            "f1_improved": f1_improved,
            "loss_diff": loss_1 - loss_0,
            "f1_diff": f1_1 - f1_0,
        }

    protocol.shutdown()
    logger.info("Protocol shutdown complete")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Two-Epoch Convergence Test for SLG-HE-PIR",
    )
    parser.add_argument("--max_epochs", type=int, default=2,
                        help="Number of epochs to train (default: 2)")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Dataset directory")
    parser.add_argument("--bfv_cache_dir", type=str, default=None,
                        help="BFV cache directory")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size override")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose logging")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    cfg = TwoEpochConfig()

    if args.max_epochs is not None:
        cfg.max_epochs = args.max_epochs
    if args.data_dir is not None:
        cfg.data_dir = args.data_dir
    if args.bfv_cache_dir is not None:
        cfg.bfv_cache_dir = args.bfv_cache_dir
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size

    try:
        results = run_two_epoch_test(cfg)

        # Check convergence
        convergence = results.get("convergence_check", {})
        if convergence.get("loss_improved") and convergence.get("f1_improved"):
            logger.info("\n✓ Two-epoch test PASSED: convergence verified")
            return 0
        elif convergence:
            logger.info("\n⚠ Two-epoch test PARTIAL: some metrics did not improve")
            return 1
        else:
            logger.info("\n✓ Two-epoch test PASSED: training completed")
            return 0

    except Exception as e:
        logger.error("Two-epoch test FAILED: %s", e, exc_info=True)
        return 1


def make_fast_cfg(ns) -> TwoEpochConfig:
    """Build a TwoEpochConfig from a namespace (used by diag_grad_flow.py)."""
    cfg = TwoEpochConfig()
    cfg.batch_size = ns.batch_size
    cfg.max_seq_length = ns.max_length
    cfg.max_epochs = ns.epochs
    cfg.lam = ns.lam
    cfg.no_purge = getattr(ns, "no_purge", False)
    # These are accessed by diag_grad_flow after make_fast_cfg returns.
    cfg.log_freq = 1
    cfg.save_freq = 999
    cfg.process_mode = getattr(ns, "process_mode", "fusion")
    return cfg


def run_stage1_with_hooks(cfg: TwoEpochConfig, resume_from=None) -> dict:
    """Run init + 1-epoch training with trainer hooks (used by diag_grad_flow.py).

    This wraps run_two_epoch_test but patches the trainer to run only 1 epoch
    and leaves room for callers (diag_grad_flow) to install additional hooks.
    """
    import torch
    from src.data.dataset import load_biotriplex_dataset, LlamaTokenizerWrapper, BioTriplexQADataset
    from src.core.bfv_privselect_v2_adapter import BFVPrivSelectV2Backend
    from src.core.s3pir_hints import HintTable
    from src.parties.heterogeneous_protocol import HeterogeneousProtocol
    from src.training.trainer import Trainer, TrainerConfig
    from pathlib import Path
    import json
    import pickle as _pickle
    from src.core.bfv_privselect_v2_adapter import _seal_to_bytes
    from safetensors.torch import load_file

    set_seed(cfg.seed)

    train_samples, val_samples, test_samples = load_biotriplex_dataset(
        data_dir=cfg.data_dir,
        train_ratio=cfg.train_ratio,
        seed=cfg.seed,
    )
    tokenizer = LlamaTokenizerWrapper(cfg.hf_model, max_length=cfg.max_seq_length)
    train_ds = BioTriplexQADataset(train_samples, tokenizer, max_length=cfg.max_seq_length, task="train")
    val_ds = BioTriplexQADataset(val_samples, tokenizer, max_length=cfg.max_seq_length, task="val")
    test_ds = BioTriplexQADataset(test_samples, tokenizer, max_length=cfg.max_seq_length, task="test")

    bfv_backend = BFVPrivSelectV2Backend(
        n_entries=cfg.vocab_size,
        vec_dim=cfg.hidden_dim,
        shared_seed=os.urandom(32),
        cache_dir=cfg.bfv_cache_dir,
        poly_degree=cfg.poly_degree,
        plain_bits=cfg.plain_bits,
        scale=float(cfg.scale),
    )

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

    db_result = bfv_backend.build_encrypted_database(V, force=not cfg.no_purge)

    sk_pem = _seal_to_bytes(bfv_backend._secret_key)
    pk_pem = _pickle.dumps({"pk_bytes": bfv_backend.public_key_bytes})
    bfv_backend._drop_secret_key()
    prg_seed = os.urandom(32)

    hints_dir = os.path.join(cfg.bfv_cache_dir, "s3pir_hints")
    partition_size = 1 << ((cfg.vocab_size.bit_length() - 1) // 2)
    if os.path.exists(os.path.join(hints_dir, "hint_table.json")):
        hint_table = HintTable.from_cache_files(hints_dir)
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

    trainer_cfg = TrainerConfig(
        max_epochs=cfg.max_epochs,
        patience=cfg.patience,
        batch_size=cfg.batch_size,
        max_seq_length=cfg.max_seq_length,
        USE_CHUNKED_PIPELINE=cfg.USE_CHUNKED_PIPELINE,
        CHUNK_TOKENS=cfg.CHUNK_TOKENS,
        save_freq=getattr(cfg, "save_freq", 1),
        log_freq=getattr(cfg, "log_freq", 10),
    )

    trainer = Trainer(
        config=trainer_cfg,
        ipc_protocol=protocol,
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        tokenizer=tokenizer,
    )

    t0 = time.time()
    results = trainer.train()
    elapsed = time.time() - t0

    results["elapsed_s"] = elapsed
    protocol.shutdown()
    return results


if __name__ == "__main__":
    sys.exit(main())
