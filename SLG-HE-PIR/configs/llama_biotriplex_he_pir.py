"""
Llama-3.1-8B-Instruct + BioTriplex-QA + HE-PIR total configuration.

This module provides a single configuration dataclass that matches the
defaults specified in docs/SLG_HE_PIR_SYSTEM_DOCUMENTATION.md §6.1.

Parameters (matching docs/SLG_HE_PIR_SYSTEM_DOCUMENTATION.md §6.1):
  poly_degree = 4096
  plain_bits = 30
  scale = 10000
  u_layers = 0 (embed_tokens only on U)
  m_layers = 32 (all decoder layers + LoRA on M)
  lora_rank = 8
  lora_alpha = 16
  batch_size = 4
  max_seq_length = 128
  max_epochs = 10
  learning_rate = 3.5e-4
  patience = 999 (no early stopping by default)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LlamaBioTriplexConfig:
    """Full configuration for SLG-HE-PIR v2.0 training.

    All defaults match docs/SLG_HE_PIR_SYSTEM_DOCUMENTATION.md §6.1.
    """

    # =========================================================================
    # Model architecture
    # =========================================================================
    vocab_size: int = 128256
    hidden_dim: int = 4096
    poly_degree: int = 4096
    plain_bits: int = 30
    u_layers: int = 0    # U holds embed_tokens only (not decoder layers)
    m_layers: int = 32   # M holds all 32 decoder layers + norm + LoRA

    # =========================================================================
    # BFV encryption
    # =========================================================================
    scale: int = 10000   # Fixed-point scaling factor

    # =========================================================================
    # S3PIR
    # =========================================================================
    lam: int = 80         # Security parameter (2^{-80} false positive rate)

    # =========================================================================
    # LoRA
    # =========================================================================
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05

    # =========================================================================
    # Training
    # =========================================================================
    batch_size: int = 4          # Small batch to avoid OOM with 32-layer M
    max_seq_length: int = 128    # Max sequence length
    max_epochs: int = 10
    learning_rate: float = 3.5e-4
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    warmup_steps: int = 200
    lr_scheduler: str = "cosine_with_warmup"
    train_ratio: float = 0.9
    patience: int = 999           # No early stopping by default

    # =========================================================================
    # Paths
    # =========================================================================
    hf_model: str = "/root/autodl-tmp/hf_cache/Llama-3-1-8B-I"
    bfv_cache_dir: str = "/root/autodl-tmp/slg-bfv-cache"
    data_dir: str = "/root/slg-v2.0/data/biotriplex_qa"
    project_root: str = "/root/autodl-tmp/SLG-HE-PIR"
    checkpoint_dir: str = "/root/autodl-tmp/SLG-HE-PIR/checkpoints"
    log_dir: str = "/root/autodl-tmp/SLG-HE-PIR/logs"

    # =========================================================================
    # Validation
    # =========================================================================
    val_metric: str = "val_entity_micro_f1"   # Must match Trainer output key

    # =========================================================================
    # Flags
    # =========================================================================
    seed: int = 42
    log_freq: int = 10
    save_freq: int = 1
    dump_attacks: bool = False
    do_test_eval: bool = False   # Run test evaluation after training

    # =========================================================================
    # Heterogeneous runtime knobs
    # =========================================================================
    N_CRYPTO_U_WORKERS: int = 8
    N_CRYPTO_M_WORKERS: int = 8
    N_CRYPTO_S_WORKERS: int = 1

    # =========================================================================
    # Pipeline mode
    # =========================================================================
    USE_CHUNKED_PIPELINE: bool = True
    CHUNK_TOKENS: int = 3072

    def to_dict(self) -> dict:
        """Convert to a plain dict for serialization."""
        return {
            "vocab_size": self.vocab_size,
            "hidden_dim": self.hidden_dim,
            "poly_degree": self.poly_degree,
            "plain_bits": self.plain_bits,
            "u_layers": self.u_layers,
            "m_layers": self.m_layers,
            "scale": self.scale,
            "lam": self.lam,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "batch_size": self.batch_size,
            "max_seq_length": self.max_seq_length,
            "max_epochs": self.max_epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "gradient_clip_norm": self.gradient_clip_norm,
            "warmup_steps": self.warmup_steps,
            "lr_scheduler": self.lr_scheduler,
            "train_ratio": self.train_ratio,
            "patience": self.patience,
            "hf_model": self.hf_model,
            "bfv_cache_dir": self.bfv_cache_dir,
            "data_dir": self.data_dir,
            "project_root": self.project_root,
            "checkpoint_dir": self.checkpoint_dir,
            "log_dir": self.log_dir,
            "val_metric": self.val_metric,
            "seed": self.seed,
            "log_freq": self.log_freq,
            "save_freq": self.save_freq,
            "dump_attacks": self.dump_attacks,
            "do_test_eval": self.do_test_eval,
            "N_CRYPTO_U_WORKERS": self.N_CRYPTO_U_WORKERS,
            "N_CRYPTO_M_WORKERS": self.N_CRYPTO_M_WORKERS,
            "N_CRYPTO_S_WORKERS": self.N_CRYPTO_S_WORKERS,
            "USE_CHUNKED_PIPELINE": self.USE_CHUNKED_PIPELINE,
            "CHUNK_TOKENS": self.CHUNK_TOKENS,
        }


__all__ = ["LlamaBioTriplexConfig"]
