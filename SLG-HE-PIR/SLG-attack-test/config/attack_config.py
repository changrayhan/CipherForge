"""Attack test configuration for the SLG-HE-PIR attack suite.

Centralises all hyperparameters so that run_attack_suite.py and individual
attack modules share the same defaults.  Loaded from CLI args and merged
with environment variables.

4 core attacks based on TEST_REPORT.md:
  - L1: M-side gradient label inference (g_{H,t} = a_t - V_y)
  - L2: S-side activation label inference (a_t analysis)
  - M1: U-side model inference (evaluation phase - S's predictions)
  - M2: S-side hidden state inversion (Z_t analysis)
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class AttackConfig:
    # ── Paths ──────────────────────────────────────────────────────────────────
    project_root: str = "/home/changrayhan/hCode/SLG-HE-PIR-code/SLG-HE-PIR"
    hf_model_path: str = "/home/changrayhan/hCode/SLG-HE-PIR-code/hf_cache/models--unsloth--Llama-3.2-1B"
    bfv_cache_dir: str = "/home/changrayhan/hCode/SLG-HE-PIR-code/bfv_cache"
    data_dir: str = "/home/changrayhan/hCode/SLG-HE-PIR-code/SLG-HE-PIR/datasets/trec-qc"

    # ── BFV / S3PIR parameters (must match protocol Config) ──────────────────
    vocab_size: int = 128_256
    hidden_dim: int = 2048       # Llama-3.2-1B: 2048 (half of 8B)
    poly_degree: int = 4096
    plain_bits: int = 30
    scale: int = 10000
    lam: int = 80

    # ── Model layers (for 3.2-1B: 16 layers total) ────────────────────────────
    u_layers: int = 0             # embeddings only on U
    m_layers: int = 16          # all 16 decoder layers on M (half of 8B)

    # ── Training (attack-test is lightweight) ────────────────────────────────
    batch_size: int = 4
    max_seq_length: int = 128
    max_steps: int = 20            # total gradient samples collected
    max_epochs: int = 2
    learning_rate: float = 3.5e-4
    gradient_clip_norm: float = 1.0

    # ── LoRA ─────────────────────────────────────────────────────────────────
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05

    # ── Crypto workers ───────────────────────────────────────────────────────
    n_crypto_u_workers: int = 8
    n_crypto_m_workers: int = 8
    n_crypto_s_workers: int = 1

    # ── Pipeline ─────────────────────────────────────────────────────────────
    use_chunked_pipeline: bool = True
    chunk_tokens: int = 3072

    # ── Attack control ───────────────────────────────────────────────────────
    attacks: List[str] = field(default_factory=lambda: ["L1", "L2", "M1", "M2"])
    n_steps: int = 20          # alias for max_steps
    seed: int = 42
    output_dir: str = "SLG-attack-test/results"

    # ── L-1 specific ─────────────────────────────────────────────────────────
    l1_n_permutations: int = 10000   # permutation test iterations
    l1_alpha: float = 0.05           # significance level
    l1_n_clusters: int = 6          # K-Means K (coarse-grained classes, TREC-QC)

    # ── L-2 specific ─────────────────────────────────────────────────────────
    l2_sample_size: int = 100       # number of samples for activation analysis

    # ── M-1 specific (conditional: requires logits return) ──────────────────
    m1_query_budget: int = 1000    # number of queries for distillation
    m1_fidelity_threshold: float = 0.7  # fidelity threshold
    m1_logits_available: bool = False   # whether S returns logits to U

    # ── M-2 specific ─────────────────────────────────────────────────────────
    m2_rank_estimator_iterations: int = 50
    m2_n_permutations: int = 999     # permutation test iterations for rank/direction fingerprints

    # ── Derived ──────────────────────────────────────────────────────────────
    @property
    def trecqc_data_dir(self) -> str:
        return os.path.join(self.project_root, "datasets", "trec-qc")

    @property
    def results_dir(self) -> Path:
        return Path(self.output_dir)

    @property
    def attack_dump_dir(self) -> str:
        return os.path.join(self.output_dir, "dumps")


def add_attack_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("Attack suite")
    g.add_argument(
        "--attacks",
        type=str,
        default="L1,L2,M1,M2",
        help="Comma-separated attack IDs: L1, L2, M1, M2 (default: L1,L2,M1,M2)",
    )
    g.add_argument(
        "--n_steps",
        type=int,
        default=20,
        help="Number of training steps / gradient samples to collect (default: 20)",
    )
    g.add_argument(
        "--output_dir",
        type=str,
        default="SLG-attack-test/results",
        help="Directory for results and dumps (default: SLG-attack-test/results)",
    )
    g.add_argument(
        "--data_dir",
        type=str,
        default="",
        help="Override data directory (default: auto from project_root)",
    )
    g.add_argument(
        "--l1_n_permutations",
        type=int,
        default=10000,
        help="Permutation test iterations for L-1 (default: 10000)",
    )
    g.add_argument(
        "--l1_alpha",
        type=float,
        default=0.05,
        help="Significance level for L-1 permutation test (default: 0.05)",
    )
    g.add_argument(
        "--hf_model",
        type=str,
        default="",
        help="Override HuggingFace model path",
    )
    g.add_argument(
        "--m1_logits_available",
        action="store_true",
        help="Enable M-1 attack (requires S to return logits to U)",
    )
    g.add_argument(
        "--m2_n_permutations",
        type=int,
        default=999,
        help="Permutation test iterations for M-2 rank/direction fingerprints (default: 999)",
    )
    g.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )


def build_attack_config(args: argparse.Namespace) -> AttackConfig:
    cfg = AttackConfig()
    cfg.attacks = [a.strip().upper() for a in args.attacks.split(",") if a.strip()]
    cfg.n_steps = args.n_steps
    cfg.max_steps = args.n_steps
    cfg.output_dir = args.output_dir
    cfg.l1_n_permutations = args.l1_n_permutations
    cfg.l1_alpha = args.l1_alpha
    cfg.seed = args.seed
    cfg.m1_logits_available = args.m1_logits_available
    cfg.m2_n_permutations = args.m2_n_permutations

    if args.data_dir:
        cfg.data_dir = args.data_dir

    if args.hf_model:
        cfg.hf_model_path = args.hf_model

    # Create output dirs
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    os.makedirs(cfg.attack_dump_dir, exist_ok=True)

    return cfg
