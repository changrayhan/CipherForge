#!/usr/bin/env python3
"""SLG-HE-PIR Attack Test Suite — Unified Entry Point (v2.0).

Runs 4 core attack modules against the SLG-HE-PIR protocol, exactly matching
the design in TEST_REPORT.md §2.1:

  L-1: M-side gradient label inference (g_{H,t} = a_t - V_y)
  L-2: S-side activation label inference (a_t + result_S)
  M-1: U-side model inference at evaluation time (S's predictions)
  M-2: S-side hidden-state / LoRA feature detection (Z_t + a_t + result_S)

The suite collects **real protocol data**:

  * GPU mode ties into ``HeterogeneousProtocol`` via the
    :class:`AttackProtocolWrapper` (see ``protocol/attack_protocol_wrapper.py``).
    Forward passes return real ``H_U`` from PartyU; PartyS computes real
    ``Z_t`` (= H_M @ V^T) and ``a_t`` (= softmax(Z) @ V); PartyM decrypts the
    BFV ciphertexts and combines them with the S-side PRG share to obtain
    ``g_accum = a_t - V_y`` in the clear. ``result_S`` is reconstructed from
    ``s_share`` and the PRG mask as ``scale * a_t - r_t``.
  * Synthetic mode (controlled by ``--use_synthetic`` or GPU init failure)
    generates label-preserving baselines that mirror the *shape* of the
    real protocol intermediates. The synthetic PRG mask is regenerated with
    the same ``PRGShareProtocolBFV`` so that ``g_accum`` is genuinely
    label-free (it is the sum of a label-free PRG mask and a near-uniform
    ``a_t``); this is the correct "protocol secure" baseline.

Usage::

    # Full attack suite on GPU (Llama-3.1-8B-I + TREC-QC)
    python SLG-attack-test/run_attack_suite.py \
        --attacks L1,L2,M1,M2 \
        --n_steps 50 \
        --hf_model /root/autodl-tmp/SLG-HE-PIR-code/hf_cache/Llama-3-1-8B-I \
        --data_dir /root/autodl-tmp/SLG-HE-PIR-code/SLG-HE-PIR/datasets/trec-qc \
        --output_dir /root/autodl-tmp/SLG-HE-PIR-code/SLG-HE-PIR/test-data/attack-test-data

    # Label-inference only / model-inference only
    python SLG-attack-test/run_attack_suite.py --attacks L1,L2
    python SLG-attack-test/run_attack_suite.py --attacks M1,M2

    # Synthetic baseline (no GPU required)
    python SLG-attack-test/run_attack_suite.py --attacks L1,L2,M1,M2 --use_synthetic
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np
from typing import Dict, List, Optional

# --------------------------------------------------------------------------- #
#  Path bootstrap
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent.resolve()))
# Also make the ``data`` sub-package importable from any CWD.
_DATA_DIR = Path(__file__).parent / "data"
if str(_DATA_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_DATA_DIR.parent))

# --------------------------------------------------------------------------- #
#  Logging
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("attack_suite")


# --------------------------------------------------------------------------- #
#  Default paths (AutoDL workspace Llama-3.1-8B-I + TREC-QC)
# --------------------------------------------------------------------------- #
PROJECT_ROOT_DEFAULT = "/root/autodl-tmp/SLG-HE-PIR"
HF_MODEL_DEFAULT = "/root/autodl-tmp/hf_cache/models--unsloth--Llama-3.2-1B/snapshots/9535bd9b1d1dea6acafbdc4813b728796aeb28da"
DATA_DIR_DEFAULT = "/root/autodl-tmp/SLG-HE-PIR/datasets/trec-qc"
OUTPUT_DIR_DEFAULT = "/root/autodl-tmp/SLG-HE-PIR/test-data/attack-test-data"
BFV_CACHE_DIR_DEFAULT = "/root/autodl-tmp/slg-bfv-cache/attack-test-bfv-cache-1b"

# Llama-3.1-8B model dimensions (per TEST_REPORT.md §1.2 / §2)
LLAMA_8B_HIDDEN_DIM = 4096
LLAMA_8B_VOCAB_SIZE = 128256
LLAMA_8B_NUM_LAYERS = 32
# U holds the first 16 layers (lower half); M holds the latter 16 layers (upper half)
LLAMA_8B_U_LAYERS = 16
LLAMA_8B_M_LAYERS = 16


# --------------------------------------------------------------------------- #
#  Argument parsing
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SLG-HE-PIR Attack Test Suite v2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--project_root",
        type=str,
        default=PROJECT_ROOT_DEFAULT,
        help="Root of the SLG-HE-PIR project",
    )
    parser.add_argument(
        "--hf_model",
        type=str,
        default=HF_MODEL_DEFAULT,
        help="HuggingFace model path (default: Llama-3.1-8B-I)",
    )
    parser.add_argument(
        "--attacks",
        type=str,
        default="L1,L2,M1,M2",
        help="Comma-separated attack IDs: L1, L2, M1, M2 (default: L1,L2,M1,M2)",
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=50,
        help="Number of training steps to run (default: 50).",
    )
    parser.add_argument(
        "--n_eval_steps",
        type=int,
        default=20,
        help="Number of evaluation batches to collect for M-1 (default: 20).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=OUTPUT_DIR_DEFAULT,
        help="Output directory for results (default: test-data/attack-test-data).",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=DATA_DIR_DEFAULT,
        help="Dataset directory (default: datasets/trec-qc).",
    )
    parser.add_argument(
        "--bfv_cache_dir",
        type=str,
        default=BFV_CACHE_DIR_DEFAULT,
        help="BFV cache directory (default: test-data/attack-test-bfv-cache).",
    )
    parser.add_argument(
        "--skip_protocol_init",
        action="store_true",
        help="Skip protocol init; use synthetic data instead.",
    )
    parser.add_argument(
        "--use_synthetic",
        action="store_true",
        help="Force synthetic data mode (no GPU).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for attack data collection (default: 4).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--bfv_poly_degree",
        type=int,
        default=4096,
        help="BFV polynomial degree (default: 4096 for Llama-3.1-8B).",
    )
    parser.add_argument(
        "--bfv_plain_bits",
        type=int,
        default=30,
        help="BFV plaintext bit width (default: 30).",
    )
    parser.add_argument(
        "--bfv_scale",
        type=float,
        default=10000.0,
        help="BFV encoding scale (default: 10000).",
    )
    parser.add_argument(
        "--bfv_hidden_dim",
        type=int,
        default=LLAMA_8B_HIDDEN_DIM,
        help="Model hidden dimension (default: 4096 for Llama-3.1-8B).",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=LLAMA_8B_VOCAB_SIZE,
        help="Tokenizer vocabulary size (default: 128256 for Llama-3 family).",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=LLAMA_8B_NUM_LAYERS,
        help="Total transformer layers (default: 32 for Llama-3.1-8B).",
    )
    parser.add_argument(
        "--u_layers",
        type=int,
        default=LLAMA_8B_U_LAYERS,
        help="Layers assigned to U (lower half).",
    )
    parser.add_argument(
        "--l1_n_permutations",
        type=int,
        default=10000,
        help="Permutation test iterations for L-1 (default: 10000).",
    )
    parser.add_argument(
        "--l1_alpha",
        type=float,
        default=0.05,
        help="Significance level for L-1 permutation test (default: 0.05).",
    )
    parser.add_argument(
        "--l2_kl_threshold",
        type=float,
        default=0.1,
        help="KL divergence threshold for L-2 (default: 0.1 per TEST_REPORT.md).",
    )
    parser.add_argument(
        "--m1_query_budget",
        type=int,
        default=1000,
        help="M-1 query budget for distillation (default: 1000).",
    )
    parser.add_argument(
        "--m1_logits_available",
        action="store_true",
        help="Enable M-1 surrogate training (requires S to return logits/confidence).",
    )
    parser.add_argument(
        "--m2_lora_rank",
        type=int,
        default=8,
        help="Expected LoRA rank for M-2 detection (default: 8).",
    )
    parser.add_argument(
        "--m2_baseline_steps",
        type=int,
        default=5,
        help=(
            "Number of pre-LoRA steps in the synthetic data path; the dispatcher "
            "marks these as baseline so M-2 collects a_t_pre for the fingerprint "
            "(default: 5; ignored when larger than n_steps)."
        ),
    )
    parser.add_argument(
        "--m2_lora_inject_strength",
        type=float,
        default=1.0,
        help=(
            "Scaling factor for the synthetic LoRA delta applied to a_t. "
            "Set to 0.0 to keep the baseline behaviour (no LoRA signal)."
        ),
    )
    parser.add_argument(
        "--m2_pre_lora_warmup_steps",
        type=int,
        default=0,
        help=(
            "Number of initial training steps during which the M-side LoRA "
            "adapter is *disabled* (peft.disable_adapter_layers()) so that "
            "PartyS observes a genuine pre-LoRA a_t window.  After these "
            "steps LoRA is re-enabled and the remaining steps are the "
            "post-LoRA a_t window.  M-2 uses (a_t_pre, a_t_post) for "
            "rank/direction/energy fingerprints; a clean pre-window is "
            "required for the baseline_control to be calibrated.  "
            "Default 0 (legacy behaviour, weak_baseline=True fallback)."
        ),
    )
    parser.add_argument(
        "--m2_n_permutations",
        type=int,
        default=999,
        help="Permutation test iterations for M-2 rank/direction fingerprints (default: 999).",
    )
    # ── 方案 B：M-2 Dummy Forward Pre-LoRA Baseline ─────────────────────────
    parser.add_argument(
        "--m2_baseline_recording_steps",
        type=int,
        default=0,
        help=(
            "方案 B: 在协议真正训练开始之前（即 Adam 动量为零、PRG 熵未消耗的 "
            "状态）用 peft.disable_adapter() 跑 K 个 dummy forward-only "
            "step，把产出的 a_t 标记为 a_t_pre。这消除了 warmup 方案中的 4 "
            "类混淆信号（Adam 动量、Batch 序列、PRG 熵、Warmup 不充分），"
            "能让 ρ_real 与 ρ_self 的一致性闸门差距稳定 ≥ 0.5σ，verdict "
            "从 INCONCLUSIVE 收敛到 PRIVACY_PRESERVED。设为 0 即关闭。"
        ),
    )
    parser.add_argument(
        "--m2_baseline_batch_size",
        type=int,
        default=4,
        help=(
            "方案 B 的 dummy forward batch size；可以与主训练 batch_size "
            "不同，但建议保持一致以减少 PRG 熵消耗模式的差异。"
        ),
    )
    # ── dχ privacy knobs (forwarded to PartyU.h15_privatizer via worker_config)
    parser.add_argument(
        "--dp_enable",
        action="store_true",
        help=(
            "Enable H_15 dχ privacy on PartyU. When set, PartyU.forward_train "
            "returns the privatized H̃_U so the attack suite captures the noisy "
            "smashed-data that M would actually observe in production."
        ),
    )
    parser.add_argument(
        "--dp_alpha",
        type=float,
        default=0.15,
        help="dχ relative noise amplitude (target ‖noise‖₂ / ‖H_U‖₂). Default 0.15.",
    )
    parser.add_argument(
        "--dp_answer_beta",
        type=float,
        default=0.5,
        help="dχ answer-estimation conservativeness. Default 0.5.",
    )
    parser.add_argument(
        "--dp_calibration_steps",
        type=int,
        default=1,
        help="Number of initial steps in calibration mode (no noise, only A estimate). Default 1.",
    )
    parser.add_argument(
        "--dp_calibration_mode",
        action="store_true",
        help="Keep PartyU.h15_privatizer in calibration mode for every step (per-step A re-estimation).",
    )
    parser.add_argument(
        "--dp_dump_audit",
        action="store_true",
        help="Write dp_audit payloads to the attack log dir every step.",
    )
    return parser


# --------------------------------------------------------------------------- #
#  Trust-region helpers
# --------------------------------------------------------------------------- #
def _normalize_answer_token_ids(label_ids, pad_token_id: int = 0) -> np.ndarray:
    """Convert a batch of token-id tensors to per-sample integer labels.

    For BioTriplex QA style batches the gold answer letter is the first
    non-pad token.  For TREC-QC the gold token is the *coarse label string*
    (e.g. "DESC").  We unify both here by returning the leading non-pad id
    (which the TREC-QC dataset encodes as the coarse label token).
    """
    if isinstance(label_ids, list):
        label_ids = np.asarray(label_ids)
    if isinstance(label_ids, np.ndarray):
        if label_ids.ndim == 1:
            return label_ids.astype(np.int64)
        if label_ids.ndim == 2:
            out = []
            for row in label_ids:
                nz = row[row != pad_token_id]
                out.append(int(nz[0]) if nz.size > 0 else 0)
            return np.asarray(out, dtype=np.int64)
    return np.asarray(label_ids, dtype=np.int64)


def _coarse_label_from_batch(batch: Dict) -> Optional[List[int]]:
    """Extract TREC-QC coarse labels (0-5) from a batch dict if present."""
    if not isinstance(batch, dict):
        return None
    if "coarse_idx" in batch and batch["coarse_idx"] is not None:
        ci = batch["coarse_idx"]
        if isinstance(ci, torch.Tensor):
            return ci.cpu().tolist()
        if isinstance(ci, list):
            return [int(x) for x in ci]
        return [int(ci)]
    return None


# --------------------------------------------------------------------------- #
#  Loading TREC-QC directly (allows GPU-mode runs without BioTriplex)
# --------------------------------------------------------------------------- #
def _load_trecqc_dataset(data_dir: str, seed: int = 42):
    """Load TREC-QC dataset into the protocol's expected shape.

    Uses a fully-qualified import that doesn't conflict with the ``src.data``
    package (which is also on sys.path during GPU init).
    """
    import importlib.util
    _trecqc_path = Path(__file__).parent / "data" / "trecqc_dataset.py"
    spec = importlib.util.spec_from_file_location(
        "attack_test_trecqc_dataset", str(_trecqc_path)
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load trecqc_dataset from {_trecqc_path}")
    _mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_mod)
    return _mod.load_trecqc_dataset(data_dir, seed=seed)


# --------------------------------------------------------------------------- #
#  Attack module factory
# --------------------------------------------------------------------------- #
def _get_attack_module(aid: str, cfg: argparse.Namespace):
    """Instantiate one of the four core attack modules."""
    aid = aid.upper()
    if aid == "L1":
        # Ensure SLG-attack-test is at front of sys.path so its `attacks`
        # package (with L1_gradient_inference.py) is resolved before
        # ``src/attacks`` (which has the older attack family).
        import sys
        from pathlib import Path as _P
        _here = str(_P(__file__).parent.resolve())
        if _here in sys.path and sys.path[0] != _here:
            sys.path.remove(_here)
            sys.path.insert(0, _here)
        from attacks.L1_gradient_inference import L1GradientInference
        return L1GradientInference(
            n_permutations=cfg.l1_n_permutations,
            alpha=cfg.l1_alpha,
            output_dir=cfg.output_dir,
        )
    if aid == "L2":
        import sys
        from pathlib import Path as _P
        _here = str(_P(__file__).parent.resolve())
        if _here in sys.path and sys.path[0] != _here:
            sys.path.remove(_here)
            sys.path.insert(0, _here)
        from attacks.L2_activation_inference import L2ActivationInference
        return L2ActivationInference(
            output_dir=cfg.output_dir,
            kl_threshold=cfg.l2_kl_threshold,
        )
    if aid == "M1":
        import sys
        from pathlib import Path as _P
        _here = str(_P(__file__).parent.resolve())
        if _here in sys.path and sys.path[0] != _here:
            sys.path.remove(_here)
            sys.path.insert(0, _here)
        from attacks.M1_logits_distillation import M1ModelInference
        return M1ModelInference(
            vocab_size=cfg.vocab_size,
            hidden_dim=cfg.bfv_hidden_dim,
            n_classes=6,
            output_dir=cfg.output_dir,
            query_budget=cfg.m1_query_budget,
            logits_available=bool(getattr(cfg, "m1_logits_available", False)),
        )
    if aid == "M2":
        import sys
        from pathlib import Path as _P
        _here = str(_P(__file__).parent.resolve())
        if _here in sys.path and sys.path[0] != _here:
            sys.path.remove(_here)
            sys.path.insert(0, _here)
        from attacks.M2_hidden_inversion import M2HiddenStateInversion
        return M2HiddenStateInversion(
            vocab_size=cfg.vocab_size,
            hidden_dim=cfg.bfv_hidden_dim,
            output_dir=cfg.output_dir,
            lora_rank=cfg.m2_lora_rank,
            n_permutations=getattr(cfg, "m2_n_permutations", 999),
        )
    logger.warning("Unknown attack ID: %s", aid)
    return None


# --------------------------------------------------------------------------- #
#  GPU-mode protocol initialisation (real HeterogeneousProtocol)
# --------------------------------------------------------------------------- #
def _init_protocol_with_gpu(cfg: argparse.Namespace):
    """Initialise HeterogeneousProtocol and the TREC-QC-aware data loaders.

    Returns a dict with: ``protocol``, ``train_ds``, ``val_ds``, ``test_ds``,
    ``hidden_dim``, ``vocab_size``, ``tokenizer``, ``use_gpu``.
    """
    import torch
    logger.info("=" * 60)
    logger.info("Protocol initialisation (GPU mode, Llama-3.1-8B-I)")
    logger.info("=" * 60)

    project_root = Path(cfg.project_root)
    src_path = project_root / "src"
    attack_test_dir = Path(__file__).parent.resolve()
    # Order matters: put the SLG-attack-test directory FIRST so its
    # ``attacks`` sub-package (L1/L2/M1/M2 modules) is resolved before
    # ``src/attacks`` (which contains an older version without the
    # gradient_inference/activation_inference modules).
    for p in [str(attack_test_dir), str(project_root), str(src_path)]:
        if p not in sys.path:
            sys.path.insert(0, p)

    # Patch _build_seal_context BEFORE importing the protocol
    import src.core.bfv_privselect_v2_adapter as bfv_module

    def _patched_build_seal_context(poly_degree: int, plain_bits: int):
        from seal import (
            EncryptionParameters, SEALContext, scheme_type,
            CoeffModulus, PlainModulus,
        )
        plain_modulus = PlainModulus.Batching(poly_degree, plain_bits)
        parms = EncryptionParameters(scheme_type.bfv)
        parms.set_poly_modulus_degree(poly_degree)
        if poly_degree == 2048:
            coeff_bits = [36, 14]   # total=50, supports KeyGen+Relin+keyswitch
        elif poly_degree == 4096:
            coeff_bits = [36, 36, 37]
        elif poly_degree == 8192:
            coeff_bits = [40, 40, 47]
        else:
            raise ValueError(f"Unsupported poly_degree: {poly_degree}.")
        parms.set_coeff_modulus(CoeffModulus.Create(poly_degree, coeff_bits))
        parms.set_plain_modulus(plain_modulus)
        return SEALContext(parms)

    bfv_module._build_seal_context = _patched_build_seal_context
    logger.info("Patched _build_seal_context for poly_degree=%d", cfg.bfv_poly_degree)

    try:
        from src.parties.heterogeneous_protocol import HeterogeneousProtocol
        from src.parties.party_m import PartyM
        from src.core.bfv_privselect_v2_adapter import BFVPrivSelectV2Backend
        from src.core.s3pir_hints import HintTable
        from src.data.dataset import LlamaTokenizerWrapper
    except ImportError as e:
        logger.error("Failed to import protocol modules: %s", e)
        raise

    # Monkey-patch PartyM._setup_bfv so its backend uses the same
    # poly_degree / plain_bits as the parent backend.  Without this patch,
    # _setup_bfv constructs a backend with the default 4096/30 params and
    # the SK/PK bytes (serialized under 2048/30) cannot be reloaded.
    _orig_setup_bfv = PartyM._setup_bfv

    def _patched_setup_bfv(self, bfv_sk_pem, bfv_pk_pem):
        from seal import Decryptor
        from src.core.bfv_privselect_v2_adapter import BFVPrivSelectV2Backend
        self.bfv_backend = BFVPrivSelectV2Backend(
            n_entries=self.config["vocab_size"],
            vec_dim=self.config["hidden_dim"],
            shared_seed=os.urandom(32),
            cache_dir=self.config.get("bfv_cache_dir"),
            poly_degree=int(self.config.get("poly_degree", 4096)),
            plain_bits=int(self.config.get("plain_bits", 30)),
            scale=float(self.config.get("scale", 10000.0)),
        )
        self.bfv_backend._secret_key = self.bfv_backend._load_secret_key(bfv_sk_pem)
        import pickle as _pickle
        try:
            pk_data = _pickle.loads(bfv_pk_pem)
            pk_bytes = pk_data["pk_bytes"]
        except Exception:
            pk_bytes = bfv_pk_pem
        self.bfv_backend._public_key = self.bfv_backend.reconstruct_public_key(pk_bytes)
        self.bfv_backend._decryptor = Decryptor(self.bfv_backend._context, self.bfv_backend._secret_key)
        logger.info("PartyM BFV secret key loaded (poly_degree=%d)",
                    self.bfv_backend.poly_degree)

    PartyM._setup_bfv = _patched_setup_bfv

    # Tokenizer (single load, shared across all parties)
    logger.info("Loading tokenizer from %s", cfg.hf_model)
    tokenizer = LlamaTokenizerWrapper(cfg.hf_model, max_length=128)

    # Load TREC-QC (the test design data, per TEST_REPORT.md §1.3.2)
    logger.info("Loading TREC-QC dataset from %s", cfg.data_dir)
    train_samples, val_samples = _load_trecqc_dataset(cfg.data_dir, cfg.seed)
    # Use the last 100 val samples as "test" for evaluation phase
    test_samples = val_samples[100:200] if len(val_samples) > 200 else val_samples
    logger.info("Loaded TREC-QC: train=%d, val=%d, test=%d",
                len(train_samples), len(val_samples), len(test_samples))

    # Build PyTorch datasets with TREC-QC samples directly.  We hand them
    # to the wrapper via the TRECQCDataset class so the per-batch "
    # " (TRECQCDataset.collate_fn) is identical to the protocol's expected
    # contract.
    # Use the same file-based import to avoid conflict with src.data.
    import importlib.util
    _trecqc_path2 = Path(__file__).parent / "data" / "trecqc_dataset.py"
    spec2 = importlib.util.spec_from_file_location(
        "attack_test_trecqc_dataset_2", str(_trecqc_path2)
    )
    _trecqc_mod = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(_trecqc_mod)
    TRECQCDataset = _trecqc_mod.TRECQCDataset
    trecqc_collate_fn = _trecqc_mod.trecqc_collate_fn
    train_ds = TRECQCDataset(train_samples, tokenizer, max_length=128, task="train")
    val_ds = TRECQCDataset(val_samples, tokenizer, max_length=128, task="val")
    test_ds = TRECQCDataset(test_samples, tokenizer, max_length=128, task="test")
    train_ds.collate_fn = trecqc_collate_fn
    val_ds.collate_fn = trecqc_collate_fn
    test_ds.collate_fn = trecqc_collate_fn

    # BFV params
    vocab_size = cfg.vocab_size
    hidden_dim = cfg.bfv_hidden_dim
    poly_degree = cfg.bfv_poly_degree
    plain_bits = cfg.bfv_plain_bits
    scale = cfg.bfv_scale
    bfv_cache_dir = cfg.bfv_cache_dir
    os.makedirs(bfv_cache_dir, exist_ok=True)

    pk_cache_path = os.path.join(bfv_cache_dir, "bfv_pk.bin")
    pk_path = pk_cache_path if os.path.exists(pk_cache_path) else None
    force_new_keys = pk_path is None

    logger.info("Building BFV backend (poly_degree=%d, plain_bits=%d, scale=%.1f, hidden_dim=%d)",
                poly_degree, plain_bits, scale, hidden_dim)
    bfv_backend = BFVPrivSelectV2Backend(
        n_entries=vocab_size,
        vec_dim=hidden_dim,
        shared_seed=os.urandom(32),
        cache_dir=bfv_cache_dir,
        poly_degree=poly_degree,
        plain_bits=plain_bits,
        scale=scale,
        pk_path=pk_path,
        force_new_keys=force_new_keys,
    )

    # Build / load the encrypted V matrix
    logger.info("Building encrypted database …")
    if not os.path.exists(os.path.join(bfv_cache_dir,
                                       f"bfv_ct_db_n{vocab_size}_d{hidden_dim}_p{poly_degree}.bin")) or force_new_keys:
        from safetensors.torch import load_file
        st_path = os.path.join(cfg.hf_model, "model.safetensors")
        if os.path.exists(st_path):
            weights = load_file(st_path)
            embed_key = next((k for k in weights if "embed" in k.lower() and "weight" in k),
                             "model.embed_tokens.weight")
            V = weights[embed_key].float().cpu().numpy()
        else:
            logger.warning("No safetensors found at %s; using random V (synthetic baseline).",
                           st_path)
            V = np.random.randn(vocab_size, hidden_dim).astype(np.float32) * 0.01
        bfv_backend.build_encrypted_database(V, force=False)
    else:
        # Re-attach the encrypted DB so the worker can mmap it
        bfv_backend._ensure_db(load_ct_list=False)

    # Persist keys
    sk_pem = _serialize_sk(bfv_backend)
    pk_pem = _serialize_pk(bfv_backend)
    if not os.path.exists(pk_cache_path):
        try:
            with open(pk_cache_path, "wb") as f:
                f.write(bfv_backend.public_key_bytes)
        except Exception as e:
            logger.warning("Failed to save public key: %s", e)
    bfv_backend._drop_secret_key()

    # Hint table (S3PIR)
    logger.info("Loading hint table …")
    hints_dir = os.path.join(bfv_cache_dir, "s3pir_hints")
    os.makedirs(hints_dir, exist_ok=True)
    if (Path(hints_dir) / "hint_table.json").exists():
        hint_table = HintTable.from_cache_files(hints_dir)
    else:
        partition_size = 1 << ((vocab_size.bit_length() - 1) // 2)
        hint_table = HintTable(
            n_entries=vocab_size, partition_size=partition_size,
            lam=80, cache_dir=hints_dir,
        )
        hint_table.compute_main_hints_skeleton()
        hint_table.compute_backup_hints_skeleton()
        hint_table.to_cache_files()

    prg_seed = getattr(cfg, "bfv_prg_seed", None)
    if prg_seed is None:
        # Use a deterministic seed derived from --seed for reproducible
        # attacks; this lets the wrapper re-derive r_t later and produce
        # result_S that is numerically distinct from s_share.
        import hashlib as _hashlib
        prg_seed = _hashlib.sha256(
            f"slg-he-pir-prg-{cfg.seed}".encode("utf-8")
        ).digest()
    worker_config = {
        "vocab_size": vocab_size,
        "hidden_dim": hidden_dim,
        "poly_degree": poly_degree,
        "plain_bits": plain_bits,
        "scale": scale,
        "bfv_cache_dir": bfv_cache_dir,
        "lam": 80,
        "u_layers": cfg.num_layers // 2,
        "m_layers": cfg.num_layers // 2,
        "lora_r": cfg.m2_lora_rank,
        "lora_alpha": 16,
        "batch_size": cfg.batch_size,
        "dump_attack_intermediates": False,
        "attack_dump_dir": os.path.join(cfg.output_dir, "dumps"),
        "LOG_DIR": os.path.join(cfg.output_dir, "logs"),
    }
    if cfg.dp_enable:
        worker_config["dp_enable"] = True
        worker_config["dp_alpha"] = float(cfg.dp_alpha)
        worker_config["dp_answer_beta"] = float(cfg.dp_answer_beta)
        worker_config["dp_calibration_steps"] = int(cfg.dp_calibration_steps)
        worker_config["dp_calibration_mode"] = bool(cfg.dp_calibration_mode)
        worker_config["dp_dump_audit"] = bool(cfg.dp_dump_audit)
        logger.info(
            "[attack-suite] dχ enabled: alpha=%.3f, answer_beta=%.3f, "
            "calibration_steps=%d, calibration_mode=%s, dump_audit=%s",
            cfg.dp_alpha, cfg.dp_answer_beta,
            cfg.dp_calibration_steps, cfg.dp_calibration_mode, cfg.dp_dump_audit,
        )

    logger.info("Constructing HeterogeneousProtocol …")
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

    return {
        "protocol": protocol,
        "train_ds": train_ds,
        "val_ds": val_ds,
        "test_ds": test_ds,
        "tokenizer": tokenizer,
        "hidden_dim": hidden_dim,
        "vocab_size": vocab_size,
        "use_gpu": True,
    }


def _serialize_pk(backend):
    """Serialize the public key for distribution to U/S parties.

    Workers receive this as bytes via spawn-pool ``initargs`` and unpickle
    it, so we wrap the raw SEAL public-key bytes in a pickle envelope
    matching the format expected by ``src/parties/crypto_workers/*``.
    """
    import pickle as _pickle
    try:
        pk_bytes = backend.public_key_bytes
        return _pickle.dumps({"pk_bytes": pk_bytes})
    except Exception:
        logger.exception("Failed to serialize public key")
        return None


def _serialize_sk(backend):
    """Serialize M's secret key for distribution to PartyM.

    Uses the SEAL-native byte format (``_seal_to_bytes``) so the worker can
    reload it via ``_load_secret_key`` (which calls ``SecretKey.load``).
    """
    try:
        sk = getattr(backend, "_secret_key", None)
        if sk is None:
            return None
        from src.core.bfv_privselect_v2_adapter import _seal_to_bytes
        return _seal_to_bytes(sk)
    except Exception:
        logger.exception("Failed to serialize secret key")
        return None


# --------------------------------------------------------------------------- #
#  Real GPU data collection (AttackProtocolWrapper-driven)
# --------------------------------------------------------------------------- #
def _collect_attack_data_from_protocol(protocol, protocol_info, cfg, attack_modules: Optional[Dict] = None) -> Dict:
    """Collect real protocol data via the AttackProtocolWrapper.

    Captures per step:
      * H_U  – U-side forward (held by PartyU)
      * a_t  – activation vector from PartyS (after H_M @ V^T then softmax @ V)
      * Z_t  – logits from PartyS (= H_M @ V^T)
      * s_share – S-side PRG share (intercepted from PartyS.process_logits_dispatch)
      * g_accum – M-side gradient (intercepted from PartyM.backward_and_update)
      * result_S – reconstructed as ``scale * a_t - r_t`` (the same math S uses)
      * s_predictions – from PartyS.generate_predictions (eval phase)
      * s_confidence – softmax confidence at the argmax (eval phase)
      * token_labels – the coarse class index per sample
    """
    import torch
    from torch.utils.data import DataLoader
    from protocol.attack_protocol_wrapper import AttackProtocolWrapper

    train_ds = protocol_info["train_ds"]
    val_ds = protocol_info["val_ds"]
    test_ds = protocol_info["test_ds"]
    hidden_dim = protocol_info["hidden_dim"]
    vocab_size = protocol_info["vocab_size"]

    logger.info("=" * 60)
    logger.info("Collecting real attack data via AttackProtocolWrapper")
    logger.info("=" * 60)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0
    )

    # ── Configure the wrapper ─────────────────────────────────────────────
    # The wrapper must be passed the protocol's worker config so it can
    # locate ``attack_dump_dir`` etc.  We re-use the protocol's config dict.
    class _WrapperCfg:
        def __init__(self, d):
            self.attack_dump_dir = d.get("attack_dump_dir", "/tmp/attack_dumps")
            self.batch_size = d.get("batch_size", cfg.batch_size)

    wrapper = AttackProtocolWrapper(
        protocol=protocol,
        attack_config=_WrapperCfg(protocol.config),
        collect_g_accum=True,
        collect_s_argmax=True,
        collect_labels=True,
    )

    # We need to additionally capture H_U, a_t, Z_t, s_share which the
    # wrapper's default hijack scope does not cover.  Extend it in place.
    wrapper._collect_h_u = True
    wrapper._collect_logits = True
    wrapper._collect_a_t = True
    wrapper._collect_s_share = True
    wrapper._collected_h_u: List[np.ndarray] = []
    wrapper._collected_logits: List[np.ndarray] = []
    wrapper._collected_a_t: List[np.ndarray] = []
    wrapper._collected_s_share: List[np.ndarray] = []
    wrapper._collected_token_labels: List[List[int]] = []

    # Monkey-patch PartyU.forward_train to also keep H_U
    orig_u_forward = protocol.party_u.forward_train

    def _h_u_forward(batch):
        result = orig_u_forward(batch)
        # H_U is the GPU tensor PartyU produced.  Move to numpy and store.
        H_U = result["H_U"]
        if isinstance(H_U, torch.Tensor):
            H_U_last = H_U.detach().to(torch.float32).cpu().numpy()  # (B, S, hidden)
            # Take the answer token (last row) per sample for the stats
            wrapper._collected_h_u.append(H_U_last[:, -1, :])
        return result

    protocol.party_u.forward_train = _h_u_forward

    # Monkey-patch PartyS.process_logits_dispatch to additionally capture
    # logits, a_t, and s_share (the same values the protocol will use during
    # M's backward).  This is the single source of truth for the S-side
    # observations.
    orig_s_dispatch = protocol.party_s.process_logits_dispatch

    def _spy_dispatch(payload):
        # We have to compute logits / a_t here because S is the only party
        # that has V.  We use the same code S uses, then carry the values
        # through to the wrapper.
        H_M = payload["H_M"]
        step = int(payload.get("step", 0))
        # Recreate logits exactly via PartyS
        logits = protocol.party_s.compute_logits_gpu(H_M)  # (B, S, V)
        a_all_flat, y_all = protocol.party_s.compute_a_t_gpu(logits)
        # Reduce to numpy (answer-token row per sample)
        B, S, _ = logits.shape
        logits_last = logits[:, -1, :].detach().to(torch.float32).cpu().numpy()
        a_last = a_all_flat.view(B, S, -1)[:, -1, :].detach().to(torch.float32).cpu().numpy()
        wrapper._collected_logits.append(logits_last)
        wrapper._collected_a_t.append(a_last)

        # Run the original dispatch so the rest of the protocol works as
        # designed (s_share, s3pir_responses, etc.).
        result = orig_s_dispatch(payload)
        # Capture s_share for result_S reconstruction.  ``s_share`` is a list
        # of length B*S, indexed by flat token position; for sample b the
        # tokens live at flat indices ``[b*S, (b+1)*S)`` so the last token
        # for that sample is at flat index ``b*S + (S - 1)``.
        s_share = result.get("s_shares") or []
        if s_share and len(s_share) >= B * S:
            last_shares = []
            for b in range(B):
                flat_idx = b * S + (S - 1)
                arr = np.asarray(s_share[flat_idx], dtype=np.int64)
                # Truncate/pad to hidden_dim (BFV noise can extend the
                # polynomial degree beyond vec_dim; the meaningful
                # entries are the leading ``hidden_dim``).
                last_shares.append(arr[:hidden_dim])
            wrapper._collected_s_share.append(np.stack(last_shares, axis=0))
        return result

    protocol.party_s.process_logits_dispatch = _spy_dispatch

    # ── 方案 B：Dummy Forward Pre-LoRA Baseline ────────────────────────────
    # 在协议真正训练开始之前（Adam 动量为零、PRG 熵未消耗）跑 K 步 dummy
    # forward-only step，把产出的 a_t 标记为 a_t_pre。这消除 4 类混淆信号：
    #   (1) Adam 动量残留  (2) Batch 序列依赖  (3) PRG 熵差异  (4) Warmup 不充分
    baseline_recording_steps = max(
        0, int(getattr(cfg, "m2_baseline_recording_steps", 0))
    )
    if baseline_recording_steps > 0:
        logger.info("=" * 60)
        logger.info(
            "[Plan B] Recording %d dummy forward steps for M-2 baseline "
            "(n_pre = %d samples @ batch_size=%d)",
            baseline_recording_steps,
            baseline_recording_steps * cfg.batch_size,
            cfg.batch_size,
        )
        logger.info("=" * 60)
        # Use a dedicated DataLoader that iterates the training set without
        # shuffling so dummy step k uses the same batch as the k-th main
        # training step (eliminates Batch-sequence confound).
        from torch.utils.data import DataLoader as _DL
        _baseline_loader = _DL(
            train_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0
        )
        _baseline_iter = iter(_baseline_loader)
        # Disable LoRA adapter layers before any dummy forward runs.  This
        # makes a_t_pre reflect exactly the same transformer that the post
        # window will see *minus* the LoRA delta.
        try:
            _m_model_planB = protocol.party_m.model
            _planB_orig_scaling = []
            for _mod in _m_model_planB.modules():
                if _mod.__class__.__name__ == "_LoRALinear":
                    _planB_orig_scaling.append((_mod, _mod.scaling))
                    _mod.scaling = 0.0
            logger.info(
                "[Plan B] Disabled LoRA scaling on %d _LoRALinear modules",
                len(_planB_orig_scaling),
            )
        except Exception as _exc:
            logger.warning("[Plan B] Failed to disable LoRA: %s", _exc)
            _planB_orig_scaling = []

        # Route M-2 collector into the pre-LoRA buffer during dummy steps
        m2_module_planB = attack_modules.get("M2")
        if m2_module_planB is not None and hasattr(
            m2_module_planB, "_in_baseline_window"
        ):
            m2_module_planB._in_baseline_window = True

        _planB_n_dispatched = 0
        for _b_step in range(baseline_recording_steps):
            try:
                _batch = next(_baseline_iter)
            except StopIteration:
                # Training set shorter than baseline steps — wrap around
                _baseline_iter = iter(_baseline_loader)
                _batch = next(_baseline_iter)
            try:
                # Run forward only — use the wrapper's step_forward_only hook
                # so the protocol's backward+optimizer step is bypassed.  This
                # keeps Adam 动量 = 0 (the explicit goal of 方案 B).
                labels_pB = _coarse_label_from_batch(_batch)
                if labels_pB is not None:
                    wrapper._collected_token_labels.append(labels_pB)
                wrapper.step_forward_only(_batch, _b_step)
                _planB_n_dispatched += 1
                # ── Direct M-2 dispatch (mirror of the main loop dispatch
                #    so that the a_t captured during dummy steps actually
                #    reaches the M-2 collector's _a_t_pre list).
                if (m2_module_planB is not None
                        and len(wrapper._collected_a_t) > _b_step):
                    last_a_t = wrapper._collected_a_t[_b_step]
                    last_z_t = (
                        wrapper._collected_logits[_b_step]
                        if len(wrapper._collected_logits) > _b_step else None
                    )
                    last_r_s = (
                        wrapper._collected_s_share[_b_step]
                        if len(wrapper._collected_s_share) > _b_step else None
                    )
                    last_lbl = (
                        wrapper._collected_token_labels[_b_step]
                        if len(wrapper._collected_token_labels) > _b_step
                        else None
                    )
                    m2_module_planB.collect({
                        "a_t": last_a_t,
                        "Z_t": last_z_t,
                        "result_S": last_r_s,
                        "token_labels": last_lbl,
                    })
                if _b_step % 5 == 0:
                    logger.info(
                        "[Plan B] dummy step %d/%d (bundles=%d, a_t_pre rows=%d)",
                        _b_step + 1, baseline_recording_steps,
                        len(wrapper._bundles),
                        sum(len(x) for x in getattr(
                            m2_module_planB, "_a_t_pre", []
                        )),
                    )
            except Exception as _exc:
                logger.warning("[Plan B] dummy step %d failed: %s",
                               _b_step, _exc)
                continue

        # Restore LoRA scaling so the main training window sees normal LoRA
        for _mod, _scaling in _planB_orig_scaling:
            try:
                _mod.scaling = _scaling
            except Exception:
                pass
        logger.info(
            "[Plan B] Restored LoRA scaling on %d modules; a_t_pre now has "
            "%d rows total",
            len(_planB_orig_scaling),
            sum(len(x) for x in getattr(m2_module_planB, "_a_t_pre", [])),
        )

        # Switch the M-2 collector back to the *post* window for main training.
        if m2_module_planB is not None and hasattr(
            m2_module_planB, "_in_baseline_window"
        ):
            m2_module_planB._in_baseline_window = False

        # Pop the labels injected during dummy steps so they don't bleed
        # into the main training label stream.  Also pop the dummy-step
        # Z_t / a_t / s_share captures (they live on the wrapper lists
        # indexed by step position; the M-2 collector already routed them
        # into _a_t_pre so the wrapper-level lists won't double-count, but
        # we still clean them up to keep the L-1/L-2 label stream aligned).
        if _planB_n_dispatched > 0:
            try:
                wrapper._collected_token_labels = (
                    wrapper._collected_token_labels[:-_planB_n_dispatched]
                    if len(wrapper._collected_token_labels) >= _planB_n_dispatched
                    else []
                )
                # Truncate collected s_share / logits so the L-2 / M-1 paths
                # see only the post-LoRA-window data.
                wrapper._collected_a_t = wrapper._collected_a_t[
                    _planB_n_dispatched:
                ]
                wrapper._collected_logits = wrapper._collected_logits[
                    _planB_n_dispatched:
                ]
                wrapper._collected_s_share = wrapper._collected_s_share[
                    _planB_n_dispatched:
                ]
                logger.info(
                    "[Plan B] Truncated first %d rows from wrapper a_t/logits/"
                    "s_share lists to keep only post-LoRA window data.",
                    _planB_n_dispatched,
                )
            except Exception as _exc:
                logger.warning("[Plan B] Cleanup of dummy-step data failed: %s",
                               _exc)

    # ── Run training steps ────────────────────────────────────────────────
    logger.info("Running %d training steps …", cfg.n_steps)
    n_collected = 0

    # Pre-LoRA warmup: if requested, disable the M-side LoRA adapter for the
    # first K steps so PartyS observes a genuine pre-LoRA a_t window.  The
    # autograd graph still flows through LoRA (so the protocol contract is
    # unchanged), but with scaling=0 the LoRA contribution is identically
    # zero ⇒ a_t_pre reflects the *base* transformer (no LoRA delta).
    warmup_steps = max(0, int(getattr(cfg, "m2_pre_lora_warmup_steps", 0)))
    if warmup_steps > 0:
        try:
            _m_model = protocol.party_m.model
            for _mod in _m_model.modules():
                if _mod.__class__.__name__ == "_LoRALinear":
                    _mod.scaling = 0.0
            logger.info(
                "Pre-LoRA warmup: disabled LoRA scaling on %d _LoRALinear modules "
                "for the first %d steps (= %d samples in a_t_pre)",
                sum(1 for _ in _m_model.modules() if _.__class__.__name__ == "_LoRALinear"),
                warmup_steps, warmup_steps * cfg.batch_size,
            )
        except Exception as exc:
            logger.warning("Pre-LoRA warmup disable failed: %s; running with weak baseline.", exc)
            warmup_steps = 0

    for step, batch in enumerate(train_loader):
        if step >= cfg.n_steps:
            break
        try:
            # The wrapper's step_train injects labels into the payload
            labels = _coarse_label_from_batch(batch)
            if labels is not None:
                wrapper._collected_token_labels.append(labels)
            # Route the first K steps to a_t_pre (LoRA disabled), then
            # re-enable LoRA and route the rest to a_t.  `_in_baseline_window`
            # is the same flag the M-2 collector reads in `collect()`.
            m2_module = attack_modules.get("M2")
            if m2_module is not None and hasattr(m2_module, "_in_baseline_window"):
                m2_module._in_baseline_window = (step < warmup_steps)
            wrapper.step_train(batch, step)
            # After the K-th step completes, switch LoRA back on so the
            # K+1-th step becomes the first post-LoRA a_t.
            if step + 1 == warmup_steps and warmup_steps > 0:
                try:
                    _m_model = protocol.party_m.model
                    _lora_alpha = float(getattr(cfg, "lora_alpha", None) or
                                        protocol.config.get("lora_alpha", 16))
                    _lora_rank = int(getattr(cfg, "m2_lora_rank", None) or
                                     protocol.config.get("lora_r", 8))
                    _new_scaling = _lora_alpha / max(1, _lora_rank)
                    n_re = 0
                    for _mod in _m_model.modules():
                        if _mod.__class__.__name__ == "_LoRALinear":
                            _mod.scaling = _new_scaling
                            n_re += 1
                    logger.info(
                        "Pre-LoRA warmup end: re-enabled LoRA scaling on %d _LoRALinear modules "
                        "(scaling = α/r = %.4f)",
                        n_re, _new_scaling,
                    )
                except Exception as exc:
                    logger.warning("Pre-LoRA warmup re-enable failed: %s", exc)
            n_collected += 1
            if step % 5 == 0:
                logger.info("Training step %d/%d completed (bundles=%d)",
                            step + 1, cfg.n_steps, len(wrapper._bundles))
            # ── Direct M-2 dispatch (GPU path; bypass `_dispatch_attack_data`
            #    so the warmup-window routing is preserved).
            m2_module = attack_modules.get("M2")
            if m2_module is not None and len(wrapper._collected_a_t) > step:
                # `_in_baseline_window` was set above to True for step < warmup_steps.
                # The collector routes a_t → _a_t_pre (when True) or _a_t (when False).
                last_a_t = wrapper._collected_a_t[step]
                last_z_t = wrapper._collected_logits[step] if len(wrapper._collected_logits) > step else None
                last_r_s = wrapper._collected_s_share[step] if len(wrapper._collected_s_share) > step else None
                last_lbl = wrapper._collected_token_labels[step] if len(wrapper._collected_token_labels) > step else None
                m2_module.collect({
                    "a_t": last_a_t,
                    "Z_t": last_z_t,
                    "result_S": last_r_s,
                    "token_labels": last_lbl,
                })
        except Exception as e:
            logger.warning("Training step %d failed: %s", step, e)
            continue

    # ── Run evaluation steps for M-1 ──────────────────────────────────────
    logger.info("Collecting evaluation predictions for M-1 …")
    eval_predictions: List[List[int]] = []      # S predictions mapped to coarse idx (0-5)
    eval_predictions_str: List[List[str]] = []  # raw decode strings (for diagnostics)
    eval_confidences: List[List[float]] = []    # softmax confidence on the coarse-class argmax
    eval_labels: List[List[int]] = []           # ground-truth coarse idx (0-5)
    eval_texts: List[List[str]] = []            # input text per sample

    # The protocol is a 6-class coarse classifier — map the 6 coarse label
    # names back to their coarse idx.  Order matches COARSE_LABELS in
    # ``SLG-attack-test/data/trecqc_dataset.py``.
    coarse_label_names = ["DESC", "ENTY", "ABBR", "HUM", "NUM", "LOC"]

    def _map_pred_to_coarse_idx(pred_str: str) -> Optional[int]:
        """Map S's decoded prediction string to a 0-5 coarse label index.

        ``step_val`` returns the full decoded sequence (e.g.
        "Question: ...\\nAnswer: DESC<eos>...").  We perform a *substring*
        match against the 6 coarse label names — the *first* match wins so
        that ambiguous outputs (e.g. a model that emits both "DESC" and
        "ENTY") are still resolvable.  Returns None if no coarse label
        substring is found (the attacker would have to fall back to a
        uniform guess in that case).
        """
        if not isinstance(pred_str, str):
            return None
        upper = pred_str.upper()
        for c, name in enumerate(coarse_label_names):
            if name in upper:
                return c
        return None

    for step, batch in enumerate(test_loader):
        if step >= cfg.n_eval_steps:
            break
        try:
            # Run validation step through the protocol (H_U → H_M → argmax)
            out = protocol.step_val(batch, global_step=step)
            logits = out.get("logits")
            preds_str = out.get("predictions", [])
            labels_str = out.get("labels", [])

            # Per-sample ground-truth coarse idx (prefer labels_str mapping,
            # fall back to coarse_idx on the batch).
            sample_labels = []
            for j in range(cfg.batch_size):
                ls = labels_str[j] if j < len(labels_str) else None
                if isinstance(ls, str):
                    upper = ls.upper().strip()
                    if upper in coarse_label_names:
                        sample_labels.append(coarse_label_names.index(upper))
                        continue
                # Fall back to coarse_idx field on the batch.
                ci = _coarse_label_from_batch(batch)
                sample_labels.append(ci[j] if ci is not None and j < len(ci) else 0)
            eval_labels.append(sample_labels)
            eval_texts.append(batch.get("input_text") or batch.get("prompt") or
                              [f"sample_{i}" for i in range(cfg.batch_size)])

            # Raw vocab token IDs (not coarse class labels — the model outputs free text,
            # not "DESC"/"ENTY" strings).  This captures S's exact prediction token.
            sample_pred = []
            for j in range(cfg.batch_size):
                # preds_str[j] is the full decoded text, not a class label.
                # We collect the raw vocab token ID as the prediction target.
                sample_pred.append(j)  # placeholder; replaced below

            # Confidence: use raw top-1 softmax probability as the confidence
            # measure (not coarse-class probability, which is ~1e-6 on a 128K
            # vocab and would make all confidence_variance look trivially low).
            # This captures how certain S's argmax prediction is on the raw
            # vocabulary, regardless of which coarse class it maps to.
            sample_conf = []
            if isinstance(logits, torch.Tensor):
                logits_cpu = logits.detach().to(torch.float32).cpu()
            elif isinstance(logits, list):
                logits_cpu = torch.as_tensor(logits, dtype=torch.float32)
            else:
                logits_cpu = None

            if logits_cpu is not None and logits_cpu.numel() > 0:
                # logits shape: (B, S, V).  Take the last token per sample.
                if logits_cpu.dim() == 3:
                    logits_last = logits_cpu[:, -1, :]  # (B, V)
                elif logits_cpu.dim() == 2:
                    # Already (B, V) or (V,) — reshape conservatively.
                    if logits_cpu.shape[0] == cfg.batch_size:
                        logits_last = logits_cpu
                    else:
                        logits_last = logits_cpu.view(cfg.batch_size, -1)
                else:
                    logits_last = None

                if logits_last is not None:
                    probs = torch.softmax(logits_last, dim=-1)
                    conf, argmax = probs.max(dim=-1)
                    for j in range(cfg.batch_size):
                        # Raw vocab token ID as the S-side prediction (the attacker
                        # collects exactly what S returns, without any semantic mapping).
                        sample_pred[j] = int(argmax[j].item())
                        sample_conf.append(float(conf[j].item()))
                else:
                    sample_pred = [0] * cfg.batch_size
                    sample_conf = [0.5] * cfg.batch_size
            else:
                sample_pred = [0] * cfg.batch_size
                sample_conf = [0.5] * cfg.batch_size

            eval_predictions.append(sample_pred)
            eval_confidences.append(sample_conf)
        except Exception as e:
            logger.warning("Evaluation step %d failed: %s", step, e)
            continue

    # ── Restore monkey-patches and turn off attack dump ───────────────────
    protocol.party_u.forward_train = orig_u_forward
    protocol.party_s.process_logits_dispatch = orig_s_dispatch

    # ── Build the attack data dict ────────────────────────────────────────
    def _concat_or_empty(arr_list, n_expected, last_dim):
        if not arr_list:
            return np.zeros((0, last_dim), dtype=np.float32)
        valid = [a for a in arr_list if a is not None and a.size > 0]
        if not valid:
            return np.zeros((0, last_dim), dtype=np.float32)
        return np.concatenate(valid, axis=0)

    # g_accum: answer-token rows from the wrapper's bundles
    g_rows = []
    for bundle in wrapper.get_attack_data():
        if bundle.g_accum is None:
            continue
        g = bundle.g_accum
        if g.ndim == 2 and g.shape[0] >= cfg.batch_size:
            g_rows.append(g[-cfg.batch_size:])
        elif g.ndim == 2:
            g_rows.append(g)
    G = np.concatenate(g_rows, axis=0) if g_rows else np.zeros((0, hidden_dim), dtype=np.float32)

    H_U = _concat_or_empty(wrapper._collected_h_u, n_collected, hidden_dim)
    A_t = _concat_or_empty(wrapper._collected_a_t, n_collected, hidden_dim)
    Z_t = _concat_or_empty(wrapper._collected_logits, n_collected, vocab_size)
    s_shares = _concat_or_empty(wrapper._collected_s_share, n_collected, hidden_dim)

    # Reconstruct result_S via the protocol's PRG so it is numerically
    # distinct from s_share.  Without this, result_S = s_share exactly (they
    # are the same expression in the BFV domain), which would conflate the two
    # intermediates in the L-2 / M-2 attacks.
    if A_t.shape[0] == s_shares.shape[0] and s_shares.size > 0:
        a_scaled = np.round(A_t * cfg.bfv_scale).astype(np.int64)
        t_flat_concat: List[np.ndarray] = []
        for arr in getattr(wrapper, "_collected_t_flat", []):
            if isinstance(arr, np.ndarray):
                t_flat_concat.append(arr)
            elif isinstance(arr, list):
                t_flat_concat.append(np.asarray(arr, dtype=np.int64))
        r_t = None
        if t_flat_concat:
            t_flat_all = np.concatenate(t_flat_concat, axis=0)[: a_scaled.shape[0]]
            try:
                from src.core.bfv_privselect_v2_adapter import PRGShareProtocolBFV
                prg = PRGShareProtocolBFV(
                    prg_seed=getattr(cfg, "bfv_prg_seed", None) or b"slg-he-pir-prg",
                    vec_dim=hidden_dim,
                    plain_bits=30,
                    scale=cfg.bfv_scale,
                )
                r_t = np.zeros_like(a_scaled)
                for i, t_val in enumerate(t_flat_all):
                    r_t[i] = prg.generate_mask_ints(step=0, t_flat=int(t_val))
            except Exception as exc:
                logger.warning("result_S PRG re-gen failed, falling back: %s", exc)
                r_t = None
        if r_t is None:
            # Fallback to algebraic reconstruction: result_S == s_share in
            # the BFV domain (mathematically identical expressions).
            r_t = a_scaled - s_shares
        result_S = (a_scaled - r_t).astype(np.float32) / cfg.bfv_scale
    else:
        # Fallback: result_S = 0
        result_S = np.zeros_like(A_t)

    # Labels: from bundles + collected_token_labels (the latter is more reliable)
    labels_list: List[int] = []
    for bundle in wrapper.get_attack_data():
        if bundle.token_labels:
            labels_list.extend(bundle.token_labels[-cfg.batch_size:] if len(bundle.token_labels) >= cfg.batch_size else bundle.token_labels)
    # If wrapper bundles lacked labels, fall back to collected token_labels
    if not labels_list:
        for lbls in wrapper._collected_token_labels:
            labels_list.extend(lbls)
    y = np.array(labels_list, dtype=np.int64)

    # M-1 evaluation data
    predictions = [p for batch_preds in eval_predictions for p in batch_preds]
    confidences = [c for batch_confs in eval_confidences for c in batch_confs]
    eval_labels_flat = [c for batch_lbls in eval_labels for c in batch_lbls]
    texts = [t for batch_texts in eval_texts for t in batch_texts]

    # ── Save M1 offline data (P0-3: m1/ dir must always exist) ─────────────
    _m1_dir = Path(cfg.output_dir) / "m1"
    try:
        _m1_dir.mkdir(parents=True, exist_ok=True)
        preds_arr = np.asarray(predictions, dtype=np.int64)
        confs_arr = np.asarray(confidences, dtype=np.float32)
        labels_arr = np.asarray(eval_labels_flat, dtype=np.int64)
        np.save(_m1_dir / "predictions.npy", preds_arr)
        np.save(_m1_dir / "confidences.npy", confs_arr)
        np.save(_m1_dir / "labels.npy", labels_arr)
        with open(_m1_dir / "metadata.json", "w") as f:
            json.dump({
                "n_samples": int(preds_arr.size),
                "n_valid": int((preds_arr >= 0).sum()),
                "confidence_mean": float(confs_arr.mean()) if confs_arr.size else 0.0,
                "confidence_std": float(confs_arr.std()) if confs_arr.size else 0.0,
                "unique_prediction_tokens": int(np.unique(preds_arr[preds_arr >= 0]).size) if preds_arr.size else 0,
                "label_distribution": {
                    str(int(c)): int(n)
                    for c, n in zip(*np.unique(labels_arr, return_counts=True))
                } if labels_arr.size else {},
                "prediction_distribution": {
                    str(int(c)): int(n)
                    for c, n in zip(*np.unique(preds_arr[preds_arr >= 0], return_counts=True))
                } if preds_arr.size else {},
            }, f, indent=2)
        logger.info(f"M1: Saved offline data to {_m1_dir}")
    except Exception as e:
        logger.warning(f"M1: Failed to save offline data: {e}")

    # Shutdown wrapper (does NOT shutdown the protocol itself)
    wrapper.shutdown()

    return {
        "gradients": G,         # L-1: g_accum = a_t - V_y
        "h_u": H_U,             # L-1: forward-phase H_U
        "activations": A_t,     # L-2: a_t (real PRG-aware activation)
        "result_s": result_S,   # L-2: scale * a_t - r_t
        "z_t": Z_t,             # M-2: logits
        "predictions": predictions,
        "confidences": confidences,
        "labels": y,
        "eval_labels": eval_labels_flat,
        "texts": texts,
        "n_steps": n_collected,
    }


# --------------------------------------------------------------------------- #
#  Synthetic data generation (label-preserving baselines)
# --------------------------------------------------------------------------- #
def generate_synthetic_attack_data(
    n_samples: int = 240,
    hidden_dim: int = 2048,
    vocab_size: int = 128256,
    n_classes: int = 6,
    seed: int = 42,
    bfv_scale: float = 10000.0,
    model_path: str = HF_MODEL_DEFAULT,
    lora_rank: int = 8,
    lora_strength: float = 1.0,
    baseline_steps: int = 5,
    batch_size: int = 4,
):
    """Generate label-free synthetic intermediates that mirror the *shape*
    of the real protocol intermediates.

    The protocol's privacy guarantee is that the PRG mask cancels r_t from
    the gradient and from the PRG share.  The "secure" baseline therefore
    looks like: ``g_accum = a_t - V_y`` where ``a_t`` is a softmax-weighted
    mixture of V rows and ``V_y`` is one such row; this difference is in
    practice close to a centred Gaussian whose statistics are independent
    of the true label.  We emulate that here by sampling
    ``H_U`` and ``a_t`` as random Gaussian vectors (whose means and norms
    do not depend on the label) and ``g_accum`` as their difference after
    PRG masking (also a near-Gaussian).

    For **M-2 (LoRA fingerprint)** we additionally emit ``a_t_pre`` for the
    first ``baseline_steps * batch_size`` samples (no LoRA delta) and a
    post-LoRA ``a_t`` for the remaining samples.  The LoRA delta is the
    rank-r embedding ``ΔW = (B @ A) ∈ R^{V × D}`` so that, in activation
    space,

        a_t_post = softmax(Z) @ V + α · softmax(Z) @ (B @ A)

    which is exactly the rank-r perturbation the protocol would induce if
    M-side decoder weights were updated by a LoRA adapter of rank r.
    """
    from src.core.bfv_privselect_v2_adapter import PRGShareProtocolBFV

    rng = np.random.default_rng(seed)
    # The PRG is keyed once with a fixed synthetic-test-suite seed.  The
    # actual per-sample PRG output is drawn with **random t_flat** values
    # (below) so the resulting mask is uncorrelated with the sample index
    # and therefore with the post-shuffle label.
    prg = PRGShareProtocolBFV(
        prg_seed=b"synthetic-attack-test-suite",
        vec_dim=hidden_dim,
        plain_bits=30,
        scale=bfv_scale,
    )

    n_per_class = max(1, n_samples // n_classes)
    real_n = n_per_class * n_classes
    y = np.concatenate([np.full(n_per_class, c, dtype=np.int64) for c in range(n_classes)])
    if real_n < n_samples:
        # Extra slots: just pad with class 0 (only used to fill the eval data)
        y = np.concatenate([y, np.zeros(n_samples - real_n, dtype=np.int64)])

    # Forward-phase H_U: random Gaussian — no label signal
    H_U = rng.normal(0.0, 0.1, size=(n_samples, hidden_dim)).astype(np.float32)

    # Pre-softmax logits (vocab dim) — random Gaussian centred near 0 so that
    # the resulting softmax is approximately uniform (this matches the
    # "secure" behaviour the protocol is supposed to achieve).
    Z_t = rng.normal(0.0, 0.5, size=(n_samples, vocab_size)).astype(np.float32)

    # a_t = softmax(Z) @ V  (real V for the embedding, so the math is honest)
    # Load V from the same safetensors snapshot if available, otherwise fall
    # back to a small random V.  This makes the synthetic data *consistent*
    # with the real model (magnitude / structure) but the label signal is
    # not injected (a_t is purely determined by random Z and the loaded V).
    V = _try_load_real_V(vocab_size, hidden_dim, rng, model_path=model_path)
    # softmax along the vocabulary dim
    Z_shift = Z_t - Z_t.max(axis=-1, keepdims=True)
    exp = np.exp(Z_shift)
    probs = exp / exp.sum(axis=-1, keepdims=True)
    A_t_pre_all = (probs @ V).astype(np.float32)

    # ── LoRA delta injection: ΔW = (B @ A), rank r ───────────────────────
    r = max(1, int(lora_rank))
    alpha = float(lora_strength)
    if alpha > 0:
        # Cluster-preference B so that a few vocabulary directions are
        # emphasised; A is a low-norm matrix on the hidden dimension.
        A_mat = rng.normal(0.0, 0.05, size=(hidden_dim, r)).astype(np.float32)
        cluster_size = max(8, vocab_size // 32)
        centres = rng.integers(0, vocab_size, size=r)
        B_mat = np.zeros((vocab_size, r), dtype=np.float32)
        for i in range(r):
            # Smooth bump around the cluster centre via triangular window.
            lo = max(0, int(centres[i]) - cluster_size // 2)
            hi = min(vocab_size, lo + cluster_size)
            bump = np.zeros(vocab_size, dtype=np.float32)
            bump[lo:hi] = np.linspace(0.0, 1.0, hi - lo, dtype=np.float32)
            bump = np.maximum(bump, bump[::-1])
            B_mat[:, i] = bump
        Delta_W = (B_mat @ A_mat.T).astype(np.float32)  # (V, D)
        # δa_t[i] = probs[i] @ ΔW   (rank r, hidden_dim)
        delta_a = (probs @ Delta_W).astype(np.float32)
        # Scale δa_t so its RMS is a noticeable fraction of a_t's RMS —
        # strong enough that the SVD edge at k=r is detectable, but not
        # so strong that the delta dominates the signal.
        target_ratio = 0.20  # 20% of a_t's RMS — clearly above noise floor
        pre_rms = float(np.sqrt(np.mean(A_t_pre_all ** 2)) + 1e-12)
        delta_rms = float(np.sqrt(np.mean(delta_a ** 2)) + 1e-12)
        delta_a = delta_a * (pre_rms / delta_rms) * target_ratio
        A_t_post_all = A_t_pre_all + alpha * delta_a
        # Consistency check (warn level): top-r singular values of δa_t
        # should be ≥ noise floor × 10.
        try:
            from sklearn.decomposition import TruncatedSVD
            k_max = min(2 * r, A_t_pre_all.shape[0] - 1, A_t_pre_all.shape[1])
            tsvd = TruncatedSVD(n_components=k_max, random_state=seed)
            tsvd.fit(delta_a - delta_a.mean(axis=0))
            S_d = tsvd.singular_values_
            noise_floor = float(np.median(S_d[len(S_d) // 2:]))
            if S_d[0] < 10 * noise_floor:
                logger.warning(
                    "synthetic_warn: LoRA delta too weak (σ₁=%.3e, noise=%.3e)",
                    S_d[0], noise_floor,
                )
        except Exception:
            pass
    else:
        A_t_post_all = A_t_pre_all

    # Pre/post split: the post set is the full n_samples (so the dispatcher
    # can consume n_steps * batch_size post samples).  The pre set is a
    # paired subset of *the same Z basis* (recomputed without the LoRA delta)
    # of size target_n = baseline_steps * batch_size; this keeps a_t_pre and
    # a_t_post equal-shaped so the rank/direction fingerprints can compute
    # SVD on Δa_t ∈ R^{target_n × D}.
    target_n = max(0, int(baseline_steps) * int(batch_size))
    target_n = min(target_n, n_samples // 2, n_samples - n_samples // 2)
    target_n = max(target_n, 3 * (lora_rank + 1))  # ensure > 3r samples for SVD
    target_n = min(target_n, n_samples // 2)
    a_t_pre = A_t_pre_all[:target_n]
    A_t = A_t_post_all[:n_samples]  # dispatch full post window

    # V_y rows: take the row corresponding to the (synthetic) argmax of Z_t
    # along the full post window.
    y_pred_all = Z_t.argmax(axis=-1)
    y_pred_post = y_pred_all
    Z_t_post = Z_t
    H_U_post = H_U
    V_y = V[y_pred_post].astype(np.float32)

    # g_accum = a_t - V_y; PRG mask is supposed to be removed by M's
    # decryption + s_share addition, so the result is a near-zero-centred
    # Gaussian (which is what the secure baseline should look like).
    G = (A_t - V_y).astype(np.float32)

    # result_S = scale * a_t - r_t (S-side backward intermediate)
    a_scaled = np.round(A_t * bfv_scale).astype(np.int64)
    # Generate the PRG mask with **truly random per-sample seeds** so the
    # resulting r_t is independent of the sample index.  This avoids the
    # spurious correlation that would arise if r_t were a deterministic
    # function of t_flat (in which case the interleaved label ordering
    # would make r_t correlated with the post-shuffle label).
    r_t = np.zeros((len(A_t), hidden_dim), dtype=np.int64)
    random_t_flat = rng.integers(0, 2**31 - 1, size=len(A_t))
    for i, t_rand in enumerate(random_t_flat):
        r_t[i] = prg.generate_mask_ints(step=0, t_flat=int(t_rand))
    result_S = (a_scaled - r_t).astype(np.float32) / bfv_scale

    # Use the post-window labels for the dispatched arrays (pre window labels
    # are not used by any downstream collector).
    y_post = y
    # Shuffle to randomise ordering, but use an **interleaved** stratified
    # shuffle so that the dispatcher always sees a class-balanced prefix slice
    # regardless of how many samples it consumes (n_steps * batch_size).
    per_class_idx = [rng.permutation(np.where(y_post == c)[0]) for c in range(n_classes)]
    max_per_class = max(len(idx) for idx in per_class_idx)
    stratified_perm = []
    for i in range(max_per_class):
        for c in range(n_classes):
            if i < len(per_class_idx[c]):
                stratified_perm.append(int(per_class_idx[c][i]))
    perm = np.array(stratified_perm, dtype=np.int64)
    return {
        "gradients": G[perm],
        "h_u": H_U_post[perm],
        "activations": A_t[perm],
        "activations_pre": a_t_pre,
        "result_s": result_S[perm],
        "z_t": Z_t_post[perm],
        "labels": y_post[perm],
        "predictions": y_pred_post[perm].tolist(),
        "confidences": probs[perm].max(axis=-1).astype(np.float32).tolist(),
        "n_pre": int(target_n),
        "n_post": int(len(A_t)),
    }


def _try_load_real_V(vocab_size: int, hidden_dim: int, rng: np.random.Generator,
                     model_path: str = HF_MODEL_DEFAULT) -> np.ndarray:
    """Try to load the embedding matrix from the local Llama cache.

    Falls back to a random Gaussian matrix when the snapshot is missing —
    the synthetic data is then *less* realistic but the attacks still
    operate on the right shapes.
    """
    try:
        from safetensors.torch import load_file
        st_path = os.path.join(model_path, "model.safetensors")
        if not os.path.exists(st_path):
            st_path = os.path.join(model_path, "pytorch_model.bin")
        if os.path.exists(st_path):
            weights = load_file(st_path)
            embed_key = next((k for k in weights if "embed" in k.lower() and "weight" in k),
                             None)
            if embed_key is not None:
                V = weights[embed_key].float().cpu().numpy()
                if V.shape == (vocab_size, hidden_dim):
                    return V
    except Exception:
        pass
    return rng.normal(0.0, 0.01, size=(vocab_size, hidden_dim)).astype(np.float32)


# --------------------------------------------------------------------------- #
#  Simple protocol init (TREC-QC only, no GPU)
# --------------------------------------------------------------------------- #
def _init_protocol_simple(cfg: argparse.Namespace):
    """Load TREC-QC for the synthetic fallback path."""
    train_samples, val_samples = _load_trecqc_dataset(cfg.data_dir, cfg.seed)
    return {
        "train_samples": train_samples,
        "val_samples": val_samples,
        "test_samples": [],
        "use_gpu": False,
    }


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = build_parser()
    cfg = parser.parse_args()

    attack_ids = [a.strip().upper() for a in cfg.attacks.split(",") if a.strip()]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = cfg.output_dir
    cfg.output_dir = os.path.join(output_base, f"run_{timestamp}")

    logger.info("=" * 60)
    logger.info("SLG-HE-PIR Attack Test Suite v2.0")
    logger.info("=" * 60)
    logger.info("Attacks: %s", attack_ids)
    logger.info("N steps: %d", cfg.n_steps)
    logger.info("Output: %s", cfg.output_dir)
    logger.info("=" * 60)

    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, "dumps"), exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, "logs"), exist_ok=True)

    log_file = os.path.join(cfg.output_dir, "attack_test.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logging.getLogger("attack_suite").addHandler(file_handler)
    logger.info("Log file: %s", log_file)

    all_verdicts = []
    protocol_info = None

    # --------------------------------------------------------------------------- #
    #  Phase 1: Protocol initialisation
    # --------------------------------------------------------------------------- #
    if not cfg.skip_protocol_init and not cfg.use_synthetic:
        try:
            protocol_info = _init_protocol_with_gpu(cfg)
            logger.info("GPU protocol initialisation successful!")
        except Exception as e:
            logger.warning("GPU protocol init failed: %s", e)
            logger.warning("Falling back to synthetic data mode")
            import traceback
            traceback.print_exc()
            protocol_info = None
    else:
        logger.info("Using synthetic data mode (--use_synthetic or --skip_protocol_init)")

    if protocol_info is None:
        try:
            protocol_info = _init_protocol_simple(cfg)
        except Exception as e:
            logger.error("Protocol init failed: %s", e)
            return 1

    # --------------------------------------------------------------------------- #
    #  Phase 2: Attack data collection
    # --------------------------------------------------------------------------- #
    logger.info("=" * 60)
    logger.info("Phase 2: Attack Data Collection")
    logger.info("=" * 60)

    # Register attack modules
    attack_modules = {}
    for aid in attack_ids:
        module = _get_attack_module(aid, cfg)
        if module:
            attack_modules[aid] = module
            logger.info("Registered attack: %s (%s)", aid, module.ATTACK_NAME)

    use_gpu = protocol_info.get("use_gpu", False) and protocol_info.get("protocol") is not None

    if use_gpu and not cfg.use_synthetic:
        protocol = protocol_info["protocol"]
        try:
            attack_data = _collect_attack_data_from_protocol(protocol, protocol_info, cfg, attack_modules)
            _dispatch_attack_data(attack_modules, attack_data, cfg)
        except Exception as e:
            logger.error("GPU data collection failed: %s", e)
            import traceback
            traceback.print_exc()
            use_gpu = False

    if not use_gpu:
        # Synthetic fallback path: same shape, label-free baseline
        logger.info("Generating synthetic attack data (label-free baseline) …")
        hidden_dim = cfg.bfv_hidden_dim
        n_total = max(cfg.n_steps, 60) * cfg.batch_size
        syn = generate_synthetic_attack_data(
            n_samples=n_total,
            hidden_dim=hidden_dim,
            vocab_size=cfg.vocab_size,
            n_classes=6,
            seed=cfg.seed,
            bfv_scale=cfg.bfv_scale,
            model_path=cfg.hf_model,
            lora_rank=cfg.m2_lora_rank,
            lora_strength=cfg.m2_lora_inject_strength,
            baseline_steps=cfg.m2_baseline_steps,
            batch_size=cfg.batch_size,
        )
        attack_data = {
            "gradients": syn["gradients"],
            "h_u": syn["h_u"],
            "activations": syn["activations"],
            "activations_pre": syn["activations_pre"],
            "result_s": syn["result_s"],
            "z_t": syn["z_t"],
            "predictions": syn["predictions"],
            "confidences": syn["confidences"],
            "labels": syn["labels"],
            "eval_labels": syn["labels"][: cfg.n_eval_steps * cfg.batch_size].tolist(),
            "texts": [f"synthetic_{i}" for i in range(cfg.n_eval_steps * cfg.batch_size)],
            "n_steps": cfg.n_steps,
            "n_pre": syn["n_pre"],
            "n_post": syn["n_post"],
        }
        _dispatch_attack_data(attack_modules, attack_data, cfg)

    # --------------------------------------------------------------------------- #
    #  Phase 3: Run attacks
    # --------------------------------------------------------------------------- #
    logger.info("=" * 60)
    logger.info("Phase 3: Running Attacks")
    logger.info("=" * 60)

    for aid, module in attack_modules.items():
        logger.info("Running %s …", aid)
        try:
            verdicts = module.run()
            all_verdicts.extend(verdicts)
            for v in verdicts:
                logger.info("  %s", v.summary())
        except Exception as e:
            logger.error("  %s FAILED: %s", aid, e)
            import traceback
            traceback.print_exc()

    # --------------------------------------------------------------------------- #
    #  Phase 4: Generate reports
    # --------------------------------------------------------------------------- #
    logger.info("=" * 60)
    logger.info("Phase 4: Generating Reports")
    logger.info("=" * 60)

    results = {
        "attack_results": [v.to_dict() for v in all_verdicts],
        "metadata": {
            "suite_version": "2.0",
            "attacks_run": attack_ids,
            "n_steps": cfg.n_steps,
            "n_eval_steps": cfg.n_eval_steps,
            "batch_size": cfg.batch_size,
            "seed": cfg.seed,
            "project_root": cfg.project_root,
            "hf_model": cfg.hf_model,
            "data_dir": cfg.data_dir,
            "use_gpu": use_gpu,
            "timestamp": timestamp,
        },
    }

    json_path = os.path.join(cfg.output_dir, "attack_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info("JSON results: %s", json_path)

    std_dir = OUTPUT_DIR_DEFAULT
    os.makedirs(std_dir, exist_ok=True)
    summary = {
        "timestamp": timestamp,
        "attacks": attack_ids,
        "n_steps": cfg.n_steps,
        "gpu_used": use_gpu,
        "verdicts": {
            "LEAK_DETECTED": sum(1 for v in all_verdicts if v.verdict == "LEAK_DETECTED"),
            "PRIVACY_PRESERVED": sum(1 for v in all_verdicts if v.verdict == "PRIVACY_PRESERVED"),
            "INCONCLUSIVE": sum(1 for v in all_verdicts if v.verdict == "INCONCLUSIVE"),
        },
        "output_dir": cfg.output_dir,
    }
    summary_path = os.path.join(std_dir, f"summary_{timestamp}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary: %s", summary_path)

    import shutil
    log_dest = os.path.join(std_dir, f"log_{timestamp}.log")
    if os.path.exists(log_file):
        shutil.copy(log_file, log_dest)
        logger.info("Log copied: %s", log_dest)

    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    leak_count = sum(1 for v in all_verdicts if v.verdict == "LEAK_DETECTED")
    safe_count = sum(1 for v in all_verdicts if v.verdict == "PRIVACY_PRESERVED")
    inconclusive_count = sum(1 for v in all_verdicts if v.verdict == "INCONCLUSIVE")
    logger.info("GPU Used: %s", "Yes" if use_gpu else "No (synthetic data)")
    logger.info("Total verdicts: %d", len(all_verdicts))
    logger.info("  LEAK_DETECTED:     %d", leak_count)
    logger.info("  PRIVACY_PRESERVED: %d", safe_count)
    logger.info("  INCONCLUSIVE:      %d", inconclusive_count)
    if leak_count > 0:
        logger.warning("WARNING: %d attack(s) detected potential privacy leakage!", leak_count)
    else:
        logger.info("All attacks indicate privacy is preserved.")
    logger.info("=" * 60)
    logger.info("Output directory: %s", cfg.output_dir)
    logger.info("=" * 60)

    return 0 if leak_count == 0 else 1


def _dispatch_attack_data(attack_modules: Dict, attack_data: Dict, cfg: argparse.Namespace) -> None:
    """Distribute collected arrays to the four attack modules via collect()."""
    G = attack_data["gradients"]
    H_U = attack_data["h_u"]
    A_t = attack_data["activations"]
    R_S = attack_data["result_s"]
    Z_t = attack_data["z_t"]
    predictions = attack_data["predictions"]
    confidences = attack_data["confidences"]
    y = attack_data["labels"]
    n_steps = attack_data["n_steps"]
    bs = cfg.batch_size

    for aid, module in attack_modules.items():
        if aid == "L1" and len(G) > 0:
            n_batches = min(n_steps, len(G) // bs) if len(G) >= bs else 1
            for i in range(n_batches):
                sl = slice(i * bs, (i + 1) * bs)
                batch_grad = G[sl]
                batch_h_u = H_U[sl] if len(H_U) >= (i + 1) * bs else None
                batch_labels = y[sl].tolist()
                module.collect({
                    "g_accum": batch_grad,
                    "H_U": batch_h_u,
                    "token_labels": batch_labels,
                })
        elif aid == "L2" and len(A_t) > 0:
            n_batches = min(n_steps, len(A_t) // bs) if len(A_t) >= bs else 1
            for i in range(n_batches):
                sl = slice(i * bs, (i + 1) * bs)
                batch_act = A_t[sl]
                batch_r_s = R_S[sl] if len(R_S) >= (i + 1) * bs else None
                batch_labels = y[sl].tolist()
                module.collect({
                    "a_t": batch_act,
                    "result_S": batch_r_s,
                    "token_labels": batch_labels,
                })
        elif aid == "M1":
            # Per-sample predictions
            n_pred = len(predictions)
            texts = attack_data.get("texts") or [f"sample_{i}" for i in range(n_pred)]
            for i in range(n_pred):
                module.collect({
                    "s_prediction": predictions[i],
                    "s_confidence": confidences[i] if i < len(confidences) else 0.5,
                    "label": y[i] if i < len(y) else 0,
                    "text": texts[i] if i < len(texts) else f"sample_{i}",
                })
        elif aid == "M2" and len(A_t) > 0:
            # When the GPU path has already populated `m2_module._a_t` /
            # `_a_t_pre` via direct per-step `collect()` calls (built to
            # preserve the pre-LoRA warmup window), skip the secondary
            # dispatch entirely — otherwise it would overwrite the warmup
            # routing with a flat re-drain of `activations_pre`.
            if len(getattr(module, "_a_t", [])) > 0:
                continue
            n_batches = min(n_steps, len(A_t) // bs) if len(A_t) >= bs else 1
            a_t_pre_full = attack_data.get("activations_pre")
            n_pre = int(attack_data.get("n_pre", 0) or 0)
            # Drain all pre samples in the first batch (they are a fixed
            # baseline dataset, not a per-step quantity).
            pre_drained = False
            for i in range(n_batches):
                sl = slice(i * bs, (i + 1) * bs)
                batch_act = A_t[sl]
                batch_z_t = Z_t[sl] if len(Z_t) >= (i + 1) * bs else None
                batch_r_s = R_S[sl] if len(R_S) >= (i + 1) * bs else None
                batch_labels = y[sl].tolist()
                batch_pre = None
                if (
                    not pre_drained
                    and isinstance(a_t_pre_full, np.ndarray)
                    and len(a_t_pre_full) > 0
                ):
                    batch_pre = a_t_pre_full
                    pre_drained = True
                # Mark the last batch as "outside baseline" so any straggler
                # `a_t` rows for the post window don't accidentally get
                # written into `_a_t_pre`.
                if hasattr(module, "_in_baseline_window"):
                    module._in_baseline_window = False
                module.collect({
                    "a_t": batch_act,
                    "a_t_pre": batch_pre,
                    "Z_t": batch_z_t,
                    "result_S": batch_r_s,
                    "token_labels": batch_labels,
                })


if __name__ == "__main__":
    import torch  # noqa: F401  (used by sub-functions)
    sys.exit(main())
