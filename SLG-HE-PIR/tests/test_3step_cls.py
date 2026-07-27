#!/usr/bin/env python3
"""
3-Step CLS Validation Test
==========================
Runs exactly 3 training steps via SLG-HE-PIR HeterogeneousProtocol (classification),
then runs a full validation epoch to measure parse_failure rate and accuracy.

This is used to verify that the `generate_predictions` fix (task-aware 7-class
projection) correctly outputs "a)".."g)" strings that pass through the BioTriplex
metric parsers without 134/134 failures.

Usage:
    python test_3step_cls.py --data_path /path/to/biotriplex --output_dir /path/to/out
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Bootstrap project root
_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.biotriplex_dataset import build_biotriplex_dataset
from src.parties.heterogeneous_protocol import HeterogeneousProtocol
from src.core.bfv_privselect_v2_adapter import BFVPrivSelectV2Backend
from src.core.s3pir_hints import HintTable
from src.training.trainer import TrainerConfig
from src.training.biotriplex_metrics import compute_classification_metrics
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("test_3step_cls")


def _load_V_for_db(model_path: str, vocab_size: int, hidden_dim: int) -> np.ndarray:
    from safetensors.torch import load_file
    snap = Path(model_path)
    idx_path = snap / "model.safetensors.index.json"
    if idx_path.exists():
        with open(idx_path) as f:
            index = json.load(f)
        lm_head_files = sorted({
            str(snap / fn) for k, fn in index["weight_map"].items() if "lm_head" in k
        })
    else:
        lm_head_files = sorted(snap.glob("*.safetensors"))
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
        raise FileNotFoundError(f"lm_head.weight not found in {model_path}")
    return V


def _serialize_sk(bfv_backend) -> bytes:
    from src.core.bfv_privselect_v2_adapter import _seal_to_bytes
    return _seal_to_bytes(bfv_backend._secret_key)


def _serialize_pk(bfv_backend) -> bytes:
    import pickle as _pickle
    return _pickle.dumps({"pk_bytes": bfv_backend.public_key_bytes})


def parse_args():
    p = argparse.ArgumentParser(description="3-step CLS validation test")
    p.add_argument("--data_path", required=True)
    p.add_argument("--hf_model", default="/root/autodl-tmp/hf_cache/Llama-3-1-8B-I")
    p.add_argument("--bfv_cache_dir", default="/root/autodl-tmp/slg-bfv-cache")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--n_steps", type=int, default=3,
                   help="Number of training steps (default: 3)")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--max_seq_length", type=int, default=1024,
                   help="Max sequence length (keep small to avoid OOM)")
    p.add_argument("--log_dir", default=None,
                   help="If omitted, uses ${output_dir}/logs")
    return p.parse_args()


def run(args):
    os.makedirs(args.output_dir, exist_ok=True)
    log_dir = args.log_dir or os.path.join(args.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"3step_test_{int(time.time())}.log")

    # Re-configure root logger to write to file too
    fh = logging.FileHandler(log_file)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logging.getLogger().addHandler(fh)
    logger.info("Logging to %s", log_file)

    # ---- Seed ----
    torch.manual_seed(42)
    np.random.seed(42)

    # ---- Load tokenizer ----
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Load datasets ----
    logger.info("Loading BioTriplex classification dataset ...")
    train_ds = build_biotriplex_dataset(
        task="classification",
        data_dir=args.data_path,
        tokenizer=tokenizer,
        split="train",
        max_length=args.max_seq_length,
        return_neg_relations=False,
    )
    val_ds = build_biotriplex_dataset(
        task="classification",
        data_dir=args.data_path,
        tokenizer=tokenizer,
        split="val",
        max_length=args.max_seq_length,
        return_neg_relations=False,
    )
    logger.info("Datasets: train=%d val=%d", len(train_ds), len(val_ds))

    # Use n_entries from cached bfv_meta to ensure consistency with existing DB
    bfv_meta_path = os.path.join(args.bfv_cache_dir, "bfv_meta.json")
    import json as _json
    if os.path.exists(bfv_meta_path):
        with open(bfv_meta_path) as f:
            bfv_meta = _json.load(f)
        bfv_n_entries = bfv_meta.get("n_entries", tokenizer.vocab_size)
    else:
        bfv_n_entries = tokenizer.vocab_size
    logger.info("Using n_entries=%d (from meta=%s)", bfv_n_entries, bfv_meta_path)

    # ---- BFV backend ----
    pk_cache_path = os.path.join(args.bfv_cache_dir, "bfv_pk.bin")
    pk_path = pk_cache_path if os.path.exists(pk_cache_path) else None
    logger.info("Building BFV backend (pk_path=%s) ...", pk_path)
    bfv_backend = BFVPrivSelectV2Backend(
        n_entries=bfv_n_entries,
        vec_dim=4096,
        shared_seed=os.urandom(32),
        cache_dir=args.bfv_cache_dir,
        poly_degree=4096,
        plain_bits=30,
        scale=10000,
        pk_path=pk_path,
        force_new_keys=(pk_path is None),
    )
    V = _load_V_for_db(args.hf_model, bfv_n_entries, 4096)
    bfv_backend.build_encrypted_database(V, force=False)
    bfv_backend.drop_encrypted_db()

    sk_pem = _serialize_sk(bfv_backend)
    pk_pem = _serialize_pk(bfv_backend)
    bfv_backend._drop_secret_key()
    prg_seed = os.urandom(32)
    logger.info("BFV keys ready; sk_M dropped from main process")

    # ---- Hint table ----
    hints_dir = os.path.join(args.bfv_cache_dir, "s3pir_hints")
    partition_size = 1 << ((bfv_n_entries.bit_length() - 1) // 2)
    if os.path.exists(os.path.join(hints_dir, "hint_table.json")):
        hint_table = HintTable.from_cache_files(hints_dir)
    else:
        hint_table = HintTable(
            n_entries=bfv_n_entries,
            partition_size=partition_size,
            lam=80,
            cache_dir=hints_dir,
        )
        hint_table.compute_main_hints_skeleton()
        hint_table.compute_backup_hints_skeleton()
        hint_table.to_cache_files()
    logger.info("Hint table ready")

    # ---- Worker config ----
    worker_config = {
        "vocab_size": bfv_n_entries,
        "hidden_dim": 4096,
        "poly_degree": 4096,
        "plain_bits": 30,
        "scale": 10000,
        "bfv_cache_dir": args.bfv_cache_dir,
        "lam": 80,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "learning_rate": 1e-4,
        "weight_decay": 0.0,
        "gradient_clip_norm": 1.0,
        "warmup_steps": 200,
        "lr_scheduler": "cosine_with_warmup",
        "batch_size": args.batch_size,
        "max_epochs": 1,
        "n_train_samples": len(train_ds),
        "dump_attack_intermediates": False,
        "attack_dump_dir": os.path.join(log_dir, "attack_dumps"),
        "hf_model_path": args.hf_model,
        "u_layers": 16,
        "m_layers": 16,
        "N_CRYPTO_U_WORKERS": 8,
        "N_CRYPTO_M_WORKERS": 8,
        "N_CRYPTO_S_WORKERS": 1,
        "ENABLE_STEP_PROFILING": True,
        "LOG_DIR": log_dir,
        "use_flash_attention": True,
        "use_sage_attention": True,
        "gradient_checkpointing_style": "reentrant",
        "use_deepspeed_zero": True,
        "zero_stage": 1,
        # DP disabled for this quick test
        "dp_enable": False,
    }

    # ---- Build HeterogeneousProtocol ----
    logger.info("Constructing HeterogeneousProtocol ...")
    protocol = HeterogeneousProtocol(
        u_submodel_path=args.hf_model,
        m_submodel_path=args.hf_model,
        s_lm_head_path=args.hf_model,
        bfv_backend=bfv_backend,
        hint_table=hint_table,
        bfv_sk_pem=sk_pem,
        bfv_pk_pem=pk_pem,
        prg_seed=prg_seed,
        config=worker_config,
    )

    # ---- Run N training steps ----
    logger.info("=" * 60)
    logger.info("Running %d training steps ...", args.n_steps)
    logger.info("=" * 60)

    from src.training.trainer import make_string_safe_collate
    collate_fn = make_string_safe_collate()

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                             num_workers=0, collate_fn=collate_fn)
    step_times = []
    step_losses = []
    it = iter(train_loader)

    for step in range(args.n_steps):
        t0 = time.time()
        batch = next(it)
        result = protocol.step_train_chunked(batch, step, chunk_tokens=1024)
        elapsed_ms = (time.time() - t0) * 1000
        step_times.append(elapsed_ms)
        step_losses.append(float(result.loss))
        logger.info(
            "[Step %d/%d] loss=%.4f time=%.1fms gpu_mem=%.0fMB",
            step + 1, args.n_steps, result.loss, elapsed_ms, result.gpu_mem_mb,
        )

    logger.info("Training steps complete. avg_loss=%.4f avg_step_time=%.1fms",
                np.mean(step_losses), np.mean(step_times))

    # ---- Run validation epoch ----
    logger.info("=" * 60)
    logger.info("Running validation epoch ...")
    logger.info("=" * 60)

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, collate_fn=collate_fn)
    all_predictions = []
    all_labels = []
    all_pred_logits = []
    all_predictions_letters = []
    all_labels_letters = []

    for batch_idx, batch in enumerate(val_loader):
        from src.data.dataset import parse_answer_letter
        result = protocol.step_val(batch, global_step=0)

        preds = result.get("predictions", [])
        labs = result.get("labels", [])
        p_log = result.get("pred_logits")

        all_predictions.extend(preds)
        all_labels.extend(labs)
        if isinstance(p_log, list):
            all_pred_logits.extend(p_log)

        preds_letters = result.get("predictions_letters") or []
        labs_letters = result.get("labels_letters") or []
        if not preds_letters:
            preds_letters = [parse_answer_letter(p) for p in preds]
        if not labs_letters:
            labs_letters = [parse_answer_letter(l) for l in labs]
        all_predictions_letters.extend(preds_letters)
        all_labels_letters.extend(labs_letters)

        logger.info("[Val batch %d] n=%d preds=%s labs=%s",
                    batch_idx, len(preds), preds[:2], labs[:2])

    logger.info("Validation complete: %d samples", len(all_predictions))

    # ---- Compute BioTriplex metrics ----
    bt = compute_classification_metrics(
        all_predictions,
        all_labels,
        pred_logits=all_pred_logits or None,
    )
    m = bt.get("metrics", {})

    # ---- Compute letter-level micro-F1 (paper Table 5 style) ----
    tp = fp = fn = 0
    correct = 0
    for p, g in zip(all_predictions_letters, all_labels_letters):
        p_set = set(p.split(",")) if p else set()
        g_set = set(g.split(",")) if g else set()
        tp += len(p_set & g_set)
        fp += len(p_set - g_set)
        fn += len(g_set - p_set)
        if p_set == g_set and p:
            correct += 1
    n = len(all_predictions_letters) or 1
    micro_p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    micro_r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0
    micro_acc = correct / n

    # ---- Results summary ----
    result_summary = {
        "n_train_steps": args.n_steps,
        "n_val_samples": len(all_predictions),
        "train": {
            "avg_loss": round(float(np.mean(step_losses)), 6),
            "avg_step_time_ms": round(float(np.mean(step_times)), 1),
            "step_losses": [round(float(l), 4) for l in step_losses],
            "step_times_ms": [round(float(t), 1) for t in step_times],
        },
        "val": {
            "n_parse_failures": bt.get("n_parse_failures", 0),
            "parse_failure_rate": round(bt.get("n_parse_failures", 0) / max(n, 1), 4),
            "micro_accuracy": round(micro_acc, 6),
            "micro_f1": round(micro_f1, 6),
            "micro_precision": round(micro_p, 6),
            "micro_recall": round(micro_r, 6),
            "macro_f1": round(m.get("macro_f1", 0.0), 6),
            "weighted_f1": round(m.get("weighted_f1", 0.0), 6),
            "macro_roc_auc_ovr": m.get("macro_roc_auc_ovr"),
            "micro_roc_auc_ovr": m.get("micro_roc_auc_ovr"),
            "has_logits": bt.get("has_logits", False),
            "sample_predictions": all_predictions[:5],
            "sample_labels": all_labels[:5],
            "sample_predictions_letters": all_predictions_letters[:5],
            "sample_labels_letters": all_labels_letters[:5],
            # Full per-sample data for downstream re-evaluation
            "all_predictions": all_predictions,
            "all_labels": all_labels,
            "all_pred_logits": all_pred_logits,
            "all_predictions_letters": all_predictions_letters,
            "all_labels_letters": all_labels_letters,
            "y_true_distribution": bt.get("y_true_distribution"),
            "y_pred_distribution": bt.get("y_pred_distribution"),
            "per_class_metrics": bt.get("per_class_metrics"),
        },
    }

    # Save results
    out_path = os.path.join(args.output_dir, "3step_test_results.json")
    with open(out_path, "w") as f:
        json.dump(result_summary, f, indent=2, ensure_ascii=False)
    logger.info("Results saved → %s", out_path)

    # Print summary
    print("\n" + "=" * 60)
    print("3-STEP CLS VALIDATION TEST SUMMARY")
    print("=" * 60)
    print(f"  Training steps : {args.n_steps}")
    print(f"  Val samples    : {len(all_predictions)}")
    print(f"  Parse failures: {bt.get('n_parse_failures', 0)} / {len(all_predictions)} "
          f"({result_summary['val']['parse_failure_rate']*100:.1f}%)")
    print(f"  Micro-Acc     : {micro_acc:.4f}")
    print(f"  Micro-F1      : {micro_f1:.4f}")
    print(f"  Macro-F1      : {m.get('macro_f1', 0.0):.4f}")
    print(f"  Macro ROC-AUC : {bt.get('macro_roc_auc_ovr')}")
    print(f"  Has logits    : {bt.get('has_logits', False)}")
    print(f"  Sample preds  : {all_predictions[:3]}")
    print(f"  Sample letters: {all_predictions_letters[:3]}")
    print(f"  Sample golds  : {all_labels[:3]}")
    print(f"  Sample gold lt: {all_labels_letters[:3]}")
    print("=" * 60)

    protocol.shutdown()
    return result_summary


if __name__ == "__main__":
    args = parse_args()
    run(args)
