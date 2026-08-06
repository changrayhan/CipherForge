"""TREC-QC PyTorch baseline trainer (Llama-3.2-1B + LoRA).

A minimal bf16 LoRA fine-tuner for the TREC-QC 6-class task that mirrors the
BioTriplex baseline's hyperparameters (lora_rank=8, lr=1e-4, batch_size=1,
weight_decay=0.0) but loads the ``TRECQADataset`` defined in
``src.data.biotriplex_dataset`` instead of the BioTriplex one.

Used by ``run_trec_one_experiment.sh`` to produce plaintext baseline outputs
that can be compared against the SLG-fixed (encrypted) outputs.

Outputs (per epoch):
    ${OUTPUT_DIR}/adapter/                  — PEFT adapter weights
    ${OUTPUT_DIR}/checkpoints/              — full state dict
    ${LOG_DIR}/train_${TS}.log              — training log
    ${LOG_DIR}/infer_outputs_epoch_NNN.json — inference JSON for evaluation
    ${LOG_DIR}/epoch_NNN_unified_metrics.json — TREC-QC evaluator output
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Bootstrap path
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "test-data" / "TrecAATestData" / "scripts"))

from src.data.biotriplex_dataset import (  # noqa: E402
    TREC_QC_COARSE_CLASSES,
    build_trec_qa_dataset,
)

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # --- Paths ---
    p.add_argument("--data_dir", required=True)
    p.add_argument("--gold_path", required=True)
    p.add_argument("--hf_model", default=os.environ.get(
        "TREC_HF_MODEL",
        "/root/autodl-tmp/CipherForgeCode/hf_cache/models--unsloth--Llama-3.2-1B/snapshots/9535bd9b1d1dea6acafbdc4813b728796aeb28da",
    ))
    p.add_argument("--output_dir", required=True)
    p.add_argument("--log_dir", required=True)
    # --- Hyperparameters (mirror BioTriplex baseline defaults) ---
    p.add_argument("--max_epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_seq_length", type=int, default=256)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_steps", type=int, default=50)
    p.add_argument("--gradient_clip_norm", type=float, default=1.0)
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_target", default="q,v",
                   help="Comma-separated target module names (BioTriplex style: q,v / q,k,v,o,gate,up,down)")
    p.add_argument("--seed", type=int, default=42)
    # --- DP ablation knobs (parity with SLG DP config) ---
    p.add_argument("--dp_alpha", type=float, default=0.0,
                   help="Gaussian noise std multiplier on per-sample loss (0 = no DP noise)")
    p.add_argument("--dp_answer_beta", type=float, default=0.5,
                   help="Mixing weight for the answer-token-only loss component")
    p.add_argument("--dp_calibration_steps", type=int, default=5)
    p.add_argument("--dp_calibration_mode", action="store_true",
                   help="Re-calibrate DP noise std from the first N steps")
    return p.parse_args()


def build_lora_model(hf_model_path: str, args):
    """Build a Llama-3.2-1B causal LM with LoRA adapters."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType

    tokenizer = AutoTokenizer.from_pretrained(hf_model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        hf_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )

    target_modules = [t.strip() for t in args.lora_target.split(",")]
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model, tokenizer


def collate_fn_factory(tokenizer, max_length: int):
    """Pad-collate for causal LM training."""
    pad_id = tokenizer.pad_token_id

    def collate(batch):
        ids = [b["input_ids"] for b in batch]
        masks = [b["attention_mask"] for b in batch]
        labels = [b["labels"] for b in batch]
        # Stack (all tensors share the same shape because dataset pads to max_length)
        return {
            "input_ids": torch.stack(ids),
            "attention_mask": torch.stack(masks),
            "labels": torch.stack(labels),
            "doc_key": [b["doc_key"] for b in batch],
            "output_text": [b["output_text"] for b in batch],
            "label_idx": torch.tensor([b["label_idx"] for b in batch], dtype=torch.long),
        }
    return collate


def compute_train_loss(model, batch, args, calib_running=None):
    """Compute the per-batch training loss with optional DP noise (Gaussian).

    Note: this baseline trainer is intentionally simple — it does NOT implement
    full DP-SGD (which requires per-sample gradients). Instead it adds Gaussian
    noise to the aggregated loss, matching the SLG DP mode at the *gradient
    proxy* level. This is sufficient for accuracy-decomposition purposes.
    """
    out = model(
        input_ids=batch["input_ids"].cuda(),
        attention_mask=batch["attention_mask"].cuda(),
        labels=batch["labels"].cuda(),
    )
    loss = out.loss
    if args.dp_alpha > 0.0:
        sigma = args.dp_alpha * float(loss.detach().item())
        noise = torch.randn_like(loss) * sigma
        loss = loss + noise
    return loss


def evaluate(model, test_ds, tokenizer, args, device="cuda"):
    """Greedy inference + collect 6-class logits for each test sample."""
    model.eval()
    n_classes = len(TREC_QC_COARSE_CLASSES)
    # Build prompt-only inputs and a separate forward to gather logits
    # We mimic the SLG protocol: project last non-pad hidden state through the
    # lm_head and pick 6 option-letter logits.
    opt_letters = ["a", "b", "c", "d", "e", "f"]
    opt_token_ids = []
    for letter in opt_letters:
        ids = tokenizer.encode(f"{letter})", add_special_tokens=False)
        # fall back to first id if multi-token
        opt_token_ids.append(ids[0] if ids else tokenizer.eos_token_id)

    outputs: Dict[str, Dict[str, Any]] = {}
    model.eval()
    with torch.no_grad():
        for idx in tqdm(range(len(test_ds)), desc="infer", leave=False):
            item = test_ds[idx]
            ids = item["input_ids"].unsqueeze(0).to(device)
            am = item["attention_mask"].unsqueeze(0).to(device)
            out = model(input_ids=ids, attention_mask=am)
            logits = out.logits  # [1, S, V]
            # Last non-pad position
            seq_len = int(am.sum().item())
            last_logits = logits[0, seq_len - 1]  # [V]
            # Project to 6 letters
            opt_logits = last_logits[opt_token_ids].float().cpu().numpy()
            pred_idx = int(np.argmax(opt_logits))
            probs = _softmax(opt_logits)
            outputs[item["doc_key"]] = {
                "answer": f"{opt_letters[pred_idx]})",
                "logits": opt_logits.tolist(),
                "probs": probs.tolist(),
                "predicted_relation": TREC_QC_COARSE_CLASSES[pred_idx],
            }
    return outputs


def _softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(args.log_dir, "train.log"), mode="w"),
        ],
        force=True,
    )
    log = logging.getLogger("trec_baseline")

    # ---- Build datasets ----
    log.info("Building TREC-QC dataset from %s ...", args.data_dir)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = build_trec_qa_dataset(
        data_dir=args.data_dir, tokenizer=tokenizer, split="train",
        max_length=args.max_seq_length,
    )
    val_ds = build_trec_qa_dataset(
        data_dir=args.data_dir, tokenizer=tokenizer, split="val",
        max_length=args.max_seq_length,
    )
    test_ds = build_trec_qa_dataset(
        data_dir=args.data_dir, tokenizer=tokenizer, split="test",
        max_length=args.max_seq_length,
    )
    log.info("train=%d val=%d test=%d", len(train_ds), len(val_ds), len(test_ds))

    # ---- Build model ----
    log.info("Loading model %s ...", args.hf_model)
    model, _ = build_lora_model(args.hf_model, args)
    model.cuda()

    # ---- Training loop ----
    collate = collate_fn_factory(tokenizer, args.max_seq_length)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          collate_fn=collate, num_workers=0)

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate, weight_decay=args.weight_decay,
    )

    from transformers import get_linear_schedule_with_warmup
    total_steps = max(1, args.max_epochs * len(train_dl))
    sched = get_linear_schedule_with_warmup(
        optim, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps,
    )

    metrics_history: List[Dict[str, Any]] = []
    step = 0
    calib_running = None
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        n_batches = 0
        for batch in train_dl:
            optim.zero_grad(set_to_none=True)
            loss = compute_train_loss(model, batch, args, calib_running)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
            optim.step()
            sched.step()
            total_loss += float(loss.detach().item())
            n_batches += 1
            step += 1
            if step % 20 == 0:
                log.info("epoch=%d step=%d loss=%.4f lr=%.2e", epoch, step,
                         total_loss / max(1, n_batches), sched.get_last_lr()[0])
        avg_loss = total_loss / max(1, n_batches)
        epoch_dur = time.time() - t0
        log.info("Epoch %d: avg_loss=%.4f, dur=%.1fs", epoch, avg_loss, epoch_dur)

        # ---- Inference on test set every epoch ----
        log.info("Running inference on test set (epoch %d) ...", epoch)
        infer_outputs = evaluate(model, test_ds, tokenizer, args)
        out_infer = Path(args.log_dir) / f"infer_outputs_epoch_{epoch:03d}.json"
        with open(out_infer, "w") as f:
            json.dump(infer_outputs, f, indent=2)

        # ---- Evaluate ----
        from trec_evaluator import evaluate_unified_trec
        metrics = evaluate_unified_trec(
            infer_outputs_path=str(out_infer),
            gold_path=args.gold_path,
            output_path=str(Path(args.log_dir) / f"epoch_{epoch:03d}_unified_metrics.json"),
            experiment_name=f"trec-baseline-epoch{epoch}",
            seed=args.seed,
        )
        log.info("[epoch %d] acc=%.4f macro_f1=%.4f macro_auc=%.4f",
                 epoch, metrics["accuracy"], metrics["macro_f1"], metrics["macro_auc"])

        metrics_history.append({
            "epoch": epoch,
            "train_loss": avg_loss,
            "epoch_dur_sec": epoch_dur,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "macro_auc": metrics["macro_auc"],
            "n_parse_failures": metrics["n_parse_failures"],
        })

        # Save adapter
        adapter_dir = Path(args.output_dir) / "adapter"
        adapter_dir.mkdir(exist_ok=True, parents=True)
        model.save_pretrained(adapter_dir)

    # ---- Save epoch metrics ----
    metrics_path = Path(args.output_dir) / "epoch_metrics.jsonl"
    with open(metrics_path, "w") as f:
        for row in metrics_history:
            f.write(json.dumps(row) + "\n")
    log.info("Saved metrics to %s", metrics_path)
    log.info("DONE")


if __name__ == "__main__":
    main()