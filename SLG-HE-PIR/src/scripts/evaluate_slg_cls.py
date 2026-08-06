#!/usr/bin/env python3
"""Re-evaluate SLG-HE-PIR classification checkpoints (one per epoch) using
the fixed generate_predictions path.

Each `.pt` checkpoint contains a full LoRA adapter state_dict saved by the
SLG-HE-PIR Trainer. We rebuild a PEFT adapter from it and run the same
7-class projection inference path as the baseline, then write per-epoch
metrics into the same schema as the Baseline CLS output.

Usage:
    python evaluate_slg_cls.py \
        --base_model /root/autodl-tmp/CipherForgeCode/hf_cache/Llama-3-1-8B-I \
        --ckpt_dir /root/autodl-tmp/CipherForgeCode/SLG-HE-PIR/test-data/SLG-test-data/cls-SLG-test-data/_SAVE_20260727_0706 \
        --data_path /root/autodl-tmp/CipherForgeCode/SLG-HE-PIR/datasets/botriplex/Preprocessed\ BioTriplex/ \
        --output_dir /root/autodl-tmp/CipherForgeCode/SLG-HE-PIR/test-data/SLG-test-data/cls-SLG-test-data \
        --epochs 5
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

THIS = Path(__file__).resolve()
PROJECT_ROOT = THIS.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.biotriplex_dataset import (
    ENTITY_TYPES,
    GENERAL_RELATIONS,
    OPTION_LETTERS,
    build_biotriplex_dataset,
)
from src.training.biotriplex_metrics import compute_classification_metrics

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True)
    p.add_argument("--ckpt_dir", required=True,
                   help="Dir containing checkpoint_epoch_XXX.pt")
    p.add_argument("--data_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--max_eval_samples", type=int, default=0,
                   help="0 = use full val set; >0 = sample N")
    p.add_argument("--max_seq_length", type=int, default=2048)
    p.add_argument("--split", default="test")
    return p.parse_args()


def find_ckpt_files(ckpt_dir: str, epochs: int):
    files = []
    for e in range(epochs):
        path = os.path.join(ckpt_dir, f"checkpoint_epoch_{e:03d}.pt")
        if os.path.exists(path):
            files.append((e, path))
    return files


def find_best_ckpt(ckpt_dir: str):
    p = os.path.join(ckpt_dir, "best_checkpoint.pt")
    return p if os.path.exists(p) else None


def get_option_token_ids(tokenizer) -> List[int]:
    ids = []
    for letter in OPTION_LETTERS:
        chosen = None
        for cand in (f"{letter})", letter, f" {letter})", f" {letter}"):
            tids = tokenizer.encode(cand, add_special_tokens=False)
            if len(tids) == 1:
                chosen = tids[0]
                break
        if chosen is None:
            tids = tokenizer.encode(f"{letter})", add_special_tokens=False)
            chosen = tids[0]
        ids.append(chosen)
    return ids


def build_prompt(sample: Dict[str, Any], tokenizer) -> str:
    """Build the prompt expected by BioTriplex classification.

    Mirrors build_biotriplex_dataset formatting for QA prompts.
    """
    instruction = sample.get("instruction") or sample.get("input") or sample.get("context", "")
    question = sample.get("question") or sample.get("query", "")
    options = sample.get("options") or []
    if isinstance(options, str):
        try:
            options = json.loads(options)
        except Exception:
            options = []

    parts = []
    if instruction:
        parts.append(instruction.strip())
    if question:
        parts.append(question.strip())
    for i, opt in enumerate(options):
        letter = OPTION_LETTERS[i] if i < len(OPTION_LETTERS) else chr(ord("a") + i)
        parts.append(f"{letter}) {opt}")
    parts.append("Answer:")
    return "\n".join(parts)


def run_inference_for_ckpt(
    ckpt_path: str,
    base_model_path: str,
    data_path: str,
    split: str,
    max_eval_samples: int,
    max_seq_length: int,
    device: str = "cuda",
) -> Dict[str, Any]:
    print(f"[eval] loading tokenizer + base model from {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    )
    model = model.to(device)
    model.eval()

    # Load LoRA state from checkpoint and inject into a PEFT model
    print(f"[eval] loading LoRA state from {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # SLG checkpoints nest LoRA under party_checkpoints['M']['lora_state']
    if isinstance(state, dict) and "party_checkpoints" in state:
        lora_state = state["party_checkpoints"].get("M", {}).get("lora_state", {})
    else:
        lora_state = state
    if not lora_state:
        raise RuntimeError(f"No LoRA state found in {ckpt_path}")

    # The SLG trainer uses bare "layers.X.self_attn.Y_proj.lora_A/B" keys.
    # PEFT expects "base_model.model.model.layers.X.self_attn.Y_proj.lora_A.weight"
    # We need to remap them.
    remapped = {}
    rank = None
    target_modules = set()
    for k, v in lora_state.items():
        # Detect rank from shape
        if hasattr(v, "shape") and v.ndim == 2 and "lora_A" in k:
            r = v.shape[0]
            if rank is None or r != rank:
                rank = r
            # Extract target module
            tm = k.split(".lora_")[0].split(".")[-1]
            target_modules.add(tm)
        new_key = f"base_model.model.model.{k}.weight"
        remapped[new_key] = v
    if rank is None:
        rank = 8
    alpha = rank * 2
    print(f"[eval] remapped {len(remapped)} LoRA keys, rank={rank}, alpha={alpha}, targets={target_modules}")

    lora_cfg = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.05,
        target_modules=list(target_modules) if target_modules else [
            "q_proj", "k_proj", "v_proj", "o_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    res = set_peft_model_state_dict(model, remapped, adapter_name="default")
    if isinstance(res, tuple):
        missing, unexpected = res
        print(f"[eval] PEFT load: missing={len(missing)} unexpected={len(unexpected)}")
        if unexpected:
            print(f"  first unexpected: {unexpected[0]}")

    option_token_ids = get_option_token_ids(tokenizer)
    option_token_ids_t = torch.tensor(option_token_ids, dtype=torch.long, device=device)

    # Load dataset
    ds = build_biotriplex_dataset(
        task="classification",
        data_dir=data_path,
        tokenizer=tokenizer,
        split=split,
        max_length=max_seq_length,
    )
    samples = ds  # dataset object supports __len__ and __getitem__
    n_total = len(samples)
    if max_eval_samples and n_total > max_eval_samples:
        # Subsample
        samples = [samples[i] for i in range(min(max_eval_samples, n_total))]
    print(f"[eval] evaluating on {len(samples)} samples (of {n_total})")

    predictions = []
    labels = []
    pred_logits_all = []

    t0 = time.time()
    for i, sample in enumerate(samples):
        prompt_text = sample["prompt"]
        gold = sample.get("output_text") or sample.get("gold_letter") or sample.get("output") or ""
        if isinstance(gold, list):
            gold = gold[0] if gold else ""
        gold = str(gold).strip()

        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True,
                           max_length=max_seq_length).to(device)
        with torch.no_grad():
            out = model(**inputs)
        last_logits = out.logits[0, -1, :].float()
        option_logits = last_logits[option_token_ids_t].cpu().tolist()
        pred_idx = int(torch.tensor(option_logits).argmax().item())
        pred_letter = OPTION_LETTERS[pred_idx]
        pred_str = f"{pred_letter})"

        predictions.append(pred_str)
        labels.append(gold)
        pred_logits_all.append(option_logits)

        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            print(f"[eval] {i+1}/{len(samples)} in {elapsed:.1f}s")

    # Free GPU
    del model
    torch.cuda.empty_cache()

    metrics_result = compute_classification_metrics(
        predictions=predictions,
        labels=labels,
        pred_logits=pred_logits_all,
    )
    metrics_result["checkpoint"] = ckpt_path
    metrics_result["n_samples"] = len(samples)
    metrics_result["elapsed_s"] = time.time() - t0
    return metrics_result


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    logs_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    ckpts = find_ckpt_files(args.ckpt_dir, args.epochs)
    if not ckpts:
        # Try absolute file paths
        ckpts = []
        for e in range(args.epochs):
            candidates = glob.glob(os.path.join(args.ckpt_dir, f"checkpoint_epoch_{e:03d}.*"))
            if candidates:
                ckpts.append((e, candidates[0]))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint files in {args.ckpt_dir}")

    all_metrics = []
    for epoch, ckpt in ckpts:
        print(f"\n===== Evaluating epoch {epoch} checkpoint: {ckpt} =====")
        m = run_inference_for_ckpt(
            ckpt_path=ckpt,
            base_model_path=args.base_model,
            data_path=args.data_path,
            split=args.split,
            max_eval_samples=args.max_eval_samples,
            max_seq_length=args.max_seq_length,
        )
        m["epoch"] = epoch
        all_metrics.append(m)

        # Save per-epoch metrics
        epoch_file = os.path.join(logs_dir, f"epoch_{epoch:03d}_evaluate_metrics.json")
        with open(epoch_file, "w") as f:
            json.dump(m, f, indent=2)
        print(f"[eval] epoch {epoch} metrics: micro_f1={m['metrics'].get('micro_accuracy'):.4f} "
              f"macro_f1={m['metrics'].get('macro_f1'):.4f} "
              f"auc={m['metrics'].get('macro_roc_auc_ovr'):.4f} "
              f"parse_failures={m['n_parse_failures']}")

    # Write a unified epoch_metrics.jsonl in the same schema as baseline
    metrics_jsonl = os.path.join(args.output_dir, "epoch_metrics.jsonl")
    with open(metrics_jsonl, "w") as f:
        for m in all_metrics:
            em = m["metrics"]
            record = {
                "epoch": m["epoch"],
                "timestamp": time.time(),
                "elapsed_s": m.get("elapsed_s"),
                "train_loss": None,
                "train_steps": 734,
                "avg_step_time_ms": None,
                "avg_gpu_mem_mb": None,
                "val_ce_loss": None,
                "val_samples": m.get("n_samples"),
                "val_bt_micro_f1": em.get("micro_f1"),
                "val_bt_macro_f1": em.get("macro_f1"),
                "val_bt_weighted_f1": em.get("weighted_f1"),
                "val_bt_multilabel_f1_samples": em.get("multilabel_f1_samples"),
                "val_bt_multilabel_f1_macro": em.get("multilabel_f1_macro"),
                "val_bt_multilabel_f1_micro": em.get("multilabel_f1_micro"),
                "val_bt_macro_roc_auc": em.get("macro_roc_auc_ovr"),
                "val_bt_micro_roc_auc": em.get("micro_roc_auc_ovr"),
                "val_bt_n_parse_failures": m.get("n_parse_failures"),
                "val_micro_accuracy": em.get("micro_accuracy"),
                "val_macro_precision": em.get("macro_precision"),
                "val_macro_recall": em.get("macro_recall"),
                "per_class": m.get("per_class_metrics"),
            }
            f.write(json.dumps(record) + "\n")
    print(f"[eval] wrote {metrics_jsonl}")

    # Also copy to logs/
    logs_metrics = os.path.join(logs_dir, "epoch_metrics.jsonl")
    shutil.copy(metrics_jsonl, logs_metrics)
    print(f"[eval] wrote {logs_metrics}")


if __name__ == "__main__":
    main()
