"""
Test-set evaluation with standard forward (no PIR).

After training, load the best checkpoint and evaluate on the held-out test set
using a merged model (LoRA + base). This uses standard forward — no privacy
constraints since we're evaluating, not training.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

__all__ = ["evaluate_test_set", "merge_lora_and_evaluate"]

logger = logging.getLogger(__name__)


def evaluate_test_set(
    checkpoint_path: str,
    test_ds: Any,
    model_path: str = "/root/autodl-tmp/hf_cache/Llama-3-1-8B-I",
    max_new_tokens: int = 128,
    device: str = "cuda",
    batch_size: int = 1,
) -> Dict[str, Any]:
    """Evaluate merged LoRA + base model on the test set.

    Loads the best checkpoint, merges LoRA weights into base model,
    then performs standard forward evaluation on the test set.

    Args:
        checkpoint_path: Path to best_checkpoint.pt
        test_ds: BioTriplexQADataset
        model_path: HF model cache path
        max_new_tokens: Max tokens to generate per sample
        device: "cuda" or "cpu"
        batch_size: Batch size for evaluation

    Returns:
        Dict with all computed metrics
    """
    logger.info("Loading best checkpoint from %s", checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    # Load base model + LoRA adapter
    model, tokenizer = load_merged_model(model_path, ckpt, device)

    model.eval()

    all_predictions = []
    all_labels = []
    all_inputs = []

    logger.info("Evaluating %d test samples", len(test_ds))

    with torch.no_grad():
        for i in range(len(test_ds)):
            sample = test_ds[i]
            input_text = sample["prompt"]
            label = sample.get("label", sample["output_text"])

            input_ids = tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(device)

            output_ids = model.generate(
                **input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

            prediction = tokenizer.decode(output_ids[0], skip_special_tokens=True)

            all_predictions.append(prediction)
            all_labels.append(label)
            all_inputs.append(input_text)

            if (i + 1) % 10 == 0:
                logger.info("Evaluated %d/%d samples", i + 1, len(test_ds))

    # Compute metrics
    metrics = compute_test_metrics(all_predictions, all_labels)

    logger.info(
        "Test set results: micro_f1=%.4f, precision=%.4f, recall=%.4f, rouge_l=%.4f",
        metrics["entity_micro_f1"],
        metrics["entity_micro_precision"],
        metrics["entity_micro_recall"],
        metrics["rouge_l"],
    )

    return {
        **metrics,
        "n_samples": len(all_predictions),
        "checkpoint": checkpoint_path,
        "predictions": all_predictions,
        "labels": all_labels,
    }


def load_merged_model(
    model_path: str,
    checkpoint: Dict,
    device: str = "cuda",
) -> tuple:
    """Load base model and merge LoRA adapter from checkpoint.

    Args:
        model_path: HF cache path for base model
        checkpoint: loaded best_checkpoint.pt
        device: device to load model on

    Returns:
        (merged_model, tokenizer)
    """
    from peft import PeftModel
    from ..core.key_remapping import remap_lora_keys

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # Load LoRA adapter from checkpoint
    party_ckpts = checkpoint.get("party_checkpoints", {})
    m_ckpt = party_ckpts.get("M", {})
    lora_state = m_ckpt.get("lora_state", {})

    if lora_state:
        # Remap LoRA keys to match live PeftModel paths
        lora_state = remap_lora_keys(lora_state)

        # Load as PeftModel
        from peft import PeftConfig
        # NOTE: If the LoRA adapter wasn't saved via PeftModel.save_pretrained,
        # we need to manually inject the weights
        adapter_path = checkpoint.get("adapter_path", None)
        if adapter_path and os.path.exists(adapter_path):
            model = PeftModel.from_pretrained(
                base_model,
                adapter_path,
                is_trainable=False,
            )
        else:
            # Manual injection: create PeftModel structure and load weights
            model = _inject_lora_manually(base_model, lora_state)
    else:
        model = base_model

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def _inject_lora_manually(base_model, lora_state: Dict) -> "AutoModelForCausalLM":
    """Inject LoRA weights directly into the base model without PeftModel."""
    from peft import inject_adapter_in_model, LoraConfig
    from peft.tuners.lora import LoraLayer

    # Create LoRA config matching the training config
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = inject_adapter_in_model(lora_config, base_model, adapter_name="default")

    # Load state dict
    missing, unexpected = model.load_state_dict(lora_state, strict=False)
    if missing:
        logger.warning("Missing LoRA keys: %s", missing[:5])
    if unexpected:
        logger.debug("Unexpected keys ignored: %s", unexpected[:5])

    return model


def compute_test_metrics(
    predictions: List[str],
    labels: List[str],
) -> Dict[str, float]:
    """Compute all evaluation metrics on test set."""
    from ..data.dataset import parse_gold_entities

    # Entity-level micro P/R/F1
    tp = fp = fn = 0
    for pred, label in zip(predictions, labels):
        pred_ents = set(parse_gold_entities(pred))
        label_ents = set(parse_gold_entities(label))
        tp += len(pred_ents & label_ents)
        fp += len(pred_ents - label_ents)
        fn += len(label_ents - pred_ents)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # ROUGE-L
    rouge_l = _avg_rouge_l(predictions, labels)

    return {
        "entity_micro_precision": precision,
        "entity_micro_recall": recall,
        "entity_micro_f1": f1,
        "rouge_l": rouge_l,
    }


def _avg_rouge_l(predictions: List[str], labels: List[str]) -> float:
    """Compute average ROUGE-L over all samples."""
    total = 0.0
    for pred, label in zip(predictions, labels):
        lcs = _lcs_len(pred, label)
        denom = max(len(pred), len(label))
        total += lcs / denom if denom > 0 else 0.0
    return total / len(predictions) if predictions else 0.0


def _lcs_len(a: str, b: str) -> int:
    """Compute LCS length."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    dp = [[0] * (n + 1) for _ in range(2)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i % 2][j] = dp[(i - 1) % 2][j - 1] + 1
            else:
                dp[i % 2][j] = max(dp[(i - 1) % 2][j], dp[i % 2][j - 1])
    return dp[m % 2][n]


def save_test_results(results: Dict, output_path: str) -> None:
    """Save test evaluation results to file."""
    # Separate out predictions/labels (too verbose for main output)
    summary = {k: v for k, v in results.items()
               if k not in ("predictions", "labels")}
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Test results saved → %s", output_path)
