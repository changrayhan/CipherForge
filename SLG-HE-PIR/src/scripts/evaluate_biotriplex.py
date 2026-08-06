#!/usr/bin/env python3
"""evaluate_biotriplex.py — Stage 2 evaluation for BioTriplex tasks.

After ``src/scripts/biotriplex_finetune.py --stage 1`` finishes, this
script loads ``best_checkpoint.pt``, merges the LoRA adapter into the
base model, runs **standard forward** inference on the test set (no
PIR / no BFV), and writes the metrics JSON file matching the
``docs/BIOTRIPLEX_FINETUNE_README.md`` schema.

The two tasks use the same script with different ``--task_type``:

* ``classification`` (GenRel QA) — outputs ``genrel_<TS>_evaluate_metrics.json``
  with multi-label F1, macro F1, macro ROC AUC, etc.
* ``generation`` (NER JSON) — outputs ``ner_<TS>_evaluate_metrics.json``
  with per-class span P/R/F1 + macro/weighted/micro aggregates.

Inference parameters follow the README exactly:

* Task A — ``max_new_tokens=50, top_p=1.0, top_k=50, temperature=0.6,
  repetition_penalty=2.0``. We also capture last-token logits at
  ``a)/b)/.../g)`` token ids and softmax them — that's the probability
  source for ROC AUC.

* Task B — ``max_new_tokens=2000, top_p=1.0, top_k=200, temperature=0.6,
  repetition_penalty=1.0``. No logits are captured; the JSON text is the
  prediction.

The LoRA merge path mirrors ``src/training/evaluation.py::load_merged_model``
but reuses the freshly-saved HF PEFT adapter when available
(see ``biotriplex_finetune.py`` which calls ``model.save_pretrained``).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Bootstrap project root so that ``src.*`` imports work when this file
# is invoked directly.
_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.biotriplex_dataset import (  # noqa: E402
    ENTITY_TYPES,
    GENERAL_RELATIONS,
    OPTION_LETTERS,
    build_biotriplex_dataset,
)
from src.training.biotriplex_metrics import (  # noqa: E402
    compute_classification_metrics,
    compute_ner_metrics,
    load_ner_gold_entities,
)

logger = logging.getLogger("biotriplex_eval")


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def _get_option_token_ids(tokenizer) -> List[int]:
    """Map ``a)/b)/.../g)`` to a token id each (matches baseline)."""
    token_ids: List[int] = []
    for letter in OPTION_LETTERS:
        chosen = None
        for cand in (f"{letter})", letter, f" {letter})", f" {letter}"):
            ids = tokenizer.encode(cand, add_special_tokens=False)
            if len(ids) == 1:
                chosen = ids[0]
                break
        if chosen is None:
            ids = tokenizer.encode(f"{letter})", add_special_tokens=False)
            chosen = ids[0]
        token_ids.append(chosen)
    return token_ids


def _load_merged_model_from_adapter_dir(adapter_dir: str, base_path: str, device: str = "cuda"):
    """Load base model + apply LoRA adapter saved via ``PeftModel.save_pretrained``.

    Args:
        adapter_dir: directory containing ``adapter_config.json`` and
            ``adapter_model.safetensors``.
        base_path: HF cache path for the base Llama model.
        device: "cuda" or "cpu".

    Returns:
        ``(model, tokenizer)`` — model is a ``PeftModel`` (not merged).
    """
    from peft import PeftModel

    logger.info("Loading base model from %s", base_path)
    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    logger.info("Attaching PEFT adapter from %s", adapter_dir)
    model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(base_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return model, tokenizer


def _inference_loop_classification(
    model,
    tokenizer,
    dataset,
    option_token_ids: List[int],
    batch_size: int = 1,
    max_new_tokens: int = 50,
    top_p: float = 1.0,
    top_k: int = 50,
    temperature: float = 0.6,
    repetition_penalty: float = 2.0,
    max_input_length: int = 4096,
) -> Dict[str, Any]:
    """Run classification inference on a dataset.

    Returns:
        ``{doc_key: {answer, logits, probs, predicted_relation}}``.
    """
    outputs: Dict[str, Any] = {}
    for idx in range(len(dataset)):
        sample = dataset[idx]
        prompt = sample["prompt"]
        doc_key = sample["doc_key"]

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_length,
        ).to(model.device)

        with torch.no_grad():
            # First pass: get logits at last token (no generate needed for
            # multi-class classification — we just need last token logits).
            # Bug fix (2026-08-01): ``fwd.logits[0, -1, :]`` predicts the token
            # AFTER the prompt. Since the prompt ends with ``assistant`` (suffix),
            # this is the right position to predict the letter token. The previous
            # baseline bug was treating ``-1`` as "last input token", which only
            # holds when the input INCLUDES the gold answer. Here we tokenize
            # only the prompt, so ``-1`` correctly targets the next token.
            fwd = model(**inputs)
            last_logits = fwd.logits[0, -1, :].float().cpu()
            option_logits = [float(last_logits[tid]) for tid in option_token_ids]
            probs = torch.softmax(torch.tensor(option_logits), dim=0).tolist()
            best_idx = int(torch.tensor(probs).argmax().item())
            answer_letter = OPTION_LETTERS[best_idx]

            # Second pass: greedy generate the answer text (for letter-level
            # accuracy / multilabel F1 fallback).
            gen = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                pad_token_id=tokenizer.eos_token_id,
            )
            new_tokens = gen[0, inputs["input_ids"].shape[1] :]
            gen_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        outputs[doc_key] = {
            "answer": f"{answer_letter})",
            "logits": option_logits,
            "probs": probs,
            "predicted_relation": GENERAL_RELATIONS[best_idx],
            "generated_text": gen_text,
        }
    return outputs


def _inference_loop_generation(
    model,
    tokenizer,
    dataset,
    batch_size: int = 1,
    max_new_tokens: int = 2000,
    top_p: float = 1.0,
    top_k: int = 200,
    temperature: float = 0.6,
    repetition_penalty: float = 1.0,
    max_input_length: int = 4096,
) -> Dict[str, Any]:
    """Run NER generation inference on a dataset.

    Returns:
        ``{doc_key: raw_generated_text}``.
    """
    outputs: Dict[str, Any] = {}
    from tqdm import tqdm
    for idx in tqdm(range(len(dataset)), desc="ner-generate"):
        sample = dataset[idx]
        prompt = sample["prompt"]
        doc_key = sample["doc_key"]

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_length,
        ).to(model.device)

        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                pad_token_id=tokenizer.eos_token_id,
            )
            new_tokens = gen[0, inputs["input_ids"].shape[1] :]
            raw = tokenizer.decode(new_tokens, skip_special_tokens=True)
        outputs[doc_key] = raw.strip()
    return outputs


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="BioTriplex Stage-2 evaluation")
    p.add_argument("--task_type", choices=["classification", "generation"], required=True)
    p.add_argument("--base_model", default="/root/autodl-tmp/hf_cache/Llama-3-1-8B-I")
    p.add_argument("--adapter_dir", required=True,
                   help="Path to LoRA adapter directory (contains adapter_config.json)")
    p.add_argument("--data_path", required=True,
                   help="Path to Preprocessed BioTriplex/ directory")
    p.add_argument("--split", default="test")
    p.add_argument("--output_dir", default=None,
                   help="Where to write infer_outputs_<TS>.json + evaluate_metrics.json")
    p.add_argument("--save_prefix", default=None,
                   help="Prefix for the metrics JSON file (e.g. genrel_<TS>_ or ner_<TS>_). "
                        "If omitted, derived from task_type + timestamp.")
    p.add_argument("--max_seq_length", type=int, default=10000)
    p.add_argument("--max_eval_samples", type=int, default=-1,
                   help="If >0, truncate to this many samples (debug only).")
    p.add_argument("--device", default="cuda")
    p.add_argument("--log_file", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(args.log_file) if args.log_file else logging.StreamHandler(),
        ],
    )

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir or os.path.join(os.path.dirname(args.adapter_dir), "logs")
    os.makedirs(out_dir, exist_ok=True)

    # ----- Load tokenizer first (needed for dataset build) -----
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # ----- Build dataset (writes gold JSONL as a side-effect) -----
    logger.info("Building dataset from %s (split=%s)", args.data_path, args.split)
    dataset = build_biotriplex_dataset(
        task=args.task_type,
        data_dir=args.data_path,
        tokenizer=tokenizer,
        split=args.split,
        max_length=args.max_seq_length,
    )
    logger.info("Dataset size: %d", len(dataset))
    if args.max_eval_samples > 0:
        # Truncate dataset via a wrapper for quick debugging
        class _Subset(torch.utils.data.Dataset):  # type: ignore
            def __init__(self, base, n):
                self.base = base
                self.n = n
            def __len__(self):
                return min(self.n, len(self.base))
            def __getitem__(self, i):
                return self.base[i]
        dataset = _Subset(dataset, args.max_eval_samples)
        logger.info("Truncated to %d samples", len(dataset))

    # ----- Load merged LoRA + base model -----
    model, tokenizer = _load_merged_model_from_adapter_dir(
        adapter_dir=args.adapter_dir,
        base_path=args.base_model,
        device=args.device,
    )

    # ----- Run inference -----
    infer_out_path = os.path.join(out_dir, f"infer_outputs_{ts}.json")
    if args.task_type == "classification":
        option_token_ids = _get_option_token_ids(tokenizer)
        logger.info("Option token mapping: %s", list(zip(OPTION_LETTERS, option_token_ids)))
        outputs = _inference_loop_classification(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            option_token_ids=option_token_ids,
            max_new_tokens=50,
            top_p=1.0,
            top_k=50,
            temperature=0.6,
            repetition_penalty=2.0,
            max_input_length=args.max_seq_length,
        )
    else:
        outputs = _inference_loop_generation(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            max_new_tokens=2000,
            top_p=1.0,
            top_k=200,
            temperature=0.6,
            repetition_penalty=1.0,
            max_input_length=args.max_seq_length,
        )

    with open(infer_out_path, "w") as f:
        json.dump(outputs, f, indent=2)
    logger.info("Wrote inference outputs → %s", infer_out_path)

    # ----- Free GPU before metric computation -----
    del model
    torch.cuda.empty_cache()

    # ----- Compute metrics -----
    if args.task_type == "classification":
        predictions: List[str] = []
        labels: List[str] = []
        for idx in range(len(dataset)):
            sample = dataset[idx]
            dk = sample["doc_key"]
            out = outputs.get(dk, {})
            # Prefer the answer letter derived from option logits (matches the
            # training-time supervision exactly: predict the letter token
            # immediately after the ``assistant`` suffix). The generated text is
            # kept only as a fallback for multi-letter answers or as a debugging
            # artifact in the output JSON.
            answer = out.get("answer", "")
            # ``answer`` is e.g. ``"a)"`` already (see _inference_loop_classification).
            # We prefer it over the greedy-generated text because the latter may
            # pick unrelated tokens when the prompt is long.
            pred_text = answer or out.get("generated_text", "")
            gold_text = sample["output_text"]
            predictions.append(pred_text)
            labels.append(gold_text)
        metrics = compute_classification_metrics(predictions, labels)
    else:
        # NER
        gold_path = os.path.join(args.data_path, f"{args.split}_gold_ner.txt")
        gold_map = load_ner_gold_entities(gold_path)
        predictions = []
        golds_aligned = []
        for idx in range(len(dataset)):
            sample = dataset[idx]
            dk = sample["doc_key"]
            pred_text = outputs.get(dk, "")
            predictions.append(pred_text)
            if dk in gold_map:
                golds_aligned.append(gold_map[dk])
            else:
                # Fallback: parse the JSON label stored in output_text
                from src.training.biotriplex_metrics import _parse_entities_json
                golds_aligned.append(_parse_entities_json(sample["output_text"]))
        metrics = compute_ner_metrics(predictions, golds_aligned, list(outputs.keys()))

    save_prefix = args.save_prefix or (
        f"genrel_{ts}_" if args.task_type == "classification" else f"ner_{ts}_"
    )
    metrics_path = os.path.join(out_dir, f"{save_prefix}evaluate_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Wrote metrics → %s", metrics_path)

    # Console summary
    print()
    print("=" * 60)
    print(f"Task: {args.task_type}  split: {args.split}  samples: {len(dataset)}")
    print("=" * 60)
    print(json.dumps(metrics.get("metrics", {}), indent=2))
    print()
    if args.task_type == "generation":
        print("Per-class metrics:")
        print(json.dumps(metrics.get("per_class_metrics", {}), indent=2))
    else:
        print("Per-class metrics:")
        print(json.dumps(metrics.get("per_class_metrics", {}), indent=2))
    print("=" * 60)
    print(f"Metrics file: {metrics_path}")


if __name__ == "__main__":
    main()