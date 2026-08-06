"""BioTriplex 1B baseline trainer v2 — Phase 1.6 quant injection on gradients.

A drop-in replacement of ``bio_baseline_trainer.py`` that implements the
Phase 1.6 (2026-08-02) redesign of scale/g_H_dtype injection. The
legacy trainer injects these on the **loss scalar**, which is **inert**
(leaf-tensor noise vanishes from the gradient; bf16/fp32 cast of a
scalar is bit-exact). This v2 trainer moves both injections onto the
**trainable gradients** after ``loss.backward()`` and before
``clip_grad_norm_``, matching the DP-SGD convention used by
``_add_dp_noise_to_grads``.

Key differences from `bio_baseline_trainer.py`:
  * `compute_train_loss` returns the **clean** CE loss (no quant touch).
  * Training loop calls ``_add_quant_noise_to_grads`` unconditionally;
    the function no-ops when ``--quant_off`` is set or when
    ``scale=0`` / ``g_H_dtype='none'``.
  * Order: DP noise → g_H_dtype cast → scale noise → clip.
  * Protocol order matches: DP-noised gradient → BFV-decrypt dtype cast
    → BFV-quantize scale noise → gradient clipping.

Identical to ``bio_baseline_trainer.py`` for: model, dataset, optimizer,
DP-noise injection, evaluator, metrics output.

Outputs (per epoch):
    ${OUTPUT_DIR}/adapter/                       — PEFT adapter weights
    ${LOG_DIR}/train.log                         — training log
    ${LOG_DIR}/infer_outputs_epoch_NNN.json      — inference JSON for evaluation
    ${LOG_DIR}/epoch_NNN_bio_metrics.json        — BioTriplex 7-class evaluator output
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

REPO_ROOT = Path("/root/autodl-tmp/CipherForgeCode/SLG-HE-PIR")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "test-data" / "BioTriplex1BTestData" / "scripts"))

from src.data.biotriplex_dataset import (  # noqa: E402
    GENERAL_RELATIONS,
    GENERAL_REL_TO_IDX,
    OPTION_LETTERS,
    build_biotriplex_dataset,
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
    p.add_argument("--gold_path", required=True,
                   help="Path to test_gold_general_qa.txt (for evaluator)")
    p.add_argument("--hf_model", default=os.environ.get(
        "BIO_HF_MODEL",
        "/root/autodl-tmp/CipherForgeCode/hf_cache/models--unsloth--Llama-3.2-1B/snapshots/9535bd9b1d1dea6acafbdc4813b728796aeb28da",
    ))
    p.add_argument("--output_dir", required=True)
    p.add_argument("--log_dir", required=True)
    # --- Hyperparameters ---
    p.add_argument("--max_epochs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--max_seq_length", type=int, default=1024,
                   help="BioTriplex sentence + few-shot prompt. p95=757 tokens; "
                        "1024 covers 99.2%% of train samples (4/491 still empty-masked).")
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_steps", type=int, default=50)
    p.add_argument("--gradient_clip_norm", type=float, default=1.0)
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_target", default="q,v")
    p.add_argument("--seed", type=int, default=42)
    # --- DP ablation knobs ---
    p.add_argument("--dp_alpha", type=float, default=0.0)
    p.add_argument("--dp_answer_beta", type=float, default=0.5)
    # --- Quantization ablation knobs (Phase 1.5 → 1.6 redesign) ---
    #
    # Phase 1.5 history (2026-08-01): the original implementation injected
    # scale round-trip noise and g_H_dtype cast on the **loss scalar**:
    #
    #     if args.scale > 0:
    #         quant_noise = torch.randn_like(loss) * (1.0 / (2.0 * args.scale))
    #         loss = loss + quant_noise
    #
    #     if args.g_H_dtype in ("bf16", "fp16"):
    #         loss = loss.to(bf16).to(fp32)
    #
    # That implementation was **inert**:
    #   1. `torch.randn_like(loss)` produces a 0-d leaf tensor (loss is a
    #      scalar), so the noise term vanishes from d(loss)/dθ — gradient
    #      is bit-exact identical to clean loss.
    #   2. bf16/fp32 cast of a single scalar (loss ≈ 1.0) is bit-exact —
    #      no precision change.
    #
    # Phase 1.6 (2026-08-02) redesign: move both injections onto the
    # **trainable gradients** (after `loss.backward()` and before
    # `clip_grad_norm_`), matching the DP-SGD convention used by
    # ``_add_dp_noise_to_grads``.
    p.add_argument("--scale", type=int, default=10000,
                   help="Quantization scale: round(V · scale) / scale simulates BFV round-trip tax. "
                        "In Phase 1.6 the noise is injected on gradients (per-element std=1/(2·scale)).")
    p.add_argument("--g_H_dtype", choices=["bf16", "fp32", "fp16", "none"], default="bf16",
                   help="Gradient-injection precision. 'none' = no cast (clean fp32 gradient). "
                        "Phase 1.6: applied to .grad tensor (was loss scalar in Phase 1.5).")
    p.add_argument("--quant_off", action="store_true",
                   help="Phase 1.6 control group: completely disable both --scale noise and "
                        "--g_H_dtype cast (equivalent to scale=0 and g_H_dtype=none).")
    return p.parse_args()


def build_lora_model(hf_model_path: str, args):
    """Build a Llama-3.2-1B causal LM with LoRA adapters."""
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, TaskType

    # bf16 can numerically overflow with long BioTriplex sequences (we hit
    # loss=nan on the first step). fp32 is safer and the 1B model fits.
    model = AutoModelForCausalLM.from_pretrained(
        hf_model_path,
        torch_dtype=torch.float32,
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
    return model


def collate_fn_factory(tokenizer, max_length: int):
    """Pad-collate for causal LM training."""
    def collate(batch):
        ids = [b["input_ids"] for b in batch]
        masks = [b["attention_mask"] for b in batch]
        labels = [b["labels"] for b in batch]
        return {
            "input_ids": torch.stack(ids),
            "attention_mask": torch.stack(masks),
            "labels": torch.stack(labels),
            "doc_key": [b["doc_key"] for b in batch],
            "output_text": [b["output_text"] for b in batch],
            "label_idx": torch.tensor([b["label_idx"] for b in batch], dtype=torch.long),
        }
    return collate


def compute_train_loss(model, batch, args):
    """Per-batch CE loss (Phase 1.6 trainer: clean loss, quant happens on grads).

    Returns a tuple ``(loss, is_valid)``. ``is_valid=False`` signals that the
    batch has no supervision signal (all label positions are -100) and the
    loss should be skipped — otherwise HuggingFace CE returns NaN, which
    poisons the optimizer.

    Phase 1.6 (2026-08-02): loss is returned **clean**. Quantization tax
    (--scale noise + --g_H_dtype cast) is applied to the trainable
    gradients by ``_add_quant_noise_to_grads`` after ``loss.backward()``.
    This is the correct path that Phase 1.5 lacked (its loss-scalar
    injection was inert — see ``bio_baseline_trainer.py`` for the
    archived legacy path).
    """
    labels = batch["labels"].cuda()
    n_valid = int((labels != -100).sum().item())
    if n_valid == 0:
        # No supervision in this batch — return zero loss (caller will skip).
        zero = torch.zeros((), device=labels.device)
        return zero, False

    out = model(
        input_ids=batch["input_ids"].cuda(),
        attention_mask=batch["attention_mask"].cuda(),
        labels=labels,
    )
    loss = out.loss
    return loss, True


def _add_dp_noise_to_grads(model, sigma: float, max_norm: float,
                             answer_beta: float = 1.0) -> None:
    """DP-SGD-style gradient perturbation.

    Adds i.i.d. Gaussian noise ``N(0, sigma^2)`` to every trainable
    parameter's ``.grad``. Run *after* ``loss.backward()`` and *before*
    ``clip_grad_norm_`` (so that the noise scale is calibrated on a clipped
    gradient vector — matching the original DP-SGD prescription).

    Args:
        model: the LoRA-augmented causal LM. Only ``requires_grad=True``
            parameters receive noise (the frozen base model is silent
            regardless of its grad state).
        sigma: Gaussian noise stddev per parameter element. Convention:
            ``sigma = dp_alpha`` so the relative noise scale matches
            ``dp_alpha=0.15`` ≈ 15% relative to a unit-magnitude gradient.
            Callers can rescale if they want absolute clipping.
        max_norm: gradient clipping norm, exposed so we can compute the
            noise scale per the ``sigma * max_norm`` DP-SGD convention if
            desired. Currently unused (we add absolute sigma noise), but
            kept as a hook for future ``sigma_dp * clip_norm`` modes.
        answer_beta: SLG-style answer-position noise multiplier (0..1).
            In SLG, β < 1 reduces the noise on answer positions (lower
            leakage on the actual target tokens); in baseline we lack the
            per-token UI machinery, so we treat β as a *global* gradient
            noise scaler that mimics the average answer-vs-prompt UI gap.
            Default 1.0 = no rescaling (matches the historical behaviour
            before the parameter was wired up).
    """
    if sigma <= 0.0:
        return
    # Beta rescales the per-parameter noise std: β=0 silences DP entirely,
    # β=1 leaves it at sigma. Clamp to [0, 1] defensively.
    beta = max(0.0, min(1.0, float(answer_beta)))
    eff_sigma = sigma * beta
    if eff_sigma <= 0.0:
        return
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            p.grad.add_(torch.randn_like(p.grad) * eff_sigma)


def _add_quant_noise_to_grads(model, scale: int, g_H_dtype: str,
                                quant_off: bool = False) -> None:
    """Phase 1.6 quantization tax simulation on gradients.

    Implements two protocol-stack effects on the **trainable gradients**,
    in order, after ``loss.backward()`` and before ``clip_grad_norm_``:

    1. ``g_H_dtype`` cast (party_m gradient injection precision):
       round each ``.grad`` to bf16 / fp16 (or skip if 'none' / 'fp32').

    2. ``scale`` round-trip noise: per-element Gaussian noise with
       std = 1 / (2 · scale) added to each ``.grad``. scale=100 →
       std=0.005 (5‰ of unit magnitude); scale=100k → std=5e-6
       (negligible).

    Args:
        model: LoRA-augmented causal LM. Only trainable params touched.
        scale: BFV round-trip quantization scale. 0 or negative = off.
        g_H_dtype: 'bf16' / 'fp16' / 'fp32' / 'none'.
        quant_off: control-group flag; if True, no-op (independent of
            scale / g_H_dtype). Equivalent to scale=0 and g_H_dtype='none'.

    Phase 1.6 design notes (2026-08-02):
        * Moved from ``compute_train_loss`` (where it was applied to the
          loss scalar and was inert) to here (per-element gradient
          injection — matches DP-SGD convention).
        * ``g_H_dtype`` cast is applied **before** ``scale`` noise so
          that the noise lives on the *quantized* gradient (matching
          the BFV-decrypt-then-cast-then-add-DP-noise protocol order).
        * Both stages use ``torch.no_grad``-equivalent in-place ops to
          avoid re-allocating the autograd graph.
    """
    if quant_off:
        return
    # Stage 1: dtype cast (party_m injection precision)
    if g_H_dtype in ("bf16", "fp16"):
        target = torch.bfloat16 if g_H_dtype == "bf16" else torch.float16
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                p.grad.data = p.grad.data.to(target).to(torch.float32)
    # Stage 2: per-element round-trip quantization noise
    if scale and scale > 0:
        sigma = 1.0 / (2.0 * float(scale))
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                p.grad.add_(torch.randn_like(p.grad) * sigma)


def evaluate(model, test_ds, tokenizer, args, device="cuda"):
    """Greedy inference + collect 7-class logits for each test sample.

    Bug fix (2026-08-01): the original implementation took ``logits[0, seq_len-1]``,
    but in HuggingFace causal-LM ``logits[i]`` predicts ``input_ids[i+1]`` (shift-by-1).
    The actual letter token is at ``seq_len-3`` (e.g. ``c)`` followed by EOS at
    ``seq_len-1``), so the trained letter prediction lives at ``logits[0, seq_len-4]``,
    NOT ``logits[0, seq_len-1]`` (which predicts an EOS-following position and
    is never supervised). Result: every prediction collapsed to ``f`` (no relation)
    and macro-F1 ≈ 0.005 (random).

    We now locate the letter position from the supervised labels (``labels > 0 and
    != pad_token_id``) and read ``logits[letter_pos - 1]``. This matches the
    training-time supervision exactly.
    """
    opt_letters = OPTION_LETTERS  # ["a", "b", "c", "d", "e", "f", "g"]
    opt_token_ids = []
    for letter in opt_letters:
        ids = tokenizer.encode(f"{letter})", add_special_tokens=False)
        opt_token_ids.append(ids[0] if ids else tokenizer.eos_token_id)
    # The letter sub-token (e.g. ``c`` for label "c)") is what we want to predict;
    # the matching ``)`` and EOS tokens come after. So we use the FIRST letter
    # token id as the canonical "this is the answer letter" position.
    letter_token_ids = set(opt_token_ids)
    pad_token_id = tokenizer.pad_token_id

    outputs: Dict[str, Dict[str, Any]] = {}
    model.eval()
    with torch.no_grad():
        for idx in tqdm(range(len(test_ds)), desc="infer", leave=False):
            item = test_ds[idx]
            ids = item["input_ids"].unsqueeze(0).to(device)
            am = item["attention_mask"].unsqueeze(0).to(device)
            out = model(input_ids=ids, attention_mask=am)
            logits = out.logits
            # Locate the letter token's supervised position. The labels tensor has
            # the supervised positions > 0 (not -100, not pad). We want the
            # rightmost supervised position whose token id matches one of the
            # 7 option letters.
            label_ids = item["labels"].tolist()
            letter_pos = None
            for p in range(len(label_ids) - 1, -1, -1):
                if label_ids[p] in letter_token_ids:
                    letter_pos = p
                    break
            if letter_pos is None or letter_pos == 0:
                # Fallback: use the last non-pad position - 1.
                seq_len = int(am.sum().item())
                letter_pos = max(0, seq_len - 4)
            # HF causal LM: logits[i] predicts input_ids[i+1].
            # We want letter at position p → logits at position p-1.
            last_logits = logits[0, letter_pos - 1]
            opt_logits = last_logits[opt_token_ids].float().cpu().numpy()
            pred_idx = int(np.argmax(opt_logits))
            probs = _softmax(opt_logits)
            outputs[item["doc_key"]] = {
                "answer": f"{opt_letters[pred_idx]}",
                "logits": opt_logits.tolist(),
                "probs": probs.tolist(),
                "predicted_relation": GENERAL_RELATIONS[pred_idx],
                "label_idx": pred_idx,
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
    log = logging.getLogger("bio_baseline")

    # ---- Build datasets ----
    log.info("Building BioTriplex classification dataset from %s ...", args.data_dir)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = build_biotriplex_dataset(
        task="classification",
        data_dir=args.data_dir, tokenizer=tokenizer, split="train",
        max_length=args.max_seq_length, return_neg_relations=True,
    )
    val_ds = build_biotriplex_dataset(
        task="classification",
        data_dir=args.data_dir, tokenizer=tokenizer, split="val",
        max_length=args.max_seq_length, return_neg_relations=True,
    )
    test_ds = build_biotriplex_dataset(
        task="classification",
        data_dir=args.data_dir, tokenizer=tokenizer, split="test",
        max_length=args.max_seq_length, return_neg_relations=True,
    )
    log.info("train=%d val=%d test=%d", len(train_ds), len(val_ds), len(test_ds))

    # ---- Build model ----
    log.info("Loading model %s ...", args.hf_model)
    model = build_lora_model(args.hf_model, args)
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
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        n_batches = 0
        for batch in train_dl:
            optim.zero_grad(set_to_none=True)
            loss, is_valid = compute_train_loss(model, batch, args)
            if not is_valid or not torch.isfinite(loss):
                # Skip empty-mask / NaN batches (BioTriplex ~40% of samples
                # have no labels after our truncation — common for long
                # prompts where the answer token falls outside max_length).
                continue
            loss.backward()
            # DP-SGD-style: add Gaussian noise to trainable gradients
            # *before* clipping, so the noise scale stays independent of
            # the gradient norm and the clip bound acts only on the
            # *clean* signal portion. Calibration: ``dp_alpha=0.05`` adds
            # 5% relative Gaussian noise per parameter element.
            _add_dp_noise_to_grads(model, sigma=args.dp_alpha,
                                    max_norm=args.gradient_clip_norm,
                                    answer_beta=args.dp_answer_beta)
            # Phase 1.6 trainer: quantization tax (g_H_dtype cast + scale round-trip
            # noise) lives on the trainable gradient, applied *after* DP
            # noise and *before* clipping. Matches the protocol order:
            # DP-noised gradient → cast to g_H_dtype → quantize to scale →
            # clip. ``--quant_off`` short-circuits both stages (control
            # group).
            _add_quant_noise_to_grads(
                model, scale=args.scale, g_H_dtype=args.g_H_dtype,
                quant_off=getattr(args, "quant_off", False),
            )
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
            optim.step()
            sched.step()
            total_loss += float(loss.detach().item())
            n_batches += 1
            step += 1
            if step % 50 == 0:
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

        # ---- Run BioTriplex 7-class evaluator ----
        log.info("Running 7-class BioTriplex evaluator ...")
        metrics = _run_evaluator(args, infer_outputs)
        out_metrics = Path(args.log_dir) / f"epoch_{epoch:03d}_bio_metrics.json"
        with open(out_metrics, "w") as f:
            json.dump(metrics, f, indent=2)
        metrics_history.append({"epoch": epoch, **metrics})
        log.info(
            "Epoch %d metrics: macro_F1=%.4f micro_F1=%.4f accuracy=%.4f",
            epoch, metrics.get("macro_f1", 0.0), metrics.get("micro_f1", 0.0),
            metrics.get("accuracy", 0.0),
        )

    # ---- Write summary ----
    summary_path = Path(args.log_dir) / "metrics_history.json"
    with open(summary_path, "w") as f:
        json.dump(metrics_history, f, indent=2)

    # ---- Save PEFT adapter ----
    adapter_dir = Path(args.output_dir) / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    log.info("Saving PEFT adapter to %s ...", adapter_dir)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    log.info("DONE.")


def _run_evaluator(args, infer_outputs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Run the BioTriplex 7-class evaluator as a subprocess (keeps this file simple)."""
    import subprocess
    infer_path = Path(args.log_dir) / "_tmp_infer.json"
    with open(infer_path, "w") as f:
        json.dump(infer_outputs, f)
    eval_script = REPO_ROOT / "test-data" / "BioTriplex1BTestData" / "scripts" / "bio_evaluator.py"
    proc = subprocess.run(
        [
            sys.executable, str(eval_script),
            "--infer_path", str(infer_path),
            "--gold_path", str(args.gold_path),
        ],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        logger.error("evaluator failed: %s", proc.stderr)
        return {"error": "evaluator_failed", "stderr": proc.stderr[-1000:]}
    return json.loads(proc.stdout)


if __name__ == "__main__":
    main()