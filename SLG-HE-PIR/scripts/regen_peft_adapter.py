#!/usr/bin/env python3
"""Regenerate PEFT adapter from a saved SLG-HE-PIR checkpoint.

Reads M's lora_state from best_checkpoint.pt, injects it into a transient
base model via inject_adapter_in_model, then exports just the LoRA adapter
weights via PeftModel's internal adapter state extraction (saves only the
adapter, not the full merged model).
"""
import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

import torch

_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger("regen_adapter")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to best_checkpoint.pt")
    p.add_argument("--base_model", default="/root/autodl-tmp/hf_cache/Llama-3-1-8B-I")
    p.add_argument("--output_dir", required=True, help="Output adapter directory")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    lora_state = ckpt.get("party_checkpoints", {}).get("M", {}).get("lora_state", {})
    if not lora_state:
        raise RuntimeError(f"No lora_state in {args.checkpoint}")
    logger.info("Loaded lora_state with %d keys", len(lora_state))

    # Remap keys (add .default suffix to LoRA keys, add base_model.model.model. prefix)
    # Note: PEFT's LoraLayer uses the same A=[rank, in], B=[out, rank] shape as
    # SLG's custom _LoRALinear — no transpose needed.
    canon = {}
    for k, v in lora_state.items():
        if "lora_A" in k or "lora_B" in k:
            nk = k.replace(".lora_A", ".lora_A.default.weight").replace(".lora_B", ".lora_B.default.weight")
        else:
            continue
        canon[f"base_model.model.model.{nk}"] = v
    logger.info("Mapped to %d canonical LoRA keys", len(canon))

    # Build base model + wrap with PEFT get_peft_model
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model

    logger.info("Loading base model from %s ...", args.base_model)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="cpu",
    )

    lora_config = LoraConfig(
        r=8, lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_config)

    # Load state with strict=False (layers 16-31 LoRA will remain random init
    # since SLG's lora_state only covers layers 0-15).
    missing, unexpected = model.load_state_dict(canon, strict=False)
    n_lora_missing = sum(1 for k in missing if "lora_A" in k or "lora_B" in k)
    n_lora_unexpected = sum(1 for k in unexpected if "lora_A" in k or "lora_B" in k)
    logger.info("Load: total_missing=%d total_unexpected=%d | LoRA_missing=%d LoRA_unexpected=%d",
                len(missing), len(unexpected), n_lora_missing, n_lora_unexpected)

    # Extract just the LoRA adapter weights (filter to LoRA keys only)
    adapter_state = {k: v.cpu() for k, v in model.state_dict().items()
                     if "lora_A" in k or "lora_B" in k}
    logger.info("Extracted %d adapter weights", len(adapter_state))

    # Sanity: assert loaded keys are non-zero (real training, not random init)
    loaded_keys = [k for k in adapter_state if k in canon]
    n_nonzero = sum(1 for k in loaded_keys if adapter_state[k].abs().sum() > 0)
    logger.info("Loaded LoRA keys with non-zero weights: %d / %d", n_nonzero, len(loaded_keys))
    if n_nonzero == 0:
        raise RuntimeError("No non-zero LoRA weights — adapter export failed")

    # Save as standard PEFT adapter format
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    # Write adapter_config.json
    config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "bias": "none",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
        "base_model_name_or_path": args.base_model,
    }
    import json as _json
    with open(os.path.join(args.output_dir, "adapter_config.json"), "w") as f:
        _json.dump(config, f, indent=2)

    # Write adapter_model.safetensors
    from safetensors.torch import save_file
    save_file(adapter_state, os.path.join(args.output_dir, "adapter_model.safetensors"),
              metadata={"format": "pt"})

    logger.info("PEFT adapter saved → %s", args.output_dir)
    logger.info("Files: %s", sorted(os.listdir(args.output_dir)))


if __name__ == "__main__":
    main()
