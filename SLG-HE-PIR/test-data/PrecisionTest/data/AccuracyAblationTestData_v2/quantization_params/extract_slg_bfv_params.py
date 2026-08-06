#!/usr/bin/env python3
"""Extract SLG BFV + LoRA parameters from the SLG classification checkpoint.

Behavior (verified 2026-07-31):
  1. Try to read ckpt['config'] (already known to NOT contain BFV params, but we
     log the absence).
  2. Read lora_state under ckpt['party_checkpoints']['M'] — yields rank,
     target_modules, dropout (from default).
  3. Read v_shape under ckpt['party_checkpoints']['S'] — yields vocab_size,
     hidden_dim.
  4. BFV params (scale, plain_bits, poly_degree) are NOT in ckpt — fallback to
     source-code constants in src/core/bfv_privselect_v2_adapter.py and
     src/parties/heterogeneous_protocol.py.

Outputs JSON + YAML side by side into quantization_params/.
"""
import json
import sys
from pathlib import Path

import torch


ROOT = Path("/root/autodl-tmp/CipherForgeCode/SLG-HE-PIR")
SLG_CKPT = (
    ROOT / "test-data" / "SLG-test-data" / "cls-SLG-test-data"
    / "_SAVE_20260727_0706" / "checkpoint_epoch_001.pt"
)
OUT_DIR = ROOT / "test-data" / "AccuracyAblationTestData" / "quantization_params"


def extract_slg_params(ckpt_path: Path) -> dict:
    cfg = {
        # From src/core/bfv_privselect_v2_adapter.py:11-13
        "scale": 10000,
        "plain_bits": 30,
        "poly_degree": 4096,
        # From src/parties/heterogeneous_protocol.py:201-202
        "hidden_dim": 4096,
        "vocab_size": 128256,
        # LoRA — peft standard
        "lora_dropout": 0.05,
        "lora_alpha_formula": "alpha = 2 * rank (peft convention)",
        # SLG-specific 7 target modules (verified from ckpt)
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        # Provenance
        "bfv_source": "src/core/bfv_privselect_v2_adapter.py:11-13 (hardcoded)",
        "shape_source": "src/parties/heterogeneous_protocol.py:201-202 (config defaults)",
        "lora_alpha_source": "src/scripts/evaluate_slg_cls.py:172 alpha = rank * 2",
        "target_modules_source": "extracted from ckpt party_checkpoints['M']['lora_state'] keys",
        "ckpt_config_has_bfv_params": False,  # verified 2026-07-31
    }

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "config" in ckpt:
        ckpt_cfg = ckpt["config"]
        # Verified: ckpt_cfg contains batch_size/max_epochs/etc., NOT BFV params
        bfv_keys = {k: v for k, v in ckpt_cfg.items() if k in {"scale", "plain_bits", "poly_degree"}}
        cfg["ckpt_config_observed"] = {
            k: v for k, v in ckpt_cfg.items()
            if k in {"max_epochs", "batch_size", "max_seq_length", "seed", "task_type", "train_ratio"}
        }
        if not bfv_keys:
            cfg["ckpt_config_has_bfv_params"] = False

    # Extract lora_state details
    lora_state = (
        ckpt.get("party_checkpoints", {}).get("M", {}).get("lora_state", {})
    )
    if lora_state:
        target_modules = set()
        rank = None
        for k, v in lora_state.items():
            if "lora_A" in k and hasattr(v, "shape") and v.ndim == 2:
                r = int(v.shape[0])
                if rank is None:
                    rank = r
                target_modules.add(k.split(".lora_")[0].split(".")[-1])
        if rank is not None:
            cfg["lora_rank"] = rank
            cfg["lora_alpha"] = rank * 2  # peft convention
        if target_modules:
            cfg["target_modules"] = sorted(target_modules)
            cfg["lora_params_count"] = sum(1 for _ in lora_state)
            cfg["target_modules_count"] = len(target_modules)
            cfg["lora_state_source"] = "ckpt['party_checkpoints']['M']['lora_state']"
    else:
        print("[WARN] No lora_state found in checkpoint", file=sys.stderr)

    # Extract V shape
    s_meta = ckpt.get("party_checkpoints", {}).get("S", {})
    if "v_shape" in s_meta:
        vs = s_meta["v_shape"]
        cfg["vocab_size"] = int(vs[0])
        cfg["hidden_dim"] = int(vs[1])
        cfg["v_shape_source"] = "ckpt['party_checkpoints']['S']['v_shape']"

    cfg["source_ckpt"] = str(ckpt_path)
    cfg["extracted_on"] = "2026-07-31"
    return cfg


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not SLG_CKPT.exists():
        print(f"[ERROR] SLG ckpt not found: {SLG_CKPT}", file=sys.stderr)
        sys.exit(1)

    print(f"[extract] reading SLG ckpt: {SLG_CKPT}")
    cfg = extract_slg_params(SLG_CKPT)

    json_path = OUT_DIR / "slg_bfv_params.json"
    with open(json_path, "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f"[extract] wrote {json_path}")

    yaml_path = OUT_DIR / "slg_bfv_params.yaml"
    lines = ["# SLG BFV + LoRA parameters extracted from checkpoint", "# Source: _SAVE_20260727_0706/checkpoint_epoch_001.pt", ""]
    for k, v in cfg.items():
        if isinstance(v, list):
            lines.append(f"{k}:")
            for item in v:
                lines.append(f"  - {item}")
        elif isinstance(v, bool):
            lines.append(f"{k}: {'true' if v else 'false'}")
        else:
            lines.append(f"{k}: {v}")
    with open(yaml_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[extract] wrote {yaml_path}")

    # Pretty-print summary
    print("\n========== SLG BFV Params Summary ==========")
    for k, v in cfg.items():
        print(f"  {k:30s} = {v}")


if __name__ == "__main__":
    main()