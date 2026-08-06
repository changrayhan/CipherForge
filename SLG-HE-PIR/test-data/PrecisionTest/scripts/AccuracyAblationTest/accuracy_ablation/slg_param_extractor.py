"""slg_param_extractor — 从 SLG checkpoint 提取 BFV + LoRA 配置参数。

SLG ckpt 是 torch.save 的 dict，但实测发现：
  - ckpt['config'] 只有训练参数 (batch_size, max_epochs, ...)
  - BFV 参数 (scale/plain_bits/poly_degree) 不在 ckpt 中
  - party_checkpoints['S']['v_shape'] 有 V 矩阵维度
  - party_checkpoints['M']['lora_state'] 有 LoRA 权重 shape

因此本模块：尝试从 ckpt 读 LoRA/V 维度，BFV 参数从源码 fallback。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch

from .quant_config import QuantConfig

logger = logging.getLogger(__name__)


# 从源码 fallback 的 BFV 参数（来源见 .cursor/plans/...plan.md §"关键 SLG 实现细节"）
SRC_FALLBACK = {
    "scale": 10000,
    "plain_bits": 30,
    "poly_degree": 4096,
    "hidden_dim": 4096,
    "vocab_size": 128256,
    "source": "src_fallback",
}


def _detect_lora_rank(lora_state: dict) -> Optional[int]:
    """从 lora_state 的第一对 lora_A 推断 rank。"""
    for k, v in lora_state.items():
        if "lora_A" in k and hasattr(v, "shape"):
            return int(v.shape[0])
    return None


def _detect_target_modules(lora_state: dict) -> list[str]:
    """从 lora_state 的 key 推断 target_modules unique 集合。"""
    modules = set()
    for k in lora_state.keys():
        if "lora_A" in k:
            # key 形如 base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
            # 取最后一段的父名
            parts = k.split(".lora_")[0].split(".")
            modules.add(parts[-1])
    return sorted(modules)


def extract_slg_params(ckpt_path: str | Path) -> QuantConfig:
    """从 SLG ckpt 提取配置参数。

    Args:
        ckpt_path: SLG checkpoint 路径
            (例: test-data/SLG-test-data/cls-SLG-test-data/_SAVE_20260727_0706/checkpoint_epoch_001.pt)

    Returns:
        QuantConfig 实例，BFV 参数从源码 fallback，LoRA/V 维度尝试从 ckpt 读。
    """
    cfg_dict = dict(SRC_FALLBACK)

    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        # 读 lora_state
        lora_state = (
            ckpt.get("party_checkpoints", {}).get("M", {}).get("lora_state", {})
        )
        if lora_state:
            rank = _detect_lora_rank(lora_state)
            if rank is not None:
                cfg_dict["lora_rank"] = int(rank)
                cfg_dict["lora_alpha"] = int(rank) * 2  # peft 默认 alpha=2*rank
            modules = _detect_target_modules(lora_state)
            if modules:
                cfg_dict["target_modules"] = modules

        # 读 v_shape
        s_meta = ckpt.get("party_checkpoints", {}).get("S", {})
        if "v_shape" in s_meta:
            vocab_size, hidden_dim = s_meta["v_shape"]
            cfg_dict["vocab_size"] = int(vocab_size)
            cfg_dict["hidden_dim"] = int(hidden_dim)

        cfg_dict["source"] = "ckpt"
        cfg_dict["source_ckpt"] = str(ckpt_path)
        logger.info(
            "[slg_param_extractor] Successfully read from ckpt: rank=%s, modules=%s",
            cfg_dict.get("lora_rank"), cfg_dict.get("target_modules"),
        )
    except Exception as e:
        logger.warning(
            "[slg_param_extractor] Failed to read from ckpt (%s); "
            "BFV params using src_fallback. Error: %s",
            ckpt_path, e,
        )
        cfg_dict["source_ckpt"] = str(ckpt_path)

    # default lora_dropout
    cfg_dict.setdefault("lora_dropout", 0.05)

    return QuantConfig(**cfg_dict)