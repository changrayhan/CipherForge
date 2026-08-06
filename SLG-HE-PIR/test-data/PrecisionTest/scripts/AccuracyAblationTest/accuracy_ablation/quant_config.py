"""QuantConfig — 量化配置 dataclass + yaml 加载。

变体定义集中化。Q1/Q2/Q2'/Q3 都从 Q0 出发，依次叠加约束。
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class QuantConfig:
    """BFV fixed-point + 反向协议 量化的核心配置。

    Attributes:
        scale: BFV 编码的 fixed-point 缩放因子（SLG = 10000）。
        plain_bits: BFV plaintext modulus bit width（SLG = 30）。
        poly_degree: BFV polynomial modulus degree（SLG = 4096）。
        hidden_dim: Llama-3-1-8B 中部 hidden_size（= 4096）。
        vocab_size: Llama-3-1-8B lm_head 输出维度（= 128256）。
        lora_rank: LoRA rank（SLG = 8）。
        lora_alpha: LoRA alpha（SLG = 16 = 2*rank）。
        lora_dropout: LoRA dropout（= 0.05）。
        target_modules: LoRA 注入的目标模块列表。
        use_lora_ablation: 是否模拟"7-target vs 2-target"差异（Q0=True, Q0'=False）。
        g_h_quant_scale: g_H int64 量化 scale（= scale, 同 V）。
        emulated_lora_param_count: 7-target ≈ 7M, 2-target ≈ 2M.
        seed_list: 多 seed 实验的随机种子列表。
        epochs: 每个变体评估的 epoch 数（= 5）。
        bf16_round: Q3 是否启用 g_H → bf16 的 round-to-nearest。
        source_ckpt: BFV 参数提取来源 SLG ckpt 路径（debug 用）。
        source: BFV 参数来源标识（"ckpt" / "src_fallback"）。
    """

    # BFV 参数（fixed-point 量化税）
    scale: int = 10000
    plain_bits: int = 30
    poly_degree: int = 4096

    # 模型维度
    hidden_dim: int = 4096
    vocab_size: int = 128256

    # LoRA 超参
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # Q0/Q0' 区分
    use_lora_ablation: bool = True  # True=Q0 (7-target), False=Q0' (2-target)

    # 实验设计
    seed_list: List[int] = field(default_factory=lambda: [42, 123, 456])
    epochs: int = 5

    # Q3 专用
    bf16_round: bool = True

    # Debug
    source_ckpt: Optional[str] = None
    source: str = "src_fallback"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "QuantConfig":
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        # target_modules 可能是 list / 空 / None
        tm = raw.get("target_modules")
        if isinstance(tm, str):
            tm = [s.strip() for s in tm.split(",") if s.strip()]
        elif tm is None:
            tm = None
        kwargs = {k: v for k, v in raw.items() if k in (
            "scale", "plain_bits", "poly_degree",
            "hidden_dim", "vocab_size",
            "lora_rank", "lora_alpha", "lora_dropout",
            "use_lora_ablation", "bf16_round",
            "seed_list", "epochs",
            "source_ckpt", "source",
        )}
        if tm is not None:
            kwargs["target_modules"] = tm
        return cls(**kwargs)

    def to_yaml(self, path: str | Path) -> None:
        d = dataclasses.asdict(self)
        with open(path, "w") as f:
            yaml.safe_dump(d, f, sort_keys=False, allow_unicode=True)

    def variant(self, name: str) -> "QuantConfig":
        """返回指定变体的配置副本（仅修改 ablation flag）。

        支持：Q0, Q0', Q1, Q2, Q2', Q3。
        """
        cfg = dataclasses.replace(self)
        if name == "Q0":
            cfg.use_lora_ablation = True
        elif name == "Q0'":
            cfg.use_lora_ablation = False
        else:
            # Q1/Q2/Q2'/Q3 在 Q0 基础上叠加
            cfg.use_lora_ablation = True
        return cfg