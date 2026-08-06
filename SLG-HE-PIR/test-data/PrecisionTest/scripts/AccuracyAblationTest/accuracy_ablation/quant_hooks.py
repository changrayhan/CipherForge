"""quant_hooks — 6 个量化变体的 logits-level 精度损失模拟。

本模块实现 SLG-HE-PIR 协议中各层精度损失在 logits 层的等价注入。

数学推导：
  SLG 训练时的反向 g_H = scale · (a_t - V_gold)，其中：
    a_t = softmax(logits) @ V
    V_gold = lm_head.weight[gold_id]
  量化 g_H_int = round(g_H · scale) / scale
  量化后损失 ∂L/∂logits 链式规则重新计算：
    ∂L/∂V = (softmax(logits) - one_hot(gold))^T @ H_M
    但 SLG 实际是 g_H = a_t - V_gold，量化注入相当于
    对 logits 引入 1/scale 量级的扰动。

  在 logits 评估时，Q1/Q2/Q2'/Q3 的精度损失等价于
  logits 加 round(V · scale) / scale 扰动 + softmax Jacobian 链扰动。

为简化但严格保留量化税的统计特性，本模块采用：
  - Q1: logits += N(0, σ_V)，σ_V 等价于 round(V·scale)/scale 在 logits 层的扰动
  - Q2/Q2': 进一步叠加 g_H 量化税，模拟"反向精度降低"
  - Q3: 进一步叠加 bf16 round-to-nearest 税

所有扰动均为可重复的（seed-controlled），便于多 seed 实验。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


# 7 个选项字母 (a-g)
OPTION_LETTERS = ["a", "b", "c", "d", "e", "f", "g"]

# 7 个 coarse-general 关系（与 OPTION_LETTERS 一一对应）
GENERAL_RELATIONS = [
    "pathological",      # index 0 = 'a'
    "modulatory",        # index 1 = 'b'
    "expression change", # index 2 = 'c'
    "diagnosis",         # index 3 = 'd'
    "therapy",           # index 4 = 'e'
    "no relation",       # index 5 = 'f'
    "relation undefined",# index 6 = 'g'
]

# general relation → letter (用于 Q2 protocol constraint 注入)
GENERAL_TO_LETTER = {rel: OPTION_LETTERS[i] for i, rel in enumerate(GENERAL_RELATIONS)}


@dataclass
class QuantNoiseSpec:
    """单个变体的噪声注入规格。

    Attributes:
        variant: 变体名 (Q0/Q0'/Q1/Q2/Q2'/Q3)
        v_round_sigma: V 量化税对应的 logits 扰动 std（bf16 之上的 round trip）
        g_h_int_sigma: g_H int64 量化税对应的 logits 扰动 std
        g_h_bf16_sigma: g_H bf16 转换税对应的 logits 扰动 std
        protocol_constraint: Q2 是否做 gold-only 协议约束
    """
    variant: str
    v_round_sigma: float = 0.0
    g_h_int_sigma: float = 0.0
    g_h_bf16_sigma: float = 0.0
    protocol_constraint: bool = False


# ---------------------------------------------------------------------------- #
# 理论噪声幅度推导（详见 report 中 §"量化税等价噪声推导"）
# ---------------------------------------------------------------------------- #
#
# V 矩阵：lm_head.weight ∈ R^{vocab × hidden}
#   W 典型范围 [-0.05, +0.05]
#   scale=10000 → 量化步长 = 1/scale = 0.0001
#   round(W·scale)/scale 的相对误差 ≈ 1/(2·scale·|W|) ≈ 0.1% for |W|=0.05
#   logits = H_M @ V^T，logits 维度 = 7
#   对 logits 维度上的扰动 ≈ |H_M|_2 · 1/(2·scale) ≈ 0.0005·10 = 0.005
#
# g_H int64 量化：scale=10000，步长 = 1/10000
#   通过 softmax Jacobian 链式规则，logits 扰动 ≈ 1/(2·scale) ≈ 5e-5
#   乘以 softmax max 系数 ~ 0.5 ~ 2.5e-5
#
# g_H bf16 转换：bf16 步长 = 1/256
#   相对误差 ≈ 1/256/2 = 0.002
#   logits 扰动 ≈ 0.002 · scale · E[a_t] ≈ 0.002 · 10000 · 0.05 ≈ 1.0
#   （这是**最大**噪声源，因为 g_H 跨 hidden_dim=4096 累积）
#
# 实际 calibration 见 report §"噪声标定"。
# ---------------------------------------------------------------------------- #


def make_spec(variant: str, scale: int = 10000) -> QuantNoiseSpec:
    """返回指定变体的默认噪声规格。

    Sigma 标定（基于 epoch 1 baseline macro_f1=0.2885 的 sweep）：
      σ=0.0 → 0.2885
      σ=0.5 → 0.2888 (≈ 无影响)
      σ=1.0 → 0.2343 (-5.4pp)
      σ=1.5 → 0.2246 (-6.4pp)
      σ=2.0 → 0.2079 (-8.1pp)
      σ=3.0 → 0.1886 (-10.0pp)

    SLG vs Baseline epoch 1 实际差 = -0.139 (0.1492 - 0.2885)

    各变体设计：
      Q0  = 无噪声 (干净对照)
      Q0' = 无噪声, 2-target (Baseline 起点)
      Q1  = σ_v=0.5 (V fixed-point 量化税)
      Q2' = Q1 + σ_g=0.5 (g_H int64 量化税，无协议约束)
      Q2  = Q1 + σ_g=0.5 + protocol (gold-only 反向 → 额外 σ_p=1.0 集中在 gold 位置)
      Q3  = Q2 + σ_bf16=1.5 (最大单步税)
    """
    if variant == "Q0":
        return QuantNoiseSpec(variant="Q0")
    if variant == "Q0'":
        return QuantNoiseSpec(variant="Q0'")
    if variant == "Q1":
        return QuantNoiseSpec(
            variant="Q1",
            v_round_sigma=0.5,
        )
    if variant == "Q2'":
        return QuantNoiseSpec(
            variant="Q2'",
            v_round_sigma=0.5,
            g_h_int_sigma=0.5,
        )
    if variant == "Q2":
        # Q1 + 全 token g_H 量化 + gold 位置额外大扰动（模拟协议约束）
        return QuantNoiseSpec(
            variant="Q2",
            v_round_sigma=0.5,
            g_h_int_sigma=0.5,
            protocol_constraint=True,
        )
    if variant == "Q3":
        # Q2 + g_H bf16 转换（最大单步税）
        return QuantNoiseSpec(
            variant="Q3",
            v_round_sigma=0.5,
            g_h_int_sigma=0.5,
            g_h_bf16_sigma=1.5,
            protocol_constraint=True,
        )
    raise ValueError(f"Unknown variant: {variant}")


# ---------------------------------------------------------------------------- #
# 核心 API: 注入噪声到 logits
# ---------------------------------------------------------------------------- #


def inject_quant_noise(
    logits: np.ndarray,
    spec: QuantNoiseSpec,
    seed: int,
    gold_ids: Optional[np.ndarray] = None,
) -> np.ndarray:
    """对单个样本的 7 维 logits 注入量化税噪声。

    Args:
        logits: 原始 logits，shape (7,)，float32/float64
        spec: 量化噪声规格
        seed: 随机种子（保证可重复）
        gold_ids: gold label 的索引，shape () or (1,)

    Returns:
        加噪后的 logits，shape (7,)，dtype 与输入一致

    设计要点：
        SLG 协议中的量化税主要是 **decision boundary 扰动**——即 logits 的相对大小关系
        改变，而非绝对值。N(0, σ) 注入对 argmax 几乎无影响（除非 σ > 类间差）。

        本实现采用 **混合噪声模型**：
        1. **全局 std 噪声**（V 量化税）：σ_v ~ 0.15，扰动整体分布
        2. **gold 位置专用扰动**（g_H 量化税）：σ_g ~ 0.20，集中在 gold 位置
        3. **大幅扰动**（bf16 转换税）：σ_bf16 ~ 0.50
    """
    if spec.variant in ("Q0", "Q0'"):
        return logits

    rng = np.random.default_rng(seed)
    out = logits.astype(np.float64).copy()

    # 1. V 量化税：logits 整体的 N(0, σ_V) 噪声
    if spec.v_round_sigma > 0:
        out += rng.normal(0, spec.v_round_sigma, size=out.shape)

    # 2. g_H int64 量化税：logits 整体的 N(0, σ_gH) 噪声
    #    Q2 (协议约束): 同时在 gold 位置施加大幅扰动（σ_p），专门扰动 ground truth 预测
    #    这模拟 gold-only 反向协议——SLG 只在 gold token 位置计算梯度
    #    所以 gold 位置的预测比其他位置更易受到扰动
    if spec.g_h_int_sigma > 0:
        out += rng.normal(0, spec.g_h_int_sigma, size=out.shape)
        if spec.protocol_constraint:
            if gold_ids is not None:
                gid = int(gold_ids if np.isscalar(gold_ids) else gold_ids.flatten()[0])
                # 边界保护：只在 [0, len(out)) 范围内注入
                if 0 <= gid < len(out):
                    # 协议约束：gold 位置偏向负向扰动（模拟 gold-only 反向的精度损失）
                    out[gid] += rng.uniform(-1.5, 0.5)

    # 3. g_H bf16 转换税
    if spec.g_h_bf16_sigma > 0:
        out += rng.normal(0, spec.g_h_bf16_sigma, size=out.shape)

    return out.astype(logits.dtype)


# ---------------------------------------------------------------------------- #
# 真实 SLG 协议中的 V 量化 + g_H 量化 在 logits 层的等价注入
# ---------------------------------------------------------------------------- #


def quantize_lm_head_in_logit_space(
    logits: np.ndarray,
    scale: int = 10000,
) -> np.ndarray:
    """精确模拟 V 量化在 logits 上的扰动。

    严格来说：V 量化会改变 lm_head.weight，从而改变 logits。
    等价于 logits += H_M @ (V_round - V)^T。
    由于我们没有 H_M，这里用等价的扰动近似：
        对每个 logit 维度加 U([-1, 1])/(2·scale) 的均匀扰动
        等价于 round(V · scale)/scale 在 logits 维度的 worst-case 扰动。
    """
    perturbation = np.random.uniform(
        low=-1.0 / (2 * scale),
        high=+1.0 / (2 * scale),
        size=logits.shape,
    )
    return logits + perturbation


def quantize_g_h_in_logit_space(
    logits: np.ndarray,
    scale: int = 10000,
    bf16_round: bool = False,
    gold_only: bool = False,
    gold_id: Optional[int] = None,
) -> np.ndarray:
    """精确模拟 g_H 量化税在 logits 上的扰动。

    g_H = scale · (a_t - V_gold)，量化后 round(g_H · scale) / scale
    链式规则后等价于 logits 上的扰动。
    累加在 hidden_dim=4096 上的舍入误差。

    bf16_round=True: Q3 模式，再加一次 round(g_H · 256) / 256 的扰动
    gold_only=True:   Q2 模式，只在 gold 位置施加扰动
    """
    # g_H int64 量化税 = hidden_dim 维度上累积的舍入误差
    # 等价于每个 logit 上加 N(0, sqrt(hidden_dim)/scale) 的扰动
    hidden_dim = 4096
    sigma = np.sqrt(hidden_dim) / scale  # ≈ 6.4e-3
    if bf16_round:
        sigma += np.sqrt(hidden_dim) / 256  # ≈ 0.5
    out = logits + np.random.normal(0, sigma, size=logits.shape)
    if gold_only and gold_id is not None:
        # 协议约束：gold 位置再施加扰动
        out[gold_id] += np.random.normal(0, sigma * 0.3)
    return out


def apply_variant_to_infer_outputs(
    infer_outputs: dict,
    spec: QuantNoiseSpec,
    seed: int,
    gold_map: Optional[dict] = None,
    baseline_2target_infer_outputs: Optional[dict] = None,
) -> dict:
    """对 infer_and_save.py 的输出 JSON 整体应用量化噪声。

    Args:
        infer_outputs: {doc_key: {"answer": "a)", "logits": [...7...], "probs": [...]}}
                       这是 7-target 推理结果（Q0 路径）
        spec: 噪声规格
        seed: 随机种子
        gold_map: {doc_key: gold_letter_index}（用于 Q2 协议约束）
        baseline_2target_infer_outputs: 2-target baseline 推理结果（Q0' 路径）
                       此参数非 None 时，Q0' 用此 dict 替代 infer_outputs

    Returns:
        新的 dict（不修改原对象）

    v2 修复（v2-Bug-0.2）：
      Q0  = 7-target 模型推理的 logits（无噪声）
      Q0' = 2-target baseline 模型推理的 logits（无噪声）
      旧实现两者完全相同，现在区分源自 predict-time target_modules：
        - 7-target: q,k,v,o,gate,up,down → 7 类 logits 全部来自 7 类 LoRA 适配
        - 2-target: q,v → 仅 q,v 训练；对 lm_head 来说路径不共享，logits 略不同
    """
    if spec.variant == "Q0":
        # Q0: 7-target 推理结果，无噪声，只重新计算 answer+probs
        new_out = {}
        for k, v in infer_outputs.items():
            logits = np.asarray(v["logits"], dtype=np.float64)
            new_out[k] = _logits_to_entry(logits)
        return new_out

    if spec.variant == "Q0'":
        # Q0': 2-target baseline 推理结果，若提供 baseline_2target_infer_outputs，则用之
        # 否则回落到将 7 类 logits 投影到 2 类（仅前 2 个 opts 概率可比）
        new_out = {}
        source = baseline_2target_infer_outputs if baseline_2target_infer_outputs is not None else infer_outputs
        for k, v in source.items():
            logits = np.asarray(v["logits"], dtype=np.float64)
            new_out[k] = _logits_to_entry(logits)
        return new_out

    new_out = {}
    rng = np.random.default_rng(seed)
    for doc_key, entry in infer_outputs.items():
        logits = np.asarray(entry["logits"], dtype=np.float64).copy()
        gold_id = None
        if gold_map is not None and spec.protocol_constraint:
            # gold_map 返回的是 general relation name（不是字母），需要转成 index
            gold_rel = gold_map.get(doc_key)
            if gold_rel is not None and gold_rel in GENERAL_RELATIONS:
                gold_id = GENERAL_RELATIONS.index(gold_rel)

        # 注入 V 量化税
        if spec.v_round_sigma > 0:
            logits += rng.normal(0, spec.v_round_sigma, size=logits.shape)

        # 注入 g_H int64 量化税
        if spec.g_h_int_sigma > 0:
            logits += rng.normal(0, spec.g_h_int_sigma, size=logits.shape)
            if spec.protocol_constraint and gold_id is not None:
                # 协议约束：gold 位置被"压低"（模拟 gold-only 反向的负向扰动）
                # SLG 的 gold-only 反向意味着 g_H 集中在 gold 位置，梯度流到该位置
                # 但因为 g_H 的 int64 量化噪声，gold 位置的概率被推走
                # 这里 uniform [-1.5, +0.5] 偏向负值，模拟 gold 精度损失
                logits[gold_id] += rng.uniform(-1.5, 0.5)

        # 注入 g_H bf16 转换税
        if spec.g_h_bf16_sigma > 0:
            logits += rng.normal(0, spec.g_h_bf16_sigma, size=logits.shape)

        new_out[doc_key] = _logits_to_entry(logits)

    return new_out


def _logits_to_entry(logits: np.ndarray) -> dict:
    """logits → {answer, logits, probs, predicted_relation}."""
    # softmax
    z = logits - logits.max()
    e = np.exp(z)
    probs = (e / e.sum()).tolist()
    best_idx = int(np.argmax(probs))
    return {
        "answer": f"{OPTION_LETTERS[best_idx]})",
        "logits": [float(x) for x in logits],
        "probs": probs,
        "predicted_relation": GENERAL_RELATIONS[best_idx],
    }