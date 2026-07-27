"""M-3: LoRA 参数推断攻击（M 侧内部审计）。

攻击逻辑（docs/攻击类测试方案.md §2.3）：
- M-3A: Δz ≈ V·J_top·Δδ·H_U 反解 Δδ
- M-3B: 累积 g_δ 审计

注：该攻击仅在 M 主机内部可实施，属于内部审计场景。
"""
from __future__ import annotations

import numpy as np

from SLG_attack_test.attacks.base import BaseAttack
from SLG_attack_test.evaluation.metrics import AttackVerdict


class M3LoraInternalsAttack(BaseAttack):
    """M-3: LoRA 参数推断（M 侧内部审计）。"""

    ATTACK_ID = "M3"
    ATTACK_NAME = "LoRA 参数推断"
    TARGET = "M 的 LoRA 权重 Δδ"
    THREAT_LEVEL = "LOW"
    GPU_REQUIRED = False

    def run(self) -> list:
        verdicts = []
        attack_data = self.load_attack_data()

        if "g_delta" not in attack_data:
            return [AttackVerdict(
                attack_id=self.ATTACK_ID,
                metric="availability",
                value=0.0,
                chance_level=0.0,
                p_value=None,
                n_samples=0,
                verdict="INCONCLUSIVE",
                notes="g_delta 仅在 M 内部可观测（内部审计场景）",
            )]

        g_delta_list = attack_data["g_delta"]
        n_samples = len(g_delta_list)

        if g_delta_list:
            delta_norms = [float(np.linalg.norm(np.array(g))) for g in g_delta_list]
            mean_norm = float(np.mean(delta_norms))
        else:
            mean_norm = 0.0

        verdicts.append(AttackVerdict(
            attack_id=self.ATTACK_ID,
            metric="mean_g_delta_norm",
            value=mean_norm,
            chance_level=0.0,
            p_value=None,
            n_samples=n_samples,
            verdict="INCONCLUSIVE",
            notes="M-3 属于内部审计，外部攻击者无法访问 g_delta。"
                  "M 侧可通过 Δδ 轨迹监控 LoRA 更新方向。",
        ))

        return verdicts
