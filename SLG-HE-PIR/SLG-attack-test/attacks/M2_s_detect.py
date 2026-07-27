"""M-2: S 方模型推断攻击。

攻击逻辑（docs/攻击类测试方案.md §2.2）：
- M-2A: 层数/结构探测（Jacobian 秩估计 → LoRA 秩）
- M-2B: H_M 分布 MMD vs 公开预训练模型
"""
from __future__ import annotations

from attacks.base import BaseAttack
from evaluation.metrics import AttackVerdict


class M2SDetectAttack(BaseAttack):
    """M-2: S 方模型推断。"""

    ATTACK_ID = "M2"
    ATTACK_NAME = "S 方模型推断"
    TARGET = "M 后 16 层结构（层数、LoRA 秩）"
    THREAT_LEVEL = "LOW"
    GPU_REQUIRED = False

    def run(self) -> list:
        verdicts = []
        attack_data = self.load_attack_data()

        if "H_M" not in attack_data:
            return [AttackVerdict(
                attack_id=self.ATTACK_ID,
                metric="availability",
                value=0.0,
                chance_level=0.0,
                p_value=None,
                n_samples=0,
                verdict="INCONCLUSIVE",
                notes="缺少 H_M 数据",
            )]

        H_M_list = attack_data["H_M"]
        n_samples = len(H_M_list)

        verdicts.append(AttackVerdict(
            attack_id=f"{self.ATTACK_ID}-A",
            metric="jacobian_rank_estimate",
            value=0.0,
            chance_level=8.0,
            p_value=None,
            n_samples=n_samples,
            verdict="INCONCLUSIVE",
            notes="Jacobian 秩估计需 GPU + 数值微分（stub）",
        ))

        verdicts.append(AttackVerdict(
            attack_id=f"{self.ATTACK_ID}-B",
            metric="mmd_vs_pretrained",
            value=0.0,
            chance_level=0.0,
            p_value=None,
            n_samples=n_samples,
            verdict="INCONCLUSIVE",
            notes="MMD 需公开预训练模型对照（stub）",
        ))

        return verdicts
