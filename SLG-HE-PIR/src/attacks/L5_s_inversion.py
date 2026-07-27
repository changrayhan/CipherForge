"""L-5: S 方输入重构攻击。

攻击逻辑（docs/攻击类测试方案.md §5）：
S 持有 H_M + V，无 M 的后 16 层 + LoRA（关键盲区）。
- L-5A: 公开预训练后 16 层反演（弱白盒）
- L-5B: H_M 残差反演（无 LoRA 校准）
"""
from __future__ import annotations

from SLG_attack_test.attacks.base import BaseAttack
from SLG_attack_test.evaluation.metrics import AttackVerdict


class L5SInversionAttack(BaseAttack):
    """L-5: S 方输入重构。"""

    ATTACK_ID = "L5"
    ATTACK_NAME = "S 方输入重构"
    TARGET = "输入 token 序列 x"
    THREAT_LEVEL = "MEDIUM"
    GPU_REQUIRED = True

    def __init__(self, variant: str = "public_backbone", **kwargs):
        super().__init__(**kwargs)
        self.variant = variant

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
                notes="缺少 H_M 数据（S 无 LoRA，需公开预训练 backbone）",
            )]

        H_M_list = attack_data["H_M"]
        n_samples = len(H_M_list)

        verdicts.append(AttackVerdict(
            attack_id=f"{self.ATTACK_ID}-{self.variant}",
            metric="token_reconstruction_rate",
            value=0.0,
            chance_level=0.01,
            p_value=None,
            n_samples=n_samples,
            verdict="INCONCLUSIVE",
            notes=f"variant={self.variant}，S 无 LoRA 校准，预期 Token Rec Rate << L-4A。",
        ))

        return verdicts
