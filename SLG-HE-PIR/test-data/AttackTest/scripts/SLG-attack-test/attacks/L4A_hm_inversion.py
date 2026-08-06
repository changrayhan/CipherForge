"""L-4A: H_M Smashed-Data Inversion。

攻击逻辑（docs/攻击类测试方案.md §4.2）：
用 H_M（后 16 层输出）替代 H_U 作为 inverter 输入。
预期 Token Reconstruction Rate < 5%。
"""
from __future__ import annotations

from attacks.base import BaseAttack
from evaluation.metrics import AttackVerdict


class L4AHMInversionAttack(BaseAttack):
    """L-4A: H_M Smashed-Data Inversion。"""

    ATTACK_ID = "L4A"
    ATTACK_NAME = "H_M Smashed-Data Inversion"
    TARGET = "输入 token 序列 x"
    THREAT_LEVEL = "HIGH"
    GPU_REQUIRED = True

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
                notes="缺少 H_M 数据（需 GPU 训练 inverter）",
            )]

        H_M_list = attack_data["H_M"]
        n_samples = len(H_M_list)

        verdicts.append(AttackVerdict(
            attack_id=self.ATTACK_ID,
            metric="token_reconstruction_rate",
            value=0.0,
            chance_level=0.01,
            p_value=None,
            n_samples=n_samples,
            verdict="INCONCLUSIVE",
            notes="需 GPU 训练 inverter。预期 Token Rec Rate < 5%（H_M 距 token embedding 远）。",
        ))

        return verdicts
