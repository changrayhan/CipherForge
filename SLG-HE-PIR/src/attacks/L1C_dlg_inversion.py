"""L-1C: SAP/DLG 风格梯度反演攻击。

攻击逻辑（docs/攻击类测试方案.md §1.3）：
M 仅靠 g_H = a_t - V_y 能否反推输入 token id。

使用 DLG/TAG 风格的随机 dummy input 迭代优化。
"""
from __future__ import annotations

from SLG_attack_test.attacks.base import BaseAttack
from SLG_attack_test.evaluation.metrics import AttackVerdict


class L1CDLGInversionAttack(BaseAttack):
    """L-1C: DLG/TAG 梯度反演攻击。"""

    ATTACK_ID = "L1C"
    ATTACK_NAME = "SAP/DLG 梯度反演"
    TARGET = "输入 token 序列 x"
    THREAT_LEVEL = "HIGH"
    GPU_REQUIRED = True

    def __init__(
        self,
        n_optimization_steps: int = 1000,
        lambda_lm: float = 0.1,
        gumbel_T: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_steps = n_optimization_steps
        self.lambda_lm = lambda_lm
        self.gumbel_T = gumbel_T

    def run(self) -> list:
        verdicts = []
        attack_data = self.load_attack_data()

        if "g_H" not in attack_data or "H_M" not in attack_data:
            return [AttackVerdict(
                attack_id=self.ATTACK_ID,
                metric="availability",
                value=0.0,
                chance_level=0.0,
                p_value=None,
                n_samples=0,
                verdict="INCONCLUSIVE",
                notes="缺少 g_H 或 H_M 数据（需 GPU + LLM 模型）",
            )]

        g_H_list = attack_data["g_H"]
        H_M_list = attack_data["H_M"]
        labels = attack_data.get("labels", [None] * len(g_H_list))

        verdicts.append(AttackVerdict(
            attack_id=self.ATTACK_ID,
            metric="token_reconstruction_rate",
            value=0.0,
            chance_level=0.01,
            p_value=None,
            n_samples=len(g_H_list),
            verdict="INCONCLUSIVE",
            notes=f"需要 GPU + LLM 前向/反向传播（n_steps={self.n_steps}，λ={self.lambda_lm}）。"
                  f"SLG-HE-PIR 的梯度不是 one-hot，反演目标为输入 token，预期重构率 ≈ 0。",
        ))

        return verdicts
