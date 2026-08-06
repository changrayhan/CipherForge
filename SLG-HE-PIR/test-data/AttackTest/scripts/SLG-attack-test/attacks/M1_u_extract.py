"""M-1: U 方模型提取攻击。

攻击逻辑（docs/攻击类测试方案.md §2.1）：
U 通过 logits 蒸馏提取 M 的模型能力。

注：当前协议不把 logits 返回给 U，需 MockPartyU(logits回流=True)。
"""
from __future__ import annotations

from attacks.base import BaseAttack
from evaluation.metrics import AttackVerdict


class M1UExtractAttack(BaseAttack):
    """M-1: U 方模型提取。"""

    ATTACK_ID = "M1"
    ATTACK_NAME = "U 方模型提取"
    TARGET = "M 后 16 层模型权重"
    THREAT_LEVEL = "HIGH"
    GPU_REQUIRED = False

    def __init__(
        self,
        query_budgets: list[int] = [100, 1000, 5000],
        distill_T: float = 1.0,
        distill_epochs: int = 20,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.query_budgets = query_budgets
        self.distill_T = distill_T
        self.distill_epochs = distill_epochs

    def run(self) -> list:
        verdicts = []
        attack_data = self.load_attack_data()

        if "logits" not in attack_data:
            return [AttackVerdict(
                attack_id=self.ATTACK_ID,
                metric="availability",
                value=0.0,
                chance_level=0.0,
                p_value=None,
                n_samples=0,
                verdict="INCONCLUSIVE",
                notes="协议默认不返回 logits 给 U，需 MockPartyU(logits回流=True)。"
                      f"query_budgets={self.query_budgets}，distill_T={self.distill_T}。",
            )]

        logits = attack_data["logits"]
        n_samples = len(logits)

        verdicts.append(AttackVerdict(
            attack_id=self.ATTACK_ID,
            metric="fidelity",
            value=0.0,
            chance_level=0.7,
            p_value=None,
            n_samples=n_samples,
            verdict="INCONCLUSIVE",
            notes=f"需训练 surrogate model（logits 蒸馏），当前为 stub。"
                  f"Fidelity > 0.7 → LEAK_DETECTED。",
        ))

        return verdicts
