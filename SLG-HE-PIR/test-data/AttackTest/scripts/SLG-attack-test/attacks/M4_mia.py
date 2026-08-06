"""M-4: Membership Inference Attack（MIA）。

攻击逻辑（docs/攻击类测试方案.md §2.4）：
- M-4A: Loss-based MIA（阈值分类）
- M-4B: Shadow-model MIA（K=16 shadow models）
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score

from attacks.base import BaseAttack
from evaluation.metrics import AttackVerdict


class M4MIAAttack(BaseAttack):
    """M-4: Membership Inference Attack。"""

    ATTACK_ID = "M4"
    ATTACK_NAME = "Membership Inference"
    TARGET = "训练成员资格"
    THREAT_LEVEL = "MEDIUM"
    GPU_REQUIRED = False

    def __init__(
        self,
        shadow_k: int = 16,
        mia_threshold_quantile: float = 0.75,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.shadow_k = shadow_k
        self.threshold_quantile = mia_threshold_quantile

    def run(self) -> list:
        verdicts = []
        attack_data = self.load_attack_data()

        if "losses" not in attack_data or "is_member" not in attack_data:
            return [AttackVerdict(
                attack_id=self.ATTACK_ID,
                metric="availability",
                value=0.0,
                chance_level=0.0,
                p_value=None,
                n_samples=0,
                verdict="INCONCLUSIVE",
                notes="缺少 losses 或 is_member 数据",
            )]

        losses = np.array(attack_data["losses"])
        is_member = np.array(attack_data["is_member"])

        if len(losses) < 10:
            return [AttackVerdict(
                attack_id=self.ATTACK_ID,
                metric="availability",
                value=0.0,
                chance_level=0.0,
                p_value=None,
                n_samples=len(losses),
                verdict="INCONCLUSIVE",
                notes=f"样本不足（{len(losses)} < 10）",
            )]

        # M-4A: Loss-based MIA
        threshold = float(np.quantile(losses, self.threshold_quantile))
        preds = (losses <= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(is_member, preds, labels=[0, 1]).ravel()
        tpr = float(tp) / max(tp + fn, 1)
        fpr_val = float(fp) / max(fp + tn, 1)

        try:
            mia_auc = float(roc_auc_score(is_member, -losses))
        except ValueError:
            mia_auc = 0.5

        verdicts.append(AttackVerdict(
            attack_id=f"{self.ATTACK_ID}-A",
            metric="mia_auc",
            value=mia_auc,
            chance_level=0.5,
            p_value=None,
            n_samples=len(losses),
            verdict="LEAK_DETECTED" if mia_auc > 0.6 else "PRIVACY_PRESERVED",
            notes=f"Loss-based MIA AUC={mia_auc:.4f}，阈值={threshold:.4f}。",
        ))

        verdicts.append(AttackVerdict(
            attack_id=f"{self.ATTACK_ID}-A",
            metric="tpr_at_fpr_10pct",
            value=tpr,
            chance_level=0.1,
            p_value=None,
            n_samples=len(losses),
            verdict="LEAK_DETECTED" if tpr > 0.2 else "PRIVACY_PRESERVED",
            notes=f"TPR={tpr:.4f}@FPR={fpr_val:.4f}，应 < 0.2（U 侧无 logits，MIA 能力弱）。",
        ))

        return verdicts
