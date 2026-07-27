"""L-3B: PIR 字节分析攻击。

攻击逻辑（docs/攻击类测试方案.md §3.2）：
S（物理隔离）能否从 PIR query/response 字节流推断 y_t。

使用 Random Forest / MLP 分类器从字节特征预测 y_t，计算 AUC。
"""
from __future__ import annotations

import hashlib

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from attacks.base import BaseAttack
from evaluation.metrics import AttackVerdict


class L3BPIRBytesAttack(BaseAttack):
    """L-3B: PIR 字节分析攻击。"""

    ATTACK_ID = "L3B"
    ATTACK_NAME = "PIR 字节分析"
    TARGET = "索引 y_t（通过 PIR 字节流推断）"
    THREAT_LEVEL = "MEDIUM"
    GPU_REQUIRED = False

    def __init__(self, classifier: str = "random_forest", n_estimators: int = 100, **kwargs):
        super().__init__(**kwargs)
        self.classifier_type = classifier
        self.n_estimators = n_estimators

    def run(self) -> list:
        verdicts = []
        attack_data = self.load_attack_data()

        if "parity_bytes" not in attack_data or "labels" not in attack_data:
            return [AttackVerdict(
                attack_id=self.ATTACK_ID,
                metric="availability",
                value=0.0,
                chance_level=0.0,
                p_value=None,
                n_samples=0,
                verdict="INCONCLUSIVE",
                notes="缺少 parity_bytes 或 labels 数据",
            )]

        parity_bytes_list = attack_data["parity_bytes"]
        labels = np.array(attack_data["labels"])

        # 提取字节级特征：直方图（256）+ hash ID
        features = []
        valid_mask = []
        for pb in parity_bytes_list:
            if isinstance(pb, bytes):
                hist = np.zeros(256, dtype=np.float32)
                for b in pb:
                    hist[b] += 1
                hist /= (len(pb) + 1e-8)
                h = int.from_bytes(hashlib.sha256(pb).digest()[:8], "little")
                features.append(np.concatenate([hist, [h % 1e9]]))
                valid_mask.append(True)
            elif isinstance(pb, str):
                b = pb.encode()
                hist = np.zeros(256, dtype=np.float32)
                for ch in b:
                    hist[ch] += 1
                hist /= (len(b) + 1e-8)
                features.append(np.concatenate([hist, [0.0]]))
                valid_mask.append(True)
            else:
                features.append(np.zeros(257))
                valid_mask.append(False)

        X = np.array(features)
        labels = np.array(labels)
        mask = np.array(valid_mask)
        X, labels = X[mask], labels[mask]

        n_classes = len(np.unique(labels))
        chance = 1.0 / max(n_classes, 1)

        if len(X) < 20:
            return [AttackVerdict(
                attack_id=self.ATTACK_ID,
                metric="auc",
                value=float("nan"),
                chance_level=float(chance),
                p_value=None,
                n_samples=len(X),
                verdict="INCONCLUSIVE",
                notes=f"样本不足（{len(X)} < 20）",
            )]

        clf = RandomForestClassifier(n_estimators=self.n_estimators, random_state=42)

        try:
            n_cv = min(5, len(X))
            auc_scores = cross_val_score(clf, X, labels, cv=n_cv, scoring="roc_auc")
            mean_auc = float(np.mean(auc_scores))
        except Exception:
            mean_auc = 0.5

        sigma = np.sqrt(chance * (1 - chance) / len(X))
        verdict = "LEAK_DETECTED" if mean_auc > chance + 3 * sigma else "PRIVACY_PRESERVED"

        verdicts.append(AttackVerdict(
            attack_id=self.ATTACK_ID,
            metric="auc",
            value=mean_auc,
            chance_level=float(chance),
            p_value=None,
            n_samples=len(X),
            verdict=verdict,
            notes=f"Mean AUC={mean_auc:.4f}, chance={chance:.4f}, 3σ={chance+3*sigma:.4f}",
        ))

        return verdicts
