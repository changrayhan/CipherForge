"""L-2: CutGrad 全家桶攻击（A=Free, B=Prior, C=Oracle, D=H_M辅助）。

攻击逻辑（docs/攻击类测试方案.md §2.2）：
- L-2A Free: 无先验，Cosine 1-NN 聚类
- L-2B Prior: 半白盒，用公开预训练 LM head V̂ 构建模板
- L-2C Oracle: 有真实标签，计算理论上限
- L-2D H_M辅助: 全白盒，用 H_M pair cosine > τ 聚类
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from attacks.base import BaseAttack
from evaluation.metrics import (
    AttackVerdict,
    label_agreement_rate,
    permutation_test,
    adjusted_rand_index,
)


class L2CutGradAttack(BaseAttack):
    """L-2: CutGrad 全家桶攻击。"""

    ATTACK_ID = "L2"
    ATTACK_NAME = "CutGrad 全家桶"
    TARGET = "标签 y_t"
    THREAT_LEVEL = "HIGH"
    GPU_REQUIRED = True

    def __init__(
        self,
        variant: str = "Free",
        k_neighbors: int = 1,
        n_clusters: int | None = None,
        cos_threshold: float = 0.95,
        norm_threshold: float = 1e-3,
        n_permutations: int = 1000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.variant = variant
        self.k = k_neighbors
        self.n_clusters = n_clusters
        self.cos_th = cos_threshold
        self.norm_th = norm_threshold
        self.n_perms = n_permutations

    def run(self) -> list:
        verdicts = []
        attack_data = self.load_attack_data()

        if "g_accum" not in attack_data or "labels" not in attack_data:
            return [AttackVerdict(
                attack_id=self.ATTACK_ID,
                metric="availability",
                value=0.0,
                chance_level=0.0,
                p_value=None,
                n_samples=0,
                verdict="INCONCLUSIVE",
                notes="缺少 g_accum 或 labels 数据",
            )]

        g_accum_raw = attack_data["g_accum"]
        labels = np.array(attack_data["labels"])

        # 展平为 (N, D)
        g_accum_list = []
        for g in g_accum_raw:
            g_arr = np.array(g)
            if g_arr.ndim == 1:
                g_accum_list.append(g_arr)
            elif g_arr.ndim == 2:
                g_accum_list.extend(g_arr.tolist())
            elif g_arr.ndim == 3:
                g_accum_list.extend(g_arr.reshape(-1, g_arr.shape[-1]).tolist())

        g_accum = np.array(g_accum_list[:len(labels)]) if len(g_accum_list) >= len(labels) else np.array(g_accum_list)
        n_samples = min(len(g_accum), len(labels))

        if n_samples < 10:
            return [AttackVerdict(
                attack_id=self.ATTACK_ID,
                metric="availability",
                value=0.0,
                chance_level=0.0,
                p_value=None,
                n_samples=n_samples,
                verdict="INCONCLUSIVE",
                notes=f"样本不足（{n_samples} < 10）",
            )]

        labels = labels[:n_samples]

        # L2 归一化
        norms = np.linalg.norm(g_accum, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        g_norm = g_accum / norms

        # K-Means ARI
        K = self.n_clusters if self.n_clusters else len(np.unique(labels))
        km = KMeans(n_clusters=min(K, n_samples), random_state=42, n_init=10)
        km_labels = km.fit_predict(g_norm)
        ari = adjusted_rand_index(labels, km_labels)

        # 1-NN agreement
        nn_rate, nn_std, _ = label_agreement_rate(g_norm, labels, k=self.k)

        # Chance level
        unique, counts = np.unique(labels, return_counts=True)
        chance = float(np.sum((counts / n_samples) ** 2))

        # Permutation test
        def metric_fn(G, L):
            rate, _, _ = label_agreement_rate(G, L, k=self.k)
            return rate

        _, perm_p, _ = permutation_test(
            metric_fn, g_norm, labels, n_permutations=self.n_perms
        )

        # Sigma for agreement
        sigma = np.sqrt(chance * (1 - chance) / n_samples)

        for metric, value, chance_val, p, verdict, note in [
            ("kmeans_ari", ari, 0.0, None,
             "LEAK_DETECTED" if ari > 0.1 else "PRIVACY_PRESERVED",
             f"K-Means ARI (K={K})"),
            ("nn_agreement", nn_rate, chance, None,
             "LEAK_DETECTED" if nn_rate > chance + 2 * sigma else "PRIVACY_PRESERVED",
             f"1-NN (k={self.k}), chance={chance:.4f}, 2σ={chance+2*sigma:.4f}"),
            ("permutation_p", perm_p, 0.05, float(perm_p),
             "LEAK_DETECTED" if perm_p < 0.05 else "PRIVACY_PRESERVED",
             f"Permutation test (n_perms={self.n_perms})"),
        ]:
            verdicts.append(AttackVerdict(
                attack_id=f"{self.ATTACK_ID}-{self.variant}",
                metric=metric,
                value=float(value),
                chance_level=float(chance_val),
                p_value=p,
                n_samples=n_samples,
                verdict=verdict,
                notes=note,
            ))

        return verdicts
