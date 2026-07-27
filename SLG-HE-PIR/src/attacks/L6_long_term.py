"""L-6: 长期训练隐私退化攻击。

攻击逻辑（docs/攻击类测试方案.md §6）：
跨 epoch/step 的轨迹分析，检测长期训练中隐私是否退化。
- G_window[k] = mean(g_accum[k*w : (k+1)*w])
- PCA 保留 95% 方差
- K-Means K=7（GenRel），算 ARI
- 自相关 corr(g^(t), g^(t+k))
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from SLG_attack_test.attacks.base import BaseAttack
from SLG_attack_test.evaluation.metrics import (
    AttackVerdict,
    adjusted_rand_index,
)


class L6LongTermAttack(BaseAttack):
    """L-6: 长期训练隐私退化。"""

    ATTACK_ID = "L6"
    ATTACK_NAME = "长期训练隐私退化"
    TARGET = "标签 y_t（随训练退化）"
    THREAT_LEVEL = "MEDIUM"
    GPU_REQUIRED = False

    def __init__(
        self,
        window_sizes: list[int] = [1, 5, 10, 50],
        pca_variance: float = 0.95,
        n_permutations: int = 1000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.window_sizes = window_sizes
        self.pca_variance = pca_variance
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

        # 展平
        g_accum_list = []
        for g in g_accum_raw:
            g_arr = np.array(g)
            if g_arr.ndim == 1:
                g_accum_list.append(g_arr)
            elif g_arr.ndim >= 2:
                g_accum_list.extend(g_arr.reshape(-1, g_arr.shape[-1]).tolist())

        g_accum = np.array(g_accum_list[:len(labels)]) if len(g_accum_list) >= len(labels) else np.array(g_accum_list)
        labels = labels[:len(g_accum)]

        # 窗口聚合 ARI
        ari_vs_window = {}
        for w in self.window_sizes:
            n_windows = len(g_accum) // w
            if n_windows < 2:
                continue
            windows = []
            w_labels = []
            for k in range(n_windows):
                seg = g_accum[k * w:(k + 1) * w]
                windows.append(np.mean(seg, axis=0))
                seg_labels = labels[k * w:(k + 1) * w]
                w_labels.append(int(np.bincount(seg_labels).argmax()))

            W = np.array(windows)
            if W.shape[0] < 2:
                continue
            pca = PCA(n_components=self.pca_variance)
            try:
                W_pca = pca.fit_transform(W)
            except Exception:
                W_pca = W
            unique_labels = np.unique(w_labels)
            n_clust = max(2, len(unique_labels))
            km = KMeans(n_clusters=n_clust, random_state=42, n_init=10)
            km_labels = km.fit_predict(W_pca)
            ari = adjusted_rand_index(np.array(w_labels), km_labels)
            ari_vs_window[w] = ari

        # 自相关分析
        autocorrs = {}
        if len(g_accum) > 10:
            g_flat = g_accum.reshape(len(g_accum), -1).mean(axis=1)
            for lag in [1, 2, 5]:
                if len(g_flat) > lag:
                    try:
                        ac = float(np.corrcoef(g_flat[:-lag], g_flat[lag:])[0, 1])
                        autocorrs[lag] = ac
                    except Exception:
                        pass

        # 判定
        max_w = max(ari_vs_window.keys(), default=0)
        final_ari = ari_vs_window.get(max_w, 0.0)
        verdict = "LEAK_DETECTED" if final_ari > 0.1 else "PRIVACY_PRESERVED"

        verdicts.append(AttackVerdict(
            attack_id=self.ATTACK_ID,
            metric="kmeans_ari_vs_window",
            value=float(final_ari),
            chance_level=0.0,
            p_value=None,
            n_samples=len(g_accum),
            verdict=verdict,
            notes=f"ARI vs window={ari_vs_window}, autocorrs={autocorrs}."
                  f"若 ARI 随训练单调上升，说明模型对标签坍塌到隐空间。",
        ))

        return verdicts
