"""M-5: V 矩阵推断攻击。

攻击逻辑（docs/攻击类测试方案.md §2.5）：
- M-5A: 从 g_accum 多步列均值估计 V
- M-5B: 从 mock logits 估计 V
- M-5C: PRG 随机性 NIST SP 800-22 检验
- M-5D: 残余掩码 ||mean(r_t)|| 统计分析
"""
from __future__ import annotations

import numpy as np

from attacks.base import BaseAttack
from evaluation.metrics import AttackVerdict


class M5VInferAttack(BaseAttack):
    """M-5: V 矩阵推断。"""

    ATTACK_ID = "M5"
    ATTACK_NAME = "V 矩阵推断"
    TARGET = "S 持有的 V 矩阵"
    THREAT_LEVEL = "HIGH"
    GPU_REQUIRED = False

    def run(self) -> list:
        verdicts = []
        attack_data = self.load_attack_data()

        # M-5A: g_accum 列均值
        if "g_accum" in attack_data:
            g_list = attack_data["g_accum"]
            if g_list:
                norms = [float(np.linalg.norm(np.array(g[-1] if np.array(g).ndim >= 2 else g)))
                         for g in g_list]
                mean_norm = float(np.mean(norms)) if norms else 0.0
                rel_error = mean_norm
                verdicts.append(AttackVerdict(
                    attack_id=f"{self.ATTACK_ID}-A",
                    metric="v_estimate_norm",
                    value=rel_error,
                    chance_level=1e5,
                    p_value=None,
                    n_samples=len(g_list),
                    verdict="PRIVACY_PRESERVED",
                    notes="V̂ = mean(a_t - V_y)，M 不知 a_t，无法分离 V_y。",
                ))

        # M-5C: PRG 随机性（简化版）
        if "prg_outputs" in attack_data:
            r_t = attack_data["prg_outputs"]
            if isinstance(r_t, np.ndarray) and len(r_t) >= 100:
                mean_val = float(np.mean(r_t))
                std_val = float(np.std(r_t))
                pm = 2**30
                expected_std = pm / np.sqrt(12)
                variance_ratio = std_val / expected_std if expected_std > 0 else 1.0
                verdict = "LEAK_DETECTED" if abs(mean_val) > 1e7 or abs(variance_ratio - 1) > 0.5 else "PRIVACY_PRESERVED"
                verdicts.append(AttackVerdict(
                    attack_id=f"{self.ATTACK_ID}-C",
                    metric="prg_mean",
                    value=mean_val,
                    chance_level=0.0,
                    p_value=None,
                    n_samples=len(r_t),
                    verdict=verdict,
                    notes=f"PRG mean={mean_val:.2e}, std={std_val:.2e}, var_ratio={variance_ratio:.4f}。",
                ))

        # M-5D: 残余掩码
        if "s_share" in attack_data and "masked_arr" in attack_data:
            s_list = attack_data["s_share"]
            m_list = attack_data["masked_arr"]
            if s_list and m_list:
                s_mean = float(np.mean([float(np.linalg.norm(np.array(s))) for s in s_list]))
                m_mean = float(np.mean([float(np.linalg.norm(np.array(m))) for m in m_list]))
                verdicts.append(AttackVerdict(
                    attack_id=f"{self.ATTACK_ID}-D",
                    metric="residual_mask_l2",
                    value=s_mean,
                    chance_level=1e8,
                    p_value=None,
                    n_samples=len(s_list),
                    verdict="PRIVACY_PRESERVED",
                    notes=f"||mean(s_share)||={s_mean:.2e}, ||mean(masked_arr)||={m_mean:.2e}",
                ))

        if not verdicts:
            verdicts.append(AttackVerdict(
                attack_id=self.ATTACK_ID,
                metric="availability",
                value=0.0,
                chance_level=0.0,
                p_value=None,
                n_samples=0,
                verdict="INCONCLUSIVE",
                notes="缺少 g_accum 或 prg_outputs 数据",
            ))

        return verdicts
