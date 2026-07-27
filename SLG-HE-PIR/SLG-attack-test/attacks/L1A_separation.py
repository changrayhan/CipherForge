"""L-1A: g_accum 分量分离攻击。

攻击逻辑（docs/攻击类测试方案.md §1.1）：
M 观测 s_share = scale·a_t - r_t（S→M 明文）与 masked_arr = -V_y·scale + r_t（U→M 解密后），
验证无法从 r_t 的量级分离出 V_y。

PRG 输出 r_t ∈ (-pm/2, +pm/2)，pm ≈ 2^30，scale = 10000
→ 误差量级 ≈ 10^5，远超 V_y ∈ [-0.1, +0.1]
"""
from __future__ import annotations

import numpy as np
from attacks.base import BaseAttack
from evaluation.metrics import AttackVerdict


class L1ASeparationAttack(BaseAttack):
    """L-1A-Sep: 分量分离攻击（验证 PRG 掩码不可分离）。

    攻击目标：验证无法从 g_accum = a_t - V_y + noise 中分离出标签嵌入 V_y。
    """

    ATTACK_ID = "L1A_SEP"
    ATTACK_NAME = "g_accum 分量分离攻击"
    TARGET = "标签 y_t（通过 V_y 推断）"
    THREAT_LEVEL = "HIGH"
    GPU_REQUIRED = True

    def run(self) -> list:
        verdicts = []
        attack_data = self.load_attack_data()

        if "s_share" not in attack_data or "masked_arr" not in attack_data:
            return [AttackVerdict(
                attack_id=self.ATTACK_ID,
                metric="availability",
                value=0.0,
                chance_level=0.0,
                p_value=None,
                n_samples=0,
                verdict="INCONCLUSIVE",
                notes="缺少 s_share 或 masked_arr 数据",
            )]

        s_shares = attack_data["s_share"]
        masked_arrs = attack_data["masked_arr"]
        V_y_true = attack_data.get("V_y", None)

        all_mse = []
        all_linf = []
        scale = 10000.0

        for s_share, masked_arr in zip(s_shares, masked_arrs):
            V_hat = -np.array(masked_arr) / scale
            if V_y_true is not None:
                mse = float(np.mean((V_hat - np.array(V_y_true)) ** 2))
                linf = float(np.max(np.abs(V_hat - np.array(V_y_true))))
                all_mse.append(mse)
                all_linf.append(linf)

        if all_mse:
            mean_mse = float(np.mean(all_mse))
            mean_linf = float(np.mean(all_linf))
            # V_y ∈ [-0.1, +0.1]，||V_y||² ≈ 0.01 量级
            threshold_mse = 0.1 ** 2
            verdict = "LEAK_DETECTED" if mean_mse > threshold_mse else "PRIVACY_PRESERVED"
            verdicts.append(AttackVerdict(
                attack_id=self.ATTACK_ID,
                metric="reconstruction_mse",
                value=mean_mse,
                chance_level=threshold_mse,
                p_value=None,
                n_samples=len(all_mse),
                verdict=verdict,
                notes=f"MSE={mean_mse:.2e}, L-inf={mean_linf:.2e}, "
                      f"threshold={threshold_mse:.2e}. "
                      f"r_t/scale ≈ 10^5 >> V_y（≈ 0.1），PRG 不可分离。",
            ))
        else:
            # 无 ground truth 时，验证 r_t 量级是否远超 V_y
            norms = [float(np.linalg.norm(np.array(m) / scale)) for m in masked_arrs]
            mean_r = float(np.mean(norms))
            # r_t ∈ (-2^29, +2^29)，scale=10000，期望 mean_r ≈ 2^29/√3/10000 ≈ 10^5
            verdict = "LEAK_DETECTED" if mean_r < 1e4 else "PRIVACY_PRESERVED"
            verdicts.append(AttackVerdict(
                attack_id=self.ATTACK_ID,
                metric="mean_V_hat_norm",
                value=mean_r,
                chance_level=1e4,
                p_value=None,
                n_samples=len(masked_arrs),
                verdict=verdict,
                notes=f"mean ||V_hat||={mean_r:.2e}（期望 ≈ 10^5，远超 V_y ≈ 0.1）",
            ))

        return verdicts
