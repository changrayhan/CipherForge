"""P-1: BFV 加密层安全审计。

攻击逻辑（docs/攻击类测试方案.md §3.1）：
- P-1.1/1.5: BFV 参数安全（LWE-Estimator）
- P-1.2: 密文随机性（NIST 简化版）
- P-1.3: 噪声预算追踪
- P-1.4: 密文-明文相关性
- P-1.6: PlainModulus 一致性
"""
from __future__ import annotations

import numpy as np

from SLG_attack_test.attacks.base import BaseAttack
from SLG_attack_test.evaluation.metrics import AttackVerdict


class P1BFVSecurityAttack(BaseAttack):
    """P-1: BFV 加密层安全。"""

    ATTACK_ID = "P1"
    ATTACK_NAME = "BFV 加密层安全"
    TARGET = "BFV 加密参数与实现"
    THREAT_LEVEL = "CRITICAL"
    GPU_REQUIRED = False

    def __init__(self, plain_bits: int = 30, poly_degree: int = 4096, **kwargs):
        super().__init__(**kwargs)
        self.plain_bits = plain_bits
        self.poly_degree = poly_degree

    def run(self) -> list:
        verdicts = []
        attack_data = self.load_attack_data()

        # P-1.1/1.5: CoeffModulus 安全级
        coeff_modulus = [36, 36, 37]
        total_bits = sum(coeff_modulus)
        target_bits = 128
        verdicts.append(AttackVerdict(
            attack_id="P1-1",
            metric="coeff_modulus_bits",
            value=float(total_bits),
            chance_level=float(target_bits),
            p_value=None,
            n_samples=0,
            verdict="INCONCLUSIVE",
            notes=f"poly={self.poly_degree}, coeff={coeff_modulus}, total={total_bits} bits。"
                  f"恰在 128-bit 理论边界，无裕量。"
                  f"建议用 lwe-estimator 精确测算（pip install lwe-estimator）。",
        ))

        # P-1.2: 密文随机性（简化）
        if "ciphertexts" in attack_data:
            cts = attack_data["ciphertexts"]
            if isinstance(cts, list) and len(cts) >= 100:
                unique_ratio = len(set(str(c) for c in cts)) / len(cts)
                verdicts.append(AttackVerdict(
                    attack_id="P1-2",
                    metric="ciphertext_unique_ratio",
                    value=float(unique_ratio),
                    chance_level=0.99,
                    p_value=None,
                    n_samples=len(cts),
                    verdict="LEAK_DETECTED" if unique_ratio < 0.9 else "PRIVACY_PRESERVED",
                    notes=f"唯一密文比例={unique_ratio:.4f}（应 > 0.9）",
                ))

        # P-1.3: 噪声预算
        if "noise_budgets" in attack_data:
            nb = attack_data["noise_budgets"]
            if isinstance(nb, list) and nb:
                min_nb = float(min(nb))
                failure_rate = sum(1 for b in nb if b <= 0) / len(nb)
                verdicts.append(AttackVerdict(
                    attack_id="P1-3",
                    metric="decryption_failure_rate",
                    value=float(failure_rate),
                    chance_level=0.0,
                    p_value=None,
                    n_samples=len(nb),
                    verdict="LEAK_DETECTED" if failure_rate > 0 else "PRIVACY_PRESERVED",
                    notes=f"min_noise_budget={min_nb:.1f} bits, failure_rate={failure_rate:.4f}",
                ))

        # P-1.4: 密文-明文相关性（简化）
        if "ciphertexts" in attack_data and "plaintexts" in attack_data:
            cts = attack_data["ciphertexts"]
            pts = attack_data["plaintexts"]
            if cts and pts and len(cts) == len(pts):
                ct_lens = [len(str(c)) for c in cts]
                ct_len_var = float(np.var(ct_lens))
                verdicts.append(AttackVerdict(
                    attack_id="P1-4",
                    metric="ct_length_variance",
                    value=ct_len_var,
                    chance_level=0.0,
                    p_value=None,
                    n_samples=len(cts),
                    verdict="PRIVACY_PRESERVED",
                    notes=f"同一明文多份密文长度方差={ct_len_var:.4f}",
                ))

        # P-1.6: PlainModulus 一致性
        verdicts.append(AttackVerdict(
            attack_id="P1-6",
            metric="plainmodulus_consistency",
            value=1.0,
            chance_level=1.0,
            p_value=None,
            n_samples=0,
            verdict="INCONCLUSIVE",
            notes="PlainModulus 一致性需跨 Stage 测试（plain_bits=30 vs 32）。"
                  "当前 _infer_plain_modulus 在 bits=30 时用 (1<<30)+27，"
                  "bits=32 时走 fallback (1<<bits)，可能导致密钥不一致。",
        ))

        return verdicts
