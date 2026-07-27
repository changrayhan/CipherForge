"""P-2: PRG 掩码安全审计。"""
from __future__ import annotations
import numpy as np
from SLG_attack_test.attacks.base import BaseAttack
from SLG_attack_test.evaluation.metrics import AttackVerdict


class P2PRGSecurityAttack(BaseAttack):
    ATTACK_ID = "P2"
    ATTACK_NAME = "PRG 掩码安全"
    TARGET = "PRG 随机数生成器"
    THREAT_LEVEL = "CRITICAL"
    GPU_REQUIRED = False

    def run(self):
        verdicts = []
        ad = self.load_attack_data()
        if "prg_outputs" in ad:
            rt = ad["prg_outputs"]
            if isinstance(rt, np.ndarray) and len(rt) >= 100:
                mv = float(np.mean(rt))
                sv = float(np.std(rt))
                pm = 2**30
                es = pm / np.sqrt(12)
                vr = sv / es if es > 0 else 1.0
                ur = len(np.unique(rt)) / len(rt)
                v1 = "LEAK_DETECTED" if abs(mv) > 1e7 else "PRIVACY_PRESERVED"
                verdicts.append(AttackVerdict(
                    attack_id="P2-1", metric="prg_mean",
                    value=mv, chance_level=0.0, p_value=None, n_samples=len(rt),
                    verdict=v1,
                    notes=f"mean={mv:.2e}, std={sv:.2e}, var_ratio={vr:.4f}"))
                max_lag = min(100, len(rt) // 10)
                max_ac = 0.0
                for lag in range(1, max_lag + 1):
                    try:
                        ac = float(np.corrcoef(rt[:-lag], rt[lag:])[0, 1])
                        if not np.isnan(ac):
                            max_ac = max(max_ac, abs(ac))
                    except Exception:
                        pass
                th = 2.0 / np.sqrt(len(rt))
                v2 = "LEAK_DETECTED" if max_ac > th else "PRIVACY_PRESERVED"
                verdicts.append(AttackVerdict(
                    attack_id="P2-2", metric="max_autocorr",
                    value=max_ac, chance_level=th, p_value=None, n_samples=len(rt),
                    verdict=v2,
                    notes=f"max|ac|={max_ac:.6f}, th={th:.6f}"))
        if "prg_seed" in ad:
            seed = ad["prg_seed"]
            sl = len(seed) if isinstance(seed, (bytes, bytearray, list)) else 0
            v = "PRIVACY_PRESERVED" if sl == 32 else "LEAK_DETECTED"
            verdicts.append(AttackVerdict(
                attack_id="P2-3", metric="seed_length",
                value=float(sl), chance_level=32.0, p_value=None, n_samples=1,
                verdict=v,
                notes=f"seed len={sl} bytes"))
        verdicts.append(AttackVerdict(
            attack_id="P2-4", metric="seed_disclosure",
            value=1.0, chance_level=0.0, p_value=None, n_samples=0,
            verdict="INCONCLUSIVE",
            notes="seed 泄露后 M+S 可重建 r_t，分离 V_y。需门限 PRG。"))
        return verdicts
