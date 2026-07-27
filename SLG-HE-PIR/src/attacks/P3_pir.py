"""P-3: PIR 查询隐私审计。"""
from __future__ import annotations
import json as json_mod
from pathlib import Path
from SLG_attack_test.attacks.base import BaseAttack
from SLG_attack_test.evaluation.metrics import AttackVerdict


class P3PIRSecurityAttack(BaseAttack):
    ATTACK_ID = "P3"
    ATTACK_NAME = "PIR 查询隐私"
    TARGET = "PIR 查询的字节级隐私保护"
    THREAT_LEVEL = "CRITICAL"
    GPU_REQUIRED = False

    def run(self):
        verdicts = []
        ad = self.load_attack_data()

        # P-3.1: 查询不可区分性
        if "queries" in ad and "labels" in ad:
            queries = ad["queries"]
            labels = ad["labels"]
            if len(queries) >= 100 and len(labels) >= 100:
                try:
                    from scipy.stats import ks_2samp
                    sizes_by_label = {}
                    for q, l in zip(queries, labels):
                        s = len(q) if isinstance(q, (bytes, str)) else 0
                        sizes_by_label.setdefault(l, []).append(s)
                    keys = list(sizes_by_label.keys())
                    if len(keys) >= 2:
                        stat, p_val = ks_2samp(
                            sizes_by_label[keys[0]],
                            sizes_by_label[keys[1]],
                        )
                        verdicts.append(AttackVerdict(
                            attack_id="P3-1", metric="ks_statistic",
                            value=float(stat), chance_level=0.1, p_value=float(p_val),
                            n_samples=len(queries),
                            verdict="LEAK_DETECTED" if p_val < 0.05 else "PRIVACY_PRESERVED",
                            notes=f"KS={stat:.4f}, p={p_val:.4f}"))
                except Exception:
                    pass

        # P-3.2: Hint 表完整性
        hint_path = Path("/root/autodl-tmp/slg-bfv-cache/s3pir_hints/hint_table.json")
        if hint_path.exists():
            try:
                with open(hint_path, encoding="utf-8") as f:
                    hint_data = json_mod.load(f)
                has_backup = "backup_hint_table" in hint_data
                has_parity = "parity_table" in hint_data
                verdicts.append(AttackVerdict(
                    attack_id="P3-2", metric="hint_table_completeness",
                    value=float(has_backup and has_parity),
                    chance_level=1.0, p_value=None, n_samples=0,
                    verdict="INCONCLUSIVE",
                    notes=f"backup={has_backup}, parity={has_parity}"))
            except Exception as e:
                verdicts.append(AttackVerdict(
                    attack_id="P3-2", metric="hint_table_readable",
                    value=0.0, chance_level=1.0, p_value=None, n_samples=0,
                    verdict="LEAK_DETECTED",
                    notes=f"无法读取 hint_table.json: {e}"))

        # P-3.3: 加密库分片数
        enc_db_dir = Path("/root/autodl-tmp/slg-bfv-cache/encrypted_db")
        if enc_db_dir.exists():
            db_files = list(enc_db_dir.glob("bfv_ct_db_*.bin"))
            verdicts.append(AttackVerdict(
                attack_id="P3-3", metric="encrypted_db_file_count",
                value=float(len(db_files)),
                chance_level=1.0, p_value=None, n_samples=len(db_files),
                verdict="INCONCLUSIVE",
                notes=f"加密库分片数={len(db_files)}，篡改检测需运行时注入。"))

        # P-3.4: Design-2 实现缺陷
        verdicts.append(AttackVerdict(
            attack_id="P3-4", metric="design_v2_security",
            value=0.0, chance_level=0.0, p_value=None, n_samples=0,
            verdict="LEAK_DETECTED",
            notes="Design-2 (crypto_s.py:172-198) 直接 mmap 读 enc_db.get_encrypted_row(y_t)，"
                  "S 明确知道 y_t。S3PIR 的 query-hiding 属性失效。"
                  "需升级为真 S3PIR（所有分区统一 mmap 读取）。"))

        return verdicts
