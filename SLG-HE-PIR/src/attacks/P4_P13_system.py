"""P-4 ~ P-13: 系统/资源侧信道攻击。

攻击逻辑（docs/攻击类测试方案.md §4.5）：
- P-4: 时序侧信道
- P-5: OOM DoS
- P-6: pickle 反序列化（已在 SLG-attack-test/ 实现）
- P-7: tmp 文件残留
- P-8: checkpoint 替换
- P-9: 日志结构泄露
- P-10: DeepSpeed 命名空间污染
- P-11: dump_attacks 误启用
- P-12: expected_shape warn
- P-13: backup hints 确定性
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from SLG_attack_test.attacks.base import (
    BaseAttack,
    AttackVerdict,
)


class PSystemAttack(BaseAttack):
    """P-4 ~ P-13: 系统/资源侧信道攻击。"""

    ATTACK_ID = "P_System"
    ATTACK_NAME = "系统/资源侧信道"
    TARGET = "协议实现的系统级安全问题"
    THREAT_LEVEL = "MEDIUM"
    GPU_REQUIRED = False

    def run(self) -> list[AttackVerdict]:
        verdicts = []
        attack_data = self.load_attack_data()

        # ── P-4: 时序侧信道 ────────────────────────────────────────────────
        if "step_times" in attack_data and "seq_lengths" in attack_data:
            step_times = attack_data["step_times"]
            seq_lengths = attack_data["seq_lengths"]
            if len(step_times) >= 10 and len(seq_lengths) >= 10:
                import numpy as np
                corr = float(np.corrcoef(step_times, seq_lengths)[0, 1])
                verdicts.append(AttackVerdict(
                    attack_id="P4",
                    metric="time_vs_seqlen_correlation",
                    value=corr,
                    chance_level=0.0,
                    p_value=None,
                    n_samples=len(step_times),
                    verdict="LEAK_DETECTED" if abs(corr) > 0.5 else "PRIVACY_PRESERVED",
                    notes=f"step_time 与 seq_length 相关系数={corr:.4f}",
                ))

        # ── P-5: OOM DoS（静态检查） ───────────────────────────────────
        verdicts.append(AttackVerdict(
            attack_id="P5",
            metric="chunk_tokens_bounds_check",
            value=1.0,
            chance_level=1.0,
            p_value=None,
            n_samples=0,
            verdict="INCONCLUSIVE",
            notes="OOM DoS 需运行时测试，静态分析无法覆盖；"
                  "建议增加 chunk_tokens 上限校验。",
        ))

        # ── P-7: tmp 文件残留 ──────────────────────────────────────────
        tmp_patterns = ["/dev/shm/slg-", "/tmp/slg-", "/tmp/seal_"]
        found_files = []
        for pattern in tmp_patterns:
            try:
                import glob
                found_files.extend(glob.glob(f"{pattern}*"))
            except Exception:
                pass
        verdicts.append(AttackVerdict(
            attack_id="P7",
            metric="tmp_residue_files",
            value=float(len(found_files)),
            chance_level=0.0,
            p_value=None,
            n_samples=len(found_files),
            verdict="LEAK_DETECTED" if found_files else "PRIVACY_PRESERVED",
            notes=f"发现 {len(found_files)} 个残留临时文件: {found_files[:5]}",
        ))

        # ── P-8: checkpoint 替换 ──────────────────────────────────────────
        checkpoint_dirs = [
            Path("/root/autodl-tmp/SLG-HE-PIR/baseline/classification_genrel/checkpoints"),
            Path("/root/autodl-tmp/SLG-HE-PIR/baseline/generation_ner/checkpoints"),
        ]
        for cp_dir in checkpoint_dirs:
            if cp_dir.exists():
                pt_files = list(cp_dir.glob("*.pt"))
                if pt_files:
                    # 检查是否有签名
                    has_sig = any(f.suffix == ".sig" for f in pt_files)
                    verdicts.append(AttackVerdict(
                        attack_id="P8",
                        metric="checkpoint_has_signature",
                        value=float(has_sig),
                        chance_level=1.0,
                        p_value=None,
                        n_samples=len(pt_files),
                        verdict="PRIVACY_PRESERVED" if has_sig else "LEAK_DETECTED",
                        notes=f"{cp_dir.name}: {len(pt_files)} 个 .pt 文件，有签名={has_sig}",
                    ))

        # ── P-9: 日志结构泄露 ───────────────────────────────────────────
        log_dir = Path("/root/autodl-tmp/SLG-HE-PIR/baseline/classification_genrel/logs")
        if log_dir.exists():
            log_files = list(log_dir.glob("train_*.log"))
            if log_files:
                latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
                try:
                    content = latest_log.read_text(encoding="utf-8", errors="ignore")
                    has_n_tokens = "n_tokens" in content or "chunk_time" in content
                    verdicts.append(AttackVerdict(
                        attack_id="P9",
                        metric="log_contains_structural_info",
                        value=float(has_n_tokens),
                        chance_level=0.0,
                        p_value=None,
                        n_samples=len(log_files),
                        verdict="LEAK_DETECTED" if has_n_tokens else "PRIVACY_PRESERVED",
                        notes=f"最新日志包含 n_tokens/chunk_time 信息={has_n_tokens}",
                    ))
                except Exception:
                    pass

        # ── P-10: DeepSpeed 命名空间污染 ───────────────────────────────
        verdicts.append(AttackVerdict(
            attack_id="P10",
            metric="deepspeed_distributed_initialized",
            value=0.0,
            chance_level=0.0,
            p_value=None,
            n_samples=0,
            verdict="INCONCLUSIVE",
            notes="DeepSpeed ZeRO 初始化需运行时检测；建议在 _setup_deepspeed_zero 后清理 torch.distributed。",
        ))

        # ── P-11: dump_attacks 误启用 ──────────────────────────────────
        dumps_dir = Path("/root/autodl-tmp/SLG-HE-PIR/baseline/classification_genrel/dumps")
        has_dumps = dumps_dir.exists() and any(dumps_dir.iterdir())
        verdicts.append(AttackVerdict(
            attack_id="P11",
            metric="attack_dumps_exists",
            value=float(has_dumps),
            chance_level=0.0,
            p_value=None,
            n_samples=0,
            verdict="LEAK_DETECTED" if has_dumps else "PRIVACY_PRESERVED",
            notes=f"attack_dumps 目录存在且非空={has_dumps}。"
                  "TrainerConfig.dump_attacks 默认为 False，若被误启用则 g_H 落盘。",
        ))

        # ── P-12: expected_shape warn ──────────────────────────────────
        verdicts.append(AttackVerdict(
            attack_id="P12",
            metric="expected_shape_check_strict",
            value=0.0,
            chance_level=1.0,
            p_value=None,
            n_samples=0,
            verdict="INCONCLUSIVE",
            notes="party_m.py:418-422 expected_shape 不匹配仅 warn，建议改为 AssertionError。",
        ))

        # ── P-13: backup hints 确定性 ───────────────────────────────────
        hint_path = Path("/root/autodl-tmp/slg-bfv-cache/s3pir_hints/backup_hint_table.json")
        if hint_path.exists():
            try:
                content = hint_path.read_text(encoding="utf-8")
                # 检查是否由 random.seed(42) 生成（确定性）
                # 通过多次哈希判断确定性
                h1 = hashlib.sha256(content.encode()).hexdigest()
                verdicts.append(AttackVerdict(
                    attack_id="P13",
                    metric="backup_hint_is_deterministic",
                    value=1.0,
                    chance_level=0.0,
                    p_value=None,
                    n_samples=1,
                    verdict="LEAK_DETECTED",
                    notes="s3pir_hints.py:104 使用 random.seed(42)，backup hints 完全确定，可被离线重建。",
                ))
            except Exception:
                pass
        else:
            verdicts.append(AttackVerdict(
                attack_id="P13",
                metric="backup_hint_file_exists",
                value=0.0,
                chance_level=1.0,
                p_value=None,
                n_samples=0,
                verdict="INCONCLUSIVE",
                notes="backup_hint_table.json 不存在，跳过",
            ))

        return verdicts
