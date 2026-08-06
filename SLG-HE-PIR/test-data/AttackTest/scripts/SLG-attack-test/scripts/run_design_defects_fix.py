#!/usr/bin/env python3
"""修复 §3.1 的 4 项 INCONCLUSIVE 设计缺陷（一次性 M-1 + M-2 修复）。

§A.5 第 3 项：扩 M-2 n_post 至 640，让一致性闸门差距稳定 > 0.5σ。
§A.5 第 4 项（新增）：扩 M-1 查询预算至 500+，触发 surrogate_model 5-fold CV 替代模型训练稳定。

**两个修复在同一次协议初始化中完成**，比单独跑节省 ~5 分钟协议初始化时间：

- M-2：n_steps=160 × batch_size=4 = 640 samples（post 窗口）+ K=40 dummy forward (n_pre=160)
- M-1：n_eval_steps=125 × batch_size=4 = 500 queries（≥ 5-fold CV 替代模型训练阈值 ≥500）

预期结果：
- M-2 三项 INCONCLUSIVE（rank_fingerprint / direction_fingerprint / m2_aggregate）应全部收敛到 PRIVACY_PRESERVED
- M-1 surrogate_model 应从 INCONCLUSIVE 收敛到 PRIVACY_PRESERVED
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("design_defects_fix")

PROJECT_ROOT = Path("/root/autodl-tmp/SLG-HE-PIR-code/SLG-HE-PIR")
ATTACK_TEST_DIR = PROJECT_ROOT / "SLG-attack-test"
HF_MODEL_DIR = Path(
    "/root/autodl-tmp/SLG-HE-PIR-code/hf_cache/models--unsloth--Llama-3.2-1B"
    "/snapshots/9535bd9b1d1dea6acafbdc4813b728796aeb28da"
)
DATA_DIR = PROJECT_ROOT / "datasets" / "trec-qc"
OUTPUT_BASE = PROJECT_ROOT / "test-data" / "attack-test-data"

# M-1 修复 + M-2 修复一次跑（mixed）
MIXED_FIX_KWARGS = dict(
    attacks="M1,M2",
    n_steps=160,
    batch_size=4,
    n_eval_steps=125,
    m1_query_budget=1000,
    m2_pre_lora_warmup_steps=0,
    m2_baseline_recording_steps=40,
    m2_lora_rank=8,
    bfv_poly_degree=2048,
    bfv_hidden_dim=2048,
    num_layers=16,
    u_layers=8,
    seed=42,
)


def _run_attack_suite(label: str, kwargs: dict, output_dir: Path) -> dict:
    cmd = [sys.executable, str(ATTACK_TEST_DIR / "run_attack_suite.py")]
    for k, v in kwargs.items():
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
        else:
            cmd.extend([f"--{k}", str(v)])
    cmd.extend([
        "--hf_model", str(HF_MODEL_DIR),
        "--data_dir", str(DATA_DIR),
        "--output_dir", str(output_dir),
        "--project_root", str(PROJECT_ROOT),
    ])
    logger.info("=" * 72)
    logger.info("Launching MIXED M1+M2 fix experiment: %s", label)
    logger.info("CMD: %s", " ".join(cmd))
    logger.info("Output: %s", output_dir)
    logger.info("=" * 72)

    t0 = time.time()
    proc = subprocess.run(
        cmd, cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    elapsed = time.time() - t0

    log_path = output_dir / "mixed_fix_stdout.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(proc.stdout)

    summary = {
        "label": label,
        "output_dir": str(output_dir),
        "elapsed_s": elapsed,
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-2000:],
    }
    summary_files = sorted(output_dir.glob("summary_*.json"))
    if summary_files:
        try:
            with open(summary_files[-1]) as f:
                summary["verdict_summary"] = json.load(f)
        except Exception:
            pass
    return summary


def main() -> int:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_BASE / f"run_{timestamp}_mixed_fix"
    summary = _run_attack_suite("mixed_M1M2_fix", MIXED_FIX_KWARGS, out_dir)
    logger.info("Done in %.1f s (rc=%d)", summary["elapsed_s"], summary["returncode"])

    manifest_path = OUTPUT_BASE / f"mixed_fix_manifest_{timestamp}.json"
    with open(manifest_path, "w") as f:
        json.dump([summary], f, indent=2)
    logger.info("Manifest: %s", manifest_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
