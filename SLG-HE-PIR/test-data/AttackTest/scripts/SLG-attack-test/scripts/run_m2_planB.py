#!/usr/bin/env python3
"""方案 B：M-2 Dummy Forward Pre-LoRA Baseline — 自动化实验脚本。

方案 B 的核心思想：在协议真正训练开始之前（Adam 动量为零、PRG 未消耗
任何熵的状态），用 ``peft.disable_adapter_layers()`` 跑 K 个 dummy
forward-only step，把产出的 a_t 标记为 ``a_t_pre``。此时：

- Adam 动量为零（消除混淆信号 1）
- Batch 序列未与 post 窗口纠缠（消除混淆信号 2）
- PRG 熵消耗从共同起点开始（消除混淆信号 3）

本脚本对 ``run_attack_suite.py`` 提供的 ``--m2_baseline_recording_steps``
参数做完整的端到端实验：

1. **Baseline-A**：当前 warmup 方案（--m2_pre_lora_warmup_steps=20）— 重现
   ``run_20260726_213344`` 的实验配置，验证脚本能复现 INCONCLUSIVE 状态。
2. **Baseline-B**：方案 B（--m2_baseline_recording_steps=40）— 用 dummy
   forward 录 a_t_pre，期望 verdict 收敛到 PRIVACY_PRESERVED。
3. **Baseline-C**：方案 B 大样本（--m2_baseline_recording_steps=80）—
   在 n_pre=320、n_post=240 配置下进一步验证稳定性。

每个实验都在真实 GPU 上跑 50+ 步训练，输出落到
``test-data/attack-test-data/run_<timestamp>_m2planB_<label>/``。

使用::

    # 仅方案 B（推荐跑法）
    bash SLG-attack-test/scripts/run_m2_planB.sh

    # 重现 Baseline-A
    bash SLG-attack-test/scripts/run_m2_planB.sh --legacy

    # 同时跑 Baseline-A 和方案 B（双对照）
    bash SLG-attack-test/scripts/run_m2_planB.sh --compare
"""

from __future__ import annotations

import argparse
import json
import logging
import os
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
logger = logging.getLogger("m2_planB")


PROJECT_ROOT = Path("/root/autodl-tmp/SLG-HE-PIR-code/SLG-HE-PIR")
ATTACK_TEST_DIR = PROJECT_ROOT / "SLG-attack-test"
HF_MODEL_DIR = Path(
    "/root/autodl-tmp/SLG-HE-PIR-code/hf_cache/models--unsloth--Llama-3.2-1B"
    "/snapshots/9535bd9b1d1dea6acafbdc4813b728796aeb28da"
)
DATA_DIR = PROJECT_ROOT / "datasets" / "trec-qc"
OUTPUT_BASE = PROJECT_ROOT / "test-data" / "attack-test-data"

PYTHON = sys.executable

# ---------------------------------------------------------------------------
# Experiment configurations
# ---------------------------------------------------------------------------

# Legacy: the original run_20260726_213344 that produced INCONCLUSIVE.
# This is a *reproduction* — we expect the same verdict given the same RNG seed.
LEGACY_RUN_KWARGS = dict(
    attacks="M2",
    n_steps=80,
    batch_size=4,
    n_eval_steps=20,
    m2_pre_lora_warmup_steps=20,
    m2_baseline_recording_steps=0,
    m2_lora_rank=8,
    bfv_poly_degree=2048,
    bfv_hidden_dim=2048,
    num_layers=16,
    u_layers=8,
    seed=42,
)

# 方案 B（小样本对照）：K=40 dummy forward steps → n_pre=160
PLANB_SMALL_KWARGS = dict(
    attacks="M2",
    n_steps=80,            # 与 legacy 一致：post 窗口仍为 80 步 × 4 = 320 样本
    batch_size=4,
    n_eval_steps=20,
    m2_pre_lora_warmup_steps=0,     # 关闭 warmup 方案，避免双基线
    m2_baseline_recording_steps=40, # 方案 B：40 dummy forward 步 × 4 = 160 样本
    m2_lora_rank=8,
    bfv_poly_degree=2048,
    bfv_hidden_dim=2048,
    num_layers=16,
    u_layers=8,
    seed=42,
)

# 方案 B（大样本对照）：K=80 dummy forward steps → n_pre=320
PLANB_LARGE_KWARGS = dict(
    attacks="M2",
    n_steps=80,
    batch_size=4,
    n_eval_steps=20,
    m2_pre_lora_warmup_steps=0,
    m2_baseline_recording_steps=80,  # n_pre=320 进一步稳态
    m2_lora_rank=8,
    bfv_poly_degree=2048,
    bfv_hidden_dim=2048,
    num_layers=16,
    u_layers=8,
    seed=42,
)


def _run_attack_suite(label: str, kwargs: dict, output_dir: Path) -> dict:
    """Invoke run_attack_suite.py with the given kwargs; return parsed summary."""
    cmd = [PYTHON, str(ATTACK_TEST_DIR / "run_attack_suite.py")]
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
    logger.info("Launching experiment: %s", label)
    logger.info("CMD: %s", " ".join(cmd))
    logger.info("Output: %s", output_dir)
    logger.info("=" * 72)

    t0 = time.time()
    proc = subprocess.run(
        cmd, cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    elapsed = time.time() - t0

    log_path = output_dir / "planB_stdout.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(proc.stdout)

    summary = {
        "label": label,
        "output_dir": str(output_dir),
        "elapsed_s": elapsed,
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-2000:],
    }
    # Try to parse the per-run summary_*.json for verdicts
    summary_files = sorted(output_dir.glob("summary_*.json"))
    if summary_files:
        try:
            with open(summary_files[-1]) as f:
                summary["verdict_summary"] = json.load(f)
        except Exception:
            pass
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy", action="store_true",
                        help="Run only Baseline-A (legacy warmup).")
    parser.add_argument("--compare", action="store_true",
                        help="Run all three experiments for ablation.")
    parser.add_argument("--only", choices=["A", "B", "C"],
                        help="Run only the specified experiment.")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_out = OUTPUT_BASE

    experiments = []
    if args.compare:
        experiments = [
            ("A_legacy", LEGACY_RUN_KWARGS),
            ("B_planB_small", PLANB_SMALL_KWARGS),
            ("C_planB_large", PLANB_LARGE_KWARGS),
        ]
    elif args.legacy:
        experiments = [("A_legacy", LEGACY_RUN_KWARGS)]
    elif args.only == "A":
        experiments = [("A_legacy", LEGACY_RUN_KWARGS)]
    elif args.only == "B":
        experiments = [("B_planB_small", PLANB_SMALL_KWARGS)]
    elif args.only == "C":
        experiments = [("C_planB_large", PLANB_LARGE_KWARGS)]
    else:
        # Default: skip legacy (cost) and run only B+C.
        experiments = [
            ("B_planB_small", PLANB_SMALL_KWARGS),
            ("C_planB_large", PLANB_LARGE_KWARGS),
        ]

    results = []
    for label, kwargs in experiments:
        out_dir = base_out / f"run_{timestamp}_m2planB_{label}"
        summary = _run_attack_suite(label, kwargs, out_dir)
        results.append(summary)
        logger.info("Experiment %s done in %.1f s (rc=%d)",
                    label, summary["elapsed_s"], summary["returncode"])
        if "verdict_summary" in summary:
            logger.info("  verdicts: %s",
                        summary["verdict_summary"].get("verdicts"))

    # Persist a manifest so the report can find these runs
    manifest_path = base_out / f"planB_manifest_{timestamp}.json"
    with open(manifest_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Manifest: %s", manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())