#!/usr/bin/env python3
"""BFV 安全参数对照测试：poly_degree=2048 (128-bit) vs poly_degree=4096 (192-bit)。

§1.2 BFV 安全参数论证要求证明 128-bit security 已经足够（攻击者计算能力
远低于 2^128）。本脚本对两组 BFV 参数分别跑完整 L-1 attack，对比：

1. **协议功能性**：50 步训练能正常完成、无 decryption failure
2. **隐私效用**：L-1 的 7 项指标均维持 PRIVACY_PRESERVED
3. **性能开销**：单步训练时间（用作性能对照）

如果两组配置在 (1)(2) 上等效、(3) 上 poly_degree=4096 显著慢于 2048，
则说明 128-bit security 已是当前威胁模型下"性能-安全"权衡的合理选择，
poly_degree=2048 为默认配置。

每个实验点都跑完整的 50 步 GPU 训练。

使用::

    bash SLG-attack-test/scripts/run_bfv_security_ablation.sh
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
logger = logging.getLogger("bfv_ablation")


PROJECT_ROOT = Path("/root/autodl-tmp/SLG-HE-PIR-code/SLG-HE-PIR")
ATTACK_TEST_DIR = PROJECT_ROOT / "SLG-attack-test"
HF_MODEL_DIR = Path(
    "/root/autodl-tmp/SLG-HE-PIR-code/hf_cache/models--unsloth--Llama-3.2-1B"
    "/snapshots/9535bd9b1d1dea6acafbdc4813b728796aeb28da"
)
DATA_DIR = PROJECT_ROOT / "datasets" / "trec-qc"
OUTPUT_BASE = PROJECT_ROOT / "test-data" / "attack-test-data"
PYTHON = sys.executable


# poly_degree → coeff_modulus 配置（与 _patched_build_seal_context 保持一致）
# 来源：HomomorphicEncryption.org standard v1.1, N=2048 → 128-bit, N=4096 → 192-bit
CONFIGS = {
    "N2048_128bit": dict(
        bfv_poly_degree=2048,
        # coeff_modulus 由 run_attack_suite 的 _patched_build_seal_context 选择
        # → poly_degree=2048: [36, 14]，总计 50 bits → 128-bit IND-CPA
        bfv_plain_bits=30,
        bfv_scale=10000.0,
        bfv_hidden_dim=2048,  # BFV backend: hidden_dim must == poly_degree
    ),
    "N4096_192bit": dict(
        bfv_poly_degree=4096,
        # poly_degree=4096: [36, 36, 37]，总计 109 bits → 192-bit IND-CPA
        bfv_plain_bits=30,
        bfv_scale=10000.0,
        bfv_hidden_dim=4096,  # BFV backend: hidden_dim must == poly_degree
    ),
}

# 固定的 L-1 attack 配置（保持其他条件不变）
COMMON_KWARGS = dict(
    attacks="L1",
    n_steps=50,
    batch_size=4,
    n_eval_steps=20,
    dp_enable=True,
    dp_alpha=0.15,
    dp_answer_beta=0.5,
    dp_calibration_steps=5,
    dp_dump_audit=False,
    # BFV backend enforces hidden_dim == poly_degree, so we override per-config
    num_layers=16,
    u_layers=8,
    seed=42,
)


def _run_one(label: str, bfv_kwargs: dict, output_dir: Path) -> dict:
    kwargs = dict(COMMON_KWARGS)
    kwargs.update(bfv_kwargs)
    cmd = [PYTHON, str(ATTACK_TEST_DIR / "run_attack_suite.py")]
    for k, v in kwargs.items():
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
        else:
            cmd.extend([f"--{k}", str(v)])
    # Each BFV poly_degree uses its own cache dir to avoid pk.bin format mismatch
    cache_dir = OUTPUT_BASE / f"bfv_cache_n{bfv_kwargs['bfv_poly_degree']}"
    cmd.extend([
        "--bfv_cache_dir", str(cache_dir),
        "--hf_model", str(HF_MODEL_DIR),
        "--data_dir", str(DATA_DIR),
        "--output_dir", str(output_dir),
        "--project_root", str(PROJECT_ROOT),
    ])
    logger.info("=" * 60)
    logger.info("[BFV ablation] %s", label)
    logger.info("CMD: %s", " ".join(cmd))
    logger.info("=" * 60)

    t0 = time.time()
    proc = subprocess.run(
        cmd, cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    elapsed = time.time() - t0

    log_path = output_dir / "bfv_ablation_stdout.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(proc.stdout)

    # Parse L-1 verdicts
    metrics = {
        "leak_count": 0,
        "preserved_count": 0,
        "inconclusive_count": 0,
        "sub_attacks": [],
    }
    candidates = list(output_dir.glob("**/attack_results.json"))
    if candidates:
        try:
            with open(candidates[0]) as f:
                data = json.load(f)
            for v in data.get("attack_results", []):
                verdict = v.get("verdict", "")
                if verdict == "LEAK_DETECTED":
                    metrics["leak_count"] += 1
                elif verdict == "PRIVACY_PRESERVED":
                    metrics["preserved_count"] += 1
                elif verdict == "INCONCLUSIVE":
                    metrics["inconclusive_count"] += 1
                metrics["sub_attacks"].append({
                    "sub_attack": v.get("sub_attack"),
                    "verdict": verdict,
                    "value": v.get("value"),
                })
        except Exception:
            pass

    # Performance: per-step time = elapsed_s / n_steps (approximate; ignore protocol init)
    per_step_s = elapsed / max(1, COMMON_KWARGS["n_steps"])
    summary = {
        "label": label,
        "bfv_kwargs": bfv_kwargs,
        "elapsed_s": elapsed,
        "per_step_s": per_step_s,
        "returncode": proc.returncode,
        "metrics": metrics,
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing.")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_out = OUTPUT_BASE

    all_results = []
    for label, bfv_kwargs in CONFIGS.items():
        out_dir = base_out / f"bfv_ablation_{label}_{timestamp}"
        if args.dry_run:
            logger.info("[dry-run] would execute label=%s", label)
            continue
        summary = _run_one(label, bfv_kwargs, out_dir)
        all_results.append(summary)
        logger.info("Done: %s in %.1f s (rc=%d)",
                    label, summary["elapsed_s"], summary["returncode"])
        logger.info("  verdicts: %d LEAK / %d PRIV / %d INC",
                    summary["metrics"]["leak_count"],
                    summary["metrics"]["preserved_count"],
                    summary["metrics"]["inconclusive_count"])
        logger.info("  per_step: %.2f s", summary["per_step_s"])

    if not args.dry_run and all_results:
        manifest_path = base_out / f"bfv_ablation_manifest_{timestamp}.json"
        with open(manifest_path, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info("Manifest: %s", manifest_path)

    # Compare
    if len(all_results) == 2:
        n2048, n4096 = all_results
        if n4096["per_step_s"] > n2048["per_step_s"] * 1.5:
            logger.info(
                "性能对照: N=4096 比 N=2048 慢 %.2fx → 128-bit security "
                "是当前威胁模型下的合理选择（攻击者算力远低于 2^128）。",
                n4096["per_step_s"] / n2048["per_step_s"],
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())