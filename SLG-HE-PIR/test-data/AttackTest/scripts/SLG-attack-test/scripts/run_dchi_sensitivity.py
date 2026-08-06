#!/usr/bin/env python3
"""dχ 参数敏感性分析 — 自动化实验脚本。

L-1 attack 的 verdict 严重依赖 dχ 的三个超参数：

- ``--dp_alpha``          (相对噪声幅度, ‖noise‖₂ / ‖H_U‖₂ 目标)
- ``--dp_answer_beta``    (answer-estimation conservativeness)
- ``--dp_calibration_steps`` (校准步数，0 噪声，仅估计 ‖H_U‖₂ 分布 A)

顶刊规范要求证明结论不依赖参数调优。本脚本对三组参数做网格扫描：

**Sweep 1：alpha 敏感性（6 个点）** — 固定 (beta=0.5, cal_steps=5)
    alpha ∈ {0.05, 0.10, 0.15, 0.20, 0.30, 0.50}

**Sweep 2：beta 敏感性（5 个点）** — 固定 (alpha=0.15, cal_steps=5)
    beta ∈ {0.10, 0.25, 0.50, 0.75, 1.00}

**Sweep 3：calibration_steps 敏感性（5 个点）** — 固定 (alpha=0.15, beta=0.5)
    cal_steps ∈ {1, 3, 5, 10, 20}

**Sweep 4：alpha×beta 组合（5 个点）** — 验证二维鲁棒性
    (alpha, beta) ∈ {(0.05,0.50), (0.15,0.50), (0.30,0.50),
                      (0.15,0.25), (0.15,0.75)}

每个实验点都跑完整的 50 步 GPU 训练，提取 L-1 的核心指标：
- K-Means ARI
- 1-NN 一致率
- Cosine AUC
- 梯度幅度 ANOVA p-value
- 前向 H̃_U 类均值 ANOVA 最小 p-value（BH-FDR 调整后）

输出到 ``test-data/attack-test-data/sweep_dchi_<axis>/run_<timestamp>/``。
汇总表格落盘到 ``sensitivity_dchi_manifest.json``。

使用::

    # 跑完所有 21 个点（推荐）
    bash SLG-attack-test/scripts/run_dchi_sensitivity.sh

    # 仅跑 alpha 敏感性
    bash SLG-attack-test/scripts/run_dchi_sensitivity.sh --axis alpha

    # 跑完所有并自动生成热力图
    bash SLG-attack-test/scripts/run_dchi_sensitivity.sh --plot
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
logger = logging.getLogger("dchi_sweep")


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
# 21 grid points split into 4 sweeps
# ---------------------------------------------------------------------------

ALPHA_SWEEP = [
    ("alpha_0.05", {"dp_alpha": 0.05}),
    ("alpha_0.10", {"dp_alpha": 0.10}),
    ("alpha_0.15", {"dp_alpha": 0.15}),   # baseline
    ("alpha_0.20", {"dp_alpha": 0.20}),
    ("alpha_0.30", {"dp_alpha": 0.30}),
    ("alpha_0.50", {"dp_alpha": 0.50}),
]
BETA_SWEEP = [
    ("beta_0.10", {"dp_answer_beta": 0.10}),
    ("beta_0.25", {"dp_answer_beta": 0.25}),
    ("beta_0.50", {"dp_answer_beta": 0.50}),   # baseline
    ("beta_0.75", {"dp_answer_beta": 0.75}),
    ("beta_1.00", {"dp_answer_beta": 1.00}),
]
CAL_SWEEP = [
    ("cal_1",  {"dp_calibration_steps": 1}),
    ("cal_3",  {"dp_calibration_steps": 3}),
    ("cal_5",  {"dp_calibration_steps": 5}),   # baseline
    ("cal_10", {"dp_calibration_steps": 10}),
    ("cal_20", {"dp_calibration_steps": 20}),
]
GRID_SWEEP = [
    ("grid_05_50", {"dp_alpha": 0.05, "dp_answer_beta": 0.50}),
    ("grid_15_50", {"dp_alpha": 0.15, "dp_answer_beta": 0.50}),  # baseline
    ("grid_30_50", {"dp_alpha": 0.30, "dp_answer_beta": 0.50}),
    ("grid_15_25", {"dp_alpha": 0.15, "dp_answer_beta": 0.25}),
    ("grid_15_75", {"dp_alpha": 0.15, "dp_answer_beta": 0.75}),
]

ALL_SWEEPS = {
    "alpha": ALPHA_SWEEP,
    "beta": BETA_SWEEP,
    "cal": CAL_SWEEP,
    "grid": GRID_SWEEP,
}

# Fixed baseline (mirrors the production L-1 run that produced 7/7 PRIVACY_PRESERVED)
BASELINE_KWARGS = dict(
    attacks="L1",
    n_steps=50,
    batch_size=4,
    n_eval_steps=20,
    dp_enable=True,
    dp_alpha=0.15,
    dp_answer_beta=0.5,
    dp_calibration_steps=5,
    dp_dump_audit=False,
    bfv_poly_degree=2048,
    bfv_hidden_dim=2048,
    num_layers=16,
    u_layers=8,
    seed=42,
)


def _l1_metrics_from_attack_results(run_dir: Path) -> dict:
    """Parse L-1's run/attack_results.json and extract verdict metrics."""
    out = {
        "ari": None,
        "ari_verdict": None,
        "nn_agreement": None,
        "nn_verdict": None,
        "cosine_auc": None,
        "cosine_verdict": None,
        "perm_p": None,
        "perm_verdict": None,
        "magnitude_anova_p": None,
        "magnitude_eta2": None,
        "magnitude_verdict": None,
        "h_u_mean_anova_min_p": None,
        "h_u_mean_anova_verdict": None,
        "h_u_norm_anova_p": None,
        "h_u_norm_anova_eta2": None,
        "h_u_norm_anova_verdict": None,
    }
    candidates = list(run_dir.glob("**/attack_results.json"))
    if not candidates:
        return out
    try:
        with open(candidates[0]) as f:
            data = json.load(f)
    except Exception:
        return out
    for v in data.get("attack_results", []):
        sub = v.get("sub_attack") or ""
        verdict = v.get("verdict", "")
        val = v.get("value")
        p_val = v.get("p_value")
        if sub == "kmeans_ari":
            out["ari"] = val
            out["ari_verdict"] = verdict
        elif sub == "nn_agreement":
            out["nn_agreement"] = val
            out["nn_verdict"] = verdict
        elif sub == "cosine_auc":
            out["cosine_auc"] = val
            out["cosine_verdict"] = verdict
        elif sub == "permutation_test":
            out["perm_p"] = val
            out["perm_verdict"] = verdict
        elif sub == "magnitude_anova":
            out["magnitude_anova_p"] = val
            out["magnitude_verdict"] = verdict
            # eta² is encoded in notes
        elif sub == "h_u_mean_anova":
            out["h_u_mean_anova_min_p"] = val
            out["h_u_mean_anova_verdict"] = verdict
        elif sub == "h_u_norm_anova":
            out["h_u_norm_anova_p"] = val
            out["h_u_norm_anova_verdict"] = verdict
    return out


def _run_one(label: str, sweep_axis: str, override: dict, output_dir: Path) -> dict:
    kwargs = dict(BASELINE_KWARGS)
    kwargs.update(override)
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
    logger.info("=" * 60)
    logger.info("[sweep=%s] %s", sweep_axis, label)
    logger.info("CMD: %s", " ".join(cmd))
    logger.info("=" * 60)

    t0 = time.time()
    proc = subprocess.run(
        cmd, cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    elapsed = time.time() - t0

    log_path = output_dir / "dchi_sweep_stdout.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(proc.stdout)

    metrics = _l1_metrics_from_attack_results(output_dir)
    summary = {
        "label": label,
        "sweep_axis": sweep_axis,
        "override": override,
        "elapsed_s": elapsed,
        "returncode": proc.returncode,
        "metrics": metrics,
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axis", choices=list(ALL_SWEEPS.keys()) + ["all"],
                        default="all",
                        help="Which sweep to run (default: all 21 points).")
    parser.add_argument("--no-gpu-dry-run", action="store_true",
                        help="Print commands without executing.")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_out = OUTPUT_BASE
    if args.axis == "all":
        sweeps_to_run = ALL_SWEEPS
    else:
        sweeps_to_run = {args.axis: ALL_SWEEPS[args.axis]}

    all_results = []
    for axis, sweep in sweeps_to_run.items():
        logger.info("=" * 72)
        logger.info("Sweep axis: %s (%d points)", axis, len(sweep))
        logger.info("=" * 72)
        for label, override in sweep:
            out_dir = base_out / f"sweep_dchi_{axis}_{timestamp}" / label
            if args.no_gpu_dry_run:
                logger.info("[dry-run] would execute sweep=%s label=%s", axis, label)
                continue
            summary = _run_one(label, axis, override, out_dir)
            all_results.append(summary)
            logger.info("Done: %s in %.1f s (rc=%d)", label,
                        summary["elapsed_s"], summary["returncode"])
            logger.info("  metrics: %s", summary["metrics"])

    # Persist a manifest so the report can build a sensitivity table
    if not args.no_gpu_dry_run and all_results:
        manifest = {
            "timestamp": timestamp,
            "sweeps_run": list(sweeps_to_run.keys()),
            "results": all_results,
        }
        manifest_path = base_out / f"sensitivity_dchi_manifest_{timestamp}.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info("Manifest: %s", manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())