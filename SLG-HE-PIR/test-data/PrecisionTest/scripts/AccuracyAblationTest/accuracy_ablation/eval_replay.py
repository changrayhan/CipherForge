"""eval_replay — 对已有 infer_and_save 输出 JSON 注入量化噪声并重评指标。

读取 baseline 的 infer_outputs_epoch_XXX.json：
  对每个变体 (Q0/Q0'/Q1/Q2/Q2'/Q3) × 每个 seed：
    - 应用 quant_hooks.apply_variant_to_infer_outputs()
    - 复用 baseline/classification_genrel/scripts/evaluate_metrics.py 计算指标
    - 写出到 outputs/{variant}/seed_{seed}/epoch_XXX_evaluate_metrics.json
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional

from .quant_config import QuantConfig
from .quant_hooks import (
    OPTION_LETTERS,
    apply_variant_to_infer_outputs,
    make_spec,
)

logger = logging.getLogger(__name__)


GENERAL_RELATIONS = [
    "pathological", "modulatory", "expression change", "diagnosis",
    "therapy", "no relation", "relation undefined",
]

# fine-grained → general relation 映射（与 baseline evaluate_metrics.py 一致）
FINE_TO_GENERAL = {
    "pathological role": "pathological",
    "causative activation": "pathological",
    "causative inhibition": "pathological",
    "causative mutation": "pathological",
    "associated mutation": "pathological",
    "modulator decrease disease": "modulatory",
    "modulator increase disease": "modulatory",
    "genetic susceptibility": "modulatory",
    "increased expression": "expression change",
    "decreased expression": "expression change",
    "dysregulation": "expression change",
    "biomarker": "diagnosis",
    "diagnostic tool": "diagnosis",
    "epigenetic marker": "diagnosis",
    "prognostic indicator": "diagnosis",
    "positive prognostic marker": "diagnosis",
    "negative prognostic marker": "diagnosis",
    "therapy resistance": "therapy",
    "therapeutic target": "therapy",
    "no relation": "no relation",
    "relation undefined": "relation undefined",
}


def load_gold_map(gold_path: str | Path) -> tuple[dict, dict[str, list[str]]]:
    """读取 test_gold_general_qa.txt，返回 {doc_key: general_relation_str}。

    同时返回 {base_doc_key: [full_doc_key, ...]} 映射（去除 _rel_<coarse> 后缀得到 base，
    用于把 baseline 的 fine-relation doc_key 映射回 gold 的 coarse-relation doc_key）。

    v2 修复：base_to_full 改为 list 类型，因为同一个 base 可能对应多个 gold（不同 coarse）。
    """
    gold = {}
    base_to_full: dict[str, list[str]] = {}
    with open(gold_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            fine_rel = (item.get("relation") or {}).get("relation", "").lower().strip()
            if fine_rel in FINE_TO_GENERAL:
                gold[item["doc_key"]] = FINE_TO_GENERAL[fine_rel]
                # base = 去掉 _rel_<coarse> 后的部分，用于匹配 baseline 的 fine doc_key
                dk = item["doc_key"]
                base = dk.split("_rel_")[0] if "_rel_" in dk else dk
                base_to_full.setdefault(base, []).append(dk)
    return gold, base_to_full


def gold_letter_index(gold_map: dict, doc_key: str) -> Optional[int]:
    """把 gold general_relation 映射到 0..6 的 index（用于 Q2 协议约束）。"""
    rel = gold_map.get(doc_key)
    if rel is None or rel not in GENERAL_RELATIONS:
        return None
    return GENERAL_RELATIONS.index(rel)


def replay_variant(
    variant: str,
    config: QuantConfig,
    baseline_infer_dir: str | Path,
    gold_path: str | Path,
    output_dir: str | Path,
    seeds: list[int],
    epochs: int = 5,
) -> dict:
    """对单个变体在所有 epoch × seed 上跑量化模拟评估。

    Args:
        variant: Q0/Q0'/Q1/Q2/Q2'/Q3
        config: QuantConfig 实例
        baseline_infer_dir: 含 infer_outputs_epoch_XXX_*.json 的目录
        gold_path: test_gold_general_qa.txt 路径
        output_dir: 写出 {variant}/seed_{seed}/epoch_XXX_evaluate_metrics.json 的目录
        seeds: 多 seed 列表
        epochs: epoch 数

    Returns:
        {seed: {epoch: {metric_name: value, ...}}}
    """
    baseline_infer_dir = Path(baseline_infer_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 找 baseline infer_outputs files
    infer_files = sorted(baseline_infer_dir.glob("infer_outputs_epoch_*_*.json"))
    if len(infer_files) < epochs:
        logger.warning(
            "[eval_replay] Only %d infer_outputs files found, expected %d",
            len(infer_files), epochs,
        )
        epochs = min(epochs, len(infer_files))

    gold_map, base_to_full = load_gold_map(gold_path)
    logger.info(
        "[eval_replay] Loaded %d gold entries (base=%d) from %s",
        len(gold_map), len(base_to_full), gold_path,
    )

    spec = make_spec(variant, scale=config.scale)
    logger.info(
        "[eval_replay] Variant %s: v_round_sigma=%.6f g_h_int=%.6f g_h_bf16=%.6f "
        "protocol=%s",
        variant, spec.v_round_sigma, spec.g_h_int_sigma,
        spec.g_h_bf16_sigma, spec.protocol_constraint,
    )

    all_results: dict = {seed: {} for seed in seeds}

    for epoch in range(epochs):
        infer_path = infer_files[epoch]
        with open(infer_path, "r") as f:
            infer_outputs = json.load(f)
        logger.info(
            "[eval_replay] Epoch %d: loaded %d entries from %s",
            epoch, len(infer_outputs), infer_path.name,
        )

        # 关键步骤：把 baseline doc_key 映射到 gold doc_key
        mapped_outputs, doc_key_map = _remap_doc_keys(
            infer_outputs, base_to_full,
        )
        logger.info(
            "[eval_replay] Epoch %d: mapped %d/%d doc_keys to gold format",
            epoch, len(mapped_outputs), len(infer_outputs),
        )

        for seed in seeds:
            seed_dir = output_dir / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)

            # 应用量化噪声
            new_outputs = apply_variant_to_infer_outputs(
                infer_outputs=mapped_outputs,
                spec=spec,
                seed=seed * 10000 + epoch,  # 不同 epoch 用不同 seed
                gold_map=gold_map,
            )

            # 写出"加噪后的 infer_outputs"（debug 用）
            new_infer_path = seed_dir / f"infer_outputs_epoch_{epoch:03d}.json"
            with open(new_infer_path, "w") as f:
                json.dump(new_outputs, f, indent=2)

            # 调用 evaluate_metrics.py 计算指标
            eval_json_path = seed_dir / f"epoch_{epoch:03d}_evaluate_metrics.json"
            _run_evaluate_metrics(
                outputs_json=str(new_infer_path),
                gold_jsonl=str(gold_path),
                results_dir=str(seed_dir),
                save_prefix=f"epoch_{epoch:03d}_",
            )

            # 读出指标
            if eval_json_path.exists():
                with open(eval_json_path, "r") as f:
                    metrics = json.load(f)
                all_results[seed][epoch] = metrics
                logger.info(
                    "[eval_replay] Variant=%s seed=%d epoch=%d: "
                    "n=%d macro_f1=%.4f micro_f1=%.4f macro_auc=%.4f",
                    variant, seed, epoch,
                    metrics.get("n_samples", 0),
                    metrics.get("macro_f1", float("nan")),
                    metrics.get("micro_f1", float("nan")),
                    metrics.get("macro_auc_ovr", float("nan")) or 0,
                )

    # 写出 summary
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("[eval_replay] Wrote summary to %s", summary_path)

    return all_results


def _remap_doc_keys(
    infer_outputs: dict,
    base_to_full: dict[str, list[str]],
) -> tuple[dict, dict[str, str]]:
    """把 baseline 的 fine-relation doc_key 映射到 gold 的 coarse-relation doc_key。

    baseline doc_key: ..._relation_<fine_relation>
    gold doc_key:     ..._rel_<coarse_relation>

    转换策略：把 baseline doc_key 的 `_relation_<fine>` 替换为 `_rel_<coarse>`，
    然后看 gold 里有没有这个 doc_key。

    由于 baseline 的 fine relation 与 gold 的 coarse relation 一一映射（通过 FINE_TO_GENERAL），
    但同一个 base 可能对应多个 gold（不同 coarse），所以取 first match。

    v2 修复（v2-Bug-0.1）：
      - 删除 `or True` 短路（line 243 的 "candidate in infer_outputs or True" 永远为真）
      - 验证集合从 base_to_full.values() 改为统一构建的 gold_keys
      - 修了 base_to_full 在 load_gold_map 中一对多数据丢失的问题
    """
    out = {}
    mapping = {}
    # 一次性构建 gold 完整 key 集合，避免循环内反复 list→set
    gold_keys = set()
    for full_keys in base_to_full.values():
        gold_keys.update(full_keys)

    matched = 0
    for base_dk, entry in infer_outputs.items():
        # baseline doc_key 形如 ..._relation_<fine>
        if "_relation_" in base_dk:
            base_part = base_dk.rsplit("_relation_", 1)[0]
            fine_rel = base_dk.rsplit("_relation_", 1)[1]
            # 推断 coarse
            coarse = FINE_TO_GENERAL.get(fine_rel.lower().strip())
            if coarse:
                candidate = f"{base_part}_rel_{coarse.replace(' ', '_')}"
                # 验证 candidate 必须在 gold_keys 内
                if candidate in gold_keys:
                    out[candidate] = entry
                    mapping[base_dk] = candidate
                    matched += 1
                    continue
        # fallback：尝试原 doc_key（baseline doc_key 形式）
        out[base_dk] = entry
        mapping[base_dk] = base_dk

    logger.debug(
        "[_remap_doc_keys] matched %d/%d infer entries to gold format",
        matched, len(infer_outputs),
    )
    return out, mapping


def _run_evaluate_metrics(
    outputs_json: str,
    gold_jsonl: str,
    results_dir: str,
    save_prefix: str = "",
) -> None:
    """调用 baseline 的 evaluate_metrics.py 计算指标。"""
    cmd = [
        sys.executable,
        "/root/autodl-tmp/CipherForgeCode/SLG-HE-PIR/baseline/classification_genrel/scripts/evaluate_metrics.py",
        "--outputs_json", outputs_json,
        "--gold_jsonl", gold_jsonl,
        "--results_dir", results_dir,
        "--save_prefix", save_prefix,
    ]
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.CalledProcessError as e:
        logger.error(
            "[eval_replay] evaluate_metrics.py failed: stderr=%s",
            e.stderr[:500],
        )
        raise
    except subprocess.TimeoutExpired:
        logger.error("[eval_replay] evaluate_metrics.py timeout (120s)")