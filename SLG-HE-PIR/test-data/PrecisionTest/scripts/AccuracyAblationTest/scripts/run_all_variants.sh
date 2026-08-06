#!/usr/bin/env bash
# run_all_variants.sh — 串行跑 6 个量化变体 (Q0/Q0'/Q1/Q2'/Q2/Q3)
#
# 每个变体调用 run_variant.py，最后调用 generate_report.py 生成报告。
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG="${REPO_ROOT}/configs/slg_extracted.yaml"
BASELINE_INFER_DIR="/root/autodl-tmp/CipherForgeCode/SLG-HE-PIR/test-data/baseline-test-data/new-cls-baseline-test-data/logs/"
GOLD_PATH="/root/autodl-tmp/CipherForgeCode/SLG-HE-PIR/datasets/botriplex/Preprocessed BioTriplex/test_gold_general_qa.txt"
SLG_JSONL="/root/autodl-tmp/CipherForgeCode/SLG-HE-PIR/test-data/SLG-test-data/cls-SLG-test-data/epoch_metrics.jsonl"
BASELINE_JSONL="/root/autodl-tmp/CipherForgeCode/SLG-HE-PIR/test-data/baseline-test-data/new-cls-baseline-test-data/epoch_metrics.jsonl"

cd "${REPO_ROOT}"
mkdir -p configs

# Step 1: 提取 SLG 参数（如已存在则跳过）
if [ ! -f "${CONFIG}" ]; then
    SLG_CKPT="/root/autodl-tmp/CipherForgeCode/SLG-HE-PIR/test-data/SLG-test-data/cls-SLG-test-data/_SAVE_20260727_0706/checkpoint_epoch_001.pt"
    if [ -f "${SLG_CKPT}" ]; then
        echo "[run_all_variants] Extracting SLG params from ${SLG_CKPT}"
        python scripts/extract_slg_params.py \
            --ckpt_path "${SLG_CKPT}" \
            --output "${CONFIG}"
    else
        echo "[run_all_variants] No SLG ckpt found, using src_fallback defaults"
        python scripts/extract_slg_params.py \
            --ckpt_path "/tmp/nonexistent.pt" \
            --output "${CONFIG}"
    fi
fi

# Step 2: 跑 6 个变体
VARIANTS=( "Q0'" "Q0" "Q1" "Q2'" "Q2" "Q3" )
for VARIANT in "${VARIANTS[@]}"; do
    echo ""
    echo "================================================================"
    echo "[run_all_variants] Running variant: ${VARIANT}"
    echo "================================================================"
    python scripts/run_variant.py \
        --variant "${VARIANT}" \
        --config "${CONFIG}" \
        --baseline_infer_dir "${BASELINE_INFER_DIR}" \
        --gold_path "${GOLD_PATH}" \
        --seeds "42,123,456" \
        --epochs 5
done

# Step 3: 生成最终报告
echo ""
echo "================================================================"
echo "[run_all_variants] Generating final report"
echo "================================================================"
python scripts/generate_report.py \
    --outputs_root outputs \
    --baseline_jsonl "${BASELINE_JSONL}" \
    --slg_jsonl "${SLG_JSONL}" \
    --report_md outputs/QUANT_ABLATION_REPORT.md \
    --report_json outputs/quant_ablation_data.json

echo "[run_all_variants] Done. Report: outputs/QUANT_ABLATION_REPORT.md"