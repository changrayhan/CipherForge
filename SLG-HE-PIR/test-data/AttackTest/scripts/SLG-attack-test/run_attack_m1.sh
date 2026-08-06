#!/bin/bash
# ============================================================================
# M1 Model Inference Attack Test — GPU Real-Protocol Mode
# Target: U-side model inference via prediction distribution
# ============================================================================
set -e

PROJECT_ROOT="/root/autodl-tmp/SLG-HE-PIR"
HF_MODEL="/root/autodl-tmp/hf_cache/models--unsloth--Llama-3.2-1B/snapshots/9535bd9b1d1dea6acafbdc4813b728796aeb28da"
DATA_DIR="/root/autodl-tmp/SLG-HE-PIR/datasets/trec-qc"
OUTPUT_DIR="/root/autodl-tmp/SLG-HE-PIR/test-data/attack-test-data"
BFV_CACHE="/root/autodl-tmp/slg-bfv-cache/attack-test-bfv-cache-1b"

# Llama-3.2-1B config: hidden_dim=2048, num_layers=16, U=8, M=8
# BFV: poly_degree=2048 (must match hidden_dim)

cd "$PROJECT_ROOT"

echo "============================================================"
echo "M1 Model Inference Attack — GPU Real Protocol Mode"
echo "============================================================"
echo "Model:         $HF_MODEL"
echo "Output:        $OUTPUT_DIR/m1_run"
echo "BFV Cache:     $BFV_CACHE"
echo "============================================================"

python3 SLG-attack-test/run_attack_suite.py \
    --attacks M1 \
    --hf_model "$HF_MODEL" \
    --vocab_size 128256 \
    --bfv_hidden_dim 2048 \
    --num_layers 16 \
    --u_layers 8 \
    --bfv_poly_degree 2048 \
    --bfv_plain_bits 30 \
    --bfv_scale 10000.0 \
    --n_steps 10 \
    --n_eval_steps 50 \
    --batch_size 4 \
    --output_dir "$OUTPUT_DIR" \
    --data_dir "$DATA_DIR" \
    --bfv_cache_dir "$BFV_CACHE" \
    --seed 42 \
    --m1_query_budget 1000

echo ""
echo "M1 attack complete. Results in: $OUTPUT_DIR/run_*/attack_results.json"
