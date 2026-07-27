#!/bin/bash
# ============================================================================
# L-1 dχ Sensitivity Analysis — dp_alpha × dp_answer_beta × dp_calibration_steps
# Grid: 3 × 3 × 3 = 27 runs
# Output: test-data/attack-test-data/dp-ablation/
# ============================================================================
set -e

PROJECT_ROOT="/root/autodl-tmp/SLG-HE-PIR"
HF_MODEL="/root/autodl-tmp/hf_cache/models--unsloth--Llama-3.2-1B/snapshots/9535bd9b1d1dea6acafbdc4813b728796aeb28da"
DATA_DIR="/root/autodl-tmp/SLG-HE-PIR/datasets/trec-qc"
OUTPUT_DIR="/root/autodl-tmp/SLG-HE-PIR/test-data/attack-test-data/dp-ablation"
BFV_CACHE="/root/autodl-tmp/slg-bfv-cache/attack-test-bfv-cache-1b"

mkdir -p "$OUTPUT_DIR"

# Ablation grid
DP_ALPHA_VALUES=(0.05 0.15 0.30)
DP_BETA_VALUES=(0.3 0.5 0.7)
DP_CAL_STEPS_VALUES=(2 5 10)

TOTAL_RUNS=0
for alpha in "${DP_ALPHA_VALUES[@]}"; do
  for beta in "${DP_BETA_VALUES[@]}"; do
    for cal_steps in "${DP_CAL_STEPS_VALUES[@]}"; do
      TOTAL_RUNS=$((TOTAL_RUNS + 1))
    done
  done
done
echo "Total runs: $TOTAL_RUNS"

RUN_IDX=0
for alpha in "${DP_ALPHA_VALUES[@]}"; do
  for beta in "${DP_BETA_VALUES[@]}"; do
    for cal_steps in "${DP_CAL_STEPS_VALUES[@]}"; do
      RUN_IDX=$((RUN_IDX + 1))
      RUN_NAME="alpha_${alpha}_beta_${beta}_cal_${cal_steps}"
      echo ""
      echo "============================================================"
      echo "[$RUN_IDX/$TOTAL_RUNS] L-1 dp ablation: alpha=$alpha beta=$beta cal=$cal_steps"
      echo "============================================================"

      # Run L-1 attack
      python3 SLG-attack-test/run_attack_suite.py \
          --attacks L1 \
          --hf_model "$HF_MODEL" \
          --vocab_size 128256 \
          --bfv_hidden_dim 2048 \
          --num_layers 16 \
          --u_layers 8 \
          --bfv_poly_degree 2048 \
          --bfv_plain_bits 30 \
          --bfv_scale 10000.0 \
          --n_steps 50 \
          --n_eval_steps 30 \
          --batch_size 4 \
          --output_dir "$OUTPUT_DIR" \
          --data_dir "$DATA_DIR" \
          --bfv_cache_dir "$BFV_CACHE" \
          --seed 42 \
          --l1_n_permutations 10000 \
          --l1_alpha 0.05 \
          --dp_enable \
          --dp_alpha "$alpha" \
          --dp_answer_beta "$beta" \
          --dp_calibration_steps "$cal_steps" \
          2>&1 | tee "$OUTPUT_DIR/${RUN_NAME}.log"

      # Find the attack_results.json produced by this run
      RUN_DIR=$(ls -td "$OUTPUT_DIR"/run_* 2>/dev/null | head -1)
      if [ -n "$RUN_DIR" ] && [ -f "$RUN_DIR/attack_results.json" ]; then
        cp "$RUN_DIR/attack_results.json" "$OUTPUT_DIR/${RUN_NAME}_results.json"
        echo "Results saved: $OUTPUT_DIR/${RUN_NAME}_results.json"
      fi
    done
  done
done

echo ""
echo "============================================================"
echo "All $TOTAL_RUNS runs complete. Generating summary CSV..."
echo "============================================================"

# Generate aggregated CSV
python3 - << 'PYEOF'
import json, csv, os, glob
from pathlib import Path

output_dir = Path("/root/autodl-tmp/SLG-HE-PIR/test-data/attack-test-data/dp-ablation")
csv_path = output_dir / "dp_ablation_summary.csv"

rows = []
for result_file in sorted(output_dir.glob("*_results.json")):
    fname = result_file.stem  # e.g. "alpha_0.05_beta_0.3_cal_2_results"
    parts = fname.replace("_results", "").split("_")
    # parts: ["alpha", "0.05", "beta", "0.3", "cal", "2"]
    try:
        alpha = float(parts[1])
        beta = float(parts[3])
        cal = int(parts[5])
    except (IndexError, ValueError):
        print(f"Skipping {result_file.name} (parse error)")
        continue

    try:
        with open(result_file) as f:
            data = json.load(f)
    except Exception as e:
        print(f"Skipping {result_file.name}: {e}")
        continue

    # data["attack_results"] is a list; build a dict keyed by sub_attack
    attack_results = data.get("attack_results", [])
    results = {r["sub_attack"]: r for r in attack_results}

    def get(sub_attack, field):
        return results.get(sub_attack, {}).get(field, "")

    rows.append({
        "dp_alpha": alpha,
        "dp_answer_beta": beta,
        "dp_calibration_steps": cal,
        "h_u_mean_anova_p": get("h_u_mean_anova", "p_value"),
        "h_u_mean_anova_verdict": get("h_u_mean_anova", "verdict"),
        "h_u_norm_anova_p": get("h_u_norm_anova", "p_value"),
        "kmeans_ari": get("kmeans_ari", "value"),
        "1nn_agreement": get("nn_agreement", "value"),
        "1nn_permutation_p": get("permutation_test", "p_value"),
        "cosine_auc": get("cosine_auc", "value"),
        "gradient_magnitude_p": get("magnitude_anova", "p_value"),
        "gradient_magnitude_verdict": get("magnitude_anova", "verdict"),
        "log_file": f"{fname}.log",
        "result_file": result_file.name,
    })

# Sort by alpha, beta, cal
rows.sort(key=lambda r: (r["dp_alpha"], r["dp_answer_beta"], r["dp_calibration_steps"]))

fieldnames = [
    "dp_alpha", "dp_answer_beta", "dp_calibration_steps",
    "h_u_mean_anova_p", "h_u_mean_anova_verdict",
    "h_u_norm_anova_p", "kmeans_ari",
    "1nn_agreement", "1nn_permutation_p",
    "cosine_auc", "gradient_magnitude_p", "gradient_magnitude_verdict",
    "log_file", "result_file",
]

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Summary written to: {csv_path}")
print(f"Total rows: {len(rows)}")
PYEOF

echo ""
echo "Done. Summary CSV: $OUTPUT_DIR/dp_ablation_summary.csv"
echo "Individual logs: $OUTPUT_DIR/*.log"
echo "Individual results: $OUTPUT_DIR/*_results.json"
