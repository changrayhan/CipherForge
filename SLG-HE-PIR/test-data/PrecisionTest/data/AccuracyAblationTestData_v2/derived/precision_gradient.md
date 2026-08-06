# Precision Gradient: Baseline → SLG (Task A, BioTriplex 7-class)

> **⚠️ Caliber warning**: Baseline uses `evaluate_metrics.py` (sklearn, keep parse_idx=-1). SLG uses `compute_classification_metrics` (replace missing predictions with `relation undefined`, custom macro-auc). See `eval_caliber_diff.md`. Numbers below are *raw, as-reported* — they are **not** a clean encryption-tax estimate.

## Best-epoch comparison

| metric | Baseline (best) | SLG (best) | Δ (Baseline−SLG) |
|--------|----------------:|-----------:|------------------:|
| micro_f1 | 0.3662 | 0.2709 | 0.0953 |
| macro_f1 | 0.2633 | 0.1515 | 0.1118 |
| weighted_f1 | 0.3806 | 0.2759 | 0.1047 |
| micro_accuracy | 0.3662 | 0.2709 | 0.0953 |
| macro_precision | 0.3522 | 0.2067 | 0.1455 |
| macro_recall | 0.3501 | 0.2309 | 0.1192 |
| micro_auc | 0.7289 | 0.7344 | -0.0055 |
| macro_auc | 0.7771 | 0.6815 | 0.0956 |
| parse_failures | 0 | 0 | 0 |
| n_samples | 213 | 203 | 10 |
| train_steps | 596 | 734 | -138 |

Baseline best epoch = 1, SLG best epoch = 4

## 5-epoch average comparison

| metric | Baseline avg | SLG avg | Δ (Baseline−SLG) |
|--------|-------------:|--------:|------------------:|
| micro_f1 | 0.3005 | 0.2650 | 0.0354 |
| macro_f1 | 0.2171 | 0.1468 | 0.0702 |
| weighted_f1 | 0.2735 | 0.2685 | 0.0050 |
| micro_accuracy | 0.3005 | 0.2650 | 0.0354 |
| macro_precision | 0.3571 | 0.2014 | 0.1557 |
| macro_recall | 0.3283 | 0.2228 | 0.1055 |
| micro_auc | 0.7129 | 0.7345 | -0.0216 |
| macro_auc | 0.7849 | 0.6790 | 0.1060 |
| parse_failures | 0.0000 | 0.0000 | 0.0000 |
| n_samples | 213.0000 | 203.0000 | 10.0000 |
| train_steps | 596.0000 | 734.0000 | -138.0000 |

## Per-class F1 (best epoch)

| class | Baseline F1 | SLG F1 | Δ F1 |
|-------|------------:|-------:|-----:|
| pathological | 0.5794 | 0.6250 | -0.0456 |
| modulatory | 0.1515 | 0.1467 | 0.0048 |
| expression change | 0.3736 | 0.2887 | 0.0850 |
| diagnosis | 0.4848 | 0.0000 | 0.4848 |
| therapy | 0.2535 | 0.0000 | 0.2535 |
| no relation | 0.0000 | 0.0000 | 0.0000 |
| relation undefined | 0.0000 | 0.0000 | 0.0000 |

## Data caliber caveats

- **Sample count**: Baseline 213 vs SLG 203 (-10). Likely root cause: different dataset class implementations (`biotriplex_qakshot_dataset.py` vs `src/data/biotriplex_dataset.py`) yielding different doc_key sets.
- **Train steps**: Baseline 596 vs SLG 734 (+138). Same root cause.
- **Eval code paths differ**: see `eval_caliber_diff.md`.

## Interpretation

As-reported gap: ≈ 9-11 pp on macro_f1, ≈ 9-10 pp on micro_f1, ≈ 9-10 pp on macro_auc.
After accounting for target_modules (2→7) and quantization (Q0/Q1/Q2/Q3), the residual gap is the **BFV-encryption tax** — to be quantified by the AccuracyAblationTest sub-package.

