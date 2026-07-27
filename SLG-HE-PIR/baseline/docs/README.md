# `baseline/docs/` — Test report and analysis artefacts

This directory contains the comprehensive write-up of the BioTriplex
baseline reproduction that lives next to it.

## Files

| Path | Purpose |
|---|---|
| [`BIOTRIPLEX_BASELINE_TEST_REPORT.md`](./BIOTRIPLEX_BASELINE_TEST_REPORT.md) | The full report — read this first. |
| `scripts/generate_plots.py` | Re-creates every PNG in `figures/` from the JSON metric files under `baseline/{classification_genrel,generation_ner}/{checkpoints,logs}/*.json`. Idempotent; safe to re-run. |
| `figures/*.png` | 12 charts referenced by the report (training dynamics, per-class metrics, confusion matrix, topline bars, cross-task comparison). |

## How to regenerate figures only (no training)

```bash
# from repo root
python3 baseline/docs/scripts/generate_plots.py
```

Output is written to `baseline/docs/figures/`.

## How to regenerate the entire reproduction

```bash
# full ~6h end-to-end (GenRel QA → NER, serial)
nohup bash baseline/run_all.sh > /tmp/baseline_run_all.log 2>&1 &
```

See the report's [Reproduction procedure](../../docs/BIOTRIPLEX_BASELINE_TEST_REPORT.md#7-procedure)
section for the complete step-by-step.

## Dependencies

* Python 3.11
* matplotlib >= 3.5 (only for `generate_plots.py`)
* numpy (only for `generate_plots.py`)
* Everything else is part of the existing repo's training stack
  (transformers, peft, accelerate, torch — see the report's environment
  table).
