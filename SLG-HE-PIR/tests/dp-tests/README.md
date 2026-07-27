# DP H_15 tests for SLG-HE-PIR

This directory contains the unit, integration, smoke and end-to-end tests
for the d_χ-privacy mechanism applied to the 16th-layer hidden state
(`H_15`) of the U shard in SLG-HE-PIR (see `docs/差分隐私实现文档.md`).

## Layout

| File | Purpose |
|---|---|
| `conftest.py` | pytest fixtures (model, tokenizer, dataset, privatiser). |
| `trecqc_adapter.py` | Lightweight adapter on top of `SLG-attack-test/data/trecqc_dataset.py`. |
| `test_dchi_sampler.py` | Unit tests for the multivariate-Laplace sampler. |
| `test_cti_label_based.py` | Unit tests for the label-conditioned CTI. |
| `test_calibrator.py` | Unit tests for `ActivationNormCalibrator`. |
| `test_h15_privatizer.py` | End-to-end unit tests for `H15Privatizer`. |
| `test_party_u_integration.py` | Integration tests against `PartyU.forward_*`. |
| `test_protocol_smoke.py` | Smoke tests for `HeterogeneousProtocol` 1-step. |
| `test_trecqc_e2e.py` | End-to-end TREC-QC training (Llama-3.2-1B + DP). |
| `scripts/run_unit_tests.sh` | CPU-only unit tests. |
| `scripts/run_smoke_test.sh` | Single-step protocol smoke test (CUDA required). |
| `scripts/run_trecqc_e2e.sh` | TREC-QC end-to-end 10-step training (CUDA required). |

## Running

```bash
# Unit tests (CPU)
bash scripts/run_unit_tests.sh

# Protocol smoke (CUDA)
bash scripts/run_smoke_test.sh 0

# TREC-QC end-to-end (CUDA)
bash scripts/run_trecqc_e2e.sh 0 ./dp_test_output
```

The smoke and end-to-end scripts expect:

* a CUDA-capable GPU (default index `0`),
* the Llama-3.2-1B model at `/root/autodl-tmp/SLG-HE-PIR-code/hf_cache/models--unsloth--Llama-3.2-1B/`,
* the TREC-QC dataset at `SLG-HE-PIR/datasets/trec-qc/`.
