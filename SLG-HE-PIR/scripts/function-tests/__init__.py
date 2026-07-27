"""scripts.function_tests — functional test suite for SLG-HE-PIR.

Package layout
--------------
All scripts in this directory are entry points.  Run them from the repo root::

    python -m scripts.function_tests.<module_name> [args...]

or::

    python scripts/function-tests/<module_name>.py [args...]

Test inventory
--------------
+----------------------------------+-----------------------------------------------+
| Script                           | Purpose                                       |
+==================================+===============================================+
| e2e_math_verify                  | Design-2 S→U→M math (gradient recovery)      |
| e2e_correctness_recheck          | Serial vs parallel hot-path correctness       |
| chunk_correctness_test           | Chunked vs flat pipeline correctness          |
| perf_bench                       | Full-scale perf: S/U/M serial vs parallel     |
| bench_optimizations              | 5-step crypto pipeline micro-benchmarks       |
| heterogeneous_correctness_test    | v2.0 runtime: 6 correctness checks            |
| run_small_scale_test             | 7-component smoke test (N=1024)               |
| two_epoch_test                   | 2-epoch convergence validation                 |
| quick_smoke_10step               | 10-step GPU-memory smoke test                  |
| diag_grad_flow                   | 1-step gradient-flow diagnostic                |
| test_step_profiler               | StepProfiler unit tests                        |
| test_trainer_dispatch            | Trainer dispatch wiring tests                  |
| compare_step_profiles             | Flat vs chunked JSONL A/B comparison          |
| diff_step_profiles               | Task A speedup report generator                |
| _demo_profile_jsonl              | Synthetic profile data for dashboard dev       |
| run_with_pyc_finder              | Bootstrap launcher when .py sources are missing|
+----------------------------------+-----------------------------------------------+
"""
from __future__ import annotations

__all__ = []
