"""Statistical analysis utilities for SLG-HE-PIR experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

__all__ = ["StatisticsAnalyzer", "StatisticalSummary"]


@dataclass
class StatisticalSummary:
    """Statistical summary of a dataset."""
    mean: float
    std: float
    median: float
    min: float
    max: float
    n: int
    ci_95: Tuple[float, float]
    se: float


class StatisticsAnalyzer:
    """Statistical methods for experimental data."""

    @staticmethod
    def compute_summary(data: List[float]) -> StatisticalSummary:
        """Compute comprehensive statistical summary."""
        arr = np.array(data, dtype=np.float64)
        n = len(arr)
        if n == 0:
            return StatisticalSummary(
                mean=0.0, std=0.0, median=0.0, min=0.0, max=0.0, n=0,
                ci_95=(0.0, 0.0), se=0.0
            )
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1))
        median = float(np.median(arr))
        min_val = float(np.min(arr))
        max_val = float(np.max(arr))
        se = std / np.sqrt(n) if n > 0 else 0.0
        # t-critical for 95% CI (two-tailed)
        t_critical = 1.96 if n >= 30 else 2.045
        ci_lower = mean - t_critical * se
        ci_upper = mean + t_critical * se
        return StatisticalSummary(
            mean=mean, std=std, median=median,
            min=min_val, max=max_val, n=n,
            ci_95=(ci_lower, ci_upper), se=se,
        )

    @staticmethod
    def t_test(
        sample1: List[float],
        sample2: List[float],
        paired: bool = False,
    ) -> Dict[str, Any]:
        """Independent or paired t-test."""
        a, b = np.array(sample1), np.array(sample2)
        n1, n2 = len(a), len(b)
        if n1 == 0 or n2 == 0:
            return {"p_value": 1.0, "t_stat": 0.0, "significant": False}

        if paired:
            diff = a - b
            mean_diff = float(np.mean(diff))
            std_diff = float(np.std(diff, ddof=1))
            se = std_diff / np.sqrt(n1) if n1 > 0 else 0.0
            t_stat = mean_diff / se if se > 0 else 0.0
            df = n1 - 1
        else:
            m1, m2 = float(np.mean(a)), float(np.mean(b))
            s1, s2 = float(np.std(a, ddof=1)), float(np.std(b, ddof=1))
            pooled_se = np.sqrt(s1**2 / n1 + s2**2 / n2)
            t_stat = (m1 - m2) / pooled_se if pooled_se > 0 else 0.0
            # Welch-Satterthwaite degrees of freedom
            num = (s1**2 / n1 + s2**2 / n2) ** 2
            denom = (s1**2 / n1)**2 / (n1 - 1) + (s2**2 / n2)**2 / (n2 - 1)
            df = int(num / denom) if denom > 0 else n1 + n2 - 2

        from scipy.stats import ttest_ind, ttest_rel
        if paired and n1 == n2:
            _, p = ttest_rel(a, b)
        else:
            _, p = ttest_ind(a, b, equal_var=not paired)
        return {
            "t_stat": float(t_stat),
            "p_value": float(p),
            "df": int(df),
            "significant": bool(p < 0.05),
        }

    @staticmethod
    def bootstrap_ci(
        data: List[float],
        statistic: str = "mean",
        n_bootstrap: int = 10000,
        ci: float = 0.95,
    ) -> Tuple[float, float]:
        """Bootstrap confidence interval for a statistic."""
        arr = np.array(data)
        rng = np.random.default_rng()
        values = []
        for _ in range(n_bootstrap):
            sample = rng.choice(arr, size=len(arr), replace=True)
            if statistic == "mean":
                values.append(float(np.mean(sample)))
            elif statistic == "median":
                values.append(float(np.median(sample)))
        alpha = 1 - ci
        lo = float(np.percentile(values, 100 * alpha / 2))
        hi = float(np.percentile(values, 100 * (1 - alpha / 2)))
        return lo, hi

    @staticmethod
    def mannwhitney_u(
        sample1: List[float],
        sample2: List[float],
    ) -> Dict[str, Any]:
        """Mann-Whitney U test (non-parametric alternative to t-test)."""
        from scipy.stats import mannwhitneyu
        u_stat, p = mannwhitneyu(sample1, sample2, alternative="two-sided")
        return {
            "u_statistic": float(u_stat),
            "p_value": float(p),
            "significant": bool(p < 0.05),
        }

    @staticmethod
    def compare_distributions(
        sample1: List[float],
        sample2: List[float],
    ) -> Dict[str, Any]:
        """Comprehensive comparison: t-test + Mann-Whitney + bootstrap."""
        return {
            "summary1": StatisticsAnalyzer.compute_summary(sample1).__dict__,
            "summary2": StatisticsAnalyzer.compute_summary(sample2).__dict__,
            "t_test": StatisticsAnalyzer.t_test(sample1, sample2),
            "mannwhitney": StatisticsAnalyzer.mannwhitney_u(sample1, sample2),
        }
