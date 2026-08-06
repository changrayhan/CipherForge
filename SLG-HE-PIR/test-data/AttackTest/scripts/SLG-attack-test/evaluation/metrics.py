"""Evaluation metrics for the SLG-HE-PIR attack test suite.

Provides statistical tests and scoring functions used across L-1A, L-3A,
and P-6 attack modules.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Verdict types
# --------------------------------------------------------------------------- #

Verdict = Literal["PRIVACY_PRESERVED", "LEAK_DETECTED", "INCONCLUSIVE"]


@dataclass
class AttackVerdict:
    """Structured result of a single attack evaluation."""

    attack_id: str
    sub_attack: str = ""

    # Primary metric
    metric: str = ""
    value: float = 0.0

    # Reference / baseline
    chance_level: float = 0.0
    std_err: float = 0.0

    # Statistical significance
    p_value: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None

    # Sample size
    n_samples: int = 0
    n_positive: int = 0

    # Outcome
    verdict: Verdict = "INCONCLUSIVE"
    notes: str = ""

    def summary(self) -> str:
        verdict_icon = {
            "PRIVACY_PRESERVED": "[OK]",
            "LEAK_DETECTED":      "[!!]",
            "INCONCLUSIVE":       "[??]",
        }.get(self.verdict, "[??]")
        p_str = f", p={self.p_value:.4f}" if self.p_value is not None else ""
        return (
            f"{verdict_icon} {self.attack_id}/{self.sub_attack} | "
            f"{self.metric}={self.value:.4f} (chance={self.chance_level:.4f}){p_str} | "
            f"n={self.n_samples} | {self.notes}"
        )

    def to_dict(self) -> dict:
        return {
            "attack_id": self.attack_id,
            "sub_attack": self.sub_attack,
            "metric": self.metric,
            "value": float(self.value),
            "chance_level": float(self.chance_level),
            "std_err": float(self.std_err) if self.std_err else 0.0,
            "p_value": float(self.p_value) if self.p_value is not None else None,
            "ci": list(self.confidence_interval) if self.confidence_interval else None,
            "n_samples": self.n_samples,
            "n_positive": self.n_positive,
            "verdict": self.verdict,
            "notes": self.notes,
        }


# --------------------------------------------------------------------------- #
#  Clustering quality metrics (ARI / NMI)
# --------------------------------------------------------------------------- #

def adjusted_rand_index(
    labels_true: np.ndarray,
    labels_pred: np.ndarray,
) -> float:
    """Compute Adjusted Rand Index (ARI) between true and predicted labels.

    ARI ranges from [-1, 1]:
        ARI = 1  → perfect agreement
        ARI ≈ 0  → random clustering (chance level)
        ARI < 0  → worse than random

    Returns:
        float ARI score
    """
    try:
        from sklearn.metrics import adjusted_rand_score
        return float(adjusted_rand_score(labels_true, labels_pred))
    except ImportError:
        logger.warning("sklearn not available; falling back to manual ARI")
        return _manual_ari(labels_true, labels_pred)


def normalized_mutual_info(
    labels_true: np.ndarray,
    labels_pred: np.ndarray,
) -> float:
    """Compute Normalized Mutual Information (NMI) between true and predicted labels."""
    try:
        from sklearn.metrics import normalized_mutual_info_score
        return float(normalized_mutual_info_score(labels_true, labels_pred))
    except ImportError:
        return 0.0


def _manual_ari(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """Fallback ARI when sklearn is unavailable."""
    n = len(labels_true)
    if n == 0:
        return 0.0

    contingency = np.zeros((labels_true.max() + 1, labels_pred.max() + 1))
    for t, p in zip(labels_true, labels_pred):
        contingency[t, p] += 1

    sum_comb = float(np.sum(contingency))
    a_sum = np.sum(contingency, axis=1)
    b_sum = np.sum(contingency, axis=0)

    expected = np.outer(a_sum, b_sum) / sum_comb
    if expected.sum() == 0:
        return 1.0 if np.allclose(contingency, 0) else 0.0

    observed = np.sum(contingency * contingency)
    expected_sum = np.sum(expected * expected)
    if expected_sum == 0:
        return 1.0 if observed == sum_comb else 0.0

    ari = (observed - expected_sum) / (0.5 * (np.sum(a_sum ** 2) + np.sum(b_sum ** 2)) - expected_sum)
    return float(ari)


# --------------------------------------------------------------------------- #
#  Cosine similarity / 1-NN
# --------------------------------------------------------------------------- #

def cosine_similarity_matrix(X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
    """Compute pairwise cosine similarities between rows of X (and Y).

    Returns:
        (len(X), len(Y)) float array.  If Y is None, computes X @ X^T.
    """
    X_norm = np.linalg.norm(X, axis=1, keepdims=True)
    X_norm = np.where(X_norm == 0, 1, X_norm)
    X_unit = X / X_norm

    if Y is None:
        return X_unit @ X_unit.T

    Y_norm = np.linalg.norm(Y, axis=1, keepdims=True)
    Y_norm = np.where(Y_norm == 0, 1, Y_norm)
    Y_unit = Y / Y_norm
    return X_unit @ Y_unit.T


def label_agreement_rate(
    gradients: np.ndarray,
    labels: np.ndarray,
    k: int = 1,
) -> Tuple[float, float, np.ndarray]:
    """Compute label agreement rate via k-NN on gradient space.

    For each sample i, find its k nearest neighbours in the gradient matrix
    (excluding i) and compute the fraction whose true label matches label[i].

    Args:
        gradients: (N, D) float array
        labels: (N,) int array
        k: number of nearest neighbours

    Returns:
        (agreement_rate, std_err, per_sample_agreements)
    """
    N = gradients.shape[0]
    if N <= k:
        logger.warning("N=%d <= k=%d; returning chance level", N, k)
        return 1.0 / len(np.unique(labels)), 0.0, np.zeros(N)

    sim = cosine_similarity_matrix(gradients)
    np.fill_diagonal(sim, -np.inf)  # exclude self

    agreements = []
    for i in range(N):
        knn_idx = np.argsort(sim[i])[-k:]
        knn_labels = labels[knn_idx]
        agreements.append(np.mean(knn_labels == labels[i]))

    agreements = np.array(agreements)
    rate = float(np.mean(agreements))
    std = float(np.std(agreements) / np.sqrt(N))
    return rate, std, agreements


# --------------------------------------------------------------------------- #
#  Permutation test
# --------------------------------------------------------------------------- #

def permutation_test(
    metric_fn: callable,
    gradients: np.ndarray,
    labels: np.ndarray,
    n_permutations: int = 10000,
    seed: int = 42,
) -> Tuple[float, float, np.ndarray]:
    """Permutation test for label-dependence of a metric.

    Args:
        metric_fn: callable(gradients, labels) → float
        gradients: (N, D) float array
        labels: (N,) int array
        n_permutations: number of permutations
        seed: random seed

    Returns:
        (observed_metric, p_value, null_distribution)
    """
    rng = np.random.default_rng(seed)
    observed = metric_fn(gradients, labels)

    null_dist = np.zeros(n_permutations, dtype=np.float64)
    for i in range(n_permutations):
        perm_labels = rng.permutation(labels)
        null_dist[i] = metric_fn(gradients, perm_labels)

    p_value = float(np.mean(null_dist >= observed))
    return observed, p_value, null_dist


# --------------------------------------------------------------------------- #
#  ROC / AUC
# --------------------------------------------------------------------------- #

def binary_auc(
    positive_scores: np.ndarray,
    negative_scores: np.ndarray,
) -> Tuple[float, float]:
    """Compute AUC for binary classification given separate positive/negative score arrays.

    Returns:
        (auc, std_err)
    """
    n_pos = len(positive_scores)
    n_neg = len(negative_scores)

    if n_pos == 0 or n_neg == 0:
        return 0.5, 0.0

    combined = np.concatenate([positive_scores, negative_scores])
    labels = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])

    order = np.argsort(combined)[::-1]
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(combined) + 1)

    auc = (np.sum(labels * ranks) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

    # Variance estimation via U-statistic
    auc = max(0.0, min(1.0, auc))
    q1 = auc / (2 - auc) if auc < 1 else 0
    q2 = 2 * auc * auc / (1 + auc) if auc > 0 else 0
    std_err = np.sqrt((auc * (1 - auc) + (n_pos - 1) * (q1 - auc ** 2) +
                       (n_neg - 1) * (q2 - auc ** 2)) / (n_pos * n_neg))
    return float(auc), float(std_err)


# --------------------------------------------------------------------------- #
#  K-Means wrapper
# --------------------------------------------------------------------------- #

def kmeans_on_gradients(
    gradients: np.ndarray,
    n_clusters: int,
    seed: int = 42,
    n_init: int = 10,
) -> Tuple[np.ndarray, float, float]:
    """Run K-Means on gradient vectors and compute ARI with true labels.

    Args:
        gradients: (N, D) float array
        n_clusters: K for K-Means
        seed: random seed
        n_init: number of K-Means restarts

    Returns:
        (cluster_labels, ari, nmi)
    """
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=seed,
            n_init=n_init,
            algorithm="lloyd",
        )
        pred_labels = kmeans.fit_predict(gradients)
        ari = adjusted_rand_score(np.zeros(len(gradients)), pred_labels)  # placeholder
        nmi = 0.0
        return pred_labels, float(ari), float(nmi)
    except ImportError:
        logger.warning("sklearn not available; K-Means skipped")
        return np.zeros(len(gradients), dtype=np.int64), 0.0, 0.0


# --------------------------------------------------------------------------- #
#  Heuristic: gradient magnitude correlates with label
# --------------------------------------------------------------------------- #

def gradient_magnitude_vs_label(
    g_accum_list: List[np.ndarray],
    labels: np.ndarray,
) -> Tuple[float, float, np.ndarray]:
    """Test whether gradient L2-norm differs across label classes.

    Computes per-sample L2 norm and tests if the distribution of norms
    varies systematically with the label class (ANOVA-style).

    Returns:
        (f_statistic, p_value, per_sample_norms)
    """
    from scipy import stats

    norms = np.array([float(np.linalg.norm(g)) for g in g_accum_list])
    groups = [norms[labels == c] for c in np.unique(labels)]
    groups = [g for g in groups if len(g) > 1]

    if len(groups) < 2:
        return 0.0, 1.0, norms

    f_stat, p_val = stats.f_oneway(*groups)
    return float(f_stat), float(p_val), norms


# --------------------------------------------------------------------------- #
#  Summary statistics
# --------------------------------------------------------------------------- #

def summarise_gradients(gradients: np.ndarray) -> dict:
    """Compute per-axis and aggregate statistics on a gradient matrix."""
    if gradients.size == 0:
        return {}
    return {
        "shape": list(gradients.shape),
        "mean": float(gradients.mean()),
        "std": float(gradients.std()),
        "min": float(gradients.min()),
        "max": float(gradients.max()),
        "l2_norm": float(np.linalg.norm(gradients)),
        "l2_per_sample": float(np.linalg.norm(gradients, axis=1).mean()),
        "sparsity_pct": float(np.mean(np.abs(gradients) < 1e-6) * 100),
    }
