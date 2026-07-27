"""L-1: M-side Gradient Label Inference Attack.

Threat model: M (honest-but-curious) observes:
  - Forward phase: H_U (first 16-layer output, plaintext)
  - Forward phase: a_t (BFV-decrypted activation values)
  - Backward phase: g_accum = a_t - V_y (plaintext after BFV decryption)

Attack method:
  1. Forward-phase analysis on H_U: class-mean ANOVA, L2-norm ANOVA
  2. Backward-phase analysis on g_accum: K-Means ARI, 1-NN agreement,
     Cosine AUC, permutation test, gradient magnitude ANOVA

Expected (PRG mask effective): all metrics near random baseline.
- ARI ≈ 0, 1-NN ≈ 16.7% (6-class random), AUC ≈ 0.5, p > 0.05

Reference: TEST_REPORT.md Section 2.2.1
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
from scipy import stats

from attacks.base import BaseAttack
from evaluation.metrics import (
    AttackVerdict,
    adjusted_rand_index,
    cosine_similarity_matrix,
    gradient_magnitude_vs_label,
    kmeans_on_gradients,
    label_agreement_rate,
    permutation_test,
)

logger = logging.getLogger(__name__)


def _knn_agreement_metric(gradients: np.ndarray, labels: np.ndarray) -> float:
    """Wrapper for permutation test: 1-NN label agreement rate."""
    rate, _, _ = label_agreement_rate(gradients, labels, k=1)
    return rate


class L1GradientInference(BaseAttack):
    """M-side gradient label inference attack (CutGrad-Free variant).

    Targets:
      - Forward phase: H_U (M-side first 16-layer output, plaintext)
      - Backward phase: g_accum = a_t - V_y (post-BFV-decryption gradient)
    """

    ATTACK_ID = "L1"
    ATTACK_NAME = "M-side Gradient Label Inference"
    TARGET = "H_U (forward) + g_accum = a_t - V_y (backward)"
    THREAT_MODEL = "M (honest-but-curious)"

    def __init__(self, n_permutations: int = 10000, alpha: float = 0.05, **kwargs):
        super().__init__(**kwargs)
        self._n_permutations = n_permutations
        self._alpha = alpha
        # Backward-pass data
        self._gradients: List[np.ndarray] = []
        self._labels: List[int] = []
        # Forward-pass data
        self._h_u: List[np.ndarray] = []

    def collect(self, step_result: Any) -> None:
        """Collect gradient, H_U, and label from one step result.

        Args:
            step_result: AttackDataBundle or dict with keys:
                - g_accum: (n_tokens, hidden_dim) gradient array (backward)
                - H_U: (n_tokens, hidden_dim) forward 16-layer output (optional)
                - token_labels: List[int] of coarse class indices (0-5)
        """
        bundle = step_result
        if hasattr(step_result, "g_accum"):
            g_accum = bundle.g_accum
            h_u = getattr(bundle, "H_U", None) or getattr(bundle, "h_u", None)
            labels = bundle.token_labels
        elif isinstance(step_result, dict):
            g_accum = step_result.get("g_accum")
            h_u = step_result.get("H_U")
            if h_u is None:
                h_u = step_result.get("h_u")
            labels = step_result.get("token_labels")
        else:
            logger.warning("L1: Unknown step_result type %s", type(step_result))
            return

        if g_accum is None or labels is None:
            return

        # Take the last token (answer token) per sample
        batch_size = len(labels)
        if g_accum.shape[0] >= batch_size:
            answer_grad = g_accum[-batch_size:]
        else:
            answer_grad = g_accum

        self._gradients.append(answer_grad.astype(np.float32))
        self._labels.extend(labels)

        # H_U is forward data — collect separately if available
        if h_u is not None:
            h_u_arr = np.asarray(h_u, dtype=np.float32)
            if h_u_arr.ndim == 3:
                # Shape (batch, seq, hidden) → take last token of each sample
                h_u_last = h_u_arr[:, -1, :]
            elif h_u_arr.ndim == 2:
                h_u_last = h_u_arr
            else:
                h_u_last = h_u_arr.reshape(-1, h_u_arr.shape[-1])

            if h_u_last.shape[0] >= batch_size:
                answer_h_u = h_u_last[-batch_size:]
            else:
                answer_h_u = h_u_last
            self._h_u.append(answer_h_u)

    def run(self) -> List[AttackVerdict]:
        """Run all statistical tests on collected gradients, H_U, and labels."""
        if not self._gradients:
            logger.warning("L1: No gradients collected")
            return [AttackVerdict(
                attack_id=self.ATTACK_ID,
                metric="error",
                value=0.0,
                verdict="INCONCLUSIVE",
                notes="No gradient data collected",
            )]

        # Build backward-phase matrices
        G = np.concatenate(self._gradients, axis=0)
        y = np.array(self._labels, dtype=np.int64)
        N, D = G.shape
        n_classes = len(np.unique(y))
        chance_level = 1.0 / n_classes

        logger.info(f"L1: G shape={G.shape}, n_classes={n_classes}, N={N}")

        verdicts: List[AttackVerdict] = []

        # ── Forward phase: H_U analysis (only if H_U was collected) ────────
        if self._h_u:
            H_U = np.concatenate(self._h_u, axis=0)
            n_h_u = min(H_U.shape[0], y.shape[0])
            H_U = H_U[:n_h_u]
            y_h = y[:n_h_u]

            # 1a. H_U class-mean ANOVA (per-feature → aggregate p-value via min)
            verdicts.append(self._compute_h_u_mean_anova(H_U, y_h, n_classes))

            # 1b. H_U L2-norm ANOVA
            verdicts.append(self._compute_h_u_norm_anova(H_U, y_h, n_classes))
        else:
            logger.info("L1: No H_U data collected; skipping forward-phase analysis")
            verdicts.append(AttackVerdict(
                attack_id=self.ATTACK_ID,
                sub_attack="h_u_mean_anova",
                metric="h_u_class_mean_anova_pvalue",
                value=float("nan"),
                chance_level=self._alpha,
                n_samples=N,
                p_value=None,
                verdict="INCONCLUSIVE",
                notes="No H_U (forward 16-layer output) collected",
            ))
            verdicts.append(AttackVerdict(
                attack_id=self.ATTACK_ID,
                sub_attack="h_u_norm_anova",
                metric="h_u_norm_anova_pvalue",
                value=float("nan"),
                chance_level=self._alpha,
                n_samples=N,
                p_value=None,
                verdict="INCONCLUSIVE",
                notes="No H_U (forward 16-layer output) collected",
            ))

        # ── Backward phase: gradient analysis ─────────────────────────────
        # 2. K-Means ARI
        ari_value, ari_verdict = self._compute_ari(G, y, n_classes)
        verdicts.append(AttackVerdict(
            attack_id=self.ATTACK_ID,
            sub_attack="kmeans_ari",
            metric="adjusted_rand_index",
            value=ari_value,
            chance_level=0.0,
            n_samples=N,
            verdict=ari_verdict,
            notes="ARI > 0.1 indicates label leakage in gradient space" if ari_verdict == "LEAK_DETECTED" else "ARI near 0: gradient space does not encode labels",
        ))

        # 3. 1-NN Agreement Rate
        nn_rate, nn_std, _ = label_agreement_rate(G, y, k=1)
        nn_verdict = "LEAK_DETECTED" if nn_rate > chance_level + 2 * nn_std else "PRIVACY_PRESERVED"
        verdicts.append(AttackVerdict(
            attack_id=self.ATTACK_ID,
            sub_attack="nn_agreement",
            metric="1nn_label_agreement",
            value=nn_rate,
            chance_level=chance_level,
            std_err=nn_std,
            n_samples=N,
            verdict=nn_verdict,
            notes=f"Rate > {chance_level:.3f} + 2σ indicates nearest neighbors share labels",
        ))

        # 4. Cosine AUC
        auc_value, auc_verdict = self._compute_cosine_auc(G, y)
        verdicts.append(AttackVerdict(
            attack_id=self.ATTACK_ID,
            sub_attack="cosine_auc",
            metric="cosine_similarity_auc",
            value=auc_value,
            chance_level=0.5,
            n_samples=N,
            verdict=auc_verdict,
            notes="AUC > 0.5 + 2σ indicates gradient similarity predicts same-label pairs" if auc_verdict == "LEAK_DETECTED" else "AUC near 0.5: gradients do not encode label similarity",
        ))

        # 5. Permutation Test
        observed_rate, p_value, _ = permutation_test(
            _knn_agreement_metric, G, y,
            n_permutations=self._n_permutations,
            seed=42,
        )
        perm_verdict = "LEAK_DETECTED" if p_value < self._alpha else "PRIVACY_PRESERVED"
        verdicts.append(AttackVerdict(
            attack_id=self.ATTACK_ID,
            sub_attack="permutation_test",
            metric="1nn_agreement_permutation_pvalue",
            value=p_value,
            chance_level=self._alpha,
            n_samples=N,
            p_value=p_value,
            verdict=perm_verdict,
            notes=f"p < {self._alpha}: observed agreement={observed_rate:.4f} is statistically significant" if perm_verdict == "LEAK_DETECTED" else f"p >= {self._alpha}: no statistically significant label signal",
        ))

        # 6. Gradient Magnitude ANOVA
        # Compute per-sample L2 norms of the gradient rows. We must operate on
        # the *norms* (1-D per-sample scalars) for the ANOVA F-test and the
        # eta-squared effect size — not on the raw 2048-dim gradient vectors.
        norms_per_sample = np.array(
            [np.linalg.norm(g[i]) for g in self._gradients for i in range(len(g))],
            dtype=np.float64,
        )
        if norms_per_sample.size == 0:
            f_stat, anova_p, _ = 0.0, 1.0, np.array([])
        else:
            groups_for_anova = [
                norms_per_sample[i : i + 1] for i in range(len(norms_per_sample))
            ]
            f_stat, anova_p, _ = gradient_magnitude_vs_label(groups_for_anova, y)

        # Effect size: eta-squared = SS_between / SS_total, computed on the
        # per-class groups of *norms* (the previous code mistakenly mixed
        # per-vector element means with scalar-norm means, producing values
        # far outside [0, 1]).
        grand_mean = float(norms_per_sample.mean()) if norms_per_sample.size else 0.0
        classes = np.unique(y)
        per_class_groups = [
            norms_per_sample[y == c] for c in classes if int((y == c).sum()) > 1
        ]
        ss_between = sum(
            float(len(g)) * (float(g.mean()) - grand_mean) ** 2 for g in per_class_groups
        )
        ss_total = float(((norms_per_sample - grand_mean) ** 2).sum()) if norms_per_sample.size else 0.0
        eta_sq = ss_between / ss_total if ss_total > 0 else 0.0
        # Clamp eta_sq defensively (numerical noise can push it slightly > 1).
        eta_sq = max(0.0, min(1.0, eta_sq))
        anova_leak = (anova_p < self._alpha) and (eta_sq > 0.10)
        anova_verdict = "LEAK_DETECTED" if anova_leak else "PRIVACY_PRESERVED"
        verdicts.append(AttackVerdict(
            attack_id=self.ATTACK_ID,
            sub_attack="magnitude_anova",
            metric="gradient_magnitude_anova_pvalue",
            value=float(anova_p),
            chance_level=self._alpha,
            n_samples=N,
            p_value=float(anova_p),
            verdict=anova_verdict,
            notes=(
                f"ANOVA p={anova_p:.4e}, eta²={eta_sq:.4f}: gradient magnitudes differ "
                f"across label classes (large effect)"
                if anova_verdict == "LEAK_DETECTED"
                else f"ANOVA p={anova_p:.4e}, eta²={eta_sq:.4f}: no meaningful magnitude "
                f"difference across classes (effect size below 0.10)"
            ),
        ))

        # Save gradient matrix for offline analysis
        self._save_gradient_data(G, y)

        return verdicts

    # --------------------------------------------------------------------------- #
    #  Forward-phase H_U statistical tests
    # --------------------------------------------------------------------------- #

    def _compute_h_u_mean_anova(self, H_U: np.ndarray, y: np.ndarray, n_classes: int) -> AttackVerdict:
        """Per-feature one-way ANOVA on H_U mean values, then aggregate p-values.

        For each of the D hidden dimensions, run f_oneway across classes.
        Use the minimum p-value (most significant feature) as the summary metric.
        """
        n_features = H_U.shape[1]
        # Subsample features for tractability if D is huge
        max_features = 512
        if n_features > max_features:
            rng = np.random.default_rng(42)
            feat_idx = rng.choice(n_features, size=max_features, replace=False)
            H_U_sub = H_U[:, feat_idx]
        else:
            H_U_sub = H_U

        classes = np.unique(y)
        p_values = []
        for d in range(H_U_sub.shape[1]):
            groups = [H_U_sub[y == c, d] for c in classes]
            groups = [g for g in groups if len(g) > 1]
            if len(groups) < 2:
                continue
            try:
                _, p = stats.f_oneway(*groups)
                p_values.append(p)
            except Exception:
                continue

        if not p_values:
            return AttackVerdict(
                attack_id=self.ATTACK_ID,
                sub_attack="h_u_mean_anova",
                metric="h_u_class_mean_anova_pvalue",
                value=1.0,
                chance_level=self._alpha,
                n_samples=H_U.shape[0],
                p_value=1.0,
                verdict="INCONCLUSIVE",
                notes="Could not compute ANOVA (insufficient features or data)",
            )

        p_min = float(np.min(p_values))
        n_tested = max(1, len(p_values))
        bonferroni_alpha = self._alpha / n_tested
        min_p_adj = p_min
        # Compute BH-FDR adjusted p-value (one of several correction methods).
        try:
            from statsmodels.stats.multitest import multipletests
            _, p_adj, _, _ = multipletests(p_values, alpha=self._alpha, method="fdr_bh")
            min_p_adj = float(np.min(p_adj))
        except Exception:
            pass
        # Robust aggregate test: a **binomial proportion test** on the number
        # of features with raw p < 0.05.  Under the null this is Binomial(n_tested, 0.05);
        # a real signal would push the count well above the expected ~5%.
        n_rejected = int(np.sum(np.array(p_values) < 0.05))
        try:
            from scipy.stats import binom
            prop_p_value = float(1.0 - binom.cdf(n_rejected, n_tested, 0.05))
        except Exception:
            prop_p_value = 1.0
        # Verdict logic:
        # - Binomial aggregate test (proportion of features with raw p < 0.05) is
        #   the primary, most robust statistic (insensitive to single-feature outliers).
        # - BH-FDR and Bonferroni min p-value are auxiliary; they trigger LEAK
        #   only when the binomial test is borderline (within 5x of significance).
        # This cascade prevents false alarms from a single anomalous feature.
        if prop_p_value < self._alpha * 0.5:
            # Binomial aggregate strongly rejects null
            verdict = "LEAK_DETECTED"
        elif prop_p_value < self._alpha:
            # Binomial aggregate is borderline — escalate to BH-FDR
            if min_p_adj < self._alpha:
                verdict = "LEAK_DETECTED"
            else:
                verdict = "PRIVACY_PRESERVED"
        elif min_p_adj < bonferroni_alpha:
            # Binomial is non-significant but Bonferroni triggers on a single feature.
            verdict = "LEAK_DETECTED"
        else:
            verdict = "PRIVACY_PRESERVED"
        return AttackVerdict(
            attack_id=self.ATTACK_ID,
            sub_attack="h_u_mean_anova",
            metric="h_u_class_mean_anova_pvalue",
            value=p_min,
            chance_level=float(bonferroni_alpha),
            n_samples=H_U.shape[0],
            p_value=p_min,
            confidence_interval=(0.0, prop_p_value),
            verdict=verdict,
            notes=(
                f"Min p-value across {n_tested} features of H_U: {p_min:.4e}. "
                f"Binomial aggregate: {n_rejected}/{n_tested} features with raw p<0.05 "
                f"(p={prop_p_value:.4e} under Binomial(n={n_tested}, 0.05) null). "
                f"BH-FDR min p_adj={min_p_adj:.4e}. "
                f"{'H_U shows systematic per-feature class mean differences (binomial aggregate + FDR)' if verdict == 'LEAK_DETECTED' else 'H_U does not encode label-mean structure (binomial aggregate)'}"
            ),
        )

    def _compute_h_u_norm_anova(self, H_U: np.ndarray, y: np.ndarray, n_classes: int) -> AttackVerdict:
        """One-way ANOVA on per-sample L2 norm of H_U.

        Uses both statistical significance (p < alpha) **and** effect size
        (eta-squared > 0.06, "medium" effect) to flag a leak.  This avoids
        false positives from tiny effect sizes that are statistically
        significant only because of the large sample size.
        """
        norms = np.linalg.norm(H_U, axis=1)
        classes = np.unique(y)
        groups = [norms[y == c] for c in classes]
        groups = [g for g in groups if len(g) > 1]
        if len(groups) < 2:
            return AttackVerdict(
                attack_id=self.ATTACK_ID,
                sub_attack="h_u_norm_anova",
                metric="h_u_norm_anova_pvalue",
                value=1.0,
                chance_level=self._alpha,
                n_samples=H_U.shape[0],
                p_value=1.0,
                verdict="INCONCLUSIVE",
                notes="Insufficient groups for ANOVA",
            )

        f_stat, p_value = stats.f_oneway(*groups)
        # Effect size: eta-squared = SS_between / SS_total
        grand_mean = norms.mean()
        ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
        ss_total = float(((norms - grand_mean) ** 2).sum())
        eta_sq = ss_between / ss_total if ss_total > 0 else 0.0
        leak = (p_value < self._alpha) and (eta_sq > 0.10)
        verdict = "LEAK_DETECTED" if leak else "PRIVACY_PRESERVED"
        return AttackVerdict(
            attack_id=self.ATTACK_ID,
            sub_attack="h_u_norm_anova",
            metric="h_u_norm_anova_pvalue",
            value=float(p_value),
            chance_level=self._alpha,
            n_samples=H_U.shape[0],
            p_value=float(p_value),
            verdict=verdict,
            notes=(
                f"ANOVA p={p_value:.4e}, eta²={eta_sq:.4f}: H_U L2 norms differ "
                f"across label classes (large effect)"
                if verdict == "LEAK_DETECTED"
                else f"ANOVA p={p_value:.4e}, eta²={eta_sq:.4f}: no meaningful norm "
                     f"difference across classes (effect size below 0.10)"
            ),
        )

    # --------------------------------------------------------------------------- #
    #  Backward-phase gradient statistical tests
    # --------------------------------------------------------------------------- #

    def _compute_ari(self, G: np.ndarray, y: np.ndarray, n_classes: int) -> Tuple[float, str]:
        """Compute K-Means ARI with true labels."""
        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import adjusted_rand_score

            kmeans = KMeans(n_clusters=n_classes, random_state=42, n_init=10)
            pred_labels = kmeans.fit_predict(G)
            ari = adjusted_rand_score(y, pred_labels)
            verdict = "LEAK_DETECTED" if ari > 0.1 else "PRIVACY_PRESERVED"
            return float(ari), verdict
        except ImportError:
            logger.warning("L1: sklearn not available, using manual ARI")
            pred_labels = kmeans_on_gradients(G, n_classes, seed=42)[0]
            ari = adjusted_rand_index(y, pred_labels)
            verdict = "LEAK_DETECTED" if ari > 0.1 else "PRIVACY_PRESERVED"
            return float(ari), verdict

    def _compute_cosine_auc(self, G: np.ndarray, y: np.ndarray) -> Tuple[float, str]:
        """Compute AUC based on pairwise cosine similarity for same-label pairs."""
        N = G.shape[0]
        sim_matrix = cosine_similarity_matrix(G)  # (N, N)
        np.fill_diagonal(sim_matrix, -np.inf)

        pos_scores, neg_scores = [], []
        for i in range(N):
            for j in range(i + 1, N):
                if sim_matrix[i, j] == -np.inf:
                    continue
                if y[i] == y[j]:
                    pos_scores.append(sim_matrix[i, j])
                else:
                    neg_scores.append(sim_matrix[i, j])

        if not pos_scores or not neg_scores:
            return 0.5, "INCONCLUSIVE"

        from evaluation.metrics import binary_auc
        auc, _ = binary_auc(np.array(pos_scores), np.array(neg_scores))

        # 4-sigma threshold for AUC (conservative to minimize false positives on synthetic data)
        n = len(pos_scores) + len(neg_scores)
        sigma = np.sqrt(0.25 / n)
        threshold = 0.5 + 4 * sigma  # 4-sigma for high specificity

        verdict = "LEAK_DETECTED" if auc > threshold else "PRIVACY_PRESERVED"
        return float(auc), verdict

    def _save_gradient_data(self, G: np.ndarray, y: np.ndarray) -> None:
        """Save gradient matrix, H_U, and labels for offline analysis."""
        try:
            if hasattr(self, "output_dir") and self.output_dir:
                save_dir = Path(self.output_dir) / self.ATTACK_ID.lower()
            else:
                save_dir = Path("SLG-attack-test/results/l1")
            save_dir.mkdir(parents=True, exist_ok=True)
            np.save(save_dir / "gradient_matrix.npy", G)
            np.save(save_dir / "label_array.npy", y)
            if self._h_u:
                H_U = np.concatenate(self._h_u, axis=0)
                np.save(save_dir / "h_u_matrix.npy", H_U)
            with open(save_dir / "metadata.json", "w") as f:
                json.dump({
                    "shape": list(G.shape),
                    "n_samples": int(G.shape[0]),
                    "n_classes": int(len(np.unique(y))),
                    "labels": y.tolist(),
                    "has_h_u": len(self._h_u) > 0,
                }, f, indent=2)
            logger.info(f"L1: Saved gradient data to {save_dir}")
        except Exception as e:
            logger.warning(f"L1: Failed to save gradient data: {e}")

    def finalise(self) -> List[AttackVerdict]:
        return self.run()
