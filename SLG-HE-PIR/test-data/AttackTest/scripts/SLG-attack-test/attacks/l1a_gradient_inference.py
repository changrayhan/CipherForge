"""L-1A: Gradient Label Inference Attack (CutGrad-Free variant).

Threat model
------------
An honest-but-curious M (or an auditor with access to M's BFV decryption
outputs) observes the per-token plaintext gradient vectors ``g_accum =
a_t - V_y`` that PartyM recovers during each backward pass.

The attack tests whether the gradient space leaks the class label ``y`` of
the answer token, which would undermine the PRG masking in S3PIR.

Attack logic
------------
Two complementary tests are run in parallel:

1. **K-Means ARI** — Fit K-Means (K = number of classes) on the collected
   answer-token gradient vectors.  If gradients of the same class cluster
   together (ARI >> 0), the label is leaked.  Under correct PRG masking,
   ARI ≈ 0 (chance level).

2. **1-NN Label Agreement Rate** — For each sample i, find its nearest
   neighbour in gradient space.  If that neighbour shares the same true
   label more often than chance (6 classes → ~16.7%), the gradient space
   is label-structured.  A permutation test assesses statistical
   significance.

3. **Gradient Similarity AUC** — For each pair of samples (i, j), define
   the "same-label" indicator as the binary target and the cosine
   similarity of their gradient vectors as the score.  Compute AUC.
   AUC ≈ 0.5 under masking; AUC significantly > 0.5 indicates leakage.

4. **Gradient Magnitude ANOVA** — Per-class L2-norm of gradients.
   Systematic variation across classes is a secondary leakage signal.

Reference: CutGrad (USENIX Sec'23) and GradLeak (arXiv) — adapted to
the BFV-based SLG setting where the masked gradient is already the
plaintext of the final gradient computation.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from evaluation.metrics import (
    AttackVerdict,
    cosine_similarity_matrix,
    gradient_magnitude_vs_label,
    label_agreement_rate,
    permutation_test,
    summarise_gradients,
)
from evaluation.reporter import AttackReporter
from protocol.attack_protocol_wrapper import AttackDataBundle, AttackProtocolWrapper

logger = logging.getLogger(__name__)


class L1AGradientInferenceAttack:
    """L-1A gradient label inference attack evaluator."""

    ATTACK_ID = "L1A"
    ATTACK_NAME = "Gradient Label Inference (CutGrad-Free)"
    TARGET = "PartyM → g_accum (per-token plaintext gradients)"
    THREAT_MODEL = "Honest-but-curious M with access to BFV decryption outputs"

    def __init__(
        self,
        n_classes: int = 6,
        n_permutations: int = 10000,
        alpha: float = 0.05,
        seed: int = 42,
        output_dir: str = "SLG-attack-test/results",
    ):
        """
        Args:
            n_classes: Number of label classes (TREC-QC coarse = 6).
            n_permutations: Iterations for permutation significance test.
            alpha: Significance threshold.
            seed: Random seed.
            output_dir: Directory for saved results.
        """
        self.n_classes = n_classes
        self.n_permutations = n_permutations
        self.alpha = alpha
        self.seed = seed
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._gradient_matrix: Optional[np.ndarray] = None
        self._label_array: Optional[np.ndarray] = None
        self._bundles: List[AttackDataBundle] = []
        self._verdicts: List[AttackVerdict] = []

    # ------------------------------------------------------------------------- #
    #  Data collection interface
    # ------------------------------------------------------------------------- #

    def ingest_bundles(self, bundles: List[AttackDataBundle]) -> None:
        """Ingest AttackDataBundle objects from AttackProtocolWrapper.

        Populates ``self._gradient_matrix`` and ``self._label_array`` using
        the answer-token (last batch_size tokens) gradient from each bundle.
        """
        gradients = []
        labels = []

        for bundle in bundles:
            if bundle.g_accum is None or bundle.token_labels is None:
                continue

            B = self._infer_batch_size(bundle)
            n_total = bundle.g_accum.shape[0]
            n_take = min(B, n_total)

            # Take the answer-token gradients (last n_take rows)
            g_answer = bundle.g_accum[-n_take:]
            gradients.append(g_answer)
            labels.extend(bundle.token_labels[-n_take:])

        if not gradients:
            logger.warning("L-1A: no valid bundles to ingest")
            return

        self._gradient_matrix = np.concatenate(gradients, axis=0)
        self._label_array = np.array(labels, dtype=np.int64)
        self._bundles = bundles

        logger.info(
            "L-1A: ingested %d gradient vectors, shape %s, labels distribution: %s",
            self._gradient_matrix.shape[0],
            self._gradient_matrix.shape,
            dict(zip(*np.unique(self._label_array, return_counts=True))),
        )

    def _infer_batch_size(self, bundle: AttackDataBundle) -> int:
        """Infer batch size from the bundle metadata."""
        batch_meta = bundle.batch_meta
        if "batch_size" in batch_meta:
            return int(batch_meta["batch_size"])
        if bundle.token_labels is not None:
            return len(bundle.token_labels)
        return 4  # fallback

    # ------------------------------------------------------------------------- #
    #  Static analysis (offline, no GPU needed)
    # ------------------------------------------------------------------------- #

    def run(self) -> List[AttackVerdict]:
        """Run all L-1A statistical tests and return verdicts."""
        if self._gradient_matrix is None or len(self._gradient_matrix) < self.n_classes:
            logger.warning(
                "L-1A: need at least %d samples, got %s",
                self.n_classes,
                getattr(self, "_gradient_matrix", None),
            )
            return [self._make_verdict("INCONCLUSIVE", "INSUFFICIENT_SAMPLES",
                                       "Not enough gradient samples collected")]

        verdicts = []
        verdicts.append(self._test_kmeans_ari())
        verdicts.append(self._test_1nn_agreement())
        verdicts.append(self._test_gradient_auc())
        verdicts.append(self._test_gradient_magnitude())
        verdicts.append(self._test_permutation_overall())

        self._verdicts = verdicts
        self._save_results()
        return verdicts

    # ------------------------------------------------------------------------- #
    #  Test 1: K-Means ARI
    # ------------------------------------------------------------------------- #

    def _test_kmeans_ari(self) -> AttackVerdict:
        """K-Means ARI: do gradient clusters align with label classes?"""
        from sklearn.cluster import KMeans
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

        G = self._gradient_matrix
        L = self._label_array
        n = len(G)

        # Subsample if too many samples (K-Means is O(n^2) for large n)
        max_samples = 2000
        if n > max_samples:
            idx = np.random.default_rng(self.seed).choice(n, max_samples, replace=False)
            G, L = G[idx], L[idx]
            n = max_samples

        kmeans = KMeans(
            n_clusters=self.n_classes,
            random_state=self.seed,
            n_init=10,
            algorithm="lloyd",
        )
        pred_labels = kmeans.fit_predict(G)

        ari = adjusted_rand_score(L, pred_labels)
        nmi = normalized_mutual_info_score(L, pred_labels)

        # ARI ≈ 0 under masking (chance level)
        # ARI >> 0 indicates label leakage
        chance = 0.0
        verdict = "PRIVACY_PRESERVED" if abs(ari) < 0.1 else "LEAK_DETECTED"

        return AttackVerdict(
            attack_id=self.ATTACK_ID,
            sub_attack="KMeans_ARI",
            metric="Adjusted Rand Index",
            value=float(ari),
            chance_level=chance,
            n_samples=n,
            verdict=verdict,
            notes=(
                f"NMI={nmi:.4f}. "
                f"ARI≈0 means gradients are randomly distributed w.r.t. labels. "
                f"ARI>>0 means gradient space encodes label identity."
            ),
        )

    # ------------------------------------------------------------------------- #
    #  Test 2: 1-NN Label Agreement Rate
    # ------------------------------------------------------------------------- #

    def _test_1nn_agreement(self) -> AttackVerdict:
        """1-NN agreement: do nearest-neighbour gradients share the same label?"""
        G = self._gradient_matrix
        L = self._label_array
        n = len(G)

        max_samples = 3000
        if n > max_samples:
            idx = np.random.default_rng(self.seed).choice(n, max_samples, replace=False)
            G, L = G[idx], L[idx]

        agreement_rate, std_err, _ = label_agreement_rate(G, L, k=1)
        chance = 1.0 / self.n_classes

        verdict = (
            "PRIVACY_PRESERVED"
            if abs(agreement_rate - chance) < 2 * std_err
            else "LEAK_DETECTED"
        )

        return AttackVerdict(
            attack_id=self.ATTACK_ID,
            sub_attack="1NN_Agreement",
            metric="1-NN Label Agreement Rate",
            value=float(agreement_rate),
            chance_level=float(chance),
            std_err=float(std_err),
            n_samples=n,
            verdict=verdict,
            notes=(
                f"Agreement={agreement_rate:.4f} vs chance={chance:.4f}. "
                f"Significantly above chance → gradient space encodes labels."
            ),
        )

    # ------------------------------------------------------------------------- #
    #  Test 3: Gradient Similarity AUC
    # ------------------------------------------------------------------------- #

    def _test_gradient_auc(self) -> AttackVerdict:
        """Gradient Similarity AUC: cosine(grad_i, grad_j) predicts same-label."""
        G = self._gradient_matrix
        L = self._label_array
        n = len(G)

        max_pairs = 5000
        if n * n > max_pairs * 2:
            rng = np.random.default_rng(self.seed)
            pair_idx = rng.choice(n, (max_pairs, 2), replace=True)
        else:
            pair_idx = np.array([(i, j) for i in range(n) for j in range(n) if i != j])

        same_label = np.array([1 if L[i] == L[j] else 0 for i, j in pair_idx])
        sim_matrix = cosine_similarity_matrix(G)
        similarities = np.array([sim_matrix[i, j] for i, j in pair_idx])

        auc, auc_std = self._compute_auc_from_scores(similarities, same_label)

        # AUC ≈ 0.5 under masking; AUC > 0.5 + 2*std indicates leakage
        chance = 0.5
        verdict = (
            "PRIVACY_PRESERVED"
            if abs(auc - 0.5) < 2 * auc_std
            else "LEAK_DETECTED"
        )

        return AttackVerdict(
            attack_id=self.ATTACK_ID,
            sub_attack="Gradient_Sim_AUC",
            metric="AUC (gradient cosine sim → same-label)",
            value=float(auc),
            chance_level=chance,
            std_err=float(auc_std),
            n_samples=len(similarities),
            verdict=verdict,
            notes=(
                f"AUC={auc:.4f} (chance=0.5). "
                f"AUC significantly > 0.5 means gradient similarity "
                f"predicts whether two samples share the same label."
            ),
        )

    def _compute_auc_from_scores(
        self, scores: np.ndarray, binary_labels: np.ndarray
    ) -> tuple:
        """Mann-Whitney U-based AUC computation."""
        n_pos = int(np.sum(binary_labels))
        n_neg = int(len(binary_labels) - n_pos)
        if n_pos == 0 or n_neg == 0:
            return 0.5, 0.0

        order = np.argsort(scores)[::-1]
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, len(scores) + 1)

        auc = (np.sum(binary_labels * ranks) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
        auc = max(0.0, min(1.0, auc))

        # Hanley-McNeil SE
        q1 = auc / (2 - auc) if auc < 1 else 0
        q2 = 2 * auc * auc / (1 + auc) if auc > 0 else 0
        se = np.sqrt(
            (auc * (1 - auc) +
             (n_pos - 1) * (q1 - auc ** 2) +
             (n_neg - 1) * (q2 - auc ** 2)) / (n_pos * n_neg)
        )
        return float(auc), float(se)

    # ------------------------------------------------------------------------- #
    #  Test 4: Gradient Magnitude ANOVA
    # ------------------------------------------------------------------------- #

    def _test_gradient_magnitude(self) -> AttackVerdict:
        """ANOVA: does gradient L2-norm vary systematically by class?"""
        G = self._gradient_matrix
        L = self._label_array

        norms = np.linalg.norm(G, axis=1)
        classes = np.unique(L)
        groups = [norms[L == c] for c in classes]
        groups = [g for g in groups if len(g) > 1]

        if len(groups) < 2:
            return AttackVerdict(
                attack_id=self.ATTACK_ID,
                sub_attack="Grad_Magnitude_ANOVA",
                metric="F-statistic",
                value=0.0,
                verdict="INCONCLUSIVE",
                notes="Not enough per-class samples for ANOVA",
            )

        try:
            from scipy import stats
            f_stat, p_val = stats.f_oneway(*groups)
        except Exception:
            return AttackVerdict(
                attack_id=self.ATTACK_ID,
                sub_attack="Grad_Magnitude_ANOVA",
                metric="F-statistic",
                value=0.0,
                verdict="INCONCLUSIVE",
                notes="scipy.stats.f_oneway unavailable",
            )

        verdict = (
            "PRIVACY_PRESERVED"
            if p_val > self.alpha
            else "LEAK_DETECTED"
        )

        return AttackVerdict(
            attack_id=self.ATTACK_ID,
            sub_attack="Grad_Magnitude_ANOVA",
            metric="ANOVA F-statistic",
            value=float(f_stat),
            chance_level=0.0,
            p_value=float(p_val),
            n_samples=len(G),
            verdict=verdict,
            notes=(
                f"F={f_stat:.4f}, p={p_val:.4f}. "
                f"Significant difference across classes → label info in gradient magnitude."
            ),
        )

    # ------------------------------------------------------------------------- #
    #  Test 5: Permutation test on 1-NN agreement
    # ------------------------------------------------------------------------- #

    def _test_permutation_overall(self) -> AttackVerdict:
        """Permutation test: is 1-NN agreement rate significantly above chance?"""
        G = self._gradient_matrix
        L = self._label_array
        n = len(G)

        max_samples = 1000
        if n > max_samples:
            idx = np.random.default_rng(self.seed).choice(n, max_samples, replace=False)
            G, L = G[idx], L[idx]

        def metric_fn(g, lbl):
            rate, _, _ = label_agreement_rate(g, lbl, k=1)
            return rate

        observed, p_value, _ = permutation_test(
            metric_fn, G, L,
            n_permutations=self.n_permutations,
            seed=self.seed,
        )

        chance = 1.0 / self.n_classes
        verdict = (
            "PRIVACY_PRESERVED"
            if p_value > self.alpha
            else "LEAK_DETECTED"
        )

        return AttackVerdict(
            attack_id=self.ATTACK_ID,
            sub_attack="Permutation_Test",
            metric="1-NN Agreement (permutation p-value)",
            value=observed,
            chance_level=chance,
            p_value=float(p_value),
            n_samples=len(G),
            verdict=verdict,
            notes=(
                f"Observed agreement={observed:.4f}, chance={chance:.4f}, "
                f"p={p_value:.4f} (n_perm={self.n_permutations}). "
                f"p < alpha={self.alpha} → label structure in gradient space."
            ),
        )

    # ------------------------------------------------------------------------- #
    #  Persistence
    # ------------------------------------------------------------------------- #

    def _save_results(self) -> None:
        """Save gradient matrix and label array for offline reanalysis."""
        if self._gradient_matrix is None:
            return

        out = self.output_dir / "l1a"
        out.mkdir(exist_ok=True)

        np.save(out / "gradient_matrix.npy", self._gradient_matrix)
        np.save(out / "label_array.npy", self._label_array)

        meta = {
            "n_samples": len(self._gradient_matrix),
            "hidden_dim": int(self._gradient_matrix.shape[1]),
            "n_classes": self.n_classes,
            "n_permutations": self.n_permutations,
            "alpha": self.alpha,
            "seed": self.seed,
            "chance_level": 1.0 / self.n_classes,
        }
        import json
        with open(out / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        logger.info("L-1A results saved to %s", out)

    # ------------------------------------------------------------------------- #
    #  Convenience
    # ------------------------------------------------------------------------- #

    def get_gradient_summary(self) -> Dict[str, Any]:
        if self._gradient_matrix is None:
            return {}
        return summarise_gradients(self._gradient_matrix)

    def plot_gradient_pca(self, save_path: Optional[str] = None) -> None:
        """2-D PCA projection of gradient space (optional, requires matplotlib)."""
        try:
            import matplotlib.pyplot as plt
            from sklearn.decomposition import PCA
        except ImportError:
            logger.warning("matplotlib or sklearn not available; skipping PCA plot")
            return

        G = self._gradient_matrix
        L = self._label_array

        pca = PCA(n_components=2, random_state=self.seed)
        G_2d = pca.fit_transform(G)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        colors = ["#e74c3c", "#2ecc71", "#3498db", "#9b59b6", "#f39c12", "#1abc9c"]
        label_names = ["DESC", "ENTY", "ABBR", "HUM", "NUM", "LOC"]

        ax = axes[0]
        for c in range(self.n_classes):
            mask = L == c
            ax.scatter(G_2d[mask, 0], G_2d[mask, 1], c=colors[c % len(colors)],
                       label=label_names[c], alpha=0.6, s=20)
        ax.set_title("L-1A: Gradient PCA by True Label")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        ax.legend(fontsize=8)

        ax = axes[1]
        ax.scatter(G_2d[:, 0], G_2d[:, 1], c=L, cmap="tab10", alpha=0.6, s=20)
        ax.set_title("L-1A: Gradient PCA Colored by Label Index")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("PCA plot saved to %s", save_path)
        else:
            plt.show()
        plt.close()
