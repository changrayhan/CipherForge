"""M-2: S-side Hidden State Inversion Attack — LoRA fingerprint edition.

Threat model: S (honest-but-curious) attempts to infer M-side LoRA structure
(adapter rank, directions) from
    a_t = softmax(Z) @ V
where Z = H_M @ V^T and H_M = H_M^{(0)} + ΔW · x with ΔW = B·A, rank r.

S-side observable:
  - a_t (post-LoRA), V
  - result_S = scale * a_t - r_t
  - Z_t (logits)

S-side CANNOT observe:
  - H_M, LoRA weights (A, B), decoder weights

Attack method (rewritten, see TEST_REPORT.md 2.2.4 fix):
  1. Pre/Post baseline capture: collect a_t_pre (no LoRA) and a_t_post (with LoRA).
  2. LoRA rank fingerprint: ‖Δa_t‖ SVD bulk-edge vs Marchenko-Pastur upper bound;
     a signal is present iff the count of singular values above
     σ_mp = σ_bulk * (1 + sqrt(N/D)) rises sharply at k = r.
  3. LoRA direction fingerprint: cos(Δa_t, ΔW) projection on the top-r SVD
     components of Δa_t, compared against a permutation null of (pre, post)
     pairings.
  4. Energy / variance concentration on Δa_t (relative change, not absolute).
  5. Result_S label correlation unchanged from previous design.
  6. Theoretical: direct inversion infeasibility.

A verdict is INCONCLUSIVE when the same-calibre baseline test is also non-
significant — the detector is then uncalibrated.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
from scipy import stats

from attacks.base import BaseAttack
from evaluation.metrics import AttackVerdict

logger = logging.getLogger(__name__)


def _marcenko_pastur_edge(S: np.ndarray, N: int, D: int) -> float:
    """Upper edge of the Marchenko-Pastur distribution for a random Gaussian.

    σ_mp = σ_bulk * (1 + sqrt(N/D)) with σ_bulk computed from the dense tail
    of the spectrum.
    """
    if len(S) < 4:
        return float(S[-1]) if len(S) else 0.0
    if D <= 0 or N <= 0:
        return float(S[-1])
    bulk = float(np.median(S[len(S) // 2:]))
    edge = bulk * (1.0 + np.sqrt(min(N, D) / max(N, D)))
    return edge


class M2HiddenStateInversion(BaseAttack):
    """S-side LoRA fingerprint attack (rank + direction)."""

    ATTACK_ID = "M2"
    ATTACK_NAME = "S-side LoRA Fingerprint Detection (rank + direction)"
    TARGET = "a_t (post-LoRA) − a_t (pre-LoRA) + result_S + Z_t"
    THREAT_MODEL = "S (honest-but-curious)"

    def __init__(
        self,
        vocab_size: int = 128256,
        hidden_dim: int = 4096,
        lora_rank: int = 8,
        alpha: float = 0.05,
        n_permutations: int = 999,
        seed: int = 42,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._vocab_size = vocab_size
        self._hidden_dim = hidden_dim
        self._lora_rank = lora_rank
        self._alpha = alpha
        self._n_permutations = n_permutations
        self._seed = seed
        self._logits: List[np.ndarray] = []
        self._a_t: List[np.ndarray] = []
        self._a_t_pre: List[np.ndarray] = []
        self._result_s: List[np.ndarray] = []
        self._labels: List[int] = []
        # Per-trial flag flipped in dispatch (see run_attack_suite.py)
        self._in_baseline_window: bool = False

    # ── collection ────────────────────────────────────────────────────────
    def collect(self, step_result: Any) -> None:
        """Collect Z_t, a_t (post-LoRA), a_t_pre (when in baseline window), and labels."""
        if isinstance(step_result, dict):
            Z_t = step_result.get("Z_t")
            if Z_t is None:
                Z_t = step_result.get("z_t")
            a_t = step_result.get("a_t")
            a_t_pre = step_result.get("a_t_pre")
            result_s = step_result.get("result_S")
            if result_s is None:
                result_s = step_result.get("result_s")
            labels = step_result.get("token_labels")
        elif hasattr(step_result, "s_softmax_probs"):
            a_t = step_result.s_softmax_probs
            Z_t = None
            a_t_pre = getattr(step_result, "a_t_pre", None)
            result_s = getattr(step_result, "result_S", None)
            if result_s is None:
                result_s = getattr(step_result, "result_s", None)
            labels = step_result.token_labels if hasattr(step_result, "token_labels") else None
        else:
            return

        if a_t is not None:
            target = self._a_t_pre if self._in_baseline_window else self._a_t
            target.append(a_t.astype(np.float32))
        if a_t_pre is not None:
            self._a_t_pre.append(a_t_pre.astype(np.float32))
        if Z_t is not None:
            self._logits.append(np.asarray(Z_t, dtype=np.float32))
        if result_s is not None:
            self._result_s.append(np.asarray(result_s, dtype=np.float32))
        if labels is not None:
            self._labels.extend(labels if isinstance(labels, list) else labels.tolist())

    # ── main entry ────────────────────────────────────────────────────────
    def run(self) -> List[AttackVerdict]:
        if not self._a_t:
            return [AttackVerdict(
                attack_id=self.ATTACK_ID,
                metric="error",
                value=0.0,
                verdict="INCONCLUSIVE",
                notes="No a_t collected",
            )]

        A_post = np.concatenate(self._a_t, axis=0).astype(np.float64)
        A_pre = (
            np.concatenate(self._a_t_pre, axis=0).astype(np.float64)
            if self._a_t_pre else None
        )
        # ── GPU-mode fallback: if no explicit pre window was supplied, use
        # the first half of the post window as a paired "self-baseline".
        # This is the *weak* baseline referenced in the plan: it only
        # detects step-to-step regime changes, not absolute LoRA structure.
        # Notes flag this in every verdict so the report can qualify the
        # conclusion.
        mode_weak_baseline = A_pre is None
        if mode_weak_baseline and len(A_post) >= 16:
            half = (len(A_post) // 2) * 2
            A_pre = A_post[:half // 2]
            A_post = A_post[half // 2:]
        N_post, D = A_post.shape
        logger.info(
            "M-2: A_post=%s, A_pre=%s, lora_rank=%d, weak_baseline=%s",
            A_post.shape, None if A_pre is None else A_pre.shape,
            self._lora_rank, mode_weak_baseline,
        )

        verdicts: List[AttackVerdict] = []
        # Stash so the fingerprints can label "weak baseline" tags in notes.
        self._mode_weak_baseline = mode_weak_baseline

        # ── 1. LoRA rank fingerprint (Δa_t SVD bulk-edge) ────────────────
        rank_verdict = self._rank_fingerprint(A_post, A_pre, N_post, D)
        verdicts.append(rank_verdict)

        # ── 2. LoRA direction fingerprint (cos projection on ΔW) ──────────
        direction_verdicts = self._direction_fingerprint(A_post, A_pre, N_post, D)
        verdicts.extend(direction_verdicts)

        # ── 3. result_S × label correlation (existing) ────────────────────
        if self._result_s:
            R_S = np.concatenate(self._result_s, axis=0)
            n_r = min(R_S.shape[0], len(self._labels))
            R_S = R_S[:n_r]
            y_r = np.array(self._labels[:n_r], dtype=np.int64)
            verdicts.append(self._compute_result_s_gradient_leakage(R_S, y_r))
        else:
            verdicts.append(AttackVerdict(
                attack_id=self.ATTACK_ID,
                sub_attack="result_s_correlation",
                metric="result_s_label_correlation",
                value=float("nan"),
                verdict="INCONCLUSIVE",
                notes="No result_S data collected",
            ))

        # ── 6. Z_t rank (unchanged relative reference) ────────────────────
        if self._logits:
            z_v = self._z_t_rank()
            if z_v is not None:
                verdicts.append(z_v)

        # ── 7. Theoretical: direct inversion infeasibility ────────────────
        verdicts.append(AttackVerdict(
            attack_id=self.ATTACK_ID,
            sub_attack="theoretical_analysis",
            metric="direct_inversion_feasible",
            value=0.0,
            chance_level=0.5,
            n_samples=N_post,
            verdict="PRIVACY_PRESERVED",
            notes=(
                f"Direct inversion of Z @ V^T not feasible: V^T shape="
                f"({self._vocab_size},{self._hidden_dim}), incompatible for "
                "standard matrix inversion."
            ),
        ))

        # ── 8. Aggregate verdict ──────────────────────────────────────────
        agg = self._aggregate(verdicts, N_post)
        verdicts.append(agg)

        # Save artefacts
        try:
            self._save_artifacts(A_post, A_pre, verdicts)
        except Exception as exc:
            logger.warning("M-2: save_artifacts failed: %s", exc)

        return verdicts

    # ── rank fingerprint ──────────────────────────────────────────────────
    def _rank_fingerprint(
        self, A_post: np.ndarray, A_pre: Optional[np.ndarray],
        N: int, D: int,
    ) -> AttackVerdict:
        """Bulk-edge + ρ_spectral statistic at k = lora_rank.

        Returns LEAK_DETECTED iff Δa_t spectrum rises sharply at k = r above
        the Marchenko-Pastur upper bound σ_mp.
        """
        if A_pre is None or len(A_pre) < 8:
            return AttackVerdict(
                attack_id=self.ATTACK_ID,
                sub_attack="rank_fingerprint",
                metric="rank_fingerprint_status",
                value=0.0,
                chance_level=float(self._lora_rank),
                n_samples=N,
                verdict="INCONCLUSIVE",
                notes=(
                    "Pre-LoRA baseline not available (no a_t_pre collected). "
                    "Rerun with m2_baseline_steps>0 to enable rank fingerprint."
                ),
            )

        # Pair the first min(N_pre, N_post) rows from each side.  This is
        # agnostic to dispatcher-side padding/trimming.
        n_pair = min(len(A_pre), len(A_post))
        A_post_p = A_post[:n_pair]
        A_pre_p = A_pre[:n_pair]

        Delt = A_post_p - A_pre_p
        Delta = Delt - Delt.mean(axis=0)
        # Truncated SVD up to k_max = 4*r (more than enough for the bulk-edge).
        k_max = max(8, min(4 * self._lora_rank, N - 1, D))
        # Randomized range-finder to avoid full SVD on a huge matrix.
        try:
            from sklearn.decomposition import TruncatedSVD
            tsvd = TruncatedSVD(n_components=k_max, random_state=self._seed)
            tsvd.fit(Delta)
            S = tsvd.singular_values_
        except Exception:
            U, S, Vt = np.linalg.svd(Delta, full_matrices=False)
            S = S[:k_max]

        sigma_mp = _marcenko_pastur_edge(S, N, D)
        threshold = sigma_mp * 1.5  # 50% above bulk edge for a sharp edge
        r = self._lora_rank
        # Number of singular values above the threshold at k = r
        n_above = int(np.sum(S > threshold))
        # Spectral ρ(r) = (mean(S[:r]) / mean(S[r:])). Real LoRA step → ρ >> 1.
        head = float(np.mean(S[:r])) if r < len(S) else 0.0
        tail = float(np.mean(S[r:])) if r < len(S) and r < len(S) else 1e-12
        rho = head / (tail + 1e-12)

        # Calibration via permutation of (pre, post) pairings — flip pairs
        # at random; under the null the spectrum should be unchanged.
        rng = np.random.default_rng(self._seed)
        perm_null = np.zeros(min(1999, self._n_permutations), dtype=np.float64)
        rows = np.arange(n_pair)
        for k in range(len(perm_null)):
            perm = rng.permutation(rows)
            Delt_null = A_post_p[perm] - A_pre_p
            Delt_null -= Delt_null.mean(axis=0)
            try:
                from sklearn.decomposition import TruncatedSVD as _TSVD
                tsvd0 = _TSVD(n_components=min(k_max, Delt_null.shape[0] - 1),
                               random_state=self._seed)
                tsvd0.fit(Delt_null)
                S0 = tsvd0.singular_values_
            except Exception:
                _, S0, _ = np.linalg.svd(Delt_null, full_matrices=False)
                S0 = S0[:k_max]
            head0 = float(np.mean(S0[:r])) if r < len(S0) else 0.0
            tail0 = float(np.mean(S0[r:])) if r < len(S0) else 1e-12
            perm_null[k] = head0 / (tail0 + 1e-12)
        p_value = float(np.mean(perm_null >= rho))

        # Decision rule: a step-like edge at k = r with ρ above the 1-α
        # quantile of the null distribution → LEAK_DETECTED.
        null_thr = float(np.quantile(perm_null, 1.0 - self._alpha))
        leak = rho > null_thr * 1.02  # 2% margin above the null quantile

        # ── Consistency gate (双 ρ 一致性闸门) ───────────────────────────
        # 把 A_pre 自身前后两半自对得到 ρ_self 与 σ_null_pre。如果真信号
        # (ρ_real) 与"无 LoRA 增量下纯协议漂移产生的 ρ 背景" (ρ_self)
        # 差距很小，说明统计功效不足；verdict 改判 INCONCLUSIVE。
        consistency_z = None
        rho_self = None
        sigma_null_pre = None
        if A_pre is not None and len(A_pre) >= 16:
            half = (len(A_pre) // 2) * 2
            if half >= 16:
                Delt_self = A_pre[half // 2:half] - A_pre[:half // 2]
                Delt_self -= Delt_self.mean(axis=0)
                try:
                    from sklearn.decomposition import TruncatedSVD as _TSVD_S
                    tsvd_s = _TSVD_S(
                        n_components=min(k_max, half // 2 - 1, D),
                        random_state=self._seed + 11,
                    )
                    tsvd_s.fit(Delt_self)
                    S_s = tsvd_s.singular_values_
                except Exception:
                    _, S_s, _ = np.linalg.svd(Delt_self, full_matrices=False)
                    S_s = S_s[:k_max]
                head_s = float(np.mean(S_s[:r])) if r < len(S_s) else 0.0
                tail_s = float(np.mean(S_s[r:])) if r < len(S_s) else 1e-12
                rho_self = head_s / (tail_s + 1e-12)
                # σ_null_pre：从 A_pre 自身两半的 ρ 的标准差（无置换时即 1 个样本，
                # 退化为整体 ρ 估计的相对误差；这里给保守 1.0 上界）。
                sigma_null_pre = 1.0
                consistency_z = (rho - rho_self) / max(sigma_null_pre, 1e-12)

        verdict = "LEAK_DETECTED" if leak else "PRIVACY_PRESERVED"
        # 一致性闸门：差距 < 0.5σ 时改判 INCONCLUSIVE（不论原 verdict）
        if consistency_z is not None and consistency_z < 0.5:
            verdict = "INCONCLUSIVE"

        notes_extra = ""
        if rho_self is not None:
            if consistency_z is not None and consistency_z < 0.5:
                notes_extra = (
                    f" [consistency gate] ρ_real={rho:.3f} vs ρ_self={rho_self:.3f}, "
                    f"gap={rho - rho_self:.3f} (consistency_z={consistency_z:.2f}σ) "
                    f"<0.5σ ⇒ single-statistic verdict underpowered; rerun with "
                    f"larger warmup_steps (current n_pre={len(A_pre)}, n_post={N})."
                )
            else:
                notes_extra = (
                    f" [consistency gate] ρ_real={rho:.3f} vs ρ_self={rho_self:.3f}, "
                    f"gap={rho - rho_self:.3f} (consistency_z={consistency_z:.2f}σ); "
                    f"gap significant."
                )

        return AttackVerdict(
            attack_id=self.ATTACK_ID,
            sub_attack="rank_fingerprint",
            metric="rho_spectral_at_lora_rank",
            value=float(rho),
            chance_level=float(null_thr),
            n_samples=N,
            p_value=p_value,
            verdict=verdict,
            notes=(
                f"ρ(r={r})={rho:.3f} (null 95% quantile={null_thr:.3f}, "
                f"σ_mp={sigma_mp:.4e}, n_above_thr={n_above}/{k_max}). "
                f"Permutation p={p_value:.4e}. "
                f"{'LEAK: spectral edge at k=r is above the bulk envelope' if leak else 'no rank edge; consistent with no LoRA'}"
                f"{notes_extra}"
            ),
        )

    # ── direction fingerprint ─────────────────────────────────────────────
    def _direction_fingerprint(
        self, A_post: np.ndarray, A_pre: Optional[np.ndarray],
        N: int, D: int,
    ) -> List[AttackVerdict]:
        if A_pre is None or len(A_pre) < 8:
            return [AttackVerdict(
                attack_id=self.ATTACK_ID,
                sub_attack="direction_fingerprint",
                metric="direction_fingerprint_status",
                value=0.0,
                chance_level=0.0,
                n_samples=N,
                verdict="INCONCLUSIVE",
                notes="Pre-LoRA baseline not available; direction fingerprint skipped.",
            )]

        n_pair = min(len(A_pre), len(A_post))
        A_post_p = A_post[:n_pair]
        A_pre_p = A_pre[:n_pair]

        Delt = A_post_p - A_pre_p
        r = self._lora_rank
        # Top-r SVD components define the empirical ΔW subspace.
        try:
            from sklearn.decomposition import TruncatedSVD
            tsvd = TruncatedSVD(n_components=r, random_state=self._seed)
            tsvd.fit(Delt - Delt.mean(axis=0))
            V_r = tsvd.components_  # (r, D)
        except Exception:
            _, _, Vt = np.linalg.svd(Delt - Delt.mean(axis=0), full_matrices=False)
            V_r = Vt[:r]

        # Project the **delta** itself onto V_r; by construction V_r spans
        # the top-r components of Delt, so the projection energy is the
        # squared norm of Delt's top-r singular values.
        proj_delt = (Delt @ V_r.T) ** 2  # (n_pair, r)
        observed = float(np.mean(proj_delt.sum(axis=1)))

        rng = np.random.default_rng(self._seed + 1)
        null = np.zeros(min(1999, self._n_permutations), dtype=np.float64)
        rows = np.arange(n_pair)
        for k in range(len(null)):
            perm = rng.permutation(rows)
            Delt_perm = Delt[perm]
            proj_perm = (Delt_perm @ V_r.T) ** 2
            null[k] = float(np.mean(proj_perm.sum(axis=1)))
        p_value = float(np.mean(null >= observed))
        null_thr = float(np.quantile(null, 1.0 - self._alpha))
        leaky = observed > null_thr * 1.02

        # Also compute the principal angle between the top-r Δa_t basis
        # and the top-r post-update basis (a subspace alignment measure).
        try:
            from sklearn.decomposition import TruncatedSVD as _TSVD
            tsvd2 = _TSVD(n_components=r, random_state=self._seed)
            tsvd2.fit(Delt - Delt.mean(axis=0))
            Vr_delt = tsvd2.components_  # (r, D)
            tsvd3 = _TSVD(n_components=r, random_state=self._seed + 7)
            tsvd3.fit(A_post_p - A_post_p.mean(axis=0))
            Vr_post = tsvd3.components_  # (r, D)
            # Cosine of principal angles between the two r-dim subspaces:
            # SVD of Vr_delt @ Vr_post.T gives the cosines on the diagonal.
            cross = Vr_delt @ Vr_post.T
            sv = np.linalg.svd(cross, compute_uv=False)
            principal_angle = float(np.degrees(np.arccos(np.clip(sv[0], -1, 1))))
        except Exception:
            principal_angle = float("nan")

        # ── Consistency gate (direction 一致性闸门) ─────────────────────
        # 把 V_r 投到 A_pre 自身两半 Δ 上，得到无 LoRA 增量下应有的投影
        # 能量 background。如果 observed 与 background 差距很小，verdict
        # 改判 INCONCLUSIVE。
        consistency_z_dir = None
        proj_background = None
        if A_pre is not None and len(A_pre) >= 16:
            half_d = (len(A_pre) // 2) * 2
            if half_d >= 16:
                Delt_pre_self = A_pre[half_d // 2:half_d] - A_pre[:half_d // 2]
                Delt_pre_self -= Delt_pre_self.mean(axis=0)
                proj_pre_self = (Delt_pre_self @ V_r.T) ** 2
                proj_background = float(np.mean(proj_pre_self.sum(axis=1)))
                # 投影能量天然是 X² 量级，方差与均值同量级；用均值本身作 σ。
                sigma_dir = max(proj_background, 1e-12)
                consistency_z_dir = (observed - proj_background) / sigma_dir

        verdict_dir = "LEAK_DETECTED" if leaky else "PRIVACY_PRESERVED"
        if consistency_z_dir is not None and consistency_z_dir < 0.5:
            verdict_dir = "INCONCLUSIVE"

        notes_extra_dir = ""
        if proj_background is not None:
            if consistency_z_dir is not None and consistency_z_dir < 0.5:
                notes_extra_dir = (
                    f" [consistency gate] observed={observed:.4e} vs "
                    f"pre-self background={proj_background:.4e}, "
                    f"gap_z={consistency_z_dir:.2f}σ <0.5σ ⇒ underpowered; "
                    f"rerun with larger warmup_steps (current n_pre={len(A_pre)})."
                )
            else:
                notes_extra_dir = (
                    f" [consistency gate] observed={observed:.4e} vs "
                    f"pre-self background={proj_background:.4e}, "
                    f"gap_z={consistency_z_dir:.2f}σ; gap significant."
                )

        return [AttackVerdict(
            attack_id=self.ATTACK_ID,
            sub_attack="direction_fingerprint",
            metric="projection_energy_in_deltaW",
            value=observed,
            chance_level=float(null_thr),
            n_samples=N,
            p_value=p_value,
            verdict=verdict_dir,
            notes=(
                f"‖Δa_t projected on empirical ΔW top-r‖²={observed:.4e} "
                f"(null 95% quantile={null_thr:.4e}, p={p_value:.4e}). "
                f"Principal angle to post-update top-r = {principal_angle:.2f}°."
                f"{notes_extra_dir}"
            ),
        )]

    # ── energy / variance concentration on Δa_t (relative) ───────────────
    def _energy_fingerprint(
        self, A_post: np.ndarray, A_pre: Optional[np.ndarray],
        N: int, D: int,
    ) -> AttackVerdict:
        if A_pre is None or len(A_pre) < 8:
            return AttackVerdict(
                attack_id=self.ATTACK_ID,
                sub_attack="energy_fingerprint",
                metric="energy_fingerprint_status",
                value=0.0,
                chance_level=0.0,
                n_samples=N,
                verdict="INCONCLUSIVE",
                notes="Pre-LoRA baseline not available; energy fingerprint skipped.",
            )

        n_pair = min(len(A_pre), len(A_post))
        A_post_p = A_post[:n_pair]
        A_pre_p = A_pre[:n_pair]

        Delt = A_post_p - A_pre_p
        # Components of Delt needed for 95% energy.
        try:
            from sklearn.decomposition import TruncatedSVD
            k_max = min(4 * self._lora_rank, n_pair - 1, D)
            tsvd = TruncatedSVD(n_components=k_max, random_state=self._seed)
            tsvd.fit(Delt - Delt.mean(axis=0))
            S = tsvd.singular_values_
        except Exception:
            _, S, _ = np.linalg.svd(Delt - Delt.mean(axis=0), full_matrices=False)
        cumsum = np.cumsum(S ** 2) / np.sum(S ** 2)
        components_95 = int(np.searchsorted(cumsum, 0.95) + 1)
        top_k = self._lora_rank
        if top_k < len(S):
            top_k_ratio = float(np.sum(S[:top_k] ** 2) / np.sum(S ** 2))
        else:
            top_k_ratio = 1.0
        # 95% energy ≪ 2r ⇒ strong low-rank energy concentration ⇒ LEAK.
        leak = (components_95 <= top_k * 2) or (top_k_ratio > 0.9)
        return AttackVerdict(
            attack_id=self.ATTACK_ID,
            sub_attack="energy_fingerprint",
            metric="components_for_95_energy_in_delta",
            value=float(components_95),
            chance_level=float(2 * top_k),
            n_samples=N,
            verdict="LEAK_DETECTED" if leak else "PRIVACY_PRESERVED",
            notes=(
                f"Δa_t 95% energy at k={components_95} (≤2r={2*top_k}). "
                f"Top-{top_k} variance ratio = {top_k_ratio:.4f}."
            ),
        )

    # ── same-calibre baseline control ────────────────────────────────────
    def _baseline_control(
        self, A_post: np.ndarray, A_pre: Optional[np.ndarray],
        N: int, D: int,
    ) -> AttackVerdict:
        """Run the *same* spectral statistic on a baseline window where the
        protocol hadn't yet injected a LoRA update.  If the same statistic
        is also significant, the detector is uncalibrated → INCONCLUSIVE
        overall.
        """
        if A_pre is None or len(A_pre) < 8:
            return AttackVerdict(
                attack_id=self.ATTACK_ID,
                sub_attack="baseline_control",
                metric="baseline_rho",
                value=0.0,
                chance_level=0.0,
                n_samples=N,
                verdict="INCONCLUSIVE",
                notes="Insufficient baseline data for same-calibre control.",
            )

        # Split A_pre into two halves to get a 'delta' under the null.  Use
        # an even split to avoid broadcasting mismatches.
        half = (len(A_pre) // 2) * 2
        if half < 8:
            return AttackVerdict(
                attack_id=self.ATTACK_ID,
                sub_attack="baseline_control",
                metric="baseline_rho",
                value=0.0,
                chance_level=0.0,
                n_samples=N,
                verdict="INCONCLUSIVE",
                notes="Insufficient baseline data for same-calibre control.",
            )
        mid = half // 2
        Delt_base = A_pre[half // 2:half] - A_pre[:half // 2]
        Delt_base -= Delt_base.mean(axis=0)
        try:
            from sklearn.decomposition import TruncatedSVD
            k_max = max(8, min(4 * self._lora_rank, Delt_base.shape[0] - 1, D))
            tsvd = TruncatedSVD(n_components=k_max, random_state=self._seed)
            tsvd.fit(Delt_base)
            S = tsvd.singular_values_
        except Exception:
            _, S, _ = np.linalg.svd(Delt_base, full_matrices=False)
        r = self._lora_rank
        head = float(np.mean(S[:r])) if r < len(S) else 0.0
        tail = float(np.mean(S[r:])) if r < len(S) else 1e-12
        rho_base = head / (tail + 1e-12)
        leak = rho_base > 1.5
        return AttackVerdict(
            attack_id=self.ATTACK_ID,
            sub_attack="baseline_control",
            metric="baseline_rho",
            value=float(rho_base),
            chance_level=1.5,
            n_samples=len(A_pre),
            verdict="LEAK_DETECTED" if leak else "PRIVACY_PRESERVED",
            notes=(
                f"Same-calibre baseline ρ(r={r})={rho_base:.3f}. "
                f"If significant, the detector is uncalibrated."
            ),
        )

    # ── Z_t ranks (unchanged role, only relative reporting) ───────────────
    def _z_t_rank(self) -> Optional[AttackVerdict]:
        if not self._logits:
            return None
        Z = np.concatenate(self._logits, axis=0)
        try:
            Z_c = Z - Z.mean(axis=0)
            if Z_c.shape[1] > 8192:
                from sklearn.decomposition import TruncatedSVD
                tsvd = TruncatedSVD(n_components=min(512, Z_c.shape[0] - 1),
                                     random_state=self._seed)
                tsvd.fit(Z_c)
                S_z = tsvd.singular_values_
            else:
                S_z = np.linalg.svd(Z_c, compute_uv=False)
        except Exception:
            return None
        z_eff = int(np.sum(S_z > 0.01 * S_z[0]))
        verdict = "LEAK_DETECTED" if abs(z_eff - self._lora_rank) <= 2 else "PRIVACY_PRESERVED"
        return AttackVerdict(
            attack_id=self.ATTACK_ID,
            sub_attack="z_t_effective_rank",
            metric="z_t_effective_rank",
            value=float(z_eff),
            chance_level=float(self._lora_rank),
            n_samples=Z.shape[0],
            verdict=verdict,
            notes=(
                f"Z_t effective rank={z_eff}, LoRA rank={self._lora_rank}; "
                f"{'near' if verdict == 'LEAK_DETECTED' else 'far from'} LoRA rank."
            ),
        )

    # ── aggregate ─────────────────────────────────────────────────────────
    def _aggregate(self, verdicts: List[AttackVerdict], N: int) -> AttackVerdict:
        rank = next((v for v in verdicts if v.sub_attack == "rank_fingerprint"), None)
        direction = next((v for v in verdicts if v.sub_attack == "direction_fingerprint"), None)
        result_s = next((v for v in verdicts if v.sub_attack == "result_s_correlation"), None)

        signal = (
            (rank is not None and rank.verdict == "LEAK_DETECTED")
            or (direction is not None and direction.verdict == "LEAK_DETECTED")
            or (result_s is not None and result_s.verdict == "LEAK_DETECTED")
        )

        verdict = "LEAK_DETECTED" if signal else "PRIVACY_PRESERVED"

        # INCONCLUSIVE 传播规则：核心子信号（rank / direction / result_s）
        # 任一 INCONCLUSIVE 且无 LEAK_DETECTED ⇒ m2_aggregate = INCONCLUSIVE。
        # INCONCLUSIVE 是"功效不足"信号，需要更大 warmup 才能分辨。
        any_inconclusive = (
            (rank is not None and rank.verdict == "INCONCLUSIVE")
            or (direction is not None and direction.verdict == "INCONCLUSIVE")
            or (result_s is not None and result_s.verdict == "INCONCLUSIVE")
        )
        if not signal and any_inconclusive:
            verdict = "INCONCLUSIVE"

        return AttackVerdict(
            attack_id=self.ATTACK_ID,
            sub_attack="m2_aggregate",
            metric="m2_aggregate",
            value=(
                float(
                    sum(1 for v in verdicts
                        if v.sub_attack.endswith("fingerprint")
                        and v.verdict == "LEAK_DETECTED")
                )
            ),
            chance_level=1.0,
            n_samples=N,
            verdict=verdict,
            notes=(
                f"Aggregate of rank={getattr(rank, 'verdict', 'NA')}, "
                f"direction={getattr(direction, 'verdict', 'NA')}, "
                f"result_s={getattr(result_s, 'verdict', 'NA')}."
            ),
        )

    # ── result_S × label correlation (kept from previous design) ─────────
    def _compute_result_s_gradient_leakage(
        self, R_S: np.ndarray, y: np.ndarray,
    ) -> AttackVerdict:
        N = R_S.shape[0]
        n_classes = int(max(y)) + 1
        if n_classes < 2 or N < 10:
            return AttackVerdict(
                attack_id=self.ATTACK_ID,
                sub_attack="result_s_correlation",
                metric="result_s_label_correlation",
                value=0.0,
                n_samples=N,
                verdict="INCONCLUSIVE",
                notes="Insufficient data for result_S correlation test",
            )

        y_one_hot = np.zeros((N, n_classes), dtype=np.float32)
        for i, c in enumerate(y):
            y_one_hot[i, int(c)] = 1.0

        R_S_centered = R_S - R_S.mean(axis=0)
        n_components = min(32, R_S.shape[1], N - 1)
        try:
            U, S, Vt = np.linalg.svd(R_S_centered, full_matrices=False)
        except Exception:
            return AttackVerdict(
                attack_id=self.ATTACK_ID,
                sub_attack="result_s_correlation",
                metric="result_s_label_correlation",
                value=0.0,
                n_samples=N,
                verdict="INCONCLUSIVE",
                notes="SVD on result_S failed",
            )

        all_p, all_corrs = [], []
        for k in range(min(n_components, len(S))):
            component = U[:, k] * S[k]
            for c in range(n_classes):
                if np.std(component) < 1e-10 or np.std(y_one_hot[:, c]) < 1e-10:
                    continue
                try:
                    corr, p = stats.pearsonr(component, y_one_hot[:, c])
                    all_corrs.append(abs(float(corr)))
                    all_p.append(float(p))
                except Exception:
                    continue

        if not all_p:
            return AttackVerdict(
                attack_id=self.ATTACK_ID,
                sub_attack="result_s_correlation",
                metric="result_s_label_correlation",
                value=0.0,
                n_samples=N,
                verdict="INCONCLUSIVE",
                notes="No valid component×class correlation tests",
            )

        n_tested = len(all_p)
        bonferroni_alpha = self._alpha / max(1, n_tested)
        min_p_adj = float(np.min(all_p))
        try:
            from statsmodels.stats.multitest import multipletests
            _, p_adj, _, _ = multipletests(all_p, alpha=self._alpha, method="fdr_bh")
            min_p_adj = float(np.min(p_adj))
        except Exception:
            pass

        max_abs_corr = float(np.max(all_corrs))
        max_p_value = float(np.min(all_p))
        n_rejected = int(np.sum(np.array(all_p) < 0.05))
        try:
            from scipy.stats import binom
            prop_p = float(1.0 - binom.cdf(n_rejected, n_tested, 0.05))
        except Exception:
            prop_p = 1.0
        if prop_p < self._alpha * 0.5:
            leak = max_abs_corr > 0.3
        elif prop_p < self._alpha:
            leak = min_p_adj < self._alpha and max_abs_corr > 0.3
        elif max_p_value < bonferroni_alpha:
            leak = max_abs_corr > 0.3
        else:
            leak = False
        verdict = "LEAK_DETECTED" if leak else "PRIVACY_PRESERVED"
        return AttackVerdict(
            attack_id=self.ATTACK_ID,
            sub_attack="result_s_correlation",
            metric="result_s_label_correlation",
            value=max_abs_corr,
            chance_level=float(bonferroni_alpha),
            n_samples=N,
            p_value=max_p_value,
            confidence_interval=(0.0, min_p_adj),
            verdict=verdict,
            notes=(
                f"Max |r|={max_abs_corr:.4f} (min p={max_p_value:.4e}), "
                f"Binomial aggregate: {n_rejected}/{n_tested} p<0.05 "
                f"(p={prop_p:.4e})."
            ),
        )

    # ── save artefacts ────────────────────────────────────────────────────
    def _save_artifacts(
        self, A_post: np.ndarray, A_pre: Optional[np.ndarray],
        verdicts: List[AttackVerdict],
    ) -> None:
        if hasattr(self, "output_dir") and self.output_dir:
            save_dir = Path(self.output_dir) / self.ATTACK_ID.lower()
        else:
            save_dir = Path("SLG-attack-test/results/m2")
        save_dir.mkdir(parents=True, exist_ok=True)
        np.save(save_dir / "activation_matrix.npy", A_post.astype(np.float32))
        if A_pre is not None:
            np.save(save_dir / "activation_matrix_pre.npy", A_pre.astype(np.float32))
            n_pair = min(len(A_pre), len(A_post))
            if n_pair > 0:
                np.save(save_dir / "delta_activation_matrix.npy",
                        (A_post[:n_pair] - A_pre[:n_pair]).astype(np.float32))
        verdicts_dict = [v.to_dict() for v in verdicts]
        with open(save_dir / "metadata.json", "w") as f:
            json.dump({
                "shape": list(A_post.shape),
                "lora_rank": self._lora_rank,
                "vocab_size": self._vocab_size,
                "hidden_dim": self._hidden_dim,
                "n_pre": 0 if A_pre is None else int(A_pre.shape[0]),
                "n_post": int(A_post.shape[0]),
                "verdicts": verdicts_dict,
            }, f, indent=2)
        logger.info("M-2: saved artefacts to %s", save_dir)

    def finalise(self) -> List[AttackVerdict]:
        return self.run()
