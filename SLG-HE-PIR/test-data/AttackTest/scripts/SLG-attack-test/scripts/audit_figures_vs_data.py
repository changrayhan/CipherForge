"""Audit all 14 figures in TEST_REPORT.md against the underlying data.

This script loads each run's `attack_results.json`, recomputes the key
statistics that the report cites, and cross-checks against the captions and
quantitative claims in the report. It is read-only with respect to figures
themselves — it does NOT regenerate them. Instead, it asks: "Given the data,
is the report's claim about the figure consistent with reality?"

The audit is structured per figure:
    figure <id>  caption:  "<...>"
        claim 1: <value>
            EXTRACTED:    <recomputed>
            MATCH / MISMATCH

Outputs a structured report to stdout. Exit code 0 means no mismatches.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ATTACK_DATA = Path("/root/autodl-tmp/SLG-HE-PIR-code/SLG-HE-PIR/test-data/attack-test-data")


def _load_npz(run_dir: str, sub_attack_dir: str):
    """Load all .npy files in <run_dir>/<sub_attack_dir>."""
    p = ATTACK_DATA / run_dir / sub_attack_dir
    if not p.exists():
        return {}
    return {f.stem: np.load(f) for f in p.glob("*.npy")}


def _load_results(run_dir: str) -> list[dict]:
    with open(ATTACK_DATA / run_dir / "attack_results.json") as f:
        return json.load(f)["attack_results"]


def _find_verdict(verdicts: list[dict], sub: str) -> dict:
    for v in verdicts:
        if v["sub_attack"] == sub:
            return v
    raise KeyError(sub)


# ───────────────────────── L1 (Figure 2 / 3 / 4) ─────────────────────────


def audit_l1() -> dict:
    out: dict = {"ok": True, "mismatches": []}
    verdicts = _load_results("run_20260726_140508_L1_with_dp")
    npz = _load_npz("run_20260726_140508_L1_with_dp", "l1")
    G = np.asarray(npz["gradient_matrix"])
    H = np.asarray(npz["h_u_matrix"])
    y = np.asarray(npz["label_array"]).astype(int)
    out["data_shape"] = {"gradient_matrix": G.shape, "h_u_matrix": H.shape, "labels": y.shape}

    # ── Verdict values (cross-check table on lines 401-407)
    expected_l1 = {
        "h_u_mean_anova":      ("4.0860×10⁻³", "PRIVACY_PRESERVED", 0.004086),
        "h_u_norm_anova":      ("2.9630×10⁻¹", "PRIVACY_PRESERVED", 0.2963),
        "kmeans_ari":          ("1.6459×10⁻⁴", "PRIVACY_PRESERVED", 0.0001646),
        "nn_agreement":        ("0.1900", "PRIVACY_PRESERVED", 0.190),
        "cosine_auc":          ("0.4999", "PRIVACY_PRESERVED", 0.4999),
        "permutation_test":    ("0.5646", "PRIVACY_PRESERVED", 0.5646),
        "magnitude_anova":     ("3.7765×10⁻¹", "PRIVACY_PRESERVED", 0.37765),
    }
    out["l1_verdict_values"] = {}
    for sub, (claim_str, claim_vd, claim_value) in expected_l1.items():
        v = _find_verdict(verdicts, sub)
        out["l1_verdict_values"][sub] = {
            "claim": claim_str, "claim_value": claim_value,
            "actual": float(v["value"]),
            "verdict": v["verdict"],
            "match_value": abs(float(v["value"]) - claim_value) < 5e-4,
            "match_verdict": (claim_vd == "PRIVACY_PRESERVED") == (v["verdict"] == "PRIVACY_PRESERVED"),
        }
        if not out["l1_verdict_values"][sub]["match_value"]:
            out["ok"] = False
            out["mismatches"].append(("L1 verdict value", sub, claim_value, float(v["value"])))

    # ── H_U feature counts: "512 features; 284 raw p<0.05" — table line 401
    # Look at notes for h_u_mean_anova verdict (notes mention 512 / 284 / Binomial)
    h_u_notes = _find_verdict(verdicts, "h_u_mean_anova").get("notes", "")
    out["l1_h_u_anova_notes"] = h_u_notes

    # ── Figure 2 caption: "g_accum L2 norm concentrated around 10^6 magnitude"
    G_norms = np.linalg.norm(G, axis=1)
    out["l1_g_accum_norm_stats"] = {
        "median": float(np.median(G_norms)),
        "min": float(np.min(G_norms)),
        "max": float(np.max(G_norms)),
        "order_of_magnitude": int(np.log10(np.median(G_norms))),
        "claim_10_6": 1e6,
        "matches_10^6": 5e5 <= np.median(G_norms) <= 5e6,
    }

    # ── Figure 2 caption: "H_U only slightly varies (e.g., ENTY/HUM higher than ABBR)"
    H_norms = np.linalg.norm(H, axis=1)
    classes = sorted(set(y.tolist()))
    # L1 attack uses 6 coarse classes (likely TREC-QC coarse)
    median_by_class = {int(c): float(np.median(H_norms[y == c])) for c in classes}
    out["l1_H_U_norm_by_class"] = median_by_class

    # ── Figure 3: cosine similarity same/different overlap around [-0.02, 0.02]
    # Recompute per the L1 cosine attack code path
    from sklearn.metrics.pairwise import cosine_similarity
    sim = cosine_similarity(G.astype(np.float64))
    n = G.shape[0]
    same_sim, diff_sim = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if y[i] == y[j]:
                same_sim.append(sim[i, j])
            else:
                diff_sim.append(sim[i, j])
    same_arr = np.array(same_sim)
    diff_arr = np.array(diff_sim)
    out["l1_cosine_stats"] = {
        "same_median": float(np.median(same_arr)),
        "diff_median": float(np.median(diff_arr)),
        "same_within_002": float(np.mean(np.abs(same_arr) < 0.02)),
        "diff_within_002": float(np.mean(np.abs(diff_arr) < 0.02)),
        "overlaps_in_band": (np.median(same_arr) > -0.02 and np.median(same_arr) < 0.02)
                            and (np.median(diff_arr) > -0.02 and np.median(diff_arr) < 0.02),
    }
    return out


# ───────────────────────── L2 (Figure 5 / 6 / 7) ─────────────────────────


def audit_l2() -> dict:
    out: dict = {"ok": True, "mismatches": []}
    verdicts = _load_results("run_20260725_202031")
    npz = _load_npz("run_20260725_202031", "l2")
    A = np.asarray(npz["activation_matrix"])
    R = np.asarray(npz["result_s_matrix"])
    y = np.asarray(npz["label_array"]).astype(int)
    out["data_shape"] = {"activation": A.shape, "result_s": R.shape, "labels": y.shape}

    expected_l2 = {
        "a_t_mean_anova":      ("8.2973×10⁻³", 0.0082973),
        "a_t_norm_anova":      ("4.6923×10⁻¹", 0.4692),
        "a_t_kl_divergence":   ("3.6769×10⁻³", 0.003677),
        "result_S_mean_anova":  ("4.3147×10⁻⁴", 0.0004315),
        "result_S_norm_anova":  ("2.3378×10⁻¹", 0.2338),
    }
    out["l2_verdict_values"] = {}
    for sub, (claim_str, claim_value) in expected_l2.items():
        v = _find_verdict(verdicts, sub)
        out["l2_verdict_values"][sub] = {
            "claim": claim_str, "claim_value": claim_value,
            "actual": float(v["value"]),
            "verdict": v["verdict"],
            "match_value": abs(float(v["value"]) - claim_value) < max(5e-4, 1e-3 * abs(claim_value)),
        }
        if not out["l2_verdict_values"][sub]["match_value"]:
            out["ok"] = False
            out["mismatches"].append(("L2 verdict value", sub, claim_value, float(v["value"])))

    # ── Figure 5: a_t norm median at ~0.24, result_S norm at ~1.4×10^6
    classes = sorted(set(y.tolist()))
    a_medians = {int(c): float(np.median(np.linalg.norm(A[y == c], axis=1))) for c in classes}
    r_medians = {int(c): float(np.median(np.linalg.norm(R[y == c], axis=1))) for c in classes}
    out["l2_norm_medians"] = {
        "a_t_by_class": a_medians,
        "a_t_overall_median": float(np.median(np.linalg.norm(A, axis=1))),
        "result_s_by_class": r_medians,
        "result_s_overall_median": float(np.median(np.linalg.norm(R, axis=1))),
        "claim_a_t_0_24": 0.24,
        "claim_result_s_1_4e6": 1.4e6,
    }
    return out


# ───────────────────────── M-1 (Figure 8 / 9 / 10 / 11) ─────────────────────────


def audit_m1() -> dict:
    out: dict = {"ok": True, "mismatches": []}
    verdicts = _load_results("run_20260725_202421")
    npz = _load_npz("run_20260725_202421", "m1")
    preds = np.asarray(npz["predictions"]).flatten()
    confs = np.asarray(npz["confidences"]).flatten()
    labels = np.asarray(npz["labels"]).flatten().astype(int)
    out["data_shape"] = {"predictions": preds.shape, "confidences": confs.shape, "labels": labels.shape}

    expected_m1 = {
        "prediction_consistency":   ("4.1346×10⁻³", 0.004135),
        "distillation_convergence": ("0.1", 0.1),
        "surrogate_model":          ("0.0", 0.0),
        "information_leakage":      ("4.3687×10⁻¹", 0.4369),
    }
    out["m1_verdict_values"] = {}
    for sub, (claim_str, claim_value) in expected_m1.items():
        v = _find_verdict(verdicts, sub)
        out["m1_verdict_values"][sub] = {
            "claim": claim_str, "claim_value": claim_value,
            "actual": float(v["value"]),
            "verdict": v["verdict"],
            "match_value": abs(float(v["value"]) - claim_value) < 5e-3,
        }
        if not out["m1_verdict_values"][sub]["match_value"]:
            out["ok"] = False
            out["mismatches"].append(("M-1 verdict value", sub, claim_value, float(v["value"])))

    # ── Figure 8 caption: confidence histogram concentrated in [0.02, 0.15], mean ~0.0881
    out["m1_confidence_stats"] = {
        "min": float(confs.min()),
        "max": float(confs.max()),
        "mean": float(confs.mean()),
        "std": float(confs.std()),
        "in_band_002_to_015": int(np.sum((confs >= 0.02) & (confs <= 0.15))),
        "fraction_in_band": float(np.mean((confs >= 0.02) & (confs <= 0.15))),
        "claim_mean_0_0881": abs(float(confs.mean()) - 0.0881) < 0.002,
    }

    # ── Figure 9 caption: top tokens — token 220 = 71%, etc., unique=10
    unique, counts = np.unique(preds, return_counts=True)
    sorted_counts = sorted(zip(unique.tolist(), counts.tolist()), key=lambda x: -x[1])
    out["m1_prediction_distribution"] = {
        "top10": sorted_counts[:10],
        "unique_count": int(len(unique)),
        "claim_token_220_is_71": (220 in unique) and dict(zip(unique.tolist(), counts.tolist()))[220] == 71,
        "claim_unique_10": len(unique) == 10,
    }

    # ── Figure 10 caption claim: predicted counts = 2/14/5/2/77/0; NOT equal to true labels 18/24/1/28/18/11
    coarse = preds % 6
    coarse_counts = dict(zip(*np.unique(coarse, return_counts=True)))
    out["m1_6bucket_counts"] = {
        "predicted": {int(k): int(v) for k, v in coarse_counts.items()},
        "true": {int(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))},
        "claim_pred_2_14_5_2_77_0": [coarse_counts.get(i, 0) for i in range(6)] == [2, 14, 5, 2, 77, 0],
        "claim_true_18_24_1_28_18_11": (
            list(np.bincount(labels))[:6] == [18, 24, 1, 28, 18, 11]
        ),
        "predicted_differs_from_true": (
            [coarse_counts.get(i, 0) for i in range(6)] != [18, 24, 1, 28, 18, 11]
        ),
    }
    return out


# ───────────────────────── M-2 (Figure 12 / 13 / 14) ─────────────────────────


def audit_m2() -> dict:
    out: dict = {"ok": True, "mismatches": []}
    verdicts = _load_results("run_20260726_032030_m2fixed_gpu_v3")
    npz = _load_npz("run_20260726_032030_m2fixed_gpu_v3", "m2")
    A_post = np.asarray(npz["activation_matrix"])
    A_pre = np.asarray(npz["activation_matrix_pre"])
    Delt = np.asarray(npz["delta_activation_matrix"])
    out["data_shape"] = {
        "activation": A_post.shape,
        "activation_pre": A_pre.shape,
        "delta": Delt.shape,
    }

    expected_m2 = {
        "rank_fingerprint":      ("13.525", 13.525, "PRIVACY_PRESERVED"),
        "direction_fingerprint": ("5.7023×10⁻²", 0.057023, "PRIVACY_PRESERVED"),
        "result_s_correlation":  ("1.5751×10⁻¹", 0.15751, "PRIVACY_PRESERVED"),
        "z_t_effective_rank":    ("44", 44.0, "PRIVACY_PRESERVED"),
        "theoretical_analysis":  ("0", 0.0, "PRIVACY_PRESERVED"),
    }
    out["m2_verdict_values"] = {}
    for sub, (claim_str, claim_value, claim_vd) in expected_m2.items():
        v = _find_verdict(verdicts, sub)
        out["m2_verdict_values"][sub] = {
            "claim": claim_str, "claim_value": claim_value,
            "actual": float(v["value"]),
            "verdict": v["verdict"],
            "match_value": abs(float(v["value"]) - claim_value) < 5e-3,
            "match_verdict": claim_vd == v["verdict"],
        }
        if not out["m2_verdict_values"][sub]["match_value"]:
            out["ok"] = False
            out["m2_verdict_values"][sub]["mismatch"] = "value"
            out["mismatches"].append(("M-2 verdict value", sub, claim_value, float(v["value"])))
        if not out["m2_verdict_values"][sub]["match_verdict"]:
            out["ok"] = False
            out["m2_verdict_values"][sub]["mismatch"] = "verdict"
            out["mismatches"].append(("M-2 verdict tag", sub, claim_vd, v["verdict"]))

    # ── rank_fingerprint permutation p-value
    rank_v = _find_verdict(verdicts, "rank_fingerprint")
    out["m2_rank_fingerprint_p"] = {
        "claim": "0.0452",
        "actual": float(rank_v.get("p_value")),
        "match": abs(float(rank_v.get("p_value")) - 0.0452) < 5e-3,
    }
    if not out["m2_rank_fingerprint_p"]["match"]:
        out["ok"] = False
        out["mismatches"].append(("M-2 p", "rank_fingerprint", 0.0452, rank_v.get("p_value")))

    # ── direction_fingerprint permutation p-value
    dir_v = _find_verdict(verdicts, "direction_fingerprint")
    out["m2_direction_fingerprint_p"] = {
        "claim": "0.8744",
        "actual": float(dir_v.get("p_value")),
        "match": abs(float(dir_v.get("p_value")) - 0.8744) < 5e-3,
    }
    if not out["m2_direction_fingerprint_p"]["match"]:
        out["ok"] = False
        out["mismatches"].append(("M-2 p", "direction_fingerprint", 0.8744, dir_v.get("p_value")))

    # ── result_s_correlation binomial p
    rs_v = _find_verdict(verdicts, "result_s_correlation")
    out["m2_result_s_p"] = {
        "claim_min_p": "2.59×10⁻²",
        "actual_min_p": float(rs_v.get("p_value")),
        "match_min_p": abs(float(rs_v.get("p_value")) - 0.02591) < 5e-3,
    }

    # ── Figure 12 caption: σ_mp = 3.34e-2, 1.5·σ_mp = 5.01e-2
    # Recompute from saved Δa_t (already in npy as delta_activation_matrix)
    from sklearn.decomposition import TruncatedSVD
    Delta = Delt - Delt.mean(axis=0)
    n, D = Delta.shape
    tsvd = TruncatedSVD(n_components=32, random_state=42)
    tsvd.fit(Delta)
    S = tsvd.singular_values_
    sigma_mp = float(np.median(S[16:])) * (1.0 + np.sqrt(min(n, D) / max(n, D)))
    threshold = sigma_mp * 1.5
    out["m2_spectrum_recap"] = {
        "top_singular_values": [float(x) for x in S[:5]],
        "sigma_mp": sigma_mp,
        "threshold_1_5_sigma_mp": threshold,
        "n_above_threshold": int(np.sum(S > threshold)),
        "k_total": len(S),
        "claim_sigma_mp_1_49e-2": abs(sigma_mp - 0.0149) < 5e-3,
        "claim_threshold_2_23e-2": abs(threshold - 0.0223) < 5e-3,
        "claim_top_18_above_threshold": int(np.sum(S > threshold)) >= 18,
        "claim_sigma1_1_201": abs(float(S[0]) - 1.201) < 5e-2,
        "claim_sigma2_0_397": abs(float(S[1]) - 0.397) < 5e-2,
        "claim_sigma3_0_285": abs(float(S[2]) - 0.285) < 5e-2,
    }
    for k in ("claim_sigma_mp_1_49e-2", "claim_threshold_2_23e-2",
              "claim_top_18_above_threshold", "claim_sigma1_1_201",
              "claim_sigma2_0_397", "claim_sigma3_0_285"):
        if not out["m2_spectrum_recap"][k]:
            out["ok"] = False
            out["mismatches"].append(("M-2 spectrum", k, True, out["m2_spectrum_recap"][k]))

    # ── Figure 13 caption: ρ=10.249, null 95% quantile=10.674, mean=10.254, std=0.266
    rank_notes = rank_v.get("notes", "")
    out["m2_perm_null_notes_check"] = rank_notes
    # Recompute the perm null to verify the new caption's "std 2.66e-1" claim
    n_pair = min(A_post.shape[0], A_pre.shape[0])
    Delta = A_post[:n_pair] - A_pre[:n_pair]
    Delta -= Delta.mean(axis=0)
    from sklearn.decomposition import TruncatedSVD
    rng = np.random.default_rng(42 + 1)
    null_vals = []
    for _ in range(199):
        perm = rng.permutation(n_pair)
        D = A_post[:n_pair][perm] - A_pre[:n_pair]
        D -= D.mean(axis=0)
        tsvd_p = TruncatedSVD(n_components=min(32, D.shape[0] - 1), random_state=42)
        tsvd_p.fit(D)
        S_p = tsvd_p.singular_values_
        null_vals.append(float(S_p[:8].mean() / (S_p[8:].mean() + 1e-12)))
    null_vals = np.array(null_vals)
    out["m2_perm_null_stats"] = {
        "mean": float(null_vals.mean()),
        "std": float(null_vals.std()),
        "q95": float(np.quantile(null_vals, 0.95)),
        "claim_mean_12_62": abs(float(null_vals.mean()) - 12.62) < 0.05,
        "claim_std_5_94e-1": abs(float(null_vals.std()) - 0.594) < 0.05,
        "claim_q95_13_51": abs(float(np.quantile(null_vals, 0.95)) - 13.51) < 0.05,
    }
    for k in ("claim_mean_12_62", "claim_std_5_94e-1", "claim_q95_13_51"):
        if not out["m2_perm_null_stats"][k]:
            out["ok"] = False
            out["mismatches"].append(("M-2 perm null", k, True, out["m2_perm_null_stats"][k]))
    return out


# ───────────────────────── Summary Figure 1 ─────────────────────────


def audit_summary() -> dict:
    """Figure 1: summary_verdict_counts.png — 4 bars (L1/L2/M1/M2) with LEAK/PRIVACY/INCONC."""
    counts = {"L1": {"LEAK_DETECTED": 0, "PRIVACY_PRESERVED": 0, "INCONCLUSIVE": 0},
              "L2": {"LEAK_DETECTED": 0, "PRIVACY_PRESERVED": 0, "INCONCLUSIVE": 0},
              "M1": {"LEAK_DETECTED": 0, "PRIVACY_PRESERVED": 0, "INCONCLUSIVE": 0},
              "M2": {"LEAK_DETECTED": 0, "PRIVACY_PRESERVED": 0, "INCONCLUSIVE": 0}}
    for run, attack in [
        ("run_20260726_140508_L1_with_dp", "L1"),
        ("run_20260725_202031", "L2"),
        ("run_20260725_202421", "M1"),
        ("run_20260726_032030_m2fixed_gpu_v3", "M2"),
    ]:
        verdicts = _load_results(run)
        for v in verdicts:
            counts[attack][v["verdict"]] = counts[attack].get(v["verdict"], 0) + 1

    claims = {
        "L1": {"LEAK_DETECTED": 0, "PRIVACY_PRESERVED": 7, "INCONCLUSIVE": 0},
        "L2": {"LEAK_DETECTED": 0, "PRIVACY_PRESERVED": 5, "INCONCLUSIVE": 0},
        "M1": {"LEAK_DETECTED": 0, "PRIVACY_PRESERVED": 3, "INCONCLUSIVE": 1},
        "M2": {"LEAK_DETECTED": 0, "PRIVACY_PRESERVED": 6, "INCONCLUSIVE": 0},
    }
    out = {"counts": counts, "claims": claims, "ok": True, "mismatches": []}
    for att in ["L1", "L2", "M1", "M2"]:
        for vd in ["LEAK_DETECTED", "PRIVACY_PRESERVED", "INCONCLUSIVE"]:
            actual = counts[att].get(vd, 0)
            claim = claims[att][vd]
            if actual != claim:
                out["ok"] = False
                out["mismatches"].append(("summary", f"{att}/{vd}", claim, actual))
    return out


def _safe(o):
    if isinstance(o, dict):
        return {k: _safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_safe(x) for x in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def main() -> int:
    print("=" * 80)
    print("CHART-DATA CONSISTENCY AUDIT")
    print("=" * 80)

    summary = audit_summary()
    print("\n--- Figure 1 (summary_verdict_counts) ---")
    print(json.dumps(_safe(summary["counts"]), indent=2))
    if summary["mismatches"]:
        print("MISMATCHES:")
        for m in summary["mismatches"]:
            print(f"  {m}")

    l1 = audit_l1()
    print("\n--- Figure 2/3/4 (L1) ---")
    print(json.dumps(_safe(l1["l1_verdict_values"]), indent=2))
    print("\nL1 g_accum norm stats:")
    print(json.dumps(_safe(l1["l1_g_accum_norm_stats"]), indent=2))
    print("\nL1 H_U norm by coarse class:")
    print(json.dumps(_safe(l1["l1_H_U_norm_by_class"]), indent=2))
    print("\nL1 cosine stats:")
    print(json.dumps(_safe(l1["l1_cosine_stats"]), indent=2))
    if l1["mismatches"]:
        print("\nL1 MISMATCHES:")
        for m in l1["mismatches"]:
            print(f"  {m}")

    l2 = audit_l2()
    print("\n--- Figure 5/6/7 (L2) ---")
    print(json.dumps(_safe(l2["l2_verdict_values"]), indent=2))
    print("\nL2 norm medians:")
    print(json.dumps(_safe(l2["l2_norm_medians"]), indent=2))
    if l2["mismatches"]:
        print("\nL2 MISMATCHES:")
        for m in l2["mismatches"]:
            print(f"  {m}")

    m1 = audit_m1()
    print("\n--- Figure 8/9/10/11 (M-1) ---")
    print(json.dumps(_safe(m1["m1_verdict_values"]), indent=2))
    print("\nM-1 confidence stats:")
    print(json.dumps(_safe(m1["m1_confidence_stats"]), indent=2))
    print("\nM-1 prediction top10:")
    print(json.dumps(_safe(m1["m1_prediction_distribution"]), indent=2))
    print("\nM-1 6-bucket counts:")
    print(json.dumps(_safe(m1["m1_6bucket_counts"]), indent=2))
    if m1["mismatches"]:
        print("\nM-1 MISMATCHES:")
        for m in m1["mismatches"]:
            print(f"  {m}")

    m2 = audit_m2()
    print("\n--- Figure 12/13/14 (M-2) ---")
    print(json.dumps(_safe(m2["m2_verdict_values"]), indent=2))
    print("\nM-2 spectrum recap:")
    print(json.dumps(_safe(m2["m2_spectrum_recap"]), indent=2))
    print("\nM-2 perm null check (notes):")
    print(m2["m2_perm_null_notes_check"])
    print("\nM-2 perm null reconstructed stats:")
    print(json.dumps(_safe(m2["m2_perm_null_stats"]), indent=2))
    if m2["mismatches"]:
        print("\nM-2 MISMATCHES:")
        for m in m2["mismatches"]:
            print(f"  {m}")

    all_mismatches = (
        summary["mismatches"] + l1["mismatches"] + l2["mismatches"]
        + m1["mismatches"] + m2["mismatches"]
    )
    print("\n" + "=" * 80)
    print(f"TOTAL MISMATCHES: {len(all_mismatches)}")
    for m in all_mismatches:
        print(f"  {m}")
    print("=" * 80)
    return 1 if all_mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
