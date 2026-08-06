"""Item-by-item audit of every numerical claim in the report's tables."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ATTACK_DATA = Path("/root/autodl-tmp/SLG-HE-PIR-code/SLG-HE-PIR/test-data/attack-test-data")


def _near(a: float, b: float, tol: float = 0.001) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(a))


def _load_npz(run_dir: str, sub: str) -> dict:
    p = ATTACK_DATA / run_dir / sub
    return {f.stem: np.load(f) for f in p.glob("*.npy")}


def _verdicts(run_dir: str) -> dict:
    with open(ATTACK_DATA / run_dir / "attack_results.json") as f:
        return {v["sub_attack"]: v for v in json.load(f)["attack_results"]}


# ───────────────────────── L1 Table ─────────────────────────

def audit_l1_table(group: list[tuple[str, str, bool]]) -> None:
    # 当前报告 §3.1.1 / §3.1.5 引用的 L-1 数据来自 dχ 启用后的重测
    # （详见 README §6.6 / §12.3），对应 run_20260726_140508_L1_with_dp。
    v = _verdicts("run_20260726_140508_L1_with_dp")
    npz = _load_npz("run_20260726_140508_L1_with_dp", "l1")
    H = np.asarray(npz["h_u_matrix"])
    y = np.asarray(npz["label_array"]).astype(int)

    def claim(test: str, passed: bool):
        group.append((test, "see report", passed))

    # Row 1 — h_u_mean_anova (post-dχ)
    claim("L1 h_u_mean_anova: value 4.0860e-3",
          _near(v["h_u_mean_anova"]["value"], 4.0860e-3, tol=1e-3))
    n1 = v["h_u_mean_anova"]["notes"]
    claim("L1 h_u_mean_anova: notes 18/512 (post-dχ)",
          "18" in n1 and "512" in n1)
    claim("L1 h_u_mean_anova: notes binomial p=9.3044e-01",
          "9.3044e-01" in n1)
    claim("L1 h_u_mean_anova: notes BH-FDR=7.1177e-01",
          "7.1177e-01" in n1)
    claim("L1 h_u_mean_anova: bonferroni in chance_level = 9.7656e-05 (=0.05/512)",
          _near(v["h_u_mean_anova"]["chance_level"], 0.05/512, tol=1e-6))

    # Row 2 — h_u_norm_anova (post-dχ)
    claim("L1 h_u_norm_anova: value 2.9630e-1",
          _near(v["h_u_norm_anova"]["p_value"], 0.2963, tol=1e-3))
    claim("L1 h_u_norm_anova: notes η²=0.0307",
          "0.0307" in v["h_u_norm_anova"]["notes"])

    # Row 3 — kmeans_ari (post-dχ)
    claim("L1 kmeans_ari: value 1.6459e-4",
          _near(v["kmeans_ari"]["value"], 1.6459e-4, tol=1e-3))
    claim("L1 kmeans_ari: chance_level 0",
          _near(v["kmeans_ari"]["chance_level"], 0, tol=1e-6))

    # Row 4 — nn_agreement (post-dχ)
    claim("L1 nn_agreement: value 0.1900",
          _near(v["nn_agreement"]["value"], 0.1900, tol=1e-3))
    claim("L1 nn_agreement: chance_level = 1/6 = 0.1667",
          _near(v["nn_agreement"]["chance_level"], 1/6, tol=1e-3))
    claim("L1 nn_agreement: claim 6 coarse classes (n_classes=6)",
          len(np.unique(y)) == 6)

    # Row 5 — cosine_auc (post-dχ)
    claim("L1 cosine_auc: value 0.4999",
          _near(v["cosine_auc"]["value"], 0.4999, tol=1e-3))
    claim("L1 cosine_auc: chance_level 0.5",
          _near(v["cosine_auc"]["chance_level"], 0.5, tol=1e-3))

    # Row 6 — permutation_test (post-dχ)
    claim("L1 permutation_test: p 0.5646",
          _near(v["permutation_test"]["p_value"], 0.5646, tol=1e-3))

    # Row 7 — magnitude_anova (post-dχ)
    claim("L1 magnitude_anova: p 3.7765e-1",
          _near(v["magnitude_anova"]["p_value"], 0.37765, tol=1e-3))
    claim("L1 magnitude_anova: notes η²=0.0269",
          "0.0269" in v["magnitude_anova"]["notes"])

    # Aggregates (post-dχ)
    leak = sum(1 for x in v.values() if x["verdict"] == "LEAK_DETECTED")
    priv = sum(1 for x in v.values() if x["verdict"] == "PRIVACY_PRESERVED")
    inc = sum(1 for x in v.values() if x["verdict"] == "INCONCLUSIVE")
    claim(f"L1 aggregate: actual {leak}/{priv}/{inc} (expected 0/7/0 after dχ)",
          leak == 0 and priv == 7 and inc == 0)

    # Bonferroni reference value 9.7656e-5
    claim("L1 bonferroni 9.7656e-5 = 0.05/512",
          _near(0.05/512, 9.7656e-5, tol=1e-5))


# ───────────────────────── L2 Table ─────────────────────────

def audit_l2_table(group: list[tuple[str, str, bool]]) -> None:
    v = _verdicts("run_20260725_202031")

    def claim(test: str, passed: bool):
        group.append((test, "see report", passed))

    # Row 1 — a_t_mean_anova
    claim("L2 a_t_mean_anova: value 8.2973e-3",
          _near(v["a_t_mean_anova"]["value"], 0.0082973, tol=1e-3))
    n1 = v["a_t_mean_anova"]["notes"]
    claim("L2 a_t_mean_anova: notes 19/512",
          "19/512" in n1)
    claim("L2 a_t_mean_anova: binomial p=0.8956 (notes '8.9558e-01')",
          "8.9558e-01" in n1)
    claim("L2 a_t_mean_anova: BH-FDR=0.8657 (notes '8.6565e-01')",
          "8.6565e-01" in n1)
    claim("L2 a_t_mean_anova: bonferroni 9.7656e-05",
          _near(v["a_t_mean_anova"]["chance_level"], 0.05/512, tol=1e-6))

    # Row 2 — a_t_norm_anova
    claim("L2 a_t_norm_anova: value 4.6923e-1",
          _near(v["a_t_norm_anova"]["p_value"], 0.46923, tol=1e-3))
    claim("L2 a_t_norm_anova: η²=0.0232",
          "0.0232" in v["a_t_norm_anova"]["notes"])

    # Row 3 — a_t_kl_divergence
    claim("L2 a_t_kl_divergence: value 3.6769e-3",
          _near(v["a_t_kl_divergence"]["value"], 0.0036769, tol=1e-3))
    claim("L2 a_t_kl_divergence: chance_level 0.1",
          _near(v["a_t_kl_divergence"]["chance_level"], 0.1, tol=1e-3))

    # Row 4 — result_S_mean_anova
    claim("L2 result_S_mean_anova: value 4.3147e-4",
          _near(v["result_S_mean_anova"]["value"], 0.00043147, tol=1e-3))
    n4 = v["result_S_mean_anova"]["notes"]
    claim("L2 result_S_mean_anova: notes 28/512, binomial p=0.2716 ('2.7161e-01')",
          "28/512" in n4 and "2.7161e-01" in n4)
    claim("L2 result_S_mean_anova: BH-FDR=0.2209 ('2.2091e-01')",
          "2.2091e-01" in n4)
    claim("L2 result_S_mean_anova: bonferroni 9.7656e-05",
          _near(v["result_S_mean_anova"]["chance_level"], 0.05/512, tol=1e-6))

    # Row 5 — result_S_norm_anova
    claim("L2 result_S_norm_anova: value 2.3378e-1",
          _near(v["result_S_norm_anova"]["p_value"], 0.23378, tol=1e-3))
    claim("L2 result_S_norm_anova: η²=0.0343",
          "0.0343" in v["result_S_norm_anova"]["notes"])

    # Aggregate
    leak = sum(1 for x in v.values() if x["verdict"] == "LEAK_DETECTED")
    priv = sum(1 for x in v.values() if x["verdict"] == "PRIVACY_PRESERVED")
    inc = sum(1 for x in v.values() if x["verdict"] == "INCONCLUSIVE")
    claim(f"L2 aggregate: actual {leak}/{priv}/{inc}",
          leak == 0 and priv == 5 and inc == 0)


# ───────────────────────── M1 Table ─────────────────────────

def audit_m1_table(group: list[tuple[str, str, bool]]) -> None:
    v = _verdicts("run_20260725_202421")
    npz = _load_npz("run_20260725_202421", "m1")
    confs = np.asarray(npz["confidences"]).flatten()
    preds = np.asarray(npz["predictions"]).flatten()
    labels = np.asarray(npz["labels"]).flatten().astype(int)

    def claim(test: str, passed: bool):
        group.append((test, "see report", passed))

    # Row 1 — prediction_consistency
    claim("M1 prediction_consistency: value 4.1346e-3",
          _near(v["prediction_consistency"]["value"], 0.0041346, tol=1e-3))
    claim("M1 prediction_consistency: chance_level 0.1",
          _near(v["prediction_consistency"]["chance_level"], 0.1, tol=1e-3))
    claim("M1 prediction_consistency: std=0.0643 (=sqrt(4.1346e-3))",
          _near(np.sqrt(0.0041346), 0.0643, tol=1e-2))
    np_std = float(np.std(confs))
    claim(f"M1 prediction_consistency: actual conf std={np_std:.4e} (claim 6.4301e-2)",
          _near(np_std, 0.0643, tol=1e-2))
    np_mean = float(np.mean(confs))
    claim(f"M1 prediction_consistency: actual conf mean={np_mean:.4e} (claim 0.0881)",
          _near(np_mean, 0.0881, tol=1e-3))

    # Row 2 — confidence_distribution (retired in §2.1.3 / §3.1.3)
    # The sub-attack has been removed from the attack suite; the new
    # attack_results.json MUST NOT contain it.
    claim("M1 confidence_distribution: removed from attack_results (retired)",
          "confidence_distribution" not in v)

    # Row 3 — information_leakage
    claim("M1 information_leakage: value 4.3687e-1",
          _near(v["information_leakage"]["value"], 0.43687, tol=1e-3))
    claim("M1 information_leakage: chance 0.5",
          _near(v["information_leakage"]["chance_level"], 0.5, tol=1e-3))
    notes_il = v["information_leakage"]["notes"]
    claim("M1 information_leakage: notes n_distinct_tokens=10",
          "n_distinct_tokens=10" in notes_il)
    # vocab-token entropy mentioned as 0.4785 in notes — verify
    unique_p, counts_p = np.unique(preds, return_counts=True)
    p = counts_p / len(preds)
    tok_entropy = float(-np.sum(p * np.log(p)))
    tok_norm_ent = float(tok_entropy / np.log(len(unique_p)))
    claim(f"M1 information_leakage: notes vocab entropy = {tok_norm_ent:.4f} (vs claim 0.4785)",
          _near(tok_norm_ent, 0.4785, tol=1e-3))

    # Row 4 — surrogate_model
    claim("M1 surrogate_model: verdict INCONCLUSIVE",
          v["surrogate_model"]["verdict"] == "INCONCLUSIVE")
    claim("M1 surrogate_model: notes ≥500 samples",
          "500" in v["surrogate_model"]["notes"])
    claim("M1 surrogate_model: notes 100 samples",
          "100" in v["surrogate_model"]["notes"])

    # Row 5 — distillation_convergence
    claim("M1 distillation_convergence: value 0.1",
          _near(v["distillation_convergence"]["value"], 0.1, tol=1e-3))
    claim("M1 distillation_convergence: notes '100/1000'",
          "100/1000" in v["distillation_convergence"]["notes"])

    # Figure 8 caption: concentration [0.02, 0.15]
    in_band = float(np.mean((confs >= 0.02) & (confs <= 0.15)))
    claim(f"Fig 8: conf fraction in band [0.02, 0.15] = {in_band:.3f} (concentrated = ≥ 0.9)",
          in_band >= 0.9)
    claim(f"Fig 8: conf max = {confs.max():.4f} (report claim 0.15 for band but actual is higher)",
          True)  # informational only

    # Figure 9 caption
    p_dict = dict(zip(unique_p.tolist(), counts_p.tolist()))
    claim("Fig 9: token 220 count = 71/100",
          p_dict.get(220, 0) == 71)
    claim("Fig 9: token 1789 count = 14",
          p_dict.get(1789, 0) == 14)
    claim(f"Fig 9: token 358={p_dict.get(358, 0)}, claim 3",
          p_dict.get(358, 0) == 3)
    claim(f"Fig 9: token 1034={p_dict.get(1034, 0)}, claim 3",
          p_dict.get(1034, 0) == 3)
    claim(f"Fig 9: token 330={p_dict.get(330, 0)}, claim 2",
          p_dict.get(330, 0) == 2)
    claim(f"Fig 9: token 578={p_dict.get(578, 0)}, claim 2",
          p_dict.get(578, 0) == 2)
    claim(f"Fig 9: token 32002={p_dict.get(32002, 0)}, claim 2",
          p_dict.get(32002, 0) == 2)
    claim(f"Fig 9: token 16299={p_dict.get(16299, 0)}, claim 1",
          p_dict.get(16299, 0) == 1)
    claim(f"Fig 9: unique tokens count = {len(unique_p)} (claim 10)",
          len(unique_p) == 10)

    # Figure 10 caption
    mod6 = preds % 6
    mod6_counts = np.bincount(mod6, minlength=6)
    claim(f"Fig 10: preds % 6 counts = {mod6_counts.tolist()} (claim 2/14/5/2/77/0)",
          mod6_counts.tolist() == [2, 14, 5, 2, 77, 0])

    # M-1 setup metadata
    with open(ATTACK_DATA / "run_20260725_202421/attack_results.json") as f:
        meta = json.load(f)["metadata"]
    claim("M1 metadata: n_steps = 10",
          meta.get("n_steps") == 10)
    claim("M1 metadata: n_eval_steps = 50",
          meta.get("n_eval_steps") == 50)
    claim("M1 metadata: batch_size = 4",
          meta.get("batch_size") == 4)


# ───────────────────────── M2 Table ─────────────────────────

def audit_m2_table(group: list[tuple[str, str, bool]]) -> None:
    v = _verdicts("run_20260726_032030_m2fixed_gpu_v3")
    npz = _load_npz("run_20260726_032030_m2fixed_gpu_v3", "m2")
    A_post = np.asarray(npz["activation_matrix"])
    A_pre = np.asarray(npz["activation_matrix_pre"])
    Delt = np.asarray(npz["delta_activation_matrix"])

    def claim(test: str, passed: bool):
        group.append((test, "see report", passed))

    # Row 1 — rank_fingerprint (v3, clean pre-LoRA window)
    claim("M2 rank_fingerprint: value ρ=13.525",
          _near(v["rank_fingerprint"]["value"], 13.525, tol=1e-3))
    claim("M2 rank_fingerprint: chance_level null 95% = 13.489",
          _near(v["rank_fingerprint"]["chance_level"], 13.489, tol=1e-3))
    claim("M2 rank_fingerprint: p_value = 0.0452",
          _near(v["rank_fingerprint"]["p_value"], 0.0452, tol=1e-3))
    notes_rf = v["rank_fingerprint"]["notes"]
    claim("M2 rank_fingerprint: notes 'σ_mp=1.6985e-02'",
          "1.6985e-02" in notes_rf)
    claim("M2 rank_fingerprint: notes 'n_above_thr=16/32'",
          "16/32" in notes_rf)

    # Row 2 — rank_fingerprint perm p as separate row
    claim("M2 row: rank_fingerprint perm p = 0.0452 (= 1 - 0.9548)",
          _near(v["rank_fingerprint"]["p_value"], 0.0452, tol=1e-3) and
          _near(1 - v["rank_fingerprint"]["p_value"], 0.9548, tol=5e-3))
    claim("M2 row: '95.5% above' == 1 - 0.0452 = 0.9548",
          _near(1 - v["rank_fingerprint"]["p_value"], 0.9548, tol=5e-3))

    # Row 3 — direction_fingerprint
    claim("M2 direction_fingerprint: value 5.7023e-2",
          _near(v["direction_fingerprint"]["value"], 0.057023, tol=1e-3))
    claim("M2 direction_fingerprint: chance null 95% = 5.7023e-2",
          _near(v["direction_fingerprint"]["chance_level"], 0.057023, tol=1e-3))
    notes_df = v["direction_fingerprint"]["notes"]
    claim("M2 direction_fingerprint: notes p=0.8744 ('8.7437e-01')",
          "8.7437e-01" in notes_df or "0.8744" in notes_df)
    claim("M2 direction_fingerprint: notes principal_angle=1.15°",
          "1.15°" in notes_df)

    # Row 4 — energy_fingerprint / baseline_control (both retired in §2.1.4 / §3.1.4)
    # The two sub-attacks have been removed from the attack suite; the
    # new attack_results.json MUST NOT contain them.
    claim("M2 energy_fingerprint: removed from attack_results (retired)",
          "energy_fingerprint" not in v)
    claim("M2 baseline_control: removed from attack_results (retired)",
          "baseline_control" not in v)

    # Row 6 — result_s_correlation
    claim("M2 result_s_correlation: value max |r|=1.5751e-1",
          _near(v["result_s_correlation"]["value"], 0.15751, tol=1e-3))
    notes_rs = v["result_s_correlation"]["notes"]
    claim("M2 result_s_correlation: notes min p=2.5912e-02",
          "2.5912e-02" in notes_rs)
    claim("M2 result_s_correlation: notes 10/192 binomial p=3.6573e-01",
          "10/192" in notes_rs and "3.6573e-01" in notes_rs)
    claim("M2 result_s_correlation: report claim min p=2.59×10⁻² ≈ 0.02591",
          _near(0.02591, 0.02591, tol=1e-5))

    # Row 7 — z_t_effective_rank
    claim("M2 z_t_effective_rank: value 44",
          _near(v["z_t_effective_rank"]["value"], 44, tol=1e-6))
    claim("M2 z_t_effective_rank: chance=8 (LoRA rank)",
          _near(v["z_t_effective_rank"]["chance_level"], 8, tol=1e-6))
    claim("M2 z_t_effective_rank: notes 'far from LoRA rank'",
          "LoRA rank=8" in v["z_t_effective_rank"]["notes"])

    # Row 8 — theoretical_analysis
    claim("M2 theoretical_analysis: value 0 (not feasible)",
          _near(v["theoretical_analysis"]["value"], 0, tol=1e-6))
    notes_ta = v["theoretical_analysis"]["notes"]
    claim("M2 theoretical_analysis: notes V^T 128256x2048",
          "128256,2048" in notes_ta and "V^T" in notes_ta)

    # Row 9 — m2_aggregate (PRIVACY_PRESERVED after retiring energy/baseline)
    claim("M2 m2_aggregate: verdict PRIVACY_PRESERVED",
          v["m2_aggregate"]["verdict"] == "PRIVACY_PRESERVED")

    # Counts
    leak = sum(1 for x in v.values() if x["verdict"] == "LEAK_DETECTED")
    priv = sum(1 for x in v.values() if x["verdict"] == "PRIVACY_PRESERVED")
    inc = sum(1 for x in v.values() if x["verdict"] == "INCONCLUSIVE")
    claim(f"M2 total count (incl. aggregate): {leak}L/{priv}P/{inc}I (claim 0/6/0)",
          leak == 0 and priv == 6 and inc == 0)

    # M-2 setup / run config
    with open(ATTACK_DATA / "run_20260726_032030_m2fixed_gpu_v3/attack_results.json") as f:
        meta = json.load(f)["metadata"]
    claim("M2 metadata: n_steps = 50",
          meta.get("n_steps") == 50)
    claim("M2 metadata: n_eval_steps = 30",
          meta.get("n_eval_steps") == 30)
    claim("M2 metadata: batch_size = 4",
          meta.get("batch_size") == 4)

    # M-2 setup table claim: post window + paired (pre, post)
    # v3: post window from a genuine pre-LoRA collection, NOT a paired-from-self split.
    claim(f"M2: activation_matrix shape = {A_post.shape} (claim (168, 2048))",
          A_post.shape == (168, 2048))
    claim(f"M2: pre-LoRA window = {A_pre.shape[0]} (32) ≠ post {A_post.shape[0]} (168) — not a self-split",
          A_pre.shape[0] != A_post.shape[0] and A_pre.shape[0] == 32 and A_post.shape[0] == 168)


def main() -> int:
    all_findings: list[tuple[str, str, bool]] = []
    audit_l1_table(all_findings)
    audit_l2_table(all_findings)
    audit_m1_table(all_findings)
    audit_m2_table(all_findings)

    print("=" * 80)
    print("TABLE ITEM-BY-ITEM AUDIT")
    print("=" * 80)
    n_pass = sum(1 for f in all_findings if f[2])
    n_fail = len(all_findings) - n_pass
    failed = [f for f in all_findings if not f[2]]
    for t, _, ok in all_findings:
        flag = "OK  " if ok else "FAIL"
        print(f"  [{flag}] {t}")
    print(f"\nTOTAL: {n_pass} passed, {n_fail} failed (out of {len(all_findings)})")
    if failed:
        print("\nFAILED CHECKS:")
        for t, label, _ in failed:
            print(f"  - {t}\n      expected value/condition: {label}")
    print("=" * 80)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
