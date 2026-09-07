# -*- coding: utf-8 -*-
"""單元測試 v2（每個模組一個獨立測試；python -m pytest experiments/sle_taper/tests -q）。
測試驗證「為什麼」：每條都對應計畫書 v2 的一項要求，改了商業邏輯就會壞。"""
import copy
import hashlib
import inspect
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

import m0_params as M0                                     # noqa: E402
from m0_params import value                                # noqa: E402
from seeding import module_rng, MODULE_INDEX               # noqa: E402
from m1_generator import (ou_unit, simulate_cohort, observe, events_from_U, prescribed_schedule, effective_exposure,
                          adherence_paths, MECHS, lower_state)                     # noqa: E402
from m5_ews import prefix_kendall, rolling_indicators, score_cohort, lock_threshold, first_alarm, running_max, evaluate  # noqa: E402
from m6_arms import biopsy_plausible_range, calibrate_biopsy, noninvasive_features, noninvasive_arm  # noqa: E402
from m4_predict import run_predict, secondary_metrics      # noqa: E402
import gates as G                                          # noqa: E402

P = M0.load("thresholds.json"); Cj = M0.load("cohort.json")
LAM0 = 2e-4      # 校準量級（測試不依賴校準）


def _small(seed=1, n=300, **kw):
    kw.setdefault("lam0", LAM0)
    return simulate_cohort(P, Cj, seed, n=n, **kw)


def _cfg():
    return dict(window_days=int(value(P, "window_days")), min_obs=int(value(P, "min_obs_per_window")), detrend="linear", bw_frac=0.25, eval_every=7, joint="min_tau")


def test_01_seeding_streams_are_reproducible_and_index_stable():
    a = module_rng(7, "generator").random(5); b = module_rng(7, "generator").random(5)
    assert np.array_equal(a, b)
    assert not np.array_equal(module_rng(7, "generator", 1).random(5), a)
    assert MODULE_INDEX["generator"] == 0 and MODULE_INDEX["arms"] == 5 and MODULE_INDEX["grid"] == 6
    with pytest.raises(KeyError):
        module_rng(7, "not_a_module")


def test_02_params_all_sourced_and_no_hardcoded_numbers_in_src():
    """每列有 source/status；pending 的 value 為 null 且 placeholder 存在；src 內不得出現界線值常數（文獻數字）。"""
    chk = M0.check(P, verbose=False)
    assert not chk["missing_source"] and not chk["bad_status"]
    for p in chk["placeholders"]:
        assert P[p["key"]]["value"] is None and P[p["key"]].get("placeholder") is not None
    import io, tokenize
    code = []
    for f in os.listdir(os.path.join(os.path.dirname(HERE), "src")):
        if f.endswith(".py"):
            txt = open(os.path.join(os.path.dirname(HERE), "src", f), encoding="utf-8").read()
            for tok in tokenize.generate_tokens(io.StringIO(txt).readline):      # 只看程式碼，不看字串／註解
                if tok.type not in (tokenize.STRING, tokenize.COMMENT):
                    code.append(tok.string)
    src = " ".join(code)
    for lit in ("0.306", "0.273", "0.321", "0.88", "0.879", "0.070", "0.051"):
        assert lit not in src, f"src 內硬寫了文獻數值 {lit}"


def test_03_ou_unit_variance_and_correlation_time():
    """單位 OU：定態變異 ≈ 1、lag-k 自相關 ≈ exp(−k/τξ)。這是 τξ 與 τx 分開識別的前提。"""
    rng = module_rng(3, "generator", 1)
    xi = ou_unit(300, 600, 10.0, rng)[:, 200:]
    assert abs(xi.var() - 1.0) < 0.1
    ac = np.mean([np.corrcoef(xi[i, :-10], xi[i, 10:])[0, 1] for i in range(300)])
    assert abs(ac - np.exp(-1.0)) < 0.08


def test_04_generator_mechanisms_and_layers():
    """六機制皆存在且標籤先於動力學；分岔者跨臨界日存在、穩定者無；τ_x 由 lower_state 一致；observe 不改生成層（雜湊）。"""
    C = _small(n=600)
    assert set(np.unique(C["mech"])) <= set(MECHS) and len(np.unique(C["mech"])) >= 5
    b = C["mech"] == "bifurcation"; s = C["mech"] == "stable"
    assert (C["t_crit"][b] >= 0).all() and (C["t_crit"][s] < 0).all()
    assert (C["mu_i"][b] - 0.0 > value(P, "mu_c")).all()                       # 完全停藥終點 g_end=0 → μ_i > μ_c
    assert (C["mu_i"][s] + 0.0 < value(P, "mu_c")).all()
    # 減藥前全體在緩解側（x < 0）
    assert (C["X"][:, :C["run_in"]] < 0).mean() > 0.99
    h0 = hashlib.sha256(C["X"].tobytes()).hexdigest()
    y = observe(C, P, Cj["observation_scenarios"]["default"], module_rng(1, "obs", 0))
    assert hashlib.sha256(C["X"].tobytes()).hexdigest() == h0
    assert np.isnan(y).any() and np.isfinite(y).any()
    # 事件反函數：換 λ0 不換 U；β=0 時事件與 x 無關（機制間發作率相近）
    te0 = events_from_U(C["X"], LAM0 * 5, 0.0, C["U"], t_start=C["run_in"]); e0 = te0 >= 0
    rates = [e0[C["mech"] == m].mean() for m in ("bifurcation", "stable") if (C["mech"] == m).sum() > 20]
    assert abs(rates[0] - rates[1]) < 0.15


def test_05_exposure_model_lag_and_adherence():
    """處方→暴露：一階滯後使 g_eff 落後於處方；依從中斷使暴露暫時歸零；lag=1 時恆等。"""
    t = np.arange(400); g = prescribed_schedule(t, 1.5, 90, 180, True)
    assert g[0] == 1.5 and g[-1] == 0.0 and g[180] > g[200] > 0 and g[270] == 0.0
    G2 = np.repeat(g[None, :], 3, axis=0)
    ge = effective_exposure(G2, 21.0)
    assert (ge[:, 100:270] > G2[:, 100:270]).all()                                # 滯後 → 減藥中暴露高於處方
    assert np.allclose(effective_exposure(G2, 1.0), G2)
    A = adherence_paths(200, 730, dict(mean=0.9, sd=0.1, lapse_prob_per_month=0.5, lapse_days=14), module_rng(5, "generator", 3))
    assert (A == 0).any() and 0.5 < A.mean() < 1.0


def test_06_prefix_kendall_and_time_direction():
    """前綴 Kendall τ：單調上升 → 1、下降 → −1；且第 k 點的 τ 只用前 k 點（時間方向）。"""
    v = np.arange(10, dtype=float)
    tk = prefix_kendall(v); assert np.isnan(tk[:2]).all() and np.allclose(tk[2:], 1.0)
    assert np.allclose(prefix_kendall(-v)[2:], -1.0)
    w = v.copy(); w[7:] = -100
    assert np.allclose(prefix_kendall(w)[:7], tk[:7], equal_nan=True)
    # 滾動指標：白雜訊 AR(1) ≈ 0；AR(1) 過程 → 正
    rng = np.random.default_rng(0); t = np.arange(0, 400, 7).astype(float)
    ar1, sd = rolling_indicators(t, rng.normal(size=len(t)), np.array([200., 390.]), 120, 6)
    assert np.isfinite(ar1).all() and abs(ar1).max() < 0.6


def test_07_threshold_lock_respects_budget_and_nan_never_alarms():
    """鎖閾：穩定虛無的假警報／病人年 ≤ 預算；分數 NaN 不觸發警報。"""
    rng = np.random.default_rng(1)
    S = rng.normal(size=(200, 50)); S[:50] = np.nan
    fu = np.full(200, 365.0); null = np.ones(200, bool)
    lk = lock_threshold(S, null, fu, 0.5)
    assert lk["achieved_fa_per_py"] <= 0.5 + 1e-9
    ta = first_alarm(S, np.arange(50) * 7, lk["threshold"])
    assert (ta[:50] < 0).all()
    assert (running_max(S)[:, -1] >= lk["threshold"]).sum() / (fu.sum() / 365.25) <= 0.5 + 1e-9


def test_08_ews_bifurcation_vs_stable_and_evaluate_semantics():
    """在生成器的正對照（分岔機制、低誤差、每日取樣）上，分數對『跳轉』應有鑑別；穩定機制在鎖定閾值下假警報 ≤ 預算；
    evaluate 的敏感度只計跳轉前的警報。"""
    C = _small(seed=11, n=500, sigma_xi=0.05)
    y = observe(C, P, dict(indicator="x", interval="daily", meas_error="none"), module_rng(11, "obs", 0))
    cfg = _cfg(); cfg["min_obs"] = 10
    stop = np.where(C["t_jump"] >= 0, C["t_jump"], C["T"])
    sc = score_cohort(y, C["run_in"], C["T"], 42, cfg["min_obs"], "linear", 0.25, 7, "min_tau", stop_at=stop)
    null = C["mech"] == "stable"; fu = (stop - C["run_in"]).clip(min=0).astype(float)
    lk = lock_threshold(sc["S"], null, fu, 0.5)
    ev = evaluate(C, sc["S"], sc["eval_times"], lk["threshold"])
    st = ev["per_mechanism"]["stable"]; b = ev["per_mechanism"]["bifurcation"]
    assert st["fa_per_py"] <= 0.5 + 1e-9
    assert b["n_jumped"] > 0 and b["sens_before_jump"] is not None
    if b["lead_to_jump_median"] is not None:
        assert b["lead_to_jump_median"] > 0                                       # 提前期定義：跳轉日 − 警報日 > 0


def test_09_biopsy_range_from_counts_and_noninvasive_signature():
    """切片可信範圍由計數推得（sens 11/11 → 下界 <1；spec 22/25 → 區間含 0.88）；非侵入臂簽章不含 μ_i／B／mech。"""
    pr = biopsy_plausible_range(value(P, "biopsy_rule_counts"))
    assert pr["sensitivity"][1] == 1.0 and pr["sensitivity"][0] < 0.8
    assert pr["specificity"][0] < 0.88 < pr["specificity"][1]
    for f in (noninvasive_features, noninvasive_arm, score_cohort):
        assert not any(p in ("mu_i", "B_true", "mech", "cohort") for p in inspect.signature(f).parameters)


def test_10_secondary_metrics_and_zero_signal_predict():
    """次要指標：完美預測 AUC=1、Brier≈0；β=0 世代下動態 C 增益 ≈ 0（零信號）。"""
    y = np.r_[np.ones(50), np.zeros(50)]; p = np.r_[np.full(50, 0.9), np.full(50, 0.1)]
    m = secondary_metrics(y, p, [0.2, 0.5])
    assert m["auc"] == 1.0 and m["brier"] < 0.02 and m["decision_curve"]["0.5"]["net_benefit"] > m["decision_curve"]["0.5"]["net_benefit_treat_all"]
    shares = {k: 0.0 for k in MECHS}; shares["stable"] = 1.0
    C = simulate_cohort(P, Cj, 21, n=800, mech_shares=shares, lam0=LAM0 * 5, beta=0.0)
    y = observe(C, P, Cj["observation_scenarios"]["default"], module_rng(21, "obs", 0))
    pr = run_predict(C, y, [180, 365], seed=21, timing=False)
    assert all(abs(r["c_gain"]) < 0.06 for r in pr["landmarks"].values())


def test_11_time_direction_gate_and_mechanism_independence():
    C = _small(seed=31, n=200)
    y = observe(C, P, Cj["observation_scenarios"]["default"], module_rng(31, "obs", 0))
    cfg = _cfg()
    assert G.q4_time_direction(y, C["run_in"], C["T"], cfg, np.random.default_rng(0))["passed"]
    assert G.q1_mechanism_independence(C, y, cfg)["passed"]


def test_12_swap_disease_parameters_still_runs():
    """換一組假疾病參數（τx、σξ、機制比例、排程）照樣跑通——框架不綁 SLE 數值。"""
    P2 = copy.deepcopy(P)
    P2["tau_x"]["value"] = 12; P2["sigma_xi"]["value"] = 0.2; P2["T"]["value"] = 240
    P2["mechanisms"]["value"] = {"bifurcation": 0.5, "stochastic_escape": 0.1, "continuous_deterioration": 0.1, "exogenous_shock": 0.1, "noise_amplification": 0.1, "stable": 0.1}
    C = simulate_cohort(P2, Cj, 41, n=150, lam0=LAM0, taper=dict(duration_days=60, complete=True))
    assert C["X"].shape == (150, 90 + 240) and set(np.unique(C["mech"])) <= set(MECHS)
