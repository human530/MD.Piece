# -*- coding: utf-8 -*-
import os

import numpy as np

from cohort import (load_params, derived_scale, make_cohort, assign_linear_class, assign_type, build_mu_path,
                    with_events, stratum_type_mix, _val, X_FOLD, MU_C)
from fade_components import _detrend
from risk import fit_risk_coefficients, risk_score, stratify

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = load_params(os.path.join(HERE, "params.json"), verbose=False)
KW = dict(delta_mu_median=0.6, lam0=1.2e-4, beta=1.1)      # 校準量級（測試不依賴校準）


def test_every_parameter_row_has_a_source():
    """禁止事項 6：沒有出處不得填軌跡參數。參數檔任何一列缺 source 就該在這裡被抓到。"""
    assert P["_provenance"]["missing_source"] == []
    assert any(d["derived_from"].startswith("assumption") for d in P["_provenance"]["derived"])  # 假設列會被列出


def test_reference_geometry_and_threshold():
    """v1 §2／v2 捌：翻轉型參考幾何——健康態 ↔ 90、摺疊點 ↔ 60，尺度 b 固定；門檻 15 兩型共用，
    且落在翻轉路徑之內（否則翻轉不會跨過門檻）。"""
    sc = derived_scale(P)
    assert abs(90 - sc["b"] * (X_FOLD - sc["y_healthy"]) - 60) < 1e-9
    y_thr = sc["y_healthy"] + (90 - sc["threshold_egfr"]) / sc["b"]     # 基線 90 者跨門檻時的 y
    assert X_FOLD < y_thr < 1.15


def test_shares_are_preserved_when_susceptibility_drives_assignment():
    """v1 §3／v2 捌：易感度決定型別與子類別，但邊際比例仍是指定值——否則『比例來自文獻／規格』就是假的。"""
    rng = np.random.default_rng(0)
    s = rng.normal(0, 1, 200_000)
    cls = assign_linear_class(P, s, rng)
    raw = np.array([c["share_raw"] for c in P["linear_classes"]["classes"]]); want = raw / raw.sum()
    assert np.allclose(np.bincount(cls, minlength=3) / len(cls), want, atol=0.01)
    assert cls[s > 1].mean() > cls[s < -1].mean()               # 高易感度 → 斜率較陡的類別
    f = assign_type(P, s, rng)
    assert abs(f.mean() - _val(P["sim"]["flip_share"])) < 0.01
    assert f[s > 1].mean() > f[s < -1].mean()                   # 型別由易感度決定


def test_type_is_not_determined_by_x0_and_strata_are_mixed():
    """v2 捌：兩型 x0 自同一分布抽 → 風險層內兩型混合，任一層單一型別不得 > 90%，否則 H1 無從檢驗。"""
    C = make_cohort(P, 5, n=1500, **KW)
    f = C["is_flip"]
    assert abs(C["egfr0"][f].mean() - C["egfr0"][~f].mean()) < 4      # 同一分布
    score = risk_score(C, fit_risk_coefficients(C))
    for q in (5, 10):
        _, worst = stratum_type_mix(f, stratify(score, q))
        assert worst < _val(P["risk_score"]["stratum_single_type_max"])


def test_three_time_points_and_reproducibility():
    """v2 參：t_crit（僅翻轉型）、t_threshold（首次 ≤ 15）、t_event（風險驅動）三者分開記錄；同 seed 重現。"""
    C1 = make_cohort(P, 5, n=200, **KW); C2 = make_cohort(P, 5, n=200, **KW)
    assert np.array_equal(C1["X"], C2["X"]) and np.array_equal(C1["t_event"], C2["t_event"])
    f = C1["is_flip"]
    assert (C1["t_crit"][~f] == -1).all() and (C1["t_crit"][f] >= 0).any()
    ok = C1["t_threshold"] >= 0
    thr = C1["scale"]["threshold_egfr"]
    assert (C1["X"][ok, :][np.arange(ok.sum()), C1["t_threshold"][ok]] <= thr).all()
    ok2 = ok & (C1["t_threshold"] > 0)
    assert (C1["X"][ok2, :][np.arange(ok2.sum()), C1["t_threshold"][ok2] - 1] > thr).all()   # 首次跨越
    # 風險驅動：事件不等於跨門檻——有人跨了門檻沒事件、事件也不必在跨門檻當天
    assert ((C1["t_threshold"] >= 0) & ~C1["event"]).any()


def test_hazard_beta_zero_makes_trajectory_uninformative():
    """v2 參（單元測試要求）：β=0 時事件與 x(t) 無關——同一世代換 (λ0, β=0) 後，事件時間與跨門檻日無關聯，
    地標動態模型相對靜態模型無增益（AUC 差 < 0.03，且兩者都接近 0.5）。"""
    from prediction import run_prediction
    C = make_cohort(P, 9, n=1200, **KW)
    C0 = with_events(C, 3e-4, 0.0, P)
    assert 0.1 < C0["event"].mean() < 0.9
    ev, tt = C0["event"], C0["t_threshold"]
    # 事件率在「跨過門檻者」與「未跨者」之間無差別
    assert abs(ev[tt >= 0].mean() - ev[tt < 0].mean()) < 0.06
    Pq = dict(P); Pq["prediction"] = dict(P["prediction"], landmarks_days=[365], timing_landmark_step_days=730)
    r = run_prediction(C0, Pq, 9)
    row = r["landmarks"]["365"]
    assert abs(row["auc_dynamic"] - row["auc_static"]) < 0.03
    assert abs(row["auc_dynamic"] - 0.5) < 0.06


def test_drift_onset_holds_mu_constant_before_onset():
    """v2 玖：漂移起始日之前 mu 固定（定態），之後才線性升到 mu_end；漂移期長度落在參數檔範圍。"""
    rng = np.random.default_rng(1)
    mu0 = -0.6; dmu = np.full(50, 1.2); T = 1825
    mp, onset, dur = build_mu_path(P, mu0, dmu, T, rng)
    d_lo, d_hi = _val(P["flip"]["drift_duration_range_days"])
    assert (dur >= d_lo).all() and (dur <= d_hi).all() and (onset + dur <= T + 1e-9).all()
    for i in range(50):
        o = int(np.floor(onset[i]))
        assert np.allclose(mp[i, :o], mu0)                             # 起始日前不動
        assert abs(mp[i, -1] - (mu0 + dmu[i])) < 1e-9                  # 最後到達 mu_end
        assert np.all(np.diff(mp[i]) >= -1e-12)                        # 單調不降
    C = make_cohort(P, 3, n=400, **KW)
    f = C["is_flip"]
    assert (C["t_crit"][f & (C["t_crit"] >= 0)] > C["t_onset"][f & (C["t_crit"] >= 0)]).all()   # 臨界日必在起始日之後
    # v2 附錄貳：線性型也有定態前期，起始日與翻轉型同分布；day0 變體則自第 0 天下降
    from scipy.stats import ks_2samp
    assert ks_2samp(C["t_onset"][f], C["t_onset"][~f]).pvalue > 0.01
    i = np.where(~f)[0][0]; o = int(C["t_onset"][i])
    early = C["X"][i, :max(o - 30, 30)]
    assert abs(early.mean() - C["egfr0"][i]) < 6                        # 起始日前在基線附近定態
    C0 = make_cohort(P, 3, n=100, linear_onset="day0", **KW)
    assert (C0["t_onset"][~C0["is_flip"]] == 0).all()


def test_linear_ou_and_flip_have_comparable_baseline_autocorrelation():
    """v1 §4（關鍵）：線性型用 OU 有色雜訊且基線自相關與翻轉型相當，H3 才是真的檢定。"""
    C = make_cohort(P, 11, n=400, **KW)
    f = C["is_flip"]

    def ar1(v):
        d = _detrend(v[:180].astype(float)); return np.corrcoef(d[:-1], d[1:])[0, 1]
    a_lin = np.mean([ar1(C["X"][i]) for i in np.where(~f)[0]])
    a_flip = np.mean([ar1(C["X"][i]) for i in np.where(f)[0]])
    assert a_lin > 0.5 and a_flip > 0.5                          # 都不是白雜訊
    assert abs(a_lin - a_flip) < 0.05                            # 起點相當


def test_dropout_variant_keeps_outcome_but_blanks_observations():
    """v1 §6 退出＝停止記錄（複用 FADE apply_S2 語意）：結果事件仍已知，觀測值退出後為 NaN。"""
    C = make_cohort(P, 3, n=100, dropout=True, **KW)
    d = C["drop_day"]
    i = np.where(d >= 0)[0][0]
    assert np.isnan(C["X_obs"][i, d[i]:]).all() and not np.isnan(C["X_obs"][i, :d[i]]).any()
    assert C["event"].dtype == bool and len(C["t_event"]) == 100
