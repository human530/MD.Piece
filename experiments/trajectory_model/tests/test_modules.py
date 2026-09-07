# -*- coding: utf-8 -*-
"""模組二～五各一組測試。每個測試都寫明它守住的是哪一條方法學規則。"""
import os

import numpy as np

from cohort import load_params, make_cohort
from risk import fit_risk_coefficients, risk_score, stratify
from clustering import cluster_stratum, featurize
from prediction import landmark_features, reclassification, run_prediction
from warning import alarms_from_flags, block_permute, _thresholds, residual_matrix

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = load_params(os.path.join(HERE, "params.json"), verbose=False)


# ---------------------------------------------------------------- 模組二
def test_risk_score_ignores_everything_after_baseline():
    """禁止事項 1：基線分數不得用 x(t>0)。把 t>0 的整條序列改掉，分數必須一位不變。"""
    C = make_cohort(P, 2, n=200, delta_mu_median=0.6, lam0=1.2e-4, beta=1.1)
    coefs = fit_risk_coefficients(C)
    s1 = risk_score(C, coefs)
    C2 = dict(C); C2["X"] = C["X"] + 100.0
    assert np.array_equal(s1, risk_score(C2, coefs))
    st = stratify(s1, 5)
    assert np.bincount(st).min() >= 39 and st.max() == 4        # 分層人數平衡


# ---------------------------------------------------------------- 模組三
def test_clustering_finds_two_planted_groups_and_not_more_on_noise():
    """H1 的分群規則要能找到真的存在的結構、也不能在純雜訊上幻想出群數 > 1（置換 p 應不顯著）。"""
    rng = np.random.default_rng(0)
    T = 300
    t = np.arange(T) / T
    Pq = dict(P); Pq["clustering"] = dict(P["clustering"], n_bootstrap=15, n_permutation=19)
    slopes = np.r_[np.full(60, -3.0), np.full(60, 3.0)]
    # 兩群水準與總變異相同、只有斜率方向相反：時間打亂置換保留每條序列的水準與總變異，
    # 所以只有「時間形狀」造成的分群才會被置換檢定判為顯著——這正是該檢定能回答的問題。
    X = (slopes[:, None] * (t[None, :] - 0.5) + rng.normal(0, 0.05, (120, T))).astype(np.float32)
    for m in ("A", "B"):
        r = cluster_stratum(X, m, Pq, rng, gen_labels=np.r_[np.zeros(60, int), np.ones(60, int)])
        assert r["K"] == 2, (m, r["K"], r["bic_by_k"], r["stability_by_k"])
        assert r["stability"] > 0.95 and r["perm"]["p"] < 0.1
        assert r["n_true_labels"] == 2 and r["over_split_by"] == 0 and not r["k_ceiling_hit"]
    noise = rng.normal(0, 1, (120, T)).astype(np.float32)
    r = cluster_stratum(noise, "B", Pq, rng)
    assert r["K"] == 1 or r["perm"]["p"] > 0.05


def test_featurize_returns_expected_shapes():
    X = np.random.default_rng(1).normal(size=(30, 100)).astype(np.float32)
    assert featurize(X, "A", 3).shape == (30, 4) and featurize(X, "B").shape == (30, 4)


# ---------------------------------------------------------------- 模組四
def test_landmark_features_cannot_see_the_future():
    """模組四最重要的洩漏口：地標特徵只能用 X[:, :L]。把 L 之後的值全改掉，特徵必須完全相同。"""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 500)).cumsum(axis=1).astype(np.float32)
    F1 = landmark_features(X, 200, 90)
    X2 = X.copy(); X2[:, 200:] = 1e6
    F2 = landmark_features(X2, 200, 90)
    assert np.array_equal(np.nan_to_num(F1), np.nan_to_num(F2))
    # 也不得用「未來」的分群標籤：run_prediction 的輸入根本沒有 clustering 這個欄位
    C = make_cohort(P, 4, n=150, delta_mu_median=0.6, lam0=1.2e-4, beta=1.1)
    Pq = dict(P); Pq["prediction"] = dict(P["prediction"], landmarks_days=[180], timing_landmark_step_days=365)
    r = run_prediction(C, Pq, 4)
    assert set(r["landmarks"]) == {"180"} and "auc_dynamic" in r["landmarks"]["180"]


def test_reclassification_arithmetic():
    """NRI 的定義要對：事件者移入為正、非事件者移出為正。手算例。"""
    y = np.array([1, 1, 1, 0, 0, 0])
    hs = np.array([0, 0, 1, 1, 1, 0], bool)
    hd = np.array([1, 0, 1, 0, 1, 0], bool)
    r = reclassification(y, hs, hd)
    assert abs(r["nri_event"] - (1 / 3 - 0)) < 1e-12           # 事件者：1 人移入、0 人移出
    assert abs(r["nri_nonevent"] - (1 / 3 - 0)) < 1e-12        # 非事件者：1 人移出、0 人移入
    assert abs(r["nri"] - 2 / 3) < 1e-12 and abs(r["moved_in"] - 1 / 6) < 1e-12


# ---------------------------------------------------------------- 模組五
def test_alarm_bookkeeping_first_alarm_false_alarms_and_person_time():
    """偽警報率的定義（episode 起點後 horizon 天內無事件）與人年分母，決定 H3 的偽警報比較。"""
    days = np.array([7, 14, 21, 28, 35])
    flags = np.array([[0, 1, 1, 0, 1],      # 兩個 episode：起 14 與 35；事件在 40 → 都在 horizon 內 → 0 偽警報
                      [1, 0, 0, 0, 1],      # 無事件：兩個 episode 都是偽警報
                      [0, 0, 0, 1, 1]], bool)  # 事件在 20：28、35 的評估在事件後 → 忽略 → 無警報
    t_event = np.array([40, -1, 20]); T = 100
    t_end = np.where(t_event >= 0, t_event, T)
    first, false_ct, py = alarms_from_flags(flags, days, t_end, t_event, horizon=30, T=T)
    assert list(first) == [14, 7, -1]
    assert list(false_ct) == [0, 2, 0]
    assert abs(py[0] - 40 / 365.25) < 1e-9 and abs(py[1] - 100 / 365.25) < 1e-9
    # v2 陸：burn-in 期不得警報且自分母扣除
    first_b, false_b, py_b = alarms_from_flags(flags, days, t_end, t_event, horizon=30, T=T, start_day=10)
    assert list(first_b) == [14, 35, -1] and list(false_b) == [0, 1, 0]
    assert abs(py_b[1] - 90 / 365.25) < 1e-9


def test_block_permutation_preserves_blocks_and_one_sided_thresholds():
    """區塊置換要保留窗內結構（只打亂區塊順序）；警報閾值上尾／下尾各取 alpha（v2 肆：主警報單尾上升）。"""
    x = np.arange(20.0)
    Xp = block_permute(x, 6, 5, np.random.default_rng(0))
    orig_blocks = [tuple(x[0:6]), tuple(x[6:12]), tuple(x[12:18]), tuple(x[18:20])]
    for row in Xp:
        assert sorted(row) == list(x)
        # 每個原始區塊都要以完整、連續、原順序的樣子出現在置換後的序列裡
        s = tuple(row)
        for b in orig_blocks:
            assert any(s[i:i + len(b)] == b for i in range(len(s) - len(b) + 1))
    null = np.random.default_rng(1).normal(size=(4000, 3))
    thr = _thresholds(null, null, 0.05)
    assert np.all(thr["ar_lo"] < -1.5) and np.all(thr["ar_hi"] > 1.5)          # 上尾／下尾各 5%
    M = residual_matrix(21, "linear", 0.25)
    assert np.allclose(M @ np.arange(21.0), 0, atol=1e-9)        # 直線去趨勢把直線消掉
