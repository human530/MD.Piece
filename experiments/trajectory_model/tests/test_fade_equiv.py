# -*- coding: utf-8 -*-
"""決定書 §10：複用 FADE 元件「不得重寫」。這些測試證明本專案的向量化實作與 FADE 原始碼
給出相同的數值——否則兩個研究的方法就不一致，複用就只是名義上的。"""
import importlib.util
import os

import numpy as np
import pytest
from scipy.stats import kendalltau

import fade_components as fc
from warning import rolling_indicators, prefix_tau

FADE_ORIG = r"C:\Users\tpc10\Desktop\AIMD\md_piece\proposal\md.piece\11_整合題目_FADE衰減鏈\02_程式實作\fade_sim.py"


def _load_orig():
    spec = importlib.util.spec_from_file_location("fade_sim_orig", FADE_ORIG)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


@pytest.mark.skipif(not os.path.exists(FADE_ORIG), reason="FADE 原始檔不在此機器")
def test_simulate_s0_bitwise_equal_to_fade_when_tau_is_one():
    """加了 tau 之後，預設值下必須與 FADE 原版逐位元相同：否則就是重寫而非複用。"""
    orig = _load_orig()
    Xo, tco, muo, so = orig.simulate_S0(n=40, T=60, seed=3)
    Xn, tcn, mun, sn = fc.simulate_S0(n=40, T=60, seed=3, tau=1.0)
    assert np.array_equal(Xo, Xn) and np.array_equal(tco, tcn) and np.array_equal(so, sn)


def test_tau_scales_relaxation_speed():
    """tau 是決定書 §5 的時間尺度：tau 大 → 回復慢 → 日 lag-1 自相關高。方向錯了 H3 就整個反。"""
    rng = np.random.default_rng(1)
    x0 = np.full(200, -1.2)
    kw = dict(n=200, T=400, substeps=10, mu_start=-0.6, mu_end=np.full(200, -0.6), sigma=np.full(200, 0.05), x0=x0)
    Xa, *_ = fc.simulate_S0(tau=5.0, rng=np.random.default_rng(1), **kw)
    Xb, *_ = fc.simulate_S0(tau=60.0, rng=np.random.default_rng(1), **kw)

    def ac1(X):
        d = X[:, 100:] - X[:, 100:].mean(axis=1, keepdims=True)
        return np.mean(np.sum(d[:, :-1] * d[:, 1:], 1) / np.sum(d * d, 1))
    assert ac1(Xb) > ac1(Xa) + 0.2


def test_rolling_indicators_match_fade_resilience_tau():
    """模組五的 AR(1)/SD 定義必須就是 FADE 的定義（去趨勢後 corrcoef 與 std）。
    以 FADE 的 resilience_tau 為 oracle：在它取樣的視窗位置上算 Kendall tau，兩者要一致。"""
    rng = np.random.default_rng(0)
    x = rng.normal(size=400).cumsum() * 0.1 + rng.normal(size=400)
    AR, SD = rolling_indicators(x[None, :].astype(np.float32), 21, "linear", 0.25)
    sel = np.arange(0, 400 - 21, 3)                       # FADE: for s in range(0, len(v)-win, 3)
    ta_f, ts_f = fc.resilience_tau(x, 0, 400, 21)
    ta_m, _ = kendalltau(np.arange(len(sel)), AR[0, sel]); ts_m, _ = kendalltau(np.arange(len(sel)), SD[0, sel])
    assert abs(ta_f - ta_m) < 1e-6 and abs(ts_f - ts_m) < 1e-6


def test_prefix_tau_equals_scipy_kendall_on_every_prefix():
    """逐前綴 tau 是警報規則的心臟；分桶加速若算錯，整個 H3 都是錯的。與 scipy 逐一比對。"""
    rng = np.random.default_rng(2)
    Y = rng.normal(size=(4, 257)).cumsum(axis=1) + rng.normal(size=(4, 257))
    PT = prefix_tau(Y)
    for r in range(4):
        for k in (2, 3, 5, 17, 64, 100, 257):
            tau, _ = kendalltau(np.arange(k), Y[r, :k])
            assert abs(tau - PT[r, k - 1]) < 1e-6
    assert np.isnan(PT[0, 0])
