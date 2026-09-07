# -*- coding: utf-8 -*-
"""
自 FADE 模擬程式複用的元件（規格決定書 §10：必須複用，不得重寫）。

來源：C:/Users/tpc10/Desktop/AIMD/md_piece/proposal/md.piece/11_整合題目_FADE衰減鏈/02_程式實作/fade_sim.py
複製日期：2026-08-17

除下列一處外，函式本體逐字照抄，讓兩個研究的方法一致：
- simulate_S0：依決定書 §5 加入時間尺度 tau，並允許外給 mu_end / sigma / x0 / rng；
  依決定書 v2 第玖項允許外給整條 mu_path（漂移起始日之前 mu 固定）。
  預設值下（tau=1、其餘 None）與 FADE 原版逐位元相同（tests/test_fade_equiv.py 驗證）。
"""
import numpy as np

MU_C = 2.0 / (3.0 * np.sqrt(3.0))      # 摺疊分岔臨界值 ≈ 0.3849
T_DAYS = 365
N_PATIENTS = 1200
SEED = 20260813


# =============================================================
# S0：雙穩態隨機微分方程
# =============================================================
def simulate_S0(n=N_PATIENTS, T=T_DAYS, seed=SEED, substeps=10, tau=1.0,
                mu_start=-0.60, mu_end=None, sigma=None, x0=None, rng=None, mu_path=None):
    """dx = (1/tau)(-x^3 + x + mu(t)) dt + sigma dW；mu 線性漂移，跨過 MU_C 即翻轉。

    ［軌跡模型修改，決定書 §5/§10、v2 第玖項］tau 決定系統回復速度、也就是基線自相關水準；
    FADE 原版隱含 tau=1 天，本研究主分析 60 天。mu_path 可外給（n×T），用來讓 mu 在漂移
    起始日之前固定；不給時照原版由 mu_start 線性漂移到 mu_end。其餘新增參數只是把原本
    寫死的分布改成可外給，未改變積分法（Euler–Maruyama、每日 substeps 子步、clip ±3）。
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    if mu_end is None and mu_path is None:
        mu_end = rng.uniform(-0.10, 2.00, size=n)
    n = len(mu_end) if mu_path is None else len(mu_path)
    if sigma is None:
        sigma = rng.uniform(0.08, 0.16, size=n)

    dt = 1.0 / substeps
    if x0 is None:
        x = np.full(n, -1.15) + rng.normal(0, 0.05, n)
    else:
        x = np.array(x0, dtype=float).copy()
    X = np.zeros((n, T), dtype=np.float32)
    if mu_path is None:
        mu_path = np.linspace(0.0, 1.0, T)[None, :] * (mu_end - mu_start)[:, None] + mu_start
    else:
        mu_path = np.asarray(mu_path, dtype=float)

    for t in range(T):
        mu_t = mu_path[:, t]
        for _ in range(substeps):
            drift = (-(x ** 3) + x + mu_t) / tau
            x = x + drift * dt + sigma * np.sqrt(dt) * rng.normal(0, 1, n)
            x = np.clip(x, -3.0, 3.0)
        X[:, t] = x

    # 臨界時間：mu 首次跨過 MU_C
    t_crit = np.full(n, -1)
    for i in range(n):
        idx = np.where(mu_path[i] >= MU_C)[0]
        if len(idx):
            t_crit[i] = idx[0]
    return X, t_crit, mu_path.astype(np.float32), sigma


# =============================================================
# S2：留存與退出（決定書 §6 敏感度分析用其退出邏輯）
# =============================================================
def apply_S2(P, fill_rate, dropout_hazard, inflation_per_lag, max_lag, rng):
    """漏登、退出、延遲登錄造成的強度膨脹。回傳含 NaN 的觀測序列。"""
    n, T = P.shape
    L = np.full((n, T), np.nan, dtype=np.float32)

    # 退出：幾何分布的退出日
    u = rng.random((n, T))
    dropped = np.cumsum(u < dropout_hazard, axis=1) > 0
    active = ~dropped

    logged = (rng.random((n, T)) < fill_rate) & active
    # 延遲：部分紀錄延後 1..max_lag 天寫入，強度隨延遲膨脹
    lag = rng.integers(0, max_lag + 1, size=(n, T))
    for t in range(T):
        idx = np.where(logged[:, t])[0]
        if not len(idx):
            continue
        lg = lag[idx, t]
        tgt = np.minimum(t + lg, T - 1)
        vals = P[idx, t] * (1.0 + inflation_per_lag * lg)
        L[idx, tgt] = vals
    return L, active


# =============================================================
# 特徵萃取（同一組固定規則套用於各段）
# =============================================================
def _detrend(v):
    k = np.arange(len(v))
    if len(v) < 4:
        return v - np.nanmean(v)
    A = np.vstack([k, np.ones_like(k)]).T
    coef, *_ = np.linalg.lstsq(A, v, rcond=None)
    return v - A @ coef


def window_features(series, lo, hi):
    """回傳 [level, trend, ar1, sd]；缺漏以可用值計算，不足則回傳 NaN。"""
    v = series[lo:hi]
    m = ~np.isnan(v)
    if m.sum() < 10:
        return np.array([np.nan] * 4)
    idx = np.where(m)[0].astype(float)
    val = v[m]
    level = val[-min(14, len(val)):].mean()
    A = np.vstack([idx, np.ones_like(idx)]).T
    coef, *_ = np.linalg.lstsq(A, val, rcond=None)
    trend = coef[0]
    d = _detrend(val)
    sd = d.std()
    ar1 = np.corrcoef(d[:-1], d[1:])[0, 1] if len(d) > 3 and d.std() > 1e-9 else np.nan
    return np.array([level, trend, ar1, sd])


# =============================================================
# 互資訊
# =============================================================
def mutual_information(x, y, bins=5):
    """x 連續（分位數分箱）或離散，y 二元；Miller-Madow 偏誤校正，單位 bit。"""
    ok = ~np.isnan(x)
    x, y = x[ok], y[ok]
    u = np.unique(x)
    if len(x) < 30 or len(u) < 2:
        return 0.0
    if len(u) <= bins:                      # 離散（例如二元決策）直接用原值
        xb = np.searchsorted(u, x)
    else:
        qs = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
        if len(qs) < 3:
            return 0.0
        xb = np.digitize(x, qs[1:-1])
    n = len(x)
    mi = 0.0
    nz = 0
    for a in np.unique(xb):
        for b in (0, 1):
            p_ab = np.mean((xb == a) & (y == b))
            if p_ab <= 0:
                continue
            nz += 1
            p_a = np.mean(xb == a)
            p_b = np.mean(y == b)
            mi += p_ab * np.log2(p_ab / (p_a * p_b))
    mi -= (nz - len(np.unique(xb)) - 1) / (2 * n * np.log(2))   # Miller-Madow
    return max(mi, 0.0)


def mi_null(x, y, bins=5, reps=400, seed=1):
    rng = np.random.default_rng(seed)
    vals = [mutual_information(x, rng.permutation(y), bins) for _ in range(reps)]
    return float(np.mean(vals)), float(np.quantile(vals, 0.95))


# =============================================================
# 韌性訊號（臨界減速）保留率
# =============================================================
def resilience_tau(series, lo, hi, win=21):
    """滾動 AR(1) 與 SD 的上升趨勢（Kendall tau 之簡化：Spearman on rank）。"""
    v = series[lo:hi]
    ars, sds, ts = [], [], []
    for s in range(0, len(v) - win, 3):
        seg = v[s:s + win]
        m = ~np.isnan(seg)
        if m.sum() < 8:
            continue
        d = _detrend(seg[m])
        if d.std() < 1e-9:
            continue
        a = np.corrcoef(d[:-1], d[1:])[0, 1] if len(d) > 3 else np.nan
        if np.isnan(a):
            continue
        ars.append(a); sds.append(d.std()); ts.append(s)
    if len(ars) < 5:
        return np.nan, np.nan
    def _tau(y):
        y = np.asarray(y); x = np.arange(len(y))
        num = 0; den = 0
        for i in range(len(y)):
            for j in range(i + 1, len(y)):
                s = np.sign(x[j] - x[i]) * np.sign(y[j] - y[i])
                num += s; den += 1
        return num / den if den else np.nan
    return _tau(ars), _tau(sds)
