# -*- coding: utf-8 -*-
"""模組五：預警的型態依賴（H3）。

指標定義與 FADE resilience_tau 完全相同（tests/test_fade_equiv.py 驗證）：
  每個滾動窗先去趨勢（linear = FADE _detrend；gaussian = 窗內高斯核平滑殘差；none = 不去），
  AR(1) = corrcoef(d[:-1], d[1:])、SD = d.std()。差別只在這裡向量化（FADE 的逐點 Python 迴圈
  在 3000 人 × 1825 天 × 12 組設定 × 20,000 條虛無序列的規模下跑不完）。
Kendall tau 用「截至該評估日的完整指標序列」，每 7 天評估一次（決定書 §9）。
虛無分布：抽 200 人 × 100 次區塊置換（區塊長 = 窗長）合併成共同分布，並以 KS 檢定檢核高／低
變異兩群同質性，不同質則分層（決定書 §9）。
警報規則（事前寫定；v2 肆）：主要規則單尾上升——AR(1) 與 SD 的 tau 同時 > 0 且同時超過虛無 95 分位
（韌性流失的定義就是自相關上升）；「下降型偏離」（同時 < 0 且低於 5 分位）獨立計數報告，不計入主要提前期。
假設檢定（兩型提前期比較）維持雙尾（禁止事項 5）。
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import ks_2samp

DAYS_PER_YEAR = 365.25


# ------------------------------------------------------------------ 滾動指標
def residual_matrix(win, mode, bw_frac):
    """把窗內序列映成去趨勢殘差的線性算子 M（win×win）：d = M v。"""
    k = np.arange(win, dtype=float)
    if mode == "linear":                     # 與 FADE _detrend 相同：OLS 直線殘差
        A = np.c_[k, np.ones(win)]
        return np.eye(win) - A @ np.linalg.pinv(A)
    if mode == "gaussian":                   # 高斯核平滑（Nadaraya–Watson）殘差
        bw = max(bw_frac * win, 1.0)
        W = np.exp(-0.5 * ((k[:, None] - k[None, :]) / bw) ** 2)
        W /= W.sum(axis=1, keepdims=True)
        return np.eye(win) - W
    if mode == "none":
        return np.eye(win)
    raise ValueError(mode)


def rolling_indicators(X, win, mode, bw_frac, chunk=256):
    """回傳 AR1, SD（n × m，m = T-win+1；位置 j 對應天 [j, j+win)）。"""
    M = residual_matrix(win, mode, bw_frac).T
    n, T = X.shape
    m = T - win + 1
    AR = np.empty((n, m), np.float32); SD = np.empty((n, m), np.float32)
    for s in range(0, n, chunk):
        V = sliding_window_view(X[s:s + chunk].astype(np.float32), win, axis=1)   # (b, m, win)
        R = V @ M.astype(np.float32)
        a, b = R[..., :-1], R[..., 1:]
        a = a - a.mean(axis=-1, keepdims=True); b = b - b.mean(axis=-1, keepdims=True)
        den = np.sqrt((a * a).sum(-1) * (b * b).sum(-1))
        AR[s:s + chunk] = np.where(den > 1e-12, (a * b).sum(-1) / np.maximum(den, 1e-12), np.nan)
        SD[s:s + chunk] = R.std(axis=-1)
    return AR, SD


# ------------------------------------------------------------------ 逐前綴 Kendall tau
def prefix_tau(Y, chunk=64):
    """對每列 y，回傳 tau[:, k-1] = 前 k 個值對時間的 Kendall tau（k>=2；k<2 為 NaN）。
    S_k = Σ_{j<k} (2 c_j - j)，c_j = 較早且較小的個數。c_j 以 sqrt(m) 分桶法算：
    較低桶的計數用一次 cumsum、同桶內才做兩兩比較 → O(m·sqrt(m))，比 O(m²) 快一個量級。"""
    Y = np.asarray(Y, dtype=np.float64)
    # 前提：指標為連續值（tau-a，不做 tie 校正；相等值極少見）。NaN（如零變異窗的 AR(1)）以 0 代入，避免被排到最大。
    Y = np.where(np.isfinite(Y), Y, 0.0)
    B_all, m = Y.shape
    out = np.empty((B_all, m), np.float32)
    bs = int(np.ceil(np.sqrt(m))); nb = int(np.ceil(m / bs)); pad = nb * bs - m
    lower_rank = np.tril(np.ones((bs, bs), bool), -1)
    j = np.arange(m)
    pairs = j * (j + 1) / 2.0                     # 前 k=j+1 個值的配對數
    for s in range(0, B_all, chunk):
        y = Y[s:s + chunk]; B = len(y)
        order = np.argsort(y, axis=1, kind="stable")           # 依值排序後的位置
        R = np.empty_like(order); np.put_along_axis(R, order, np.broadcast_to(j, order.shape), axis=1)
        q = R // bs
        onehot = np.zeros((B, m, nb), np.int32); np.put_along_axis(onehot, q[..., None], 1, axis=2)
        cum = np.cumsum(onehot, axis=1) - onehot               # 較早元素在各桶的個數
        lower = np.cumsum(cum, axis=2) - cum                   # 較早且桶號更低
        t1 = np.take_along_axis(lower, q[..., None], axis=2)[..., 0]
        pos = np.concatenate([order, np.full((B, pad), m + 1)], axis=1).reshape(B, nb, bs)
        earlier = pos[..., None, :] < pos[..., :, None]         # [b,q,g,g']: g' 較早
        c2 = (earlier & lower_rank).sum(axis=3)                 # 同桶、較早、rank 較低
        flat = np.zeros((B, nb * bs + 2), np.int64)
        np.put_along_axis(flat, pos.reshape(B, -1), c2.reshape(B, -1), axis=1)
        c = t1 + flat[:, :m]
        S = np.cumsum(2 * c - j, axis=1)
        out[s:s + chunk] = np.where(pairs > 0, S / np.maximum(pairs, 1), np.nan)
    return out


# ------------------------------------------------------------------ 區塊置換
def block_permute(x, win, n_perm, rng):
    """把一條序列切成長度 win 的區塊（最後一塊可短），打亂區塊順序；回傳 (n_perm, T)。"""
    T = len(x)
    starts = np.arange(0, T, win)
    nblk = len(starts)
    idx = np.empty((n_perm, T), np.int64)
    for p in range(n_perm):
        o = rng.permutation(nblk)
        idx[p] = np.concatenate([np.arange(starts[b], min(starts[b] + win, T)) for b in o])
    return x[idx]


# ------------------------------------------------------------------ 主流程
def _eval_prefix_lengths(T, win, every):
    days = np.arange(every, T + 1, every)              # 評估日（第 t 天：可用 X[:, :t]）
    mlen = days - win + 1                              # 截至該日的指標序列長度
    ok = mlen >= 5
    return days[ok], mlen[ok]


def _tau_at(AR, SD, mlen):
    ta = prefix_tau(AR); ts = prefix_tau(SD)
    return ta[:, mlen - 1], ts[:, mlen - 1]


def build_null(X, win, mode, bw_frac, mlen, n_subj, n_perm, rng, var_group):
    """共同虛無分布 + 同質性檢核。var_group：全體個案的高／低變異布林（由呼叫端以同一規則算）。
    回傳 dict(thresholds[group] = (lo_ar, hi_ar, lo_sd, hi_sd) 各為 (n_eval,), stratified, ks)。"""
    n = len(X)
    pick = rng.choice(n, size=min(n_subj, n), replace=False)
    taus_ar, taus_sd, grp = [], [], []
    for i in pick:
        Xp = block_permute(X[i].astype(np.float32), win, n_perm, rng)
        AR, SD = rolling_indicators(Xp, win, mode, bw_frac)
        ta, ts = _tau_at(AR, SD, mlen)
        taus_ar.append(ta); taus_sd.append(ts); grp.append(np.full(n_perm, var_group[i]))
    TA = np.concatenate(taus_ar); TS = np.concatenate(taus_sd); G = np.concatenate(grp)
    # 同質性：高／低變異兩群的虛無 tau 在幾個檢查點的 KS 檢定
    checks = np.unique(np.linspace(0, len(mlen) - 1, 4).astype(int))
    ks = []
    for c in checks:
        for name, TT in (("ar1", TA), ("sd", TS)):
            a, b = TT[G, c], TT[~G, c]
            a, b = a[np.isfinite(a)], b[np.isfinite(b)]
            if len(a) > 10 and len(b) > 10:
                r = ks_2samp(a, b)
                ks.append(dict(check_idx=int(c), indicator=name, D=float(r.statistic), p=float(r.pvalue)))
    stratified = bool(ks) and min(k["p"] for k in ks) < 0.05
    return TA, TS, G, dict(stratified=stratified, ks=ks, n_null=int(len(TA)),
                           n_high_var=int(G.sum()), n_low_var=int((~G).sum()))


def _thresholds(TA, TS, alpha):
    """上尾（主要警報）與下尾（下降型偏離計數）各取 alpha 分位；兩者都是單尾（v2 肆）。"""
    return dict(ar_lo=np.nanquantile(TA, alpha, axis=0), ar_hi=np.nanquantile(TA, 1 - alpha, axis=0),
                sd_lo=np.nanquantile(TS, alpha, axis=0), sd_hi=np.nanquantile(TS, 1 - alpha, axis=0))


def _q(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if len(v) == 0:
        return dict(n=0, median=np.nan, q25=np.nan, q75=np.nan, mean=np.nan)
    return dict(n=int(len(v)), median=float(np.median(v)), q25=float(np.percentile(v, 25)),
                q75=float(np.percentile(v, 75)), mean=float(v.mean()))


def alarms_from_flags(flags, days, t_end, t_event, horizon, T, start_day=0):
    """flags: n × n_eval 布林（評估日是否警報）；忽略事件後的評估日。
    回傳 first_alarm(天, -1=無)、偽警報次數（episode 起點後 horizon 天內無事件）、追蹤人年。
    start_day：燒入期（v2 陸）——之前不得警報，且自人年分母扣除。"""
    n = len(flags)
    active = (days[None, :] <= t_end[:, None]) & (days[None, :] >= start_day)
    f = flags & active
    first = np.where(f.any(axis=1), days[np.argmax(f, axis=1)], -1)
    starts = f & ~np.concatenate([np.zeros((n, 1), bool), f[:, :-1]], axis=1)   # episode 起點
    has_ev = t_event >= 0
    ev_within = has_ev[:, None] & (t_event[:, None] - days[None, :] >= 0) & (t_event[:, None] - days[None, :] <= horizon)
    false_ct = (starts & ~ev_within).sum(axis=1)
    py = np.maximum(np.where(has_ev, t_event, T) - start_day, 0) / DAYS_PER_YEAR
    return first, false_ct, py


def trend_alarm(X, threshold, days, horizon, min_history):
    """H3 後半的比較基準：截至評估日的 OLS 直線外推，預測 horizon 天內向下跨過門檻（eGFR 低＝差）即警報。
    min_history：歷史短於此不評估（幾週的 OU 雜訊就能外推出任何斜率；v2 陸）。"""
    n, T = X.shape
    Xf = X.astype(np.float64)
    t = np.arange(1, T + 1, dtype=float)
    cs_x = np.cumsum(Xf, axis=1); cs_tx = np.cumsum(Xf * t, axis=1)
    cs_t = np.cumsum(t); cs_tt = np.cumsum(t * t)
    flags = np.zeros((n, len(days)), bool)
    for k, d in enumerate(days):
        if d < min_history:
            continue
        m = d
        sx, stx, st, stt = cs_x[:, m - 1], cs_tx[:, m - 1], cs_t[m - 1], cs_tt[m - 1]
        slope = (m * stx - st * sx) / (m * stt - st * st)
        icpt = (sx - slope * st) / m
        cross = (threshold - icpt) / np.where(slope < 0, slope, np.nan)      # 預測向下跨門檻的天
        flags[:, k] = (slope < 0) & np.isfinite(cross) & (cross - d <= horizon)
    return flags


def perm_diff_median(a, b, n_perm, rng):
    """雙尾置換檢定：|中位差| 是否異常（禁止事項 5）。"""
    a = np.asarray(a, float); b = np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return dict(diff=np.nan, p=np.nan, n_a=int(len(a)), n_b=int(len(b)))
    obs = np.median(a) - np.median(b)
    pool = np.concatenate([a, b]); na = len(a); cnt = 0
    for _ in range(n_perm):
        rng.shuffle(pool)
        d = np.median(pool[:na]) - np.median(pool[na:])
        cnt += abs(d) >= abs(obs)
    return dict(diff=float(obs), p=float((cnt + 1) / (n_perm + 1)), n_a=int(na), n_b=int(len(b)))




def run_warning(C, P, win, mode, rng, verbose=False):
    """單一（窗長, 去趨勢）設定的完整模組五。"""
    W = P["warning"]
    X, T, n = C["X"], C["T"], C["n"]
    te, tc, tt, is_flip = C["t_event"], C["t_crit"], C["t_threshold"], C["is_flip"]
    bw = W["gaussian_bw_frac"]["value"]
    days, mlen = _eval_prefix_lengths(T, win, W["eval_every_days"])
    horizon = W["alarm_event_horizon_days"]["value"]

    # 高／低變異分組規則（同一規則用於虛無抽樣與全體）：第一年原始序列 SD 的中位數切
    base_sd = X[:, :min(365, T)].std(axis=1)
    var_group = base_sd > np.median(base_sd)

    TA, TS, G, nullinfo = build_null(X, win, mode, bw, mlen, W["null_subjects"], W["null_perms_per_subject"],
                                     rng, var_group)
    if nullinfo["stratified"]:
        thr = {True: _thresholds(TA[G], TS[G], W["alpha"]), False: _thresholds(TA[~G], TS[~G], W["alpha"])}
    else:
        thr = {True: _thresholds(TA, TS, W["alpha"])}
        thr[False] = thr[True]

    AR, SD = rolling_indicators(X, win, mode, bw)
    ta, ts = _tau_at(AR, SD, mlen)
    pick = lambda key: np.where(var_group[:, None], thr[True][key], thr[False][key])
    # 主要警報（單尾上升）與下降型偏離（獨立計數）
    flags_up = (ta > 0) & (ts > 0) & (ta > pick("ar_hi")) & (ts > pick("sd_hi"))
    flags_dn = (ta < 0) & (ts < 0) & (ta < pick("ar_lo")) & (ts < pick("sd_lo"))
    t_end = np.where(te >= 0, te, T)
    first, false_ct, py = alarms_from_flags(flags_up, days, t_end, te, horizon, T)
    dn_first, dn_ct, _ = alarms_from_flags(flags_dn, days, t_end, te, horizon, T)

    burn = W["trend_min_history_days"]["value"]
    tflags = trend_alarm(X, C["scale"]["threshold_egfr"], days, W["trend_alarm_horizon_days"]["value"], burn)
    tfirst, tfalse, tpy = alarms_from_flags(tflags, days, t_end, te, horizon, T, start_day=burn)

    ev = te >= 0
    res = dict(window=int(win), detrend=mode, n_eval_days=int(len(days)), null=nullinfo,
               baseline_ar1={"flip": float(np.nanmean(AR[is_flip, :max(1, 180 - win)])),
                             "linear": float(np.nanmean(AR[~is_flip, :max(1, 180 - win)]))},
               baseline_sd={"flip": float(np.nanmean(SD[is_flip, :max(1, 180 - win)])),
                            "linear": float(np.nanmean(SD[~is_flip, :max(1, 180 - win)]))},
               by_type={})
    for name, mask in (("flip", is_flip), ("linear", ~is_flip)):
        m_ev = mask & ev
        det = m_ev & (first >= 0) & (first <= te)
        tdet = m_ev & (tfirst >= 0) & (tfirst <= te)
        row = dict(n=int(mask.sum()), n_event=int(m_ev.sum()),
                   frac_any_alarm=float((mask & (first >= 0)).sum() / max(1, mask.sum())),
                   detection_rate=float(det.sum() / max(1, m_ev.sum())),
                   lead_to_event_days=_q((te - first)[det].astype(float)),
                   false_alarms_per_person_year=float(false_ct[mask].sum() / max(py[mask].sum(), 1e-9)),
                   # 下降型偏離（v2 肆）：只計數報告，不計入主要提前期
                   downward=dict(frac_any=float((mask & (dn_first >= 0)).sum() / max(1, mask.sum())),
                                 episodes_per_person_year=float(dn_ct[mask].sum() / max(py[mask].sum(), 1e-9)),
                                 frac_before_event=float(((dn_first >= 0) & (dn_first <= te))[m_ev].mean()) if m_ev.any() else np.nan),
                   trend_detection_rate=float(tdet.sum() / max(1, m_ev.sum())),
                   trend_lead_to_event_days=_q((te - tfirst)[tdet].astype(float)),
                   trend_false_alarms_per_person_year=float(tfalse[mask].sum() / max(tpy[mask].sum(), 1e-9)))
        if name == "flip":
            wc = mask & (tc >= 0) & (first >= 0)
            row["lead_to_crit_days"] = _q((tc - first)[wc].astype(float))         # 可為負（臨界日之後才警報）
            row["frac_alarm_before_crit"] = float(((tc - first)[wc] > 0).mean()) if wc.any() else np.nan
            wt = mask & (tt >= 0) & (first >= 0)
            row["lead_to_threshold_days"] = _q((tt - first)[wt].astype(float))
            wce = mask & ev & (tc >= 0)
            row["event_minus_crit_days"] = _q((te - tc)[wce].astype(float))       # v2 玖.3 兩個量並報
            row["frac_alarm_after_onset"] = float((first[mask & (first >= 0)] >= C["t_onset"][mask & (first >= 0)]).mean()) if (mask & (first >= 0)).any() else np.nan
        res["by_type"][name] = row
    a = (te - first)[is_flip & ev & (first >= 0) & (first <= te)]
    b = (te - first)[~is_flip & ev & (first >= 0) & (first <= te)]
    res["perm_test_lead_flip_vs_linear"] = perm_diff_median(a, b, W["n_permutation_leadtime"], rng)
    ta_ = (te - tfirst)[is_flip & ev & (tfirst >= 0) & (tfirst <= te)]
    tb_ = (te - tfirst)[~is_flip & ev & (tfirst >= 0) & (tfirst <= te)]
    res["perm_test_trend_lead_flip_vs_linear"] = perm_diff_median(ta_, tb_, W["n_permutation_leadtime"], rng)
    both = ev & (first >= 0) & (tfirst >= 0) & (first <= te) & (tfirst <= te)
    res["csd_minus_trend_lead_days"] = {"flip": _q((tfirst - first)[both & is_flip].astype(float)),
                                        "linear": _q((tfirst - first)[both & ~is_flip].astype(float))}
    res["_first_alarm"] = first          # 供圖 1 使用；run.py 寫 JSON 前會移除
    return res
