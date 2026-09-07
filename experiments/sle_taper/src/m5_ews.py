# -*- coding: utf-8 -*-
"""M5：臨界減速預警（計畫書 v2 伍之一～伍之三）。

指標：以「時間窗」（不規則取樣亦以天數計）滾動計算去趨勢後的 AR(1) 與 SD；
      Kendall τ 為監測起點至當下之前綴趨勢（時間方向：只用 ≤ 當下的觀測）。
聯合警報分數：S(t) = min(τ_AR1(t), τ_SD(t))（或 mean / ar1_only，事前指定）；病人分數 = 監測期間內 S 的累計最大值。
閾值鎖定：在校準集上，以穩定機制病人的真實序列為主要虛無（保留邊際分布、短期自相關、取樣與缺失、個體異質），
          選最小 s* 使假警報數 / 病人年 ≤ 事前指定的負擔；區塊置換、相位隨機化、AR(1) 擬合為敏感度替代虛無。
比較規則（同一負擔下）：水準規則（觀測值累計最大）與趨勢規則（前綴斜率累計最大）——回答「臨界減速是否優於只看數值上升」。
不做任何單序列顯著性檢定（Boettiger 2012 檢察官謬誤：警報以固定假警報負擔定義，不以 p 值定義）。"""
import numpy as np


# ------------------------------------------------------------------ 指標
def _detrend(t, y, method, bw_frac):
    if method == "none" or len(y) < 3:
        return y - y.mean()
    if method == "linear":
        A = np.vstack([t - t.mean(), np.ones_like(t)]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        return y - A @ coef
    if method == "gaussian":
        bw = max(bw_frac * (t.max() - t.min()), 1.0)
        W = np.exp(-0.5 * ((t[:, None] - t[None, :]) / bw) ** 2)
        return y - (W @ y) / W.sum(axis=1)
    raise ValueError(method)


def rolling_indicators(t_obs, y_obs, eval_times, window_days, min_obs, detrend="linear", bw_frac=0.25):
    """單一序列：回傳 (ar1, sd) 各 len(eval_times)，觀測不足處為 NaN。AR(1) 為觀測順序上的 lag-1 相關（不規則取樣下為近似）。"""
    ar1 = np.full(len(eval_times), np.nan); sd = np.full(len(eval_times), np.nan)
    for j, te in enumerate(eval_times):
        m = (t_obs > te - window_days) & (t_obs <= te)
        if m.sum() < min_obs:
            continue
        r = _detrend(t_obs[m].astype(float), y_obs[m].astype(float), detrend, bw_frac)
        if r.std() < 1e-12:
            continue
        ar1[j] = np.corrcoef(r[:-1], r[1:])[0, 1]
        sd[j] = r.std(ddof=1)
    return ar1, sd


def prefix_kendall(v):
    """前綴 Kendall τ（相對時間順序）：τ_k 用前 k 個非 NaN 值；回傳與 v 等長、每點為「至該點為止」的 τ。O(m²) 向量化。"""
    out = np.full(len(v), np.nan)
    idx = np.where(np.isfinite(v))[0]
    if len(idx) < 3:
        return out
    x = v[idx]
    sgn = np.sign(x[None, :] - x[:, None])          # sgn[j,k] = sign(x_k − x_j), j<k 為新加入 k 對舊 j
    tri = np.triu(sgn, 1)
    contrib = tri.sum(axis=0)                        # 每加入第 k 點新增的 concordant−discordant
    S = np.cumsum(contrib)
    k = np.arange(1, len(x) + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        tau = np.where(k >= 3, S / (k * (k - 1) / 2.0), np.nan)
    out[idx] = tau
    return out


def prefix_slope(t, v):
    """前綴最小平方斜率（趨勢規則比較用）。"""
    out = np.full(len(v), np.nan)
    ok = np.isfinite(v)
    if ok.sum() < 3:
        return out
    tt, vv = t[ok].astype(float), v[ok].astype(float)
    n = np.arange(1, len(vv) + 1)
    st, sv, stt, stv = np.cumsum(tt), np.cumsum(vv), np.cumsum(tt * tt), np.cumsum(tt * vv)
    with np.errstate(invalid="ignore", divide="ignore"):
        slope = (n * stv - st * sv) / (n * stt - st ** 2)
    slope[:2] = np.nan
    out[ok] = slope
    return out


def joint_score(tau_ar1, tau_sd, mode="min_tau"):
    if mode == "min_tau":                       # 兩指標皆須上升；任一缺值則該點無分數（NaN 傳遞）
        return np.minimum(tau_ar1, tau_sd)
    if mode == "mean_tau":
        return np.nanmean(np.vstack([tau_ar1, tau_sd]), axis=0)
    if mode == "ar1_only":
        return tau_ar1
    raise ValueError(mode)


def score_cohort(y_obs, run_in, T, window_days, min_obs, detrend, bw_frac, eval_every, joint="min_tau", stop_at=None):
    """對整個世代計算 (n, n_eval) 的聯合分數 S、水準規則分數、趨勢規則分數與 eval_times。
    監測自 t_onset（減藥起點）開始；指標窗可回看至減藥前觀察期。stop_at[i]：該病人分數在此日之後設 NaN（事件／跳轉後不再計）。"""
    n = y_obs.shape[0]
    eval_times = np.arange(run_in, T, eval_every)
    S = np.full((n, len(eval_times)), np.nan); LV = np.full_like(S, np.nan); TR = np.full_like(S, np.nan)
    tgrid = np.arange(T)
    for i in range(n):
        ok = np.isfinite(y_obs[i])
        t_i, y_i = tgrid[ok], y_obs[i][ok]
        ar1, sd = rolling_indicators(t_i, y_i, eval_times, window_days, min_obs, detrend, bw_frac)
        S[i] = joint_score(prefix_kendall(ar1), prefix_kendall(sd), joint)
        # 比較規則：水準（至當下觀測最大值）與趨勢（前綴斜率）
        lv = np.full(len(eval_times), np.nan); tr = np.full(len(eval_times), np.nan)
        for j, te in enumerate(eval_times):
            m = t_i <= te
            if m.sum() >= 3:
                lv[j] = y_i[m][-min(m.sum(), 3):].mean()      # 最近三次觀測平均（減少單點雜訊）
        sl = prefix_slope(t_i[t_i >= run_in - window_days], y_i[t_i >= run_in - window_days])
        tsl = t_i[t_i >= run_in - window_days]
        for j, te in enumerate(eval_times):
            m = tsl <= te
            if m.sum() >= 3:
                tr[j] = sl[m][-1]
        LV[i], TR[i] = lv, tr
        if stop_at is not None and stop_at[i] >= 0:
            S[i, eval_times > stop_at[i]] = np.nan; LV[i, eval_times > stop_at[i]] = np.nan; TR[i, eval_times > stop_at[i]] = np.nan
    return dict(S=S, level=LV, trend=TR, eval_times=eval_times)


def running_max(M):
    return np.fmax.accumulate(np.where(np.isfinite(M), M, -np.inf), axis=1)


# ------------------------------------------------------------------ 閾值鎖定（校準集）
def lock_threshold(scores_cal, null_mask, followup_days, budget_per_py):
    """scores_cal: (n, n_eval) 分數；null_mask: 虛無序列（穩定機制）；followup_days: 各人可計警報的追蹤天數。
    以病人分數（累計最大）計算：選最小 s 使 Σ_null 1[max ≥ s] / Σ_null (followup/365.25) ≤ budget。回傳 s* 與達成的負擔。"""
    M = running_max(scores_cal)[:, -1]
    m = M[null_mask]; py = followup_days[null_mask].sum() / 365.25
    if len(m) == 0 or py <= 0:
        return dict(threshold=None, achieved_fa_per_py=None, n_null=int(null_mask.sum()), note="無虛無序列")
    cand = np.sort(m[np.isfinite(m)])[::-1]
    for k, s in enumerate(cand):                     # 由高到低放寬，最多允許 budget·py 個假警報
        if (k + 1) / py > budget_per_py:
            s_star = cand[k - 1] if k > 0 else np.inf
            break
    else:
        s_star = cand[-1] if len(cand) else np.inf
    fa = float((m >= s_star).sum() / py)
    return dict(threshold=float(s_star), achieved_fa_per_py=fa, n_null=int(null_mask.sum()), null_py=float(py))


def surrogate_threshold(y_obs, run_in, T, cfg, rng, method, null_mask, followup_days, budget, block_len=30):
    """敏感度替代虛無：對虛無序列做區塊置換／相位隨機化／AR(1) 擬合後重算分數再鎖閾。"""
    ys = y_obs[null_mask].copy()
    n = ys.shape[0]
    tgrid = np.arange(T)
    for i in range(n):
        ok = np.isfinite(ys[i]); v = ys[i][ok]
        if len(v) < 6:
            continue
        if method == "block_permutation":
            # 以觀測順序切區塊（block_len 天 ≈ 每人平均間隔換算），置換區塊順序、保留取樣時點
            spacing = max(np.median(np.diff(tgrid[ok])), 1.0)
            b = max(int(round(block_len / spacing)), 2)
            blocks = [v[k:k + b] for k in range(0, len(v), b)]
            perm = rng.permutation(len(blocks))
            v2 = np.concatenate([blocks[p] for p in perm])[:len(v)]
        elif method == "phase_randomization":
            f = np.fft.rfft(v - v.mean()); ph = rng.uniform(0, 2 * np.pi, len(f)); ph[0] = 0
            v2 = np.fft.irfft(np.abs(f) * np.exp(1j * ph), n=len(v)) + v.mean()
        elif method == "ar1_fit":
            r = v - v.mean(); phi = np.corrcoef(r[:-1], r[1:])[0, 1] if r.std() > 0 else 0.0
            e_sd = np.sqrt(max(1 - phi ** 2, 1e-6)) * r.std()
            v2 = np.empty_like(r); v2[0] = r[0]
            for k in range(1, len(r)):
                v2[k] = phi * v2[k - 1] + rng.normal(0, e_sd)
            v2 = v2 + v.mean()
        else:
            raise ValueError(method)
        ys[i][ok] = v2
    sc = score_cohort(ys, run_in, T, cfg["window_days"], cfg["min_obs"], cfg["detrend"], cfg["bw_frac"], cfg["eval_every"], cfg["joint"])
    return lock_threshold(sc["S"], np.ones(n, bool), followup_days[null_mask], budget)


# ------------------------------------------------------------------ 評估（測試集）
def first_alarm(scores, eval_times, threshold):
    """每人首次警報日（累計最大 ≥ s*），無則 −1。"""
    if threshold is None or not np.isfinite(threshold):
        return np.full(scores.shape[0], -1)
    RM = running_max(scores)
    hit = RM >= threshold
    idx = hit.argmax(axis=1)
    return np.where(hit.any(axis=1), eval_times[idx], -1)


def evaluate(cohort, scores, eval_times, threshold, horizon_days=None):
    """主要指標（v2 伍之四）：各機制警報率；分岔翻轉機制之『跳轉前警報』敏感度、提前期（至 t_jump／t_crit）；
    非分岔機制警報率＝機制別假警報率；每病人年假警報負擔（穩定機制）。"""
    n, run_in = cohort["n"], cohort["run_in"]
    T_end = cohort["T"] if horizon_days is None else min(cohort["T"], run_in + horizon_days)
    t_alarm = first_alarm(scores, eval_times, threshold)
    mech = cohort["mech"]; t_jump = cohort["t_jump"]; t_crit = cohort["t_crit"]; t_event = cohort["t_event"]
    out = {}
    for m in np.unique(mech):
        sel = mech == m
        # 停止時點：跳轉或事件（先到者），否則追蹤結束
        stop = np.where(t_jump[sel] >= 0, t_jump[sel], T_end); stop = np.where((t_event[sel] >= 0) & (t_event[sel] < stop), t_event[sel], stop)
        stop = np.minimum(stop, T_end)
        al = t_alarm[sel]
        alarmed_before_stop = (al >= 0) & (al < stop)
        py = (stop - run_in).clip(min=0).sum() / 365.25
        scored = np.isfinite(scores[sel]).any(axis=1)                       # 分數可算者（觀測密度不足時為 False → 只能不警報）
        row = dict(n=int(sel.sum()), alarm_rate=float(alarmed_before_stop.mean()) if sel.sum() else None,
                   fa_per_py=float(alarmed_before_stop.sum() / py) if py > 0 else None, frac_scored=float(scored.mean()) if sel.sum() else None)
        jumped = (t_jump[sel] >= 0) & (t_jump[sel] < T_end)
        if jumped.any():
            lead = np.where(alarmed_before_stop & jumped, t_jump[sel] - al, np.nan)
            row.update(n_jumped=int(jumped.sum()), sens_before_jump=float((alarmed_before_stop & jumped).sum() / jumped.sum()),
                       lead_to_jump_median=float(np.nanmedian(lead)) if np.isfinite(lead).any() else None,
                       lead_to_jump_iqr=[float(np.nanpercentile(lead, 25)), float(np.nanpercentile(lead, 75))] if np.isfinite(lead).any() else None)
            crit_ok = jumped & (t_crit[sel] >= 0)
            if crit_ok.any():
                lead_c = np.where(alarmed_before_stop & crit_ok, t_crit[sel] - al, np.nan)
                row.update(lead_to_crit_median=float(np.nanmedian(lead_c)) if np.isfinite(lead_c).any() else None,
                           alarm_before_crit_frac=float(np.nanmean(lead_c > 0)) if np.isfinite(lead_c).any() else None)
        out[str(m)] = row
    return dict(per_mechanism=out, threshold=threshold, n_alarmed=int((t_alarm >= 0).sum()), t_alarm=t_alarm)


def alarm_auc(scores, cohort, horizon_days):
    """病人層級分數（累計最大；scores 已在跳轉／事件後設 NaN，故為跳轉前分數）對『視窗內是否跳轉』的 AUROC（次要指標）。"""
    from sklearn.metrics import roc_auc_score
    run_in, T = cohort["run_in"], cohort["T"]
    end = min(T, run_in + horizon_days)
    y = (cohort["t_jump"] >= 0) & (cohort["t_jump"] < end)
    M = running_max(scores)[:, -1]
    ok = np.isfinite(M)
    if y[ok].sum() in (0, ok.sum()):
        return None
    return float(roc_auc_score(y[ok], M[ok]))
