# -*- coding: utf-8 -*-
"""M4：靜態對動態預測（H2）與其最小版（G2 用）。

地標 L（自 t_onset 起算）：風險集 = 前 t_onset+L 天未發生事件者；特徵 = 基線 + 由 x_obs[:, :t_onset+L] 算出的
軌跡特徵（整段斜率／曲率、最近 lookback 的 [level, trend, ar1, sd]、觀測數）；結局 = 之後至 T 的事件。
只往回看；分群標籤不進來；μ_intrinsic 不進來。C 指數 = 該地標風險集上二元結局的 AUC。"""
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

from m2_risk import baseline_matrix, static_cv_pred


def window_features(series, lo, hi):
    """[level, trend, ar1, sd]（與 FADE / CKD 版相同定義：窗內去趨勢後 corrcoef(d[:-1], d[1:]) 與 std）。"""
    v = series[lo:hi]
    m = ~np.isnan(v)
    if m.sum() < 6:
        return np.array([np.nan] * 4)
    idx = np.where(m)[0].astype(float); val = v[m]
    level = val[-min(6, len(val)):].mean()
    A = np.vstack([idx, np.ones_like(idx)]).T
    coef, *_ = np.linalg.lstsq(A, val, rcond=None)
    d = val - A @ coef
    sd = d.std()
    ar1 = np.corrcoef(d[:-1], d[1:])[0, 1] if len(d) > 3 and sd > 1e-9 else np.nan
    return np.array([level, coef[0], ar1, sd])


def landmark_features(x_obs, upto, lookback):
    """只用 x_obs[:, :upto]。回傳 n × 7：[slope_all, curv_all, level_lb, trend_lb, ar1_lb, sd_lb, n_obs]。"""
    H = x_obs[:, :upto].astype(float)
    n = H.shape[0]
    out = np.full((n, 7), np.nan)
    t = np.arange(upto, dtype=float) / max(upto, 1)
    A = np.c_[np.ones(upto), t, t ** 2]
    for i in range(n):
        m = ~np.isnan(H[i])
        if m.sum() >= 3:
            c = np.linalg.lstsq(A[m], H[i, m], rcond=None)[0]
            out[i, 0], out[i, 1] = c[1], c[2]
        out[i, 2:6] = window_features(H[i], max(0, upto - lookback), upto)
    out[:, 6] = (~np.isnan(H)).sum(axis=1)
    return out


def cv_pred(F, y, folds, seed):
    y = np.asarray(y).astype(int)
    if y.min() == y.max() or min(np.bincount(y)) < folds:
        return np.full(len(y), y.mean())
    cv = StratifiedKFold(folds, shuffle=True, random_state=int(seed))
    model = make_pipeline(SimpleImputer(strategy="median"), LogisticRegression(max_iter=2000))
    return cross_val_predict(model, F, y, cv=cv, method="predict_proba")[:, 1]


def _auc(y, p):
    y = np.asarray(y).astype(int)
    return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else np.nan


def _high(p, rule, top, thr):
    if rule == "abs":
        return p >= thr
    k = max(1, int(round(top * len(p))))
    return p >= np.sort(p)[-k]


def reclassification(y, hs, hd):
    y = np.asarray(y).astype(bool)
    up, down = (~hs) & hd, hs & (~hd)
    nri_e = (up[y].mean() - down[y].mean()) if y.any() else np.nan
    nri_n = (down[~y].mean() - up[~y].mean()) if (~y).any() else np.nan
    return dict(moved_in=float(up.mean()), moved_out=float(down.mean()), nri_event=float(nri_e), nri_nonevent=float(nri_n),
                nri=float(nri_e + nri_n), static_high=float(hs.mean()), dynamic_high=float(hd.mean()))


def secondary_metrics(y, p, thresholds):
    """次要指標（v2 伍之五）：AUROC、Brier、校準斜率／截距（logit 迴歸）、決策曲線淨效益、NNT（每找出一例事件需標記幾人）。"""
    from sklearn.linear_model import LogisticRegression
    y = np.asarray(y).astype(int); p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    out = dict(auc=_auc(y, p), brier=float(brier_score_loss(y, p)), n=int(len(y)), event_rate=float(y.mean()))
    if 0 < y.sum() < len(y):
        lg = np.log(p / (1 - p))
        m = LogisticRegression(C=1e6, max_iter=2000).fit(lg[:, None], y)
        out["calibration_slope"] = float(m.coef_[0][0]); out["calibration_intercept"] = float(m.intercept_[0])
        rate = y.mean(); dc = {}
        for thr in thresholds:
            flag = p >= thr
            tp = float((flag & (y == 1)).mean()); fp = float((flag & (y == 0)).mean())
            nb = tp - fp * thr / (1 - thr)
            nb_all = rate - (1 - rate) * thr / (1 - thr)
            nnt = float(flag.sum() / max((flag & (y == 1)).sum(), 1)) if flag.any() else None
            dc[str(thr)] = dict(net_benefit=float(nb), net_benefit_treat_all=float(nb_all), flagged=float(flag.mean()), nnt_flag_per_event=nnt)
        out["decision_curve"] = dc
    return out


def run_predict(cohort, x_obs, landmarks, folds=5, seed=0, lookback=90, top=0.2, timing=True, timing_step=30, secondary_thresholds=None):
    """靜態（基線，配適一次）vs 地標動態。回傳 dict。timing=False 為最小版；secondary_thresholds 給定時附次要指標（Brier／校準／決策曲線／NNT）。"""
    te, T, onset = cohort["t_event"], cohort["T"], cohort["t_onset"]
    y_all = te >= 0
    p_static = static_cv_pred(cohort, y_all, folds, seed)
    B = baseline_matrix(cohort)
    rate_all = float(y_all.mean())
    hs = {"abs": _high(p_static, "abs", top, rate_all), "rel": _high(p_static, "rel", top, rate_all)}
    res = dict(static=dict(auc=_auc(y_all, p_static), brier=float(brier_score_loss(y_all.astype(int), p_static)), event_rate=rate_all),
               landmarks={})

    def fit_L(L):
        upto = onset + L
        risk = (te < 0) | (te >= upto)
        yL = te[risk] >= 0
        F = np.c_[B[risk], landmark_features(x_obs[risk], upto, lookback)]
        return risk, yL, cv_pred(F, yL, folds, seed + L), cv_pred(B[risk], yL, folds, seed + L)

    for L in landmarks:
        risk, yL, pd, ps_refit = fit_L(L)
        ps = p_static[risk]
        rate_L = float(yL.mean()) if len(yL) else np.nan
        row = dict(n_risk=int(risk.sum()), event_rate=rate_L, auc_static=_auc(yL, ps), auc_static_refit=_auc(yL, ps_refit), auc_dynamic=_auc(yL, pd),
                   brier_static=float(brier_score_loss(yL.astype(int), ps)) if len(yL) else np.nan,
                   brier_dynamic=float(brier_score_loss(yL.astype(int), pd)) if len(yL) else np.nan)
        row["c_gain"] = row["auc_dynamic"] - row["auc_static"]
        row["c_gain_vs_refit"] = row["auc_dynamic"] - row["auc_static_refit"]
        for rule in ("abs", "rel"):
            hd = _high(pd, rule, top, rate_L)
            row[f"reclass_{rule}"] = reclassification(yL, hs[rule][risk], hd)
            row[f"reclass_same_riskset_{rule}"] = reclassification(yL, _high(ps, rule, top, rate_L), hd)
        if secondary_thresholds and len(yL):
            row["secondary_static"] = secondary_metrics(yL, ps, secondary_thresholds)
            row["secondary_dynamic"] = secondary_metrics(yL, pd, secondary_thresholds)
        res["landmarks"][str(L)] = row
    if not timing:
        return res
    # 時點位移：每 timing_step 天一個地標，回傳整個陣列（指示：不得只回摘要統計）
    Ls = list(range(timing_step, T - onset - timing_step + 1, timing_step))
    first = {r: np.full(cohort["n"], -1) for r in ("abs", "rel")}
    for L in Ls:
        risk, yL, pd, _ = fit_L(L)
        rate_L = float(yL.mean()) if len(yL) else np.nan
        idx = np.where(risk)[0]
        for rule in ("abs", "rel"):
            hd = _high(pd, rule, top, rate_L)
            new = idx[hd & (first[rule][idx] < 0)]
            first[rule][new] = L
    res["timing"] = {rule: dict(landmarks=Ls, first_dynamic_flag_day=first[rule].tolist(), static_high=hs[rule].tolist(),
                                shift_days=[int(f) if (h and f >= 0) else None for f, h in zip(first[rule], hs[rule])])
                     for rule in ("abs", "rel")}
    return res
