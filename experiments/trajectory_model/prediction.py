# -*- coding: utf-8 -*-
"""模組四：靜態對動態預測（H2）。

資料流（每一步都只往回看，這是本模組唯一重要的事）：
  靜態：基線特徵 (age, male, x0) → 5 折交叉驗證預測 P(T 內事件)，配適一次。
  動態：地標 L 時，風險集 = 前 L 天未發生事件者；特徵 = 基線 + 由 X[:, :L]（含 L 之前）算出的
        軌跡特徵（整段斜率／曲率、最近 lookback 天的 [level, trend, ar1, sd]、觀測數）；
        結局 = (L, T] 內事件；同樣 5 折交叉驗證。
  模組三的分群標籤不進來（整條序列算的，含未來）。
比較：C 指數（=AUC，主分析僅行政刪失，決定書 §6）、Brier；淨重新分類（絕對＋相對兩種閾值，
     決定書 §7）；建議介入時點位移（每 90 天一個地標，決定書 §8）。
"""
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

from fade_components import window_features
from risk import baseline_matrix


def cv_pred(F, y, folds, seed):
    y = np.asarray(y).astype(int)
    if y.min() == y.max() or min(np.bincount(y)) < folds:
        return np.full(len(y), y.mean(), dtype=float)
    # 缺值補中位數放在 Pipeline 內：補值統計只由訓練摺估計（codex 稽核：全體先補值會讓驗證摺參與補值參數）
    cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
    model = make_pipeline(SimpleImputer(strategy="median"), LogisticRegression(max_iter=2000))
    return cross_val_predict(model, F, y, cv=cv, method="predict_proba")[:, 1]


def landmark_features(Xobs, L, lookback):
    """只用 Xobs[:, :L]。回傳 n × 7：[slope_all, curv_all, level_lb, trend_lb, ar1_lb, sd_lb, n_obs]。"""
    H = Xobs[:, :L].astype(float)
    n = len(H)
    out = np.full((n, 7), np.nan)
    t = np.arange(L, dtype=float) / L
    A = np.c_[np.ones(L), t, t ** 2]
    pinvA = np.linalg.pinv(A)
    full = ~np.isnan(H).any(axis=1)
    if full.any():                                                    # 無缺值列一次算完
        c = H[full] @ pinvA.T
        out[full, 0] = c[:, 1]; out[full, 1] = c[:, 2]
    for i in np.where(~full)[0]:                                      # 有缺值（退出）逐列
        m = ~np.isnan(H[i])
        if m.sum() >= 3:
            c = np.linalg.lstsq(A[m], H[i, m], rcond=None)[0]
            out[i, 0] = c[1]; out[i, 1] = c[2]
    lo = max(0, L - lookback)
    for i in range(n):
        out[i, 2:6] = window_features(H[i], lo, L)                    # 複用 FADE：[level, trend, ar1, sd]
    out[:, 6] = (~np.isnan(H)).sum(axis=1)
    return out


def _auc(y, p):
    y = np.asarray(y).astype(int)
    return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else np.nan


def _brier(y, p):
    return float(brier_score_loss(np.asarray(y).astype(int), p))


def _high(p, rule, top_share, abs_thr):
    if rule == "abs":
        return p >= abs_thr
    k = max(1, int(round(top_share * len(p))))
    return p >= np.sort(p)[-k]


def reclassification(y, hs, hd):
    """hs/hd：靜態／動態高風險布林。回傳移入、移出比例與 NRI（事件、非事件分開）。"""
    y = np.asarray(y).astype(bool)
    up, down = (~hs) & hd, hs & (~hd)
    ev, ne = y, ~y
    nri_e = (up[ev].mean() - down[ev].mean()) if ev.any() else np.nan
    nri_n = (down[ne].mean() - up[ne].mean()) if ne.any() else np.nan
    return dict(moved_in=float(up.mean()), moved_out=float(down.mean()),
                nri_event=float(nri_e), nri_nonevent=float(nri_n), nri=float(nri_e + nri_n),
                static_high=float(hs.mean()), dynamic_high=float(hd.mean()))


def run_prediction(C, P, seed, dropout=False, timing=True):
    Q = P["prediction"]
    T, n = C["T"], C["n"]
    Xobs = C["X_obs"] if dropout else C["X"]
    y_all = C["event"]
    te = C["t_event"]
    B = baseline_matrix(C)
    p_static = cv_pred(B, y_all, Q["cv_folds"], seed)
    top = P["prediction"]["high_risk_top_share"]["value"]
    rate_all = float(y_all.mean())
    hs = {"abs": _high(p_static, "abs", top, rate_all), "rel": _high(p_static, "rel", top, rate_all)}
    res = dict(static=dict(auc=_auc(y_all, p_static), brier=_brier(y_all, p_static), event_rate=rate_all,
                           high_abs=float(hs["abs"].mean()), high_rel=float(hs["rel"].mean())),
               landmarks={}, dropout=bool(dropout))

    def fit_landmark(L, refit_static=False):
        risk = (te < 0) | (te >= L)
        yL = (te[risk] >= 0)
        F = np.c_[B[risk], landmark_features(Xobs[risk], L, Q["lookback_days"])]
        pd = cv_pred(F, yL, Q["cv_folds"], seed + L)
        if not refit_static:
            return risk, yL, pd
        # 公平比較用的對照（codex 稽核）：同一風險集、同一結局 (L,T]、只用基線特徵重新配適
        ps_refit = cv_pred(B[risk], yL, Q["cv_folds"], seed + L)
        return risk, yL, pd, ps_refit

    for L in Q["landmarks_days"]:
        risk, yL, pd, ps_refit = fit_landmark(L, refit_static=True)
        ps = p_static[risk]
        rate_L = float(yL.mean()) if len(yL) else np.nan
        row = dict(n_risk=int(risk.sum()), event_rate=rate_L,
                   # 規格：靜態只配適一次（五年結局），在地標風險集上評估排序；auc 是五年二元結局的 AUC（二元 C 統計量，非 Harrell C）
                   auc_static=_auc(yL, ps), auc_dynamic=_auc(yL, pd),
                   brier_static=_brier(yL, ps), brier_dynamic=_brier(yL, pd),
                   # 同 estimand 對照：基線特徵在同一風險集、同一 (L,T] 結局重新配適
                   auc_static_refit=_auc(yL, ps_refit), brier_static_refit=_brier(yL, ps_refit))
        row["c_gain"] = row["auc_dynamic"] - row["auc_static"]
        row["c_gain_vs_refit"] = row["auc_dynamic"] - row["auc_static_refit"]
        row["leak_alert"] = bool(row["c_gain"] > Q["leak_alert_c_gain"]["value"])
        for rule in ("abs", "rel"):
            hd = _high(pd, rule, top, rate_L)
            # (a) 名單更替：靜態＝基線時的分類（規格語意：加入軌跡後被移入／移出高風險組）
            row[f"reclass_{rule}"] = reclassification(yL, hs[rule][risk], hd)
            # (b) 標準 NRI：兩模型在同一風險集、同一門檻下分類（abs=風險集事件率；rel=風險集內前 20%）
            hs_same = _high(ps, rule, top, rate_L)
            row[f"reclass_same_riskset_{rule}"] = reclassification(yL, hs_same, hd)
        res["landmarks"][str(L)] = row

    if not timing:
        return res
    # ---- 建議介入時點位移：每 90 天一個地標（決定書 §8），靜態只在第 0 天判定一次 ----
    step = Q["timing_landmark_step_days"]
    Ls = list(range(step, T - step + 1, step))
    first_dyn = {r: np.full(n, -1) for r in ("abs", "rel")}
    for L in Ls:
        risk, yL, pd = fit_landmark(L)
        rate_L = float(yL.mean()) if len(yL) else np.nan
        idx = np.where(risk)[0]
        for rule in ("abs", "rel"):
            hd = _high(pd, rule, top, rate_L)
            new = idx[hd & (first_dyn[rule][idx] < 0)]
            first_dyn[rule][new] = L
    timing = {}
    for rule in ("abs", "rel"):
        fd, s_high = first_dyn[rule], hs[rule]
        both = s_high & (fd >= 0); dyn_only = (~s_high) & (fd >= 0); stat_only = s_high & (fd < 0)
        hist_L = lambda mask: [int(((fd == L) & mask).sum()) for L in Ls]
        ev = y_all
        timing[rule] = dict(
            landmarks=Ls,
            n_both=int(both.sum()), n_dynamic_only=int(dyn_only.sum()), n_static_only=int(stat_only.sum()),
            n_neither=int(((~s_high) & (fd < 0)).sum()),
            # 位移 = 動態首次判定日 − 靜態判定日（=0）；只對兩者都判定者定義
            shift_days_both=dict(median=float(np.median(fd[both])) if both.any() else np.nan,
                                 q25=float(np.percentile(fd[both], 25)) if both.any() else np.nan,
                                 q75=float(np.percentile(fd[both], 75)) if both.any() else np.nan,
                                 hist=hist_L(both)),
            first_flag_hist_dynamic_only=hist_L(dyn_only),
            # 距事件的提前量：事件者中，判定日到事件日
            lead_static_days=dict(median=float(np.median(te[s_high & ev])) if (s_high & ev).any() else np.nan),
            lead_dynamic_days=dict(median=float(np.median((te - fd)[(fd >= 0) & ev])) if ((fd >= 0) & ev).any() else np.nan,
                                   q25=float(np.percentile((te - fd)[(fd >= 0) & ev], 25)) if ((fd >= 0) & ev).any() else np.nan,
                                   q75=float(np.percentile((te - fd)[(fd >= 0) & ev], 75)) if ((fd >= 0) & ev).any() else np.nan),
            event_rate_dynamic_flagged=float(ev[fd >= 0].mean()) if (fd >= 0).any() else np.nan,
            event_rate_static_flagged=float(ev[s_high].mean()) if s_high.any() else np.nan,
        )
    res["timing"] = timing
    return res
