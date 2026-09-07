# -*- coding: utf-8 -*-
"""M6：切片參考臂與非侵入臂（計畫書 v2 肆之四；v1 H4 交換率）。

切片臂：單次、於 t0（減藥起點）觀測組織活性評分 B = q[x(t0), μ_i, CI, duration] + εB（**不是直接觀測 μ**）；
        參考規則 = B ≥ 閾值 → 判定會發作。閾值與 εB 於校準集內選定，使敏感度／特異度落在 De Rosa 計數推得的
        95% 信賴區間（可信範圍）而非固定單點；範圍不可達時標記留白。
非侵入臂：高頻、高誤差、只看觀測序列 y_obs：以減藥後窗內特徵交叉驗證 logistic 給出發作機率。
        **函式簽章完全不接收 μ_i／B／機制標籤**（品質關卡「機制獨立」以簽章與雜湊為證）。
compare_arms：兩臂各自敏感度／特異度／AUC 與一致性；不合成單一分數。"""
import numpy as np
from scipy.stats import beta as beta_dist
from sklearn.metrics import roc_auc_score

from m4_predict import landmark_features, cv_pred


def clopper_pearson(k, n, alpha=0.05):
    lo = beta_dist.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta_dist.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def biopsy_plausible_range(counts):
    """De Rosa 計數 → 敏感度／特異度 95% CI（Clopper–Pearson）。sens 11/11；spec 由誤分類 8.3%（3/36）反推 22/25。"""
    n_flare = counts["flared"]; n_done = counts["n_completed"]
    n_nonflare = n_done - n_flare
    fp = int(round(counts["misclassification_reported"] * n_done))                 # 3 例誤分類，全在未發作側（敏感度 1.00）
    return dict(sensitivity=clopper_pearson(n_flare, n_flare), specificity=clopper_pearson(n_nonflare - fp, n_nonflare),
                counts=dict(tp=n_flare, fn=0, tn=n_nonflare - fp, fp=fp))


def biopsy_observation(B_true, noise_sd, rng):
    return B_true + (rng.normal(0.0, noise_sd, len(B_true)) if noise_sd > 0 else 0.0)


def biopsy_arm(B_obs, threshold):
    return B_obs >= threshold


def _sens_spec(pred, truth):
    pred = np.asarray(pred, bool); truth = np.asarray(truth, bool)
    sens = float(pred[truth].mean()) if truth.any() else np.nan
    spec = float((~pred[~truth]).mean()) if (~truth).any() else np.nan
    return sens, spec


def calibrate_biopsy(B_true, events, noise_grid, rng, plausible, n_thr=200):
    """校準集：對每個 εB 掃閾值，找 (sens, spec) 同時落在可信範圍者；回傳最接近範圍中點的組合；找不到則 unreachable。"""
    (s_lo, s_hi), (p_lo, p_hi) = plausible["sensitivity"], plausible["specificity"]
    mid = np.array([(s_lo + s_hi) / 2, (p_lo + p_hi) / 2])
    best, table = None, []
    for sd in noise_grid:
        B = biopsy_observation(B_true, sd, rng)
        for thr in np.quantile(B, np.linspace(0.3, 0.98, n_thr)):
            sens, spec = _sens_spec(B >= thr, events)
            table.append((sd, float(thr), sens, spec))
            if s_lo <= sens <= s_hi and p_lo <= spec <= p_hi:
                d = float(np.hypot(sens - mid[0], spec - mid[1]))
                if best is None or d < best["dist"]:
                    best = dict(noise_sd=float(sd), threshold=float(thr), sensitivity=sens, specificity=spec, dist=d)
    if best is None:
        # 留白＋診斷：各 εB 下 spec 最高而 sens ≥ s_lo 的點
        diag = {}
        for sd in noise_grid:
            rows = [r for r in table if r[0] == sd and r[2] >= s_lo]
            diag[str(sd)] = max(rows, key=lambda r: r[3])[2:] if rows else None
        return dict(reachable=False, plausible=plausible, diagnostics=diag)
    best.update(reachable=True, plausible=plausible)
    return best


def noninvasive_features(y_obs, upto, window):
    """只由 y_obs[:, :upto] 建構特徵。簽章裡沒有 μ_i／B／mech，也不接受 cohort dict——機制獨立關卡以此為證。"""
    return landmark_features(y_obs, upto, window)


def noninvasive_arm(y_obs, window, params):
    """params: dict(upto, y, folds, seed)。回傳 CV 發作機率（n,）。"""
    F = noninvasive_features(y_obs, params["upto"], window)
    return cv_pred(F, params["y"], params.get("folds", 5), params.get("seed", 0))


def compare_arms(biopsy_pred, noninv_prob, events, top_share=None, threshold=None):
    events = np.asarray(events, bool)
    if threshold is None:
        k = max(1, int(round((top_share if top_share else events.mean()) * len(noninv_prob))))
        thr = np.sort(noninv_prob)[-k]
    else:
        thr = threshold
    nv_pred = noninv_prob >= thr
    sb, pb = _sens_spec(biopsy_pred, events)
    sn, pn = _sens_spec(nv_pred, events)
    return dict(biopsy=dict(sensitivity=sb, specificity=pb, flagged=float(np.mean(biopsy_pred))),
                noninvasive=dict(sensitivity=sn, specificity=pn, flagged=float(np.mean(nv_pred)),
                                 auc=float(roc_auc_score(events, noninv_prob)) if 0 < events.sum() < len(events) else np.nan),
                agreement=float(np.mean(nv_pred == biopsy_pred)), both_flag=float(np.mean(nv_pred & biopsy_pred)),
                only_biopsy=float(np.mean(biopsy_pred & ~nv_pred)), only_noninvasive=float(np.mean(nv_pred & ~biopsy_pred)),
                conflict_rate=float(np.mean(nv_pred != biopsy_pred)))
