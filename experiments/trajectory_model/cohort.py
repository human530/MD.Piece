# -*- coding: utf-8 -*-
"""模組零＋模組一：外部參數檔與合成世代生成器（決定書 v1 + v2 裁決）。

為什麼分成「參數檔 → 檢核 → 校準 → 生成」四步：參數檔讓每一個數字都有出處欄位
（建置提示詞 模組零），檢核把沒有出處或屬假設／校準的列印出來並寫成附錄檔（v2 拾壹），
校準只動決定書明說要反推的量（Δμ 中位：v1 §1；λ0 與 β：v2 參），其餘一律照抄。

共同刻度（v2 捌）：兩型都以 eGFR 為序列單位（低＝差），x0 = 基線 eGFR 自同一分布抽取；
型別由潛在易感度決定、不得由 x0 決定。翻轉型的雙穩態動力學在標準座標 y 上跑（FADE simulate_S0），
再以固定尺度 b（eGFR / y 單位）平移到個案自己的基線：eGFR_i(t) = eGFR0_i − b·(y_i(t) − y_L)。
這樣翻轉的形狀與臨界減速的可偵測性與 x0 無關，不會把 x0 混進 H1／H3。

三個時點並存（v2 參）：t_crit（μ 跨 μc，僅翻轉型）、t_threshold（eGFR 首次 ≤ 門檻，描述性里程碑）、
t_event（風險驅動：λ(t) = λ0·exp(β·(門檻 − eGFR(t))/10)，僅調節不決定）。
漂移起始日（v2 玖）：翻轉型每人抽 t_drift_onset，之前 μ 固定、序列定態。
"""
import json
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from fade_components import MU_C, simulate_S0, apply_S2, window_features

X_FOLD = -1.0 / np.sqrt(3.0)          # 摺疊點座標：dV'/dx=0 → y=-1/sqrt(3)，對應 mu=MU_C
DAYS_PER_YEAR = 365.25


# ------------------------------------------------------------------ 參數檔
def _val(entry):
    """參數列可以是純數值，也可以是 {value, source, derived_from}。"""
    return entry["value"] if isinstance(entry, dict) and "value" in entry else entry


def load_params(path, verbose=True):
    with open(path, encoding="utf-8") as f:
        P = json.load(f)
    missing, derived = [], []

    def walk(node, trail):
        if isinstance(node, dict):
            if "value" in node or "share_raw" in node:
                if not node.get("source"):
                    missing.append(trail)
                if str(node.get("derived_from", "")).startswith(("assumption", "calibrated")):
                    derived.append(dict(param=trail, derived_from=node.get("derived_from"),
                                        value=node.get("value", node.get("share_raw")), source=node.get("source", "")))
            for k, v in node.items():
                if not k.startswith("_"):
                    walk(v, f"{trail}.{k}" if trail else k)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{trail}[{i}]")

    walk(P, "")
    if verbose:
        for m in missing:
            print(f"[警告] 參數列缺少出處：{m}")
        print(provenance_report(missing, derived))
    P["_provenance"] = dict(missing_source=missing, derived=derived)
    return P


def provenance_report(missing, derived):
    """啟動時列印、也寫成檔案（v2 拾壹）：哪些數值不是直接抄自文獻。"""
    lines = ["# 非直接抄錄自文獻的參數（assumption / calibrated_*）", "",
             "| 參數 | 性質 | 現值 | 說明／出處 |", "|---|---|---|---|"]
    for d in derived:
        lines.append(f"| {d['param']} | {d['derived_from']} | {d['value']} | {d['source']} |")
    if missing:
        lines += ["", "缺少出處的列：" + ", ".join(missing)]
    return "\n".join(lines)


# ------------------------------------------------------------------ 刻度與物理量
def lower_state(mu):
    """y^3 - y - mu = 0 的最小實根（mu < MU_C 時為健康穩態）。"""
    r = np.roots([1.0, 0.0, -1.0, -mu])
    return float(np.min(r[np.abs(r.imag) < 1e-9].real))


def derived_scale(P, tau=None):
    """v1 §2/§4/§5：翻轉型參考幾何（健康態 ↔ 90、摺疊點 ↔ 60）給出 b；雜訊、tau_ou、sigma。"""
    sc, fl, nz = P["scale"], P["flip"], P["noise"]
    tau = _val(fl["tau_days"]) if tau is None else tau
    mu0 = _val(fl["mu_start"])
    yL = lower_state(mu0)
    b = (_val(sc["egfr_at_healthy_state"]) - _val(sc["egfr_at_fold"])) / (X_FOLD - yL)   # eGFR / y 單位
    lam0 = 3.0 * yL ** 2 - 1.0                      # 健康態線性化回復率（單位 1/tau）
    sd_egfr = _val(nz["stationary_sd_egfr"])
    tau_ou = _val(nz["tau_ou_days"])
    if tau_ou is None:
        tau_ou = tau / lam0                          # 使兩型 lag-1 自相關 exp(-lam0/tau) 相同
    return dict(b=b, y_healthy=yL, y_fold=X_FOLD, lambda0=lam0, tau=tau, tau_ou=tau_ou,
                sd_egfr=sd_egfr, sd_y=sd_egfr / b,
                sigma_flip=(sd_egfr / b) * np.sqrt(2.0 * lam0 / tau),   # y 單位；線性化 OU 定態變異 = sigma^2 tau/(2 lam0)
                sigma_ou=sd_egfr * np.sqrt(2.0 / tau_ou),               # eGFR 單位；OU 定態變異 = sigma^2 tau_ou/2
                threshold_egfr=_val(sc["event_threshold_egfr"]), egfr_floor=_val(sc["egfr_floor"]))


# ------------------------------------------------------------------ 生成
def _thresholds_for_shares(share, beta):
    """找 theta_k 使 E_s[expit(theta_k - beta*s)] = 累積比例（s ~ N(0,1)，Gauss–Hermite），
    讓易感度進入歸屬後，邊際比例仍等於指定值。"""
    from scipy.optimize import brentq
    z, w = np.polynomial.hermite_e.hermegauss(40); w = w / w.sum()
    cum = np.cumsum(share)[:-1]
    return np.array([brentq(lambda th: np.sum(w * expit(th - beta * z)) - c, -20, 20) for c in cum])


def draw_baseline(P, n, rng, kappa=None):
    """年齡、性別、易感度 s = kappa*u + sqrt(1-kappa^2)*eps。"""
    B = P["baseline"]
    kappa = _val(B["kappa"]) if kappa is None else kappa
    age = np.clip(rng.normal(_val(B["age_mean"]), _val(B["age_sd"]), n), *B["age_range"])
    male = (rng.random(n) < _val(B["male_share"])).astype(int)
    w = B["susceptibility_weights"]
    z_age = (age - _val(B["age_mean"])) / _val(B["age_sd"])
    p = _val(B["male_share"])
    z_male = (male - p) / np.sqrt(p * (1 - p))
    u = (w["age"] * z_age + w["male"] * z_male) / np.hypot(w["age"], w["male"])
    s = kappa * u + np.sqrt(max(0.0, 1 - kappa ** 2)) * rng.normal(0, 1, n)
    return age, male, s


def assign_type(P, s, rng):
    """v2 捌：型別由易感度決定（P(flip|s) = expit(theta + beta_type*s)），邊際比例 = flip_share。"""
    share = _val(P["sim"]["flip_share"]); beta = _val(P["baseline"]["type_link_beta"])
    theta = _thresholds_for_shares(np.array([1 - share, share]), beta)[0]   # P(linear | s) = expit(theta - beta s)
    return rng.random(len(s)) >= expit(theta - beta * s)


def assign_linear_class(P, s, rng):
    """累積 logit：P(class<=k | s) = expit(theta_k - beta*s)，theta 由 O'Hare 比例決定。"""
    C = P["linear_classes"]["classes"]
    share = np.array([c["share_raw"] for c in C]); share = share / share.sum()
    theta = _thresholds_for_shares(share, _val(P["baseline"]["class_link_beta"]))
    cum = expit(theta[None, :] - _val(P["baseline"]["class_link_beta"]) * s[:, None])
    return (rng.random(len(s))[:, None] > cum).sum(axis=1)


def draw_egfr0(P, is_flip, cls, rng, mode=None):
    """v2 捌：兩型 x0 自同一基線分布抽（kdigo_g2_g4_uniform）。ohare_ranges 為示範模式（v2 貳）："""
    n = len(is_flip)
    mode = _val(P["linear_classes"]["linear_start_mode"]) if mode is None else mode
    if mode == "kdigo_g2_g4_uniform":
        lo, hi = P["linear_classes"]["uniform_start_range_egfr"]
        return rng.uniform(lo, hi, n)
    if mode == "ohare_ranges":
        # 回顧性世代（透析前兩年）的各類別起始水準搬到前瞻模擬：線性型用各類別範圍、翻轉型用災難型的 >60
        C = P["linear_classes"]["classes"]
        lo = np.where(is_flip, 60.0, [C[max(k, 0)]["egfr_start_range"][0] for k in cls])
        hi = np.where(is_flip, 90.0, [C[max(k, 0)]["egfr_start_range"][1] for k in cls])
        return rng.uniform(lo, hi)
    raise ValueError(mode)


def simulate_linear(P, sc, egfr0, cls, rng, T, substeps, onset_mode=None):
    """(甲) eGFR = eGFR0 − slope·max(t − t_slope_onset, 0) + eta，eta 為 OU（v1 §4：不得用白雜訊）。
    v2 附錄貳：下降起始日 t_slope_onset 與翻轉型漂移起始日自同一分布抽（之前為定態 OU），
    否則兩型在「早期是否平坦」上系統性不同，H3 的差異無法歸因於臨界減速；
    onset_mode="day0" 為 linear_from_day0 敏感度變體（盛行個案自第 0 天即下降）。"""
    C = P["linear_classes"]["classes"]
    n = len(cls)
    slope_yr = rng.normal([C[k]["slope_mean"] for k in cls], [C[k]["slope_sd"] for k in cls])   # eGFR/年 下降量
    slope = slope_yr / DAYS_PER_YEAR
    mode = _val(P["linear_classes"]["linear_onset_mode"]) if onset_mode is None else onset_mode
    o_lo, o_hi = _val(P["flip"]["drift_onset_range_days"])
    onset = np.zeros(n) if mode == "day0" else rng.uniform(o_lo, o_hi, n)
    dt = 1.0 / substeps
    eta = rng.normal(0, sc["sd_egfr"], n)                       # 由定態分布起始
    X = np.zeros((n, T), dtype=np.float32)
    for t in range(T):
        for _ in range(substeps):
            eta = eta - eta / sc["tau_ou"] * dt + sc["sigma_ou"] * np.sqrt(dt) * rng.normal(0, 1, n)
        X[:, t] = egfr0 - slope * np.maximum(t + 1 - onset, 0.0) + eta
    return X, slope_yr, onset


def build_mu_path(P, mu0, dmu, T, rng):
    """v2 玖：每人抽漂移起始日與漂移期長度；起始日前 mu 固定，之後線性升到 mu_end 後保持。"""
    n = len(dmu)
    d_lo, d_hi = _val(P["flip"]["drift_duration_range_days"])
    dur = rng.uniform(d_lo, d_hi, n)
    o_lo, o_hi = _val(P["flip"]["drift_onset_range_days"])
    onset = rng.uniform(o_lo, np.minimum(o_hi, T - dur), n)
    t = np.arange(T)[None, :]
    frac = np.clip((t - onset[:, None]) / dur[:, None], 0.0, 1.0)
    return mu0 + dmu[:, None] * frac, onset, dur


def simulate_flip(P, sc, egfr0, s, rng, T, substeps, delta_mu_median):
    """(乙) 雙穩態 SDE 在標準座標 y 上跑（複用 FADE simulate_S0，加 tau 與 mu_path），
    再平移到個案基線：eGFR = eGFR0 − b·(y − y_L)。mu_end 由易感度決定（v1 §3）。"""
    n = len(s)
    mu0 = _val(P["flip"]["mu_start"])
    dmu = delta_mu_median * np.exp(_val(P["flip"]["delta_mu_log_spread"]) * s)
    mu_path, onset, dur = build_mu_path(P, mu0, dmu, T, rng)
    y0 = sc["y_healthy"] + rng.normal(0, sc["sd_y"], n)
    Y, t_crit, _, _ = simulate_S0(n=n, T=T, substeps=substeps, tau=sc["tau"], mu_start=mu0,
                                  sigma=np.full(n, sc["sigma_flip"]), x0=y0, rng=rng, mu_path=mu_path)
    X = egfr0[:, None] - sc["b"] * (Y - sc["y_healthy"])
    return X.astype(np.float32), mu0 + dmu, t_crit, onset, dur, Y


def first_below(X, thr):
    """每列首次 <= thr 的索引；未跨過回傳 -1。"""
    hit = X <= thr
    idx = hit.argmax(axis=1)
    return np.where(hit.any(axis=1), idx, -1)


def hazard_events(X, lam0, beta, thr, U, h_max, floor=0.0):
    """v2 參：每日風險 λ(t) = λ0·exp(β·(門檻 − max(eGFR, floor))/10)（封頂 h_max），x 只調節不決定。
    風險在 eGFR 封底處飽和（序列本身不截斷：翻轉後低於 0 的值是抽象活動度，截斷會讓 SD 歸零、
    製造假的「下降型偏離」）。逆變換法：t_event = 首日 累積風險 >= −ln U（U 每人抽一次，校準時重複使用）。"""
    lam = np.minimum(lam0 * np.exp(beta * (thr - np.maximum(X.astype(np.float64), floor)) / 10.0), h_max)
    H = np.cumsum(lam, axis=1)
    target = -np.log(U)[:, None]
    hit = H >= target
    idx = hit.argmax(axis=1)
    return np.where(hit.any(axis=1), idx, -1)


def make_cohort(P, seed, n=None, tau=None, delta_mu_median=None, lam0=None, beta=None,
                dropout=False, start_mode=None, kappa=None, linear_onset=None):
    """產生一個世代。回傳 dict（全部 numpy 陣列＋衍生刻度）。"""
    S = P["sim"]
    n = S["n"] if n is None else n
    T, sub = S["T_days"], S["substeps"]
    sc = derived_scale(P, tau)
    dmu = _val(P["flip"]["delta_mu_median"]) if delta_mu_median is None else delta_mu_median
    lam0 = _val(P["hazard"]["lambda0_per_day"]) if lam0 is None else lam0
    beta = _val(P["hazard"]["beta_per_10_egfr"]) if beta is None else beta
    if dmu is None or lam0 is None or beta is None:
        raise ValueError("delta_mu_median / lambda0 / beta 尚未校準（先呼叫 calibrate_*）")
    rng = np.random.default_rng(seed)

    age, male, s = draw_baseline(P, n, rng, kappa)
    is_flip = assign_type(P, s, rng)
    cls = np.full(n, -1)
    li, fi = np.where(~is_flip)[0], np.where(is_flip)[0]
    if len(li):
        cls[li] = assign_linear_class(P, s[li], rng)
    egfr0 = draw_egfr0(P, is_flip, cls, rng, start_mode)
    U = rng.random(n)                                            # 事件實現用的均勻亂數（每人一個）

    X = np.zeros((n, T), dtype=np.float32)
    slope_yr = np.full(n, np.nan); mu_end = np.full(n, np.nan); t_crit = np.full(n, -1)
    t_onset = np.full(n, np.nan); drift_dur = np.full(n, np.nan)
    if len(li):
        X[li], slope_yr[li], t_onset[li] = simulate_linear(P, sc, egfr0[li], cls[li], rng, T, sub, linear_onset)
    if len(fi):
        X[fi], mu_end[fi], t_crit[fi], t_onset[fi], drift_dur[fi], _ = simulate_flip(P, sc, egfr0[fi], s[fi], rng, T, sub, dmu)

    t_thr = first_below(X, sc["threshold_egfr"])
    t_event = hazard_events(X, lam0, beta, sc["threshold_egfr"], U, _val(P["hazard"]["h_max_per_day"]), sc["egfr_floor"])
    C = dict(X=X, is_flip=is_flip, cls=cls, age=age, male=male, s=s, x0=egfr0, egfr0=egfr0, U=U,
             slope_egfr=slope_yr, mu_end=mu_end, t_crit=t_crit, t_threshold=t_thr, t_event=t_event,
             t_onset=t_onset, drift_dur=drift_dur, event=t_event >= 0, T=T, n=n, seed=seed,
             delta_mu_median=dmu, lam0=lam0, beta=beta, kappa=_val(P["baseline"]["kappa"]) if kappa is None else kappa, scale=sc)
    if dropout:
        # v1 §6：複用 FADE apply_S2 的退出邏輯（fill_rate=1、無延遲 → 只剩退出）。
        # 退出 = 停止記錄（觀測值變 NaN），結果事件仍已知（FADE 語意：S0 真實病理仍在）。
        X_obs, active = apply_S2(X, 1.0, _val(P["prediction"]["dropout_hazard"]), 0.0, 0, rng)
        C["X_obs"] = X_obs
        C["drop_day"] = np.where(active.all(axis=1), -1, (~active).argmax(axis=1))
    return C


def with_events(C, lam0, beta, P):
    """同一世代（同 X、同 U）換一組 (λ0, β) 重算事件——校準時用，不必重生成序列。"""
    C = dict(C)
    C["t_event"] = hazard_events(C["X"], lam0, beta, C["scale"]["threshold_egfr"], C["U"], _val(P["hazard"]["h_max_per_day"]), C["scale"]["egfr_floor"])
    C["event"] = C["t_event"] >= 0; C["lam0"] = lam0; C["beta"] = beta
    return C


# ------------------------------------------------------------------ 檢核與摘要
def flip_timing_summary(C):
    """翻轉型時程（v1 §1/§2、v2 參）：t_threshold − t_crit（機制參考點 → 門檻里程碑）、
    fall = t_threshold − 最後一次高於摺疊點對應 eGFR 的日期（快速下墜的實際歷時）、t_event − t_crit。"""
    f = C["is_flip"]
    out = dict(n_flip=int(f.sum()), frac_flip_crit=float((C["t_crit"][f] >= 0).mean()) if f.any() else np.nan,
               frac_flip_threshold=float((C["t_threshold"][f] >= 0).mean()) if f.any() else np.nan,
               frac_flip_event=float(C["event"][f].mean()) if f.any() else np.nan)

    def q(v):
        v = np.asarray(v, float); v = v[np.isfinite(v)]
        if len(v) < 5:
            return dict(n=int(len(v)), median=np.nan, q25=np.nan, q75=np.nan)
        return dict(n=int(len(v)), median=float(np.median(v)), q25=float(np.percentile(v, 25)), q75=float(np.percentile(v, 75)))
    idx = np.where(f & (C["t_crit"] >= 0))[0]
    tc, tt, te = C["t_crit"][idx], C["t_threshold"][idx], C["t_event"][idx]
    ok = tt >= 0
    out["threshold_minus_crit_days"] = q((tt - tc)[ok])
    out["event_minus_crit_days"] = q((te - tc)[te >= 0])
    out["onset_days"] = q(C["t_onset"][f]); out["drift_duration_days"] = q(C["drift_dur"][f])
    # v2 附錄貳：兩型定態期長度分布無系統性差異（中位、四分位距、KS p）
    from scipy.stats import ks_2samp
    lo = C["t_onset"][~f]
    out["linear_onset_days"] = q(lo)
    out["onset_ks_p_flip_vs_linear"] = float(ks_2samp(C["t_onset"][f], lo).pvalue) if (f.sum() > 10 and (~f).sum() > 10 and np.std(lo) > 0) else None
    return out


def stratum_type_mix(is_flip, strata):
    """v2 捌檢核：每層兩型比例；回傳 (各層翻轉型比例, 單一型別最大佔比)。"""
    shares = [float(is_flip[strata == s].mean()) if (strata == s).any() else np.nan for s in range(strata.max() + 1)]
    worst = max(max(p, 1 - p) for p in shares if np.isfinite(p))
    return shares, worst


def type_separability(C, days=180, folds=5, pre_onset_only=False):
    """v2 玖／附錄閘門三：機制啟動前翻轉型與線性型是否難以區分——用前 `days` 天的 [level, trend, ar1, sd]
    以 logistic 交叉驗證分辨型別的 AUC（≈0.5 = 難以區分）。pre_onset_only=True 時只納入起始日 >= days 者
    （兩型起始日同分布，此篩選對稱），即「僅取定態期資料」。"""
    idx = np.arange(C["n"])
    if pre_onset_only:
        idx = idx[np.nan_to_num(C["t_onset"], nan=np.inf) >= days]
    F = np.array([window_features(C["X"][i].astype(float), 0, days) for i in idx])
    F = np.where(np.isnan(F), np.nanmedian(F, axis=0), F)
    y = C["is_flip"][idx].astype(int)
    if len(y) < 50 or y.min() == y.max():
        return np.nan
    p = cross_val_predict(LogisticRegression(max_iter=1000), F, y, cv=StratifiedKFold(folds, shuffle=True, random_state=0),
                          method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


def static_auc(C):
    """基線特徵 → 事件的 in-sample AUC（只用於校準）。"""
    Xb = np.c_[C["age"], C["male"], C["x0"]]
    if C["event"].all() or not C["event"].any():
        return np.nan
    p = LogisticRegression(max_iter=1000).fit(Xb, C["event"]).predict_proba(Xb)[:, 1]
    return roc_auc_score(C["event"], p)


# ------------------------------------------------------------------ 校準（v1 §1、v2 參）
def calibrate_delta_mu(P, tau=None, target_days=None, seed=None, n_pilot=800, iters=14, verbose=True):
    """二分法找 Δμ 中位數，使翻轉型（參考幾何：基線 90）median(t_threshold − t_crit) = target。
    先掃格點：延遲對漂移速率不一定單調（雜訊誘發的提早逃逸會讓 t_threshold 早於 t_crit），
    只有目標被相鄰兩格夾住時才二分；否則標 unreachable（v2 壹：不用替代值填）。"""
    S = P["sim"]
    target = _val(P["flip"]["flip_time_target_days"]) if target_days is None else target_days
    seed = (S["seed"] + 1000) if seed is None else seed
    sc = derived_scale(P, tau)
    lo_b, hi_b = np.log(P["flip"]["delta_mu_bounds"])

    def timing(logm):
        rng = np.random.default_rng(seed)
        s = rng.normal(0, 1, n_pilot)
        egfr0 = np.full(n_pilot, float(_val(P["scale"]["egfr_at_healthy_state"])))
        X, _, tc, _, _, _ = simulate_flip(P, sc, egfr0, s, rng, S["T_days"], S["substeps"], np.exp(logm))
        tt = first_below(X, sc["threshold_egfr"])
        ok = (tc >= 0) & (tt >= 0)
        med = float(np.median(tt[ok] - tc[ok])) if ok.sum() >= 20 else np.nan
        return med, float(ok.mean())

    grid = np.linspace(lo_b, hi_b, 13)
    curve = [(float(np.exp(g)),) + timing(g) for g in grid]
    meds = np.array([c[1] for c in curve])
    best, feasible = None, False
    for i in range(len(grid) - 1):
        m0, m1 = meds[i], meds[i + 1]
        if np.isfinite(m0) and np.isfinite(m1) and (m0 - target) * (m1 - target) <= 0:
            lo, hi = grid[i], grid[i + 1]
            for _ in range(iters):
                mid = 0.5 * (lo + hi)
                gm, _ = timing(mid)
                if not np.isfinite(gm):
                    break
                if (gm - target) * (m0 - target) > 0:
                    lo = mid
                else:
                    hi = mid
            best, feasible = 0.5 * (lo + hi), True
            break
    out = dict(target_days=float(target), tau=sc["tau"], feasible=feasible,
               scan=[dict(delta_mu_median=m, median_threshold_minus_crit_days=d, frac_crossing=f) for m, d, f in curve])
    if feasible:
        med, frac = timing(best)
        out.update(delta_mu_median=float(np.exp(best)), achieved_median_days=med, frac_crossing=frac,
                   derived_from="calibrated_to_catastrophic_class")
        if verbose:
            print(f"[校準 Δμ] tau={sc['tau']:g} 目標 {target:g} 天 → Δμ 中位 {np.exp(best):.3f}，"
                  f"實得中位(t_threshold−t_crit) {med:.0f} 天，跨臨界比例 {frac:.2f}")
    else:
        out.update(delta_mu_median=None, achieved_median_days=None, frac_crossing=None, derived_from="unreachable")
        if verbose:
            reach = [c[1] for c in curve if np.isfinite(c[1])]
            print(f"[校準 Δμ] tau={sc['tau']:g} 目標 {target:g} 天 → unreachable（掃描可達中位 "
                  f"{min(reach) if reach else float('nan'):.0f}–{max(reach) if reach else float('nan'):.0f} 天），此格留白")
    return out


def calibrate_hazard(P, delta_mu_median, tau=None, seed=None, n_pilot=1500, iters=12, verbose=True, start_mode=None, linear_onset=None, kappa_fixed=None):
    """v2 參：λ0 校準使五年事件率落在目標區間中點；β 固定為文獻量級的假設值（hazard.beta_per_10_egfr），
    因為靜態 C 對 β 並非單調（在中等 β 附近最高、極端 β 反而下降），無法用二分法求；
    C 若不在目標區間，改在 kappa 格點（基線↔易感度相關）上挑最接近中點者。
    序列 X 與 U 只生成一次，(λ0) 換算事件不需重生成。"""
    S, Hz, B = P["sim"], P["hazard"], P["baseline"]
    seed = (S["seed"] + 1000) if seed is None else seed
    er_lo, er_hi = _val(Hz["event_rate_target"]); er_mid = 0.5 * (er_lo + er_hi)
    c_lo, c_hi = _val(B["static_c_target"]); c_mid = 0.5 * (c_lo + c_hi)
    beta = _val(Hz["beta_per_10_egfr"])
    l_lo, l_hi = np.log(Hz["lambda0_bounds_per_day"])

    def fit(kappa):
        C0 = make_cohort(P, seed, n=n_pilot, tau=tau, delta_mu_median=delta_mu_median, lam0=1e-4, beta=beta,
                         start_mode=start_mode, kappa=kappa, linear_onset=linear_onset)
        # 二分前確認目標事件率被 λ0 上下界夾住（codex 稽核）
        er_at = [with_events(C0, np.exp(bb), beta, P)["event"].mean() for bb in (l_lo, l_hi)]
        bracketed = bool(er_at[0] <= er_mid <= er_at[1])
        lo, hi = l_lo, l_hi
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            if with_events(C0, np.exp(mid), beta, P)["event"].mean() < er_mid:
                lo = mid
            else:
                hi = mid
        lam = float(np.exp(0.5 * (lo + hi))); Cb = with_events(C0, lam, beta, P)
        return dict(kappa=kappa, lambda0_per_day=lam, pilot_event_rate=float(Cb["event"].mean()), pilot_static_auc=float(static_auc(Cb)),
                    lambda0_bracketed=bracketed)

    k0 = _val(B["kappa"]) if kappa_fixed is None else kappa_fixed
    best = fit(k0); tried = [best]
    if kappa_fixed is None and not (c_lo <= best["pilot_static_auc"] <= c_hi):
        for k in _val(B["kappa_grid"]):
            if k != k0:
                tried.append(fit(k))
        best = min(tried, key=lambda r: abs(r["pilot_static_auc"] - c_mid))
    out = dict(beta_per_10_egfr=float(beta), hazard_ratio_per_10_egfr=float(np.exp(beta)), **best,
               event_rate_target=[er_lo, er_hi], static_c_target=[c_lo, c_hi],
               in_target=bool(c_lo <= best["pilot_static_auc"] <= c_hi and er_lo <= best["pilot_event_rate"] <= er_hi and best["lambda0_bracketed"]),
               kappa_tried=tried)
    if verbose:
        print(f"[校準 λ0] β={beta:.3f}/10 eGFR（HR {np.exp(beta):.2f}，固定）、kappa={best['kappa']}、λ0={best['lambda0_per_day']:.2e}/天 → "
              f"pilot 事件率 {best['pilot_event_rate']:.3f}、靜態 C {best['pilot_static_auc']:.3f}（目標 {er_lo}–{er_hi} / {c_lo}–{c_hi}）"
              + ("" if out["in_target"] else "  ※ 未落在目標區間"))
    return out
