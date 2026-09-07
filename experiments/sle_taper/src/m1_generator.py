# -*- coding: utf-8 -*-
"""M1：合成世代生成器 v2（計畫書 v2 肆之一～肆之五）。

潛在動力學（分岔翻轉主模型）：
    dx = (1/τx)[−x³ + x + μ(t)] dt + s·ξ(t) dt，   dξ = −(1/τξ) ξ dt + √(2/τξ) dW（單位變異 OU）
    μ(t) = μ_i − g_eff(t)
Euler–Maruyama 每日 10 子步。σξ 定義為緩解態 x 波動的定態 SD（x 單位），OU 力振幅 s 由線性化反推。

六種競爭機制（每個情境都含，v2 肆之二）——主分析不靠分群「發現」它們，機制標籤已知：
    bifurcation              μ_i − g_end > μ_c：減藥途中緩慢跨越分岔 → 臨界減速的正對照
    stochastic_escape        μ_i − g_end 略低於 μ_c：不達分岔點，但能障小、由雜訊促成跳轉
    continuous_deterioration 平均水準線性上升、局部回復率不變（OU 繞移動目標）
    exogenous_shock          μ_i 低；隨機時點一段 μ 脈衝（感染等）引發突然惡化，無可預警窗口
    noise_amplification      μ_i 低；雜訊振幅隨時間線性放大、穩定性不變
    stable                   μ_i 低、無趨勢（主要虛無序列）

減藥與暴露（v2 肆之五）：g_eff = 一階滯後(處方排程 × 依從)；依從 = 個人係數 × 隨機中斷。
觀測層（v2 肆之三）：y = h(x)+ε 於取樣日；規則／不規則／症狀驅動取樣、缺失、檢測下限、治療悖論（觀測異常→加藥）。
切片參考臂（v2 肆之四）：B = q[x(t0), μ_i, CI, duration] + εB（不是直接觀測 μ）。
事件：風險驅動 λ(t)=λ0·exp(β·x)（工作規則沿用），跳轉里程碑 t_jump（x 跨 0）另記。
觀測層與生成層分開：observe() 不改 X。"""
import numpy as np

from m0_params import value

MECHS = ("bifurcation", "stochastic_escape", "continuous_deterioration", "exogenous_shock", "noise_amplification", "stable")


def lower_state(mu):
    """y³ − y − μ = 0 的最小實根（μ < μ_c 時為緩解穩態）。"""
    r = np.roots([1.0, 0.0, -1.0, -mu])
    return float(np.min(r[np.abs(r.imag) < 1e-9].real))


def ou_unit(n, T, tau_xi, rng, substeps=10):
    """單位變異 OU：dξ = −ξ/τξ dt + √(2/τξ) dW；回傳每日值 (n, T)。"""
    dt = 1.0 / substeps
    xi = rng.normal(0.0, 1.0, n)
    out = np.empty((n, T), dtype=np.float32)
    coef = np.sqrt(2.0 / tau_xi) * np.sqrt(dt)
    for t in range(T):
        for _ in range(substeps):
            xi = xi - xi / tau_xi * dt + coef * rng.normal(0.0, 1.0, n)
        out[:, t] = xi
    return out


def prescribed_schedule(t, g0, onset, duration_days, complete=True, residual_fraction=0.5):
    """處方排程 g_rx(t)：自 onset 起線性遞減至 g_end。"""
    t = np.asarray(t, dtype=float)
    g_end = 0.0 if complete else g0 * residual_fraction
    frac = np.clip((t - onset) / max(duration_days, 1), 0.0, 1.0)
    return g0 + (g_end - g0) * frac


def adherence_paths(n, T, spec, rng):
    """個人依從係數 × 隨機中斷（每月機率 lapse_prob，持續 lapse_days，其間暴露為 0）。回傳 (n, T) 乘數。"""
    a = np.clip(rng.normal(spec["mean"], spec["sd"], n), 0.3, 1.0)
    A = np.repeat(a[:, None], T, axis=1)
    p_day = spec["lapse_prob_per_month"] / 30.0
    lapses = rng.random((n, T)) < p_day
    for i, t in zip(*np.where(lapses)):
        A[i, t:t + spec["lapse_days"]] = 0.0
    return A


def effective_exposure(g_rx, lag_days):
    """一階藥效滯後：g_eff[t] = g_eff[t−1] + (g_rx[t] − g_eff[t−1]) / lag。"""
    if lag_days <= 1:
        return g_rx.copy()
    g = np.empty_like(g_rx)
    g[:, 0] = g_rx[:, 0]
    k = 1.0 / lag_days
    for t in range(1, g_rx.shape[1]):
        g[:, t] = g[:, t - 1] + (g_rx[:, t] - g[:, t - 1]) * k
    return g


def _draw_baseline(Cj, n, rng):
    out = {}
    for k, row in Cj["baseline_features"].items():
        d = row["dist"]
        if d == "lognormal":
            v = np.exp(np.log(row["median"]) + row["sigma_log"] * rng.normal(0, 1, n))
        elif d == "normal":
            v = rng.normal(row["mean"], row["sd"], n)
        elif d == "poisson":
            v = rng.poisson(row["lam"], n).astype(float)
        else:
            raise ValueError(d)
        out[k] = np.clip(v, *row["clip"])
    return out


def _susceptibility(Cj, feats):
    L = Cj["susceptibility_link"]
    u = np.zeros(len(next(iter(feats.values()))))
    for k, w in L["weights"].items():
        v = np.log(feats[k]) if k in L.get("log_transform", []) else feats[k]
        u += w * (v - v.mean()) / (v.std() + 1e-12)
    return (u - u.mean()) / (u.std() + 1e-12)


def hazard_events(X, lam0, beta, U, t_start=0):
    """λ(t)=λ0·exp(β·x)；反函數抽樣，U 固定於個體（換 λ0/β 不換 U）。回傳事件日或 −1。"""
    lam = lam0 * np.exp(beta * X.astype(np.float64))
    lam[:, :t_start] = 0.0
    H = np.cumsum(lam, axis=1)
    hit = H >= (-np.log(U))[:, None]
    idx = hit.argmax(axis=1)
    return np.where(hit.any(axis=1), idx, -1)


def events_from_U(X, lam0, beta, U, t_start=0):
    return hazard_events(X, lam0, beta, U, t_start)


def first_cross(X, thr, start=0):
    hit = X[:, start:] >= thr
    idx = hit.argmax(axis=1) + start
    return np.where(hit.any(axis=1), idx, -1)


def simulate_cohort(P, Cj, master_seed, n=None, tau_x=None, tau_xi=None, sigma_xi=None, mech_shares=None,
                    taper=None, lam0=None, beta=None, kappa=None, pkpd_lag=None, substeps=10):
    """產生一個 Monte Carlo 資料集。回傳 dict（生成層 X、機制標籤、時點、基線特徵、切片評分材料、設定）。"""
    from seeding import module_rng
    n = int(value(P, "N")) if n is None else n
    tau_x = float(value(P, "tau_x")) if tau_x is None else float(tau_x)
    tau_xi = float(value(P, "tau_xi")) if tau_xi is None else float(tau_xi)
    sigma_xi = float(value(P, "sigma_xi")) if sigma_xi is None else float(sigma_xi)
    run_in, T = int(value(P, "run_in_days")), int(value(P, "T"))
    T_total = run_in + T
    mu_c = float(value(P, "mu_c"))
    shares = dict(value(P, "mechanisms") if mech_shares is None else mech_shares)
    kappa = float(value(P, "kappa")) if kappa is None else float(kappa)
    hz = value(P, "hazard")
    lam0 = hz["lambda0_per_day"] if lam0 is None else lam0
    beta = hz["beta_per_x"] if beta is None else beta
    if lam0 is None:
        raise ValueError("λ0 尚未校準（於校準集校準使 24 個月發作率落在文獻範圍）")
    tp = value(P, "taper_schedule"); taper = taper or {}
    g0 = float(value(P, "g0"))
    dur = taper.get("duration_days", tp["duration_days"]); complete = taper.get("complete", tp["complete"])
    residual = taper.get("residual_fraction", 0.5)
    pkpd_lag = float(value(P, "pkpd_lag_days")) if pkpd_lag is None else float(pkpd_lag)

    rng_b = module_rng(master_seed, "generator", 0)   # 基線特徵
    rng_x = module_rng(master_seed, "generator", 1)   # 動力學雜訊
    rng_e = module_rng(master_seed, "generator", 2)   # 事件 U
    rng_m = module_rng(master_seed, "generator", 3)   # 機制、μ、依從、衝擊

    feats = _draw_baseline(Cj, n, rng_b)
    s = _susceptibility(Cj, feats)
    # 機制指派：基線特徵只透過「機制歸屬機率」與結局相關（κ 傾斜：易感度高者較可能落在分岔／越障／連續惡化，較不可能穩定），
    # μ_i 之後純由機制決定——基線特徵不改動任何機制內的動力學。
    names = list(MECHS); base = np.array([shares[m] for m in names], float); base /= base.sum()
    tilt = np.array([1.0 if m in ("bifurcation", "stochastic_escape", "continuous_deterioration") else (-1.0 if m == "stable" else 0.0) for m in names])
    W = base[None, :] * np.exp(kappa * s[:, None] * tilt[None, :]); W /= W.sum(axis=1, keepdims=True)
    u = rng_m.random(n)
    mech = np.array(names)[(np.cumsum(W, axis=1) < u[:, None]).sum(axis=1).clip(max=len(names) - 1)]
    g_end = 0.0 if complete else g0 * residual
    lo_b, hi_b = value(P, "mu_bifurcation_margin"); lo_e, hi_e = value(P, "mu_escape_margin"); low = value(P, "mu_low_dist")
    mu_i = np.empty(n)
    nb = int((mech == "bifurcation").sum()); ne = int((mech == "stochastic_escape").sum())
    mu_i[mech == "bifurcation"] = mu_c + g_end + rng_m.uniform(lo_b, hi_b, nb)
    mu_i[mech == "stochastic_escape"] = mu_c + g_end - rng_m.uniform(lo_e, hi_e, ne)
    rest = ~np.isin(mech, ["bifurcation", "stochastic_escape"])
    mu_i[rest] = np.minimum(rng_m.normal(low["mean"], low["sd"], int(rest.sum())), mu_c + g_end - hi_e)   # 遠低於 μ_c
    mu_i = np.minimum(mu_i, g0 + mu_c - 0.05)                                  # 全劑量下必在緩解側

    t = np.arange(T_total)
    g_rx = prescribed_schedule(t, g0, run_in, dur, complete, residual)
    adh = adherence_paths(n, T_total, value(P, "adherence"), rng_m)
    g_eff = effective_exposure(np.repeat(g_rx[None, :], n, axis=0) * adh, pkpd_lag)
    mu_path = mu_i[:, None] - g_eff
    sh = value(P, "shock"); is_shock = mech == "exogenous_shock"
    t_shock = np.full(n, -1)
    if is_shock.any():
        t_shock[is_shock] = run_in + rng_m.integers(sh["onset_range_after_taper"][0], sh["onset_range_after_taper"][1], int(is_shock.sum()))
        for i in np.where(is_shock)[0]:
            mu_path[i, t_shock[i]:t_shock[i] + sh["duration_days"]] += sh["mu_pulse"]

    xL0 = np.array([lower_state(m) for m in np.round(mu_path[:, 0], 3)])
    a_ref = float(np.median(3.0 * xL0 ** 2 - 1.0)) / tau_x                    # 緩解態線性化回復率
    s_amp = sigma_xi / np.sqrt(tau_xi / (a_ref * (1.0 + a_ref * tau_xi)))     # Var(x)=s²τξ/(a(1+aτξ))
    xi = ou_unit(n, T_total, tau_xi, rng_x, substeps)
    amp = np.full((n, T_total), s_amp, dtype=np.float32)
    na = value(P, "noise_amplification")["factor_end"]; is_na = mech == "noise_amplification"
    if is_na.any():
        ramp = 1.0 + (na - 1.0) * np.clip((t - run_in) / max(T, 1), 0, 1)
        amp[is_na] = (s_amp * ramp)[None, :]
    is_cd = mech == "continuous_deterioration"
    slope_ph = value(P, "traj_slopes")
    cd_slope = np.where(is_cd, rng_m.normal(slope_ph["continuous_deterioration_x_per_year"], slope_ph["sd"], n) / 365.25, 0.0)
    lam_lin = 3.0 * xL0 ** 2 - 1.0

    x = xL0 + rng_x.normal(0, sigma_xi * 0.5, n)
    X = np.empty((n, T_total), dtype=np.float32)
    dt = 1.0 / substeps
    for k in range(T_total):
        mu_k = mu_path[:, k]
        target = xL0 + cd_slope * max(k - run_in, 0)
        force = amp[:, k] * xi[:, k]
        for _ in range(substeps):
            drift = np.where(is_cd, lam_lin * (target - x) / tau_x, (-(x ** 3) + x + mu_k) / tau_x)
            x = x + (drift + force) * dt
        X[:, k] = x

    cross = mu_path >= mu_c
    t_crit = np.where(cross.any(axis=1), cross.argmax(axis=1), -1)
    t_crit[is_cd | is_na] = -1
    x_thr = value(P, "flare_definition_threshold")["x_threshold"]
    t_jump = first_cross(X, x_thr, start=run_in)
    U = rng_e.random(n)
    t_event = hazard_events(X, lam0, beta, U, t_start=run_in)
    bw = Cj["biopsy_score"]["weights"]
    x0 = X[:, run_in - 1].astype(float)
    B_true = bw["mu"] * mu_i + bw["x0"] * (x0 - xL0) + bw["chronicity_index"] * feats["chronicity_index"] + bw["disease_duration_yr"] * feats["disease_duration_yr"]
    return dict(X=X, mu_i=mu_i, mu_path=mu_path.astype(np.float32), g_eff=g_eff.astype(np.float32), g_rx=g_rx.astype(np.float32),
                mech=mech, s=s, features=feats, t_onset=run_in, t_crit=t_crit, t_jump=t_jump, t_event=t_event, t_shock=t_shock, U=U,
                B_true=B_true, x0=x0, xL0=xL0, T=T_total, run_in=run_in, n=n, event=t_event >= 0,
                event_24m=(t_event >= 0) & (t_event < run_in + 730), event_52w=(t_event >= 0) & (t_event < run_in + 364),
                settings=dict(tau_x=tau_x, tau_xi=tau_xi, sigma_xi=sigma_xi, s_amp=float(s_amp), shares=shares, kappa=kappa,
                              lam0=lam0, beta=beta, taper=dict(duration_days=dur, complete=complete), pkpd_lag=pkpd_lag, master_seed=master_seed))


# ------------------------------------------------------------------ 觀測層
def _marker(X, indicator, P):
    m = value(P, "marker_map")
    if indicator == "x":
        return X.astype(np.float64)
    if indicator == "upcr":                                          # 對數線性映射＋檢測下限；分析用對數尺度
        y = np.exp(m["upcr"]["log_intercept"] + m["upcr"]["log_slope"] * X.astype(np.float64))
        return np.log(np.maximum(y, m["upcr"]["detection_limit"]))
    if indicator == "egfr":
        return m["egfr"]["intercept"] + m["egfr"]["slope"] * X.astype(np.float64)
    raise ValueError(indicator)


def observe(cohort, P, scenario, rng, interval_days=None, error_sd=None):
    """觀測層：回傳 y_obs（n, T），非取樣日為 NaN；以個人減藥前平均置中（x 情境再 −1 對齊緩解態）。
    情境鍵：indicator / interval / meas_error / irregular / symptom_driven。"""
    X, run_in, T = cohort["X"], cohort["run_in"], cohort["T"]
    n = X.shape[0]
    interval = interval_days if interval_days is not None else value(P, "sampling_intervals")[scenario["interval"]]
    err = error_sd if error_sd is not None else value(P, "meas_error_levels")[scenario["meas_error"]]
    y = _marker(X, scenario["indicator"], P)
    y = y - y[:, :run_in].mean(axis=1, keepdims=True) - (1.0 if scenario["indicator"] == "x" else 0.0)
    if err > 0:
        y = y + rng.normal(0.0, err, y.shape)
    if scenario.get("irregular"):
        irr = value(P, "irregular_sampling")
        mask = np.zeros((n, T), bool)
        for i in range(n):
            tt = 0
            while tt < T:
                mask[i, tt] = rng.random() > irr["missing_prob"]
                tt += max(1, int(round(interval * (1 + irr["jitter_frac"] * rng.uniform(-1, 1)))))
    else:
        row = np.zeros(T, bool); row[::max(int(interval), 1)] = True
        mask = np.repeat(row[None, :], n, axis=0)
    if scenario.get("symptom_driven"):                               # x 上升時額外回診（非隨機取樣）
        gain = value(P, "symptom_driven_sampling")["gain"]
        extra_p = np.clip(gain * np.maximum(X.astype(np.float64) + 1.0, 0) / interval, 0, 1)
        mask |= rng.random((n, T)) < extra_p
    return np.where(mask, y, np.nan).astype(np.float32)


def treatment_intervention(cohort, P, y_obs, master_seed):
    """治療悖論情境：觀測值超過門檻 → delay 天後加藥（g_eff += g_boost），以同一雜訊串流重生成之後的序列與事件。
    回傳反事實 cohort（含 t_intervention）；只用於情境比較，主分析不用。"""
    from seeding import module_rng
    ti = value(P, "treatment_intervention")
    n, T, run_in = cohort["n"], cohort["T"], cohort["run_in"]
    st = cohort["settings"]
    t_int = np.full(n, -1)
    for i in range(n):
        idx = np.where((y_obs[i] > ti["y_threshold"]) & (np.arange(T) >= run_in))[0]
        if len(idx):
            t_int[i] = idx[0] + ti["delay_days"]
    g = cohort["g_eff"].astype(np.float64).copy()
    for i in np.where(t_int >= 0)[0]:
        g[i, t_int[i]:] += ti["g_boost"]
    C2 = dict(cohort)
    C2["g_eff"] = g.astype(np.float32); C2["t_intervention"] = t_int
    C2["mu_path"] = (cohort["mu_i"][:, None] - g).astype(np.float32)
    rng_x = module_rng(master_seed, "generator", 1)
    xi = ou_unit(n, T, st["tau_xi"], rng_x)
    x = cohort["xL0"] + rng_x.normal(0, st["sigma_xi"] * 0.5, n)
    X = np.empty((n, T), dtype=np.float32); dt = 0.1
    for k in range(T):
        mu_k = C2["mu_path"][:, k]
        for _ in range(10):
            x = x + ((-(x ** 3) + x + mu_k) / st["tau_x"] + st["s_amp"] * xi[:, k]) * dt
        X[:, k] = x
    C2["X"] = X
    C2["t_jump"] = first_cross(X, value(P, "flare_definition_threshold")["x_threshold"], start=run_in)
    C2["t_event"] = hazard_events(X, st["lam0"], st["beta"], cohort["U"], t_start=run_in)
    C2["event"] = C2["t_event"] >= 0
    C2["event_24m"] = C2["event"] & (C2["t_event"] < run_in + 730)
    return C2
