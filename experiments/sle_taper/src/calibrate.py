# -*- coding: utf-8 -*-
"""校準（計畫書 v2 肆之六：校準集與測試集用不同 seed 族；文獻數值用範圍不用單點）。

步驟一：λ0（發作基準風險）——二分使校準集「完全停藥後 24 個月」發作率落在 flare_rate_24m_range
        [WIN-Lupus 0.273, De Rosa 0.306]（目標中點）；檢核 52 週對 Gopal 0.321，記錄偏差。
步驟二：切片參考臂——εB 與閾值於校準集內選定，使敏感度／特異度落在 De Rosa 計數推得的 95% CI；不可達→留白。
步驟三：靜態基線 C 落在 [0.65, 0.75]；不在區間時走 kappa 格點取最接近中點者。
步驟四（於 run.py）：預警閾值於校準集鎖定（m5.lock_threshold），測試集只讀不改。
達成值須於啟動時列印並寫入 assumptions.md。"""
import numpy as np

from m0_params import value
from m1_generator import simulate_cohort as _sim, events_from_U
from m2_risk import static_auc
from m6_arms import biopsy_plausible_range, calibrate_biopsy
from seeding import module_rng


def cal_seed(P, master_seed):
    return int(master_seed) + int(value(P, "seed_families")["calibration"])


def step1_lambda0(P, Cj, master_seed, n_cal=None, iters=16, verbose=True, **sim_kw):
    """回傳 dict(lambda0, rate_24m, rate_52w, target_range, in_range, clamped)。事件由固定 U 反函數抽樣，換 λ0 不換世代。"""
    rng_ = value(P, "flare_rate_24m_range")
    lo_t, hi_t = min(rng_["win_lupus_discontinuation"], rng_["derosa_withdrawal"]), max(rng_["win_lupus_discontinuation"], rng_["derosa_withdrawal"])
    target = 0.5 * (lo_t + hi_t)
    check52 = float(value(P, "flare_rate_52w"))
    beta = value(P, "hazard")["beta_per_x"]
    C = _sim(P, Cj, cal_seed(P, master_seed), n=n_cal, lam0=1e-4, taper=dict(complete=True), **sim_kw)
    X, U, run_in = C["X"], C["U"], C["run_in"]

    def rates(l0):
        te = events_from_U(X, l0, beta, U, t_start=run_in)
        return float(((te >= 0) & (te < run_in + 730)).mean()), float(((te >= 0) & (te < run_in + 364)).mean())

    lo, hi = 1e-7, 1e-1
    r_lo, r_hi = rates(lo)[0], rates(hi)[0]
    clamped = None
    if r_lo > target:
        best, clamped = lo, "lower_bound"
    elif r_hi < target:
        best, clamped = hi, "upper_bound"
    else:
        for _ in range(iters):
            mid = np.sqrt(lo * hi)
            if rates(mid)[0] < target:
                lo = mid
            else:
                hi = mid
        best = np.sqrt(lo * hi)
    r24, r52 = rates(best)
    out = dict(lambda0=float(best), rate_24m=r24, rate_52w=r52, target_range=[lo_t, hi_t], target_mid=target, in_range=bool(lo_t <= r24 <= hi_t),
               check_52w=check52, dev_52w=r52 - check52, clamped=clamped, n_cal=int(C["n"]))
    te = events_from_U(X, best, beta, U, t_start=run_in); e24 = (te >= 0) & (te < run_in + 730)
    out["mech_flare_24m"] = {str(m): float(e24[C["mech"] == m].mean()) for m in np.unique(C["mech"])}
    out["jump_24m"] = float(((C["t_jump"] >= 0) & (C["t_jump"] < run_in + 730)).mean())
    if verbose:
        print(f"[校準一] λ0={best:.3e} → 24 個月發作率 {r24:.3f}（目標範圍 {lo_t}–{hi_t}，{'達標' if out['in_range'] else '未達標'}）；52 週 {r52:.3f} vs Gopal {check52}（偏差 {r52-check52:+.3f}）；跳轉 {out['jump_24m']:.3f}")
    return out


def step2_biopsy(P, Cj, master_seed, lam0, n_cal=None, verbose=True):
    C = _sim(P, Cj, cal_seed(P, master_seed), n=n_cal, lam0=lam0, taper=dict(complete=True))
    plausible = biopsy_plausible_range(value(P, "biopsy_rule_counts"))
    grid = Cj["biopsy_score"]["noise_sd"]["scan"]
    rng = module_rng(master_seed, "arms", 0)
    res = calibrate_biopsy(C["B_true"], C["event_24m"], grid, rng, plausible)
    if verbose:
        if res["reachable"]:
            print(f"[校準二] 切片臂 εB={res['noise_sd']}、閾值 {res['threshold']:.3f} → 敏感度 {res['sensitivity']:.3f}／特異度 {res['specificity']:.3f}（可信範圍 sens {plausible['sensitivity']}, spec {plausible['specificity']}）")
        else:
            print(f"[校準二] 切片臂可信範圍不可達（留白）；診斷 {res['diagnostics']}")
    return res


def step3_static_c(P, Cj, master_seed, lam0, n_cal=None, verbose=True):
    lo, hi = value(P, "static_c_target"); mid = 0.5 * (lo + hi)
    k0 = float(value(P, "kappa"))
    C = _sim(P, Cj, cal_seed(P, master_seed), n=n_cal, lam0=lam0, kappa=k0)
    c0 = static_auc(C, C["event_24m"])
    out = dict(kappa=k0, c=c0, in_range=bool(lo <= c0 <= hi), grid=None)
    if not out["in_range"]:
        grid = {}
        for k in P["kappa"]["scan"]:
            Ck = _sim(P, Cj, cal_seed(P, master_seed), n=n_cal, lam0=lam0, kappa=float(k))
            grid[str(k)] = static_auc(Ck, Ck["event_24m"])
        best = min(grid, key=lambda k: abs(grid[k] - mid))
        out.update(kappa=float(best), c=grid[best], grid=grid, in_range=bool(lo <= grid[best] <= hi), fallback="kappa_grid")
    if verbose:
        print(f"[校準三] κ={out['kappa']} → 靜態 C {out['c']:.3f}（目標 {lo}–{hi}，{'達標' if out['in_range'] else '未達標'}）")
    return out
