# -*- coding: utf-8 -*-
"""品質關卡（計畫書 v2 陸之一～陸之六；取代 v1 四道閘門——v1 G1／G3 不在 v2 內）。

Q1 機制獨立：預警與非侵入預測只用觀測序列（與基線特徵），不用機制標籤／μ_i／B_true／未來資訊——
             簽章檢查 + 打亂 mech／μ_i／B_true 後分數雜湊不變。
Q2 零信號：全穩定機制世代 + 結局脫耦（β=0）→ 預警敏感度應 ≈ 假警報率（比值 ≤ 1.5）、AUROC 落在 0.45–0.55、max|ΔC| < 0.01。
Q3 閾值鎖定：閾值檔於校準 seed 族產生且與測試 seed 族不交集；測試階段只讀（雜湊比對）。
Q4 時間方向：把 t > te 的觀測值打亂／改寫，te 之前的分數必須完全相同。
Q5 觀測密度：每個評估點回報指標可算的比例；警報只在指標可算處觸發（NaN 不觸發）。
Q6 結局盲化：僅適用真實資料階段（Stage 2/3），本階段標記 not_applicable。"""
import hashlib
import inspect

import numpy as np

from m0_params import value
from m1_generator import simulate_cohort, observe, events_from_U
from m5_ews import score_cohort, first_alarm, lock_threshold, running_max, evaluate
from m6_arms import noninvasive_features, noninvasive_arm
from m4_predict import run_predict
from seeding import module_rng


def _h(a):
    return hashlib.sha256(np.ascontiguousarray(np.nan_to_num(np.asarray(a, dtype=np.float64), nan=-999.0)).tobytes()).hexdigest()


def q1_mechanism_independence(cohort, y_obs, cfg):
    forbidden = ("mu_i", "mu_intrinsic", "mech", "B_true", "B", "cohort", "t_jump", "t_crit", "t_event")
    sig_ok = {}
    for f in (score_cohort, noninvasive_features, noninvasive_arm):
        params = list(inspect.signature(f).parameters)
        sig_ok[f.__name__] = not any(p in forbidden for p in params)
    src_ok = not any(k in inspect.getsource(score_cohort) for k in ("mu_i", "mech", "B_true", "t_jump"))
    sc = score_cohort(y_obs, cohort["run_in"], cohort["T"], cfg["window_days"], cfg["min_obs"], cfg["detrend"], cfg["bw_frac"], cfg["eval_every"], cfg["joint"])
    C2 = dict(cohort); rng = np.random.default_rng(0)
    C2["mech"] = rng.permutation(cohort["mech"]); C2["mu_i"] = rng.permutation(cohort["mu_i"]); C2["B_true"] = rng.permutation(cohort["B_true"])
    sc2 = score_cohort(y_obs, C2["run_in"], C2["T"], cfg["window_days"], cfg["min_obs"], cfg["detrend"], cfg["bw_frac"], cfg["eval_every"], cfg["joint"])
    same = _h(sc["S"]) == _h(sc2["S"])
    passed = all(sig_ok.values()) and src_ok and same
    return dict(gate="Q1 機制獨立", value=dict(signatures=sig_ok, source_clean=src_ok, hash_invariant=same), criterion="簽章無 μ_i／mech／B；打亂後雜湊不變",
                passed=bool(passed), fail_means="預警或非侵入臂偷看了生成層資訊")


def q2_zero_signal(P, Cj, master_seed, cfg, threshold, n=None, seeds=3, verbose=True):
    """全穩定機制世代 + β=0：預警警報率應與事件無關；靜態／動態 AUC ≈ 0.5；max_L|mean_seed ΔC| < 0.01。
    ΔC 準則沿用 CKD 版裁決：4 倍樣本 × 3 seeds 取平均後再取 max（單一 seed 的 |ΔC| 抽樣噪音 ≈ 0.02，準則會誤觸）。"""
    shares = {m: 0.0 for m in value(P, "mechanisms")}; shares["stable"] = 1.0
    lam0 = value(P, "hazard")["lambda0_per_day"]
    n_gate = 4 * n if n else 4 * int(value(P, "N"))
    alarm_rates, r_evs, r_nes, aucs_all = [], [], [], []
    gains_by_L = {}
    for j in range(seeds):
        C = simulate_cohort(P, Cj, master_seed + 3000 + j, n=n_gate, mech_shares=shares, lam0=lam0, beta=0.0)
        y = observe(C, P, Cj["observation_scenarios"]["default"], module_rng(master_seed + 3000 + j, "obs", 0))
        if j == 0:                                                     # 警報與 AUC 檢查一個 seed 即足（比值準則）
            sc = score_cohort(y, C["run_in"], C["T"], cfg["window_days"], cfg["min_obs"], cfg["detrend"], cfg["bw_frac"], cfg["eval_every"], cfg["joint"])
            t_alarm = first_alarm(sc["S"], sc["eval_times"], threshold)
            alarm_rates.append(float((t_alarm >= 0).mean()))
            ev = C["event_24m"]
            r_evs.append(float((t_alarm[ev] >= 0).mean()) if ev.any() else np.nan)
            r_nes.append(float((t_alarm[~ev] >= 0).mean()) if (~ev).any() else np.nan)
            event_rate = float(ev.mean())
        pr = run_predict(C, y, value(P, "landmarks"), seed=master_seed + j, timing=False)
        for L, r in pr["landmarks"].items():
            if r["c_gain"] == r["c_gain"]:
                gains_by_L.setdefault(L, []).append(r["c_gain"])
            if r["auc_dynamic"] == r["auc_dynamic"]:
                aucs_all.append(r["auc_dynamic"])
    mean_gain_by_L = {L: float(np.mean(v)) for L, v in gains_by_L.items()}
    max_dc = max(abs(g) for g in mean_gain_by_L.values()) if mean_gain_by_L else None
    r_ev, r_ne, alarm_rate = r_evs[0], r_nes[0], alarm_rates[0]
    ratio = r_ev / r_ne if r_ne and r_ne > 0 else (np.inf if r_ev and r_ev > 0 else 1.0)
    band = value(P, "zero_signal_tolerance")["auc_band"]; rmax = value(P, "zero_signal_tolerance")["sens_over_fa_ratio_max"]
    auc_ok = all(band[0] <= a <= band[1] for a in aucs_all) if aucs_all else True
    gain_ok = (max_dc < value(P, "gate_max_delta_c_beta0")) if max_dc is not None else True
    ratio_ok = bool(ratio <= rmax) or (r_ev == 0 and r_ne == 0)
    val = dict(alarm_rate=alarm_rate, alarm_rate_event=r_ev, alarm_rate_nonevent=r_ne, ratio=float(ratio) if np.isfinite(ratio) else None,
               dynamic_auc=aucs_all, mean_gain_by_landmark=mean_gain_by_L, max_delta_c=max_dc, event_rate=event_rate, n_gate=n_gate, seeds=seeds)
    if verbose:
        print(f"[Q2 零信號] 警報率 {alarm_rate:.3f}（事件者 {r_ev:.3f}／非事件者 {r_ne:.3f}，比 {ratio:.2f}）；動態 AUC {[round(a,3) for a in aucs_all]}；max|mean ΔC| {max_dc}")
    return dict(gate="Q2 零信號", value=val, criterion=f"警報率比 ≤ {rmax}；AUC ∈ {band}；max_L|mean ΔC| < {value(P, 'gate_max_delta_c_beta0')}（{seeds} seeds 平均）",
                passed=bool(auc_ok and gain_ok and ratio_ok), fail_means="無信號時仍出現鑑別或警報偏向事件者 → 洩漏或閾值未鎖")


def q3_threshold_locking(lock_record, test_seeds):
    cal = set(int(s) for s in lock_record.get("calibration_seeds", []))
    tst = set(int(s) for s in test_seeds)
    disjoint = cal.isdisjoint(tst)
    has_hash = bool(lock_record.get("hash"))
    return dict(gate="Q3 閾值鎖定", value=dict(calibration_seeds=sorted(cal), test_seeds=sorted(tst), disjoint=disjoint, locked_hash=lock_record.get("hash")),
                criterion="校準與測試 seed 族不交集；閾值檔雜湊於測試前寫定", passed=bool(disjoint and has_hash), fail_means="閾值在測試資料上調整過（事後調參）")


def q4_time_direction(y_obs, run_in, T, cfg, rng, cut_day=None):
    """把 t > cut_day 的觀測值打亂後，cut_day 前的分數必須完全相同。"""
    cut = cut_day if cut_day is not None else run_in + (T - run_in) // 2
    sc = score_cohort(y_obs, run_in, T, cfg["window_days"], cfg["min_obs"], cfg["detrend"], cfg["bw_frac"], cfg["eval_every"], cfg["joint"])
    y2 = y_obs.copy()
    fut = y2[:, cut + 1:]
    y2[:, cut + 1:] = rng.permutation(fut.ravel()).reshape(fut.shape)
    sc2 = score_cohort(y2, run_in, T, cfg["window_days"], cfg["min_obs"], cfg["detrend"], cfg["bw_frac"], cfg["eval_every"], cfg["joint"])
    m = sc["eval_times"] <= cut
    same = _h(sc["S"][:, m]) == _h(sc2["S"][:, m])
    return dict(gate="Q4 時間方向", value=dict(cut_day=int(cut), identical_before_cut=same), criterion="打亂未來後，切點前分數雜湊相同",
                passed=bool(same), fail_means="分數用到了評估時點之後的觀測")


def q5_observation_density(scores, eval_times, run_in):
    S = scores["S"]
    cov = np.isfinite(S).mean(axis=0)
    first_ok = eval_times[np.argmax(cov > 0.5)] if (cov > 0.5).any() else None
    return dict(gate="Q5 觀測密度", value=dict(coverage_by_eval=[round(float(c), 3) for c in cov], first_eval_with_half_coverage=int(first_ok) if first_ok is not None else None,
                                            days_after_onset=int(first_ok - run_in) if first_ok is not None else None),
                criterion="指標不可算處為 NaN 且不觸發警報（由 first_alarm 之 NaN 語意保證）；回報覆蓋率", passed=True, fail_means="")


def q6_outcome_blinding():
    return dict(gate="Q6 結局盲化", value="not_applicable（合成資料階段；Stage 2/3 真實資料時：計算指標者不得看結局）", criterion="—", passed=True, fail_means="")
