# -*- coding: utf-8 -*-
"""單一入口（計畫書 v2 版）。
  python run.py --phase calibrate    校準集：λ0／切片臂／κ／預警閾值鎖定 → results/locked.json、assumptions.md
  python run.py --phase default      測試集：預設情境 × Monte Carlo 資料集（主要／次要指標、兩臂、六道品質關卡）
  python run.py --phase grid         敏感度格點（一次一軸）＋可辨識邊界圖（τx × 取樣間隔 × 量測誤差）
  python run.py --verify             驗收清單
選項：--seed 20260818 --n 3000 --quick
規則：校準與測試 seed 族分離；閾值於校準集鎖定並寫檔（含雜湊）後測試集只讀；不可達→留白；每個數字有出處。"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np                                                       # noqa: E402
import m0_params as M0                                                   # noqa: E402
from m0_params import value                                              # noqa: E402
from m1_generator import simulate_cohort, observe, treatment_intervention, MECHS  # noqa: E402
from m2_risk import static_auc                                           # noqa: E402
from m4_predict import run_predict                                       # noqa: E402
from m5_ews import score_cohort, lock_threshold, surrogate_threshold, evaluate, alarm_auc, first_alarm  # noqa: E402
from m6_arms import biopsy_observation, biopsy_arm, noninvasive_arm, compare_arms  # noqa: E402
from calibrate import step1_lambda0, step2_biopsy, step3_static_c, cal_seed  # noqa: E402
import gates as G                                                        # noqa: E402
from seeding import module_rng                                           # noqa: E402

RESULTS = os.path.join(ROOT, "results")


def _json_default(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    if isinstance(o, np.ndarray): return o.tolist()
    return str(o)


def _dump(obj, name):
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=_json_default)


def params_hash():
    h = hashlib.sha256()
    for name in ("thresholds.json", "cohort.json"):
        h.update(open(os.path.join(M0.PARAMS_DIR, name), "rb").read())
    return h.hexdigest()[:16]


def ews_cfg(P, **over):
    cfg = dict(window_days=int(value(P, "window_days")), min_obs=int(value(P, "min_obs_per_window")), detrend=value(P, "detrend"),
               bw_frac=float(value(P, "gaussian_bw_frac")), eval_every=int(value(P, "eval_every_days")), joint=value(P, "joint_alarm"),
               budget=float(value(P, "false_alarm_budget_per_py")), surrogate=value(P, "surrogates"), block_len=int(value(P, "block_len_days")))
    cfg.update(over)
    return cfg


def _stop_days(C):
    """每人可計警報的追蹤天數（自減藥起點至跳轉／事件／追蹤末）與停止日。"""
    T, run_in = C["T"], C["run_in"]
    stop = np.full(C["n"], T)
    stop = np.where(C["t_jump"] >= 0, np.minimum(stop, C["t_jump"]), stop)
    stop = np.where(C["t_event"] >= 0, np.minimum(stop, C["t_event"]), stop)
    return stop, (stop - run_in).clip(min=0)


# ------------------------------------------------------------------ 一個情境格（校準→鎖閾→測試）
def run_cell(P, Cj, seed, cfg, scenario, sim_kw, n=None, mc=None, verbose=False, with_arms=False, with_predict=False, treatment=False):
    """校準集（seed 族 calibration）鎖閾值 → 測試集（seed 族 test，mc 個資料集）評估。回傳 dict（含 MC 平均與 SE）。"""
    fam = value(P, "seed_families"); mc = int(value(P, "mc_datasets")) if mc is None else mc
    T, run_in = int(value(P, "T")) + int(value(P, "run_in_days")), int(value(P, "run_in_days"))
    # --- 校準集：鎖閾（主要虛無 = 穩定機制真實序列）
    s_cal = seed + fam["calibration"]
    Cc = simulate_cohort(P, Cj, s_cal, n=n, **sim_kw)
    yc = observe(Cc, P, scenario, module_rng(s_cal, "obs", 0))
    stop_c, fu_c = _stop_days(Cc)
    scc = score_cohort(yc, run_in, Cc["T"], cfg["window_days"], cfg["min_obs"], cfg["detrend"], cfg["bw_frac"], cfg["eval_every"], cfg["joint"], stop_at=stop_c)
    null = Cc["mech"] == "stable"
    locks = {k: lock_threshold(scc[k], null, fu_c, cfg["budget"]) for k in ("S", "level", "trend")}
    if cfg["surrogate"] != "empirical_null":
        locks["S"] = surrogate_threshold(yc, run_in, Cc["T"], cfg, module_rng(s_cal, "ews", 1), cfg["surrogate"], null, fu_c, cfg["budget"], cfg["block_len"])
    lock_record = dict(calibration_seeds=[s_cal], thresholds={k: v.get("threshold") for k, v in locks.items()}, achieved={k: v.get("achieved_fa_per_py") for k, v in locks.items()},
                       budget=cfg["budget"], cfg=cfg, scenario=scenario, sim_kw=sim_kw)
    lock_record["hash"] = hashlib.sha256(json.dumps(lock_record, sort_keys=True, default=_json_default).encode()).hexdigest()[:16]
    # --- 測試集
    test_seeds = [seed + fam["test"] + k for k in range(mc)]
    per = []
    for s in test_seeds:
        C = simulate_cohort(P, Cj, s, n=n, **sim_kw)
        y = observe(C, P, scenario, module_rng(s, "obs", 0))
        if treatment:
            C = treatment_intervention(C, P, y, s)
            y = observe(C, P, scenario, module_rng(s, "obs", 0))
        stop, fu = _stop_days(C)
        sc = score_cohort(y, run_in, C["T"], cfg["window_days"], cfg["min_obs"], cfg["detrend"], cfg["bw_frac"], cfg["eval_every"], cfg["joint"], stop_at=stop)
        row = dict(seed=s, event_rate_24m=float(C["event_24m"].mean()), jump_rate=float((C["t_jump"] >= 0).mean()),
                   mech_shares={m: float((C["mech"] == m).mean()) for m in MECHS})
        for k, name in (("S", "ews"), ("level", "level_rule"), ("trend", "trend_rule")):
            ev = evaluate(C, sc[k], sc["eval_times"], locks[k].get("threshold"))
            row[name] = dict(threshold=locks[k].get("threshold"), per_mechanism=ev["per_mechanism"], n_alarmed=ev["n_alarmed"],
                             auc_jump_24m=alarm_auc(sc[k], C, 730))
        row["coverage_first_half_day"] = G.q5_observation_density(sc, sc["eval_times"], run_in)["value"]["days_after_onset"]
        if with_predict:
            row["predict"] = run_predict(C, y, value(P, "landmarks"), seed=s, lookback=cfg["window_days"] * 3, top=float(value(P, "high_risk_top_share")),
                                         timing=False, secondary_thresholds=value(P, "decision_thresholds"))
        if with_arms:
            row["arms"] = arms_block(P, Cj, C, y, s, cfg)
        per.append(row)
    return dict(lock=lock_record, test_seeds=test_seeds, per_dataset=per, mc=mc, n=int(Cc["n"]), summary=summarize_cell(per))


def arms_block(P, Cj, C, y, seed, cfg):
    """兩臂比較（v1 H4 / v2 肆之四）：切片臂用鎖定的 (εB, 閾值)；非侵入臂只看 y_obs 至各地標。"""
    lk = load_locked()
    b = lk.get("biopsy") if lk else None
    out = {}
    if b and b.get("reachable"):
        Bo = biopsy_observation(C["B_true"], b["noise_sd"], module_rng(seed, "arms", 0))
        bp = biopsy_arm(Bo, b["threshold"])
        for L in value(P, "landmarks"):
            upto = C["run_in"] + int(L)
            risk = (C["t_event"] < 0) | (C["t_event"] >= upto)
            yL = C["event_24m"][risk]
            pn = noninvasive_arm(y[risk], cfg["window_days"] * 3, dict(upto=upto, y=yL, folds=5, seed=seed))
            out[str(L)] = compare_arms(bp[risk], pn, yL, top_share=float(value(P, "high_risk_top_share")))
        out["biopsy_overall"] = dict(sensitivity=float(bp[C["event_24m"]].mean()), specificity=float((~bp[~C["event_24m"]]).mean()))
    else:
        out["note"] = "切片臂可信範圍不可達或尚未校準——留白"
    return out


def summarize_cell(per):
    """MC 平均與標準誤（v2 伍之六）：對每個機制的 alarm_rate／sens_before_jump／lead 中位。"""
    def agg(getter):
        vals = [getter(r) for r in per]
        vals = [v for v in vals if v is not None and v == v]
        if not vals:
            return dict(mean=None, se=None, k=0)
        a = np.array(vals, float)
        return dict(mean=float(a.mean()), se=float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else None, k=len(a))
    out = {}
    for rule in ("ews", "level_rule", "trend_rule"):
        out[rule] = {}
        for m in MECHS:
            out[rule][m] = dict(alarm_rate=agg(lambda r: r[rule]["per_mechanism"].get(m, {}).get("alarm_rate")),
                                fa_per_py=agg(lambda r: r[rule]["per_mechanism"].get(m, {}).get("fa_per_py")),
                                frac_scored=agg(lambda r: r[rule]["per_mechanism"].get(m, {}).get("frac_scored")))
            if m in ("bifurcation", "stochastic_escape", "exogenous_shock", "continuous_deterioration"):
                out[rule][m].update(sens_before_jump=agg(lambda r: r[rule]["per_mechanism"].get(m, {}).get("sens_before_jump")),
                                    lead_to_jump_median=agg(lambda r: r[rule]["per_mechanism"].get(m, {}).get("lead_to_jump_median")),
                                    lead_to_crit_median=agg(lambda r: r[rule]["per_mechanism"].get(m, {}).get("lead_to_crit_median")))
        out[rule]["auc_jump_24m"] = agg(lambda r: r[rule]["auc_jump_24m"])
    out["event_rate_24m"] = agg(lambda r: r["event_rate_24m"])
    if per and per[0].get("predict"):
        out["predict"] = {L: dict(c_gain=agg(lambda r: r["predict"]["landmarks"][L]["c_gain"]), auc_static=agg(lambda r: r["predict"]["landmarks"][L]["auc_static"]),
                                  auc_dynamic=agg(lambda r: r["predict"]["landmarks"][L]["auc_dynamic"]),
                                  nri_rel=agg(lambda r: r["predict"]["landmarks"][L]["reclass_same_riskset_rel"]["nri"]))
                          for L in per[0]["predict"]["landmarks"]}
    return out


def load_locked():
    p = os.path.join(RESULTS, "locked.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


# ------------------------------------------------------------------ 階段
def phase_calibrate(seed, n=None, quick=False, verbose=True):
    P = M0.load("thresholds.json"); Cj = M0.load("cohort.json")
    chk = M0.check(P, verbose=verbose); M0.check_cohort(Cj, verbose=verbose)
    n_cal = 800 if quick else (n or int(value(P, "N")))
    t0 = time.time()
    c1 = step1_lambda0(P, Cj, seed, n_cal=n_cal, verbose=verbose)
    c2 = step2_biopsy(P, Cj, seed, c1["lambda0"], n_cal=n_cal, verbose=verbose)
    c3 = step3_static_c(P, Cj, seed, c1["lambda0"], n_cal=n_cal, verbose=verbose)
    # 把校準值寫回 thresholds.json 的 value（λ0）與 locked.json（其餘），並記錄雜湊
    P["hazard"]["value"]["lambda0_per_day"] = c1["lambda0"]
    P["hazard"]["source"] += f"；λ0 於校準集（seed {cal_seed(P, seed)}，n={n_cal}）校準達 24 個月 {c1['rate_24m']:.3f}"
    P["kappa"]["value"] = c3["kappa"]
    with open(os.path.join(M0.PARAMS_DIR, "thresholds.json"), "w", encoding="utf-8") as f:
        json.dump(P, f, ensure_ascii=False, indent=1)
    locked = dict(seed=seed, calibration_seed=cal_seed(P, seed), n_cal=n_cal, lambda0=c1, biopsy=c2, static=c3, params_hash=params_hash(), seconds=time.time() - t0)
    _dump(locked, "locked.json")
    calib = [dict(step="一 λ0", target=f"24 個月發作率 ∈ {c1['target_range']}", achieved=f"{c1['rate_24m']:.3f}（λ0={c1['lambda0']:.3e}）", note=f"52 週 {c1['rate_52w']:.3f} vs {c1['check_52w']}；機制別 {c1['mech_flare_24m']}"),
             dict(step="二 切片臂", target=f"sens/spec ∈ 95% CI {c2['plausible']}", achieved=(f"εB={c2['noise_sd']}, thr={c2['threshold']:.3f} → {c2['sensitivity']:.3f}/{c2['specificity']:.3f}" if c2["reachable"] else "不可達（留白）"), note=str(c2.get("diagnostics", ""))),
             dict(step="三 靜態 C", target="0.65–0.75", achieved=f"{c3['c']:.3f}（κ={c3['kappa']}）", note=str(c3.get("grid", "")))]
    with open(os.path.join(RESULTS, "assumptions.md"), "w", encoding="utf-8") as f:
        f.write(M0.assumptions_markdown(P, Cj, calib, None))
    if verbose:
        print(f"[校準] 完成（{time.time() - t0:.0f} s）：results/locked.json、assumptions.md；※ 佔位參數 {[p['key'] for p in chk['placeholders']]}——產出不得對外")
    return locked


def phase_default(seed, n=None, quick=False, verbose=True):
    P = M0.load("thresholds.json"); Cj = M0.load("cohort.json")
    lk = load_locked()
    if not lk or value(P, "hazard")["lambda0_per_day"] is None:
        raise SystemExit("先跑 --phase calibrate")
    n_use = 800 if quick else n; mc = 2 if quick else None
    cfg = ews_cfg(P)
    scen = Cj["observation_scenarios"]["default"]
    t0 = time.time()
    cell = run_cell(P, Cj, seed, cfg, scen, dict(), n=n_use, mc=mc, with_arms=True, with_predict=True)
    # 品質關卡
    s_t = cell["test_seeds"][0]
    C = simulate_cohort(P, Cj, s_t, n=n_use); y = observe(C, P, scen, module_rng(s_t, "obs", 0))
    stop, _ = _stop_days(C)
    sc = score_cohort(y, C["run_in"], C["T"], cfg["window_days"], cfg["min_obs"], cfg["detrend"], cfg["bw_frac"], cfg["eval_every"], cfg["joint"], stop_at=stop)
    gates = [G.q1_mechanism_independence(C, y, cfg),
             G.q2_zero_signal(P, Cj, seed, cfg, cell["lock"]["thresholds"]["S"], n=n_use, verbose=verbose),
             G.q3_threshold_locking(cell["lock"], cell["test_seeds"]),
             G.q4_time_direction(y, C["run_in"], C["T"], cfg, module_rng(seed, "gates", 4)),
             G.q5_observation_density(sc, sc["eval_times"], C["run_in"]),
             G.q6_outcome_blinding()]
    # 其他觀測情境（不規則／症狀驅動／治療悖論／UPCR 指標）
    scen_res = {}
    for name in Cj["observation_scenarios"]["list"]:
        if name == "default":
            continue
        s2 = Cj["observation_scenarios"][name]
        scen_res[name] = run_cell(P, Cj, seed, cfg, s2, dict(), n=n_use, mc=mc, treatment=bool(s2.get("treatment_intervention")))["summary"]
    out = dict(phase="default", seed=seed, params_hash=params_hash(), cfg=cfg, scenario=scen, cell=cell, gates=gates, scenarios=scen_res, seconds=time.time() - t0)
    _dump(out, "default.json")
    with open(os.path.join(RESULTS, "assumptions.md"), "a", encoding="utf-8") as f:
        f.write("\n## 品質關卡（v2 六道）\n\n| 關卡 | 值 | 條件 | 結果 |\n|---|---|---|---|\n")
        for g in gates:
            f.write(f"| {g['gate']} | {json.dumps(g['value'], ensure_ascii=False, default=_json_default)[:300]} | {g['criterion']} | {'通過' if g['passed'] else '未過：' + g['fail_means']} |\n")
    if verbose:
        _print_summary(cell, gates)
        print(f"[預設情境] 完成（{time.time() - t0:.0f} s）：results/default.json")
    return out


def _print_summary(cell, gates=None):
    s = cell["summary"]
    print(f"\n=== 預設情境（n={cell['n']}, MC={cell['mc']}）鎖定閾值 {cell['lock']['thresholds']}（達成假警報／病人年 {cell['lock']['achieved']}，預算 {cell['lock']['budget']}）===")
    for rule in ("ews", "level_rule", "trend_rule"):
        print(f"  [{rule}]")
        for m in MECHS:
            r = s[rule][m]
            line = f"    {m:24s} 警報率 {r['alarm_rate']['mean']}  可算 {r['frac_scored']['mean']}"
            if "sens_before_jump" in r:
                line += f"  跳轉前敏感度 {r['sens_before_jump']['mean']}  提前(至跳轉)中位 {r['lead_to_jump_median']['mean']}  提前(至臨界) {r['lead_to_crit_median']['mean']}"
            print(line)
        print(f"    AUC(跳轉) {s[rule]['auc_jump_24m']}")
    if s.get("predict"):
        for L, r in s["predict"].items():
            print(f"  H2 地標 {L}: C 靜態 {r['auc_static']['mean']:.3f} 動態 {r['auc_dynamic']['mean']:.3f} 增益 {r['c_gain']['mean']:+.3f}±{r['c_gain']['se']}  NRI(同風險集,相對) {r['nri_rel']['mean']}")
    if gates:
        print("=== 品質關卡 ===")
        for g in gates:
            print(f"  [{'通過' if g['passed'] else '未過'}] {g['gate']}")


def phase_grid(seed, n=None, quick=False, verbose=True):
    """敏感度格點（v2 拾）：一次一軸；可辨識邊界圖：τx × 取樣間隔 × 量測誤差。"""
    P = M0.load("thresholds.json"); Cj = M0.load("cohort.json")
    if value(P, "hazard")["lambda0_per_day"] is None:
        raise SystemExit("先跑 --phase calibrate")
    n_use = 600 if quick else (n or 1500); mc = 1 if quick else 3
    print(f"[格點] 每格 n={n_use}, MC={mc}（為算力設限，已記錄）")
    scen = Cj["observation_scenarios"]["default"]
    axes = {
        "tau_x": [dict(sim=dict(tau_x=v)) for v in P["tau_x"]["scan"]],
        "tau_xi": [dict(sim=dict(tau_xi=v)) for v in P["tau_xi"]["scan"]],
        "sigma_xi": [dict(sim=dict(sigma_xi=v)) for v in P["sigma_xi"]["scan"]],
        "meas_error": [dict(scen=dict(meas_error=k)) for k in value(P, "meas_error_levels")],
        "interval": [dict(scen=dict(interval=k)) for k in value(P, "sampling_intervals") if k != "half_year"],
        "T": [dict(T=v) for v in P["T"]["scan"]],
        "window_days": [dict(cfg=dict(window_days=v)) for v in P["window_days"]["scan"]],
        "detrend": [dict(cfg=dict(detrend=v)) for v in P["detrend"]["scan"]],
        "joint_alarm": [dict(cfg=dict(joint=v)) for v in P["joint_alarm"]["scan"]],
        "surrogates": [dict(cfg=dict(surrogate=v)) for v in P["surrogates"]["scan"]],
        "budget": [dict(cfg=dict(budget=v)) for v in P["false_alarm_budget_per_py"]["scan"]],
        "taper": [dict(sim=dict(taper=dict(duration_days=p["duration_days"], complete=True))) for p in value(P, "taper_schedule")["presets"]],
        "mechanisms": [dict(sim=dict(mech_shares=s)) for s in P["mechanisms"]["scan"]],
        "pkpd_lag": [dict(sim=dict(pkpd_lag=v)) for v in P["pkpd_lag_days"]["scan"]],
    }
    if quick:
        axes = {k: axes[k] for k in ("tau_x", "meas_error", "interval")}
    t0 = time.time(); res = {}
    for ax, cells in axes.items():
        res[ax] = []
        for c in cells:
            cfg = ews_cfg(P, **c.get("cfg", {})); s2 = dict(scen, **c.get("scen", {})); sim = dict(c.get("sim", {}))
            P2 = P
            if "T" in c:
                P2 = json.loads(json.dumps(P)); P2["T"]["value"] = c["T"]
            r = run_cell(P2, Cj, seed, cfg, s2, sim, n=n_use, mc=mc)
            res[ax].append(dict(cell=c, lock=r["lock"]["thresholds"], achieved=r["lock"]["achieved"], summary=r["summary"]))
            if verbose:
                b = r["summary"]["ews"]["bifurcation"]
                print(f"  {ax}={c}: 分岔敏感度 {b['sens_before_jump']['mean']} 提前中位 {b['lead_to_jump_median']['mean']} 穩定假警報/py {r['summary']['ews']['stable']['fa_per_py']['mean']}  ({time.time()-t0:.0f}s)")
    # 可辨識邊界圖：τx × interval × meas_error（quick 時縮）
    taus = P["tau_x"]["scan"] if not quick else [P["tau_x"]["scan"][0], P["tau_x"]["scan"][-1]]
    ints = [k for k in value(P, "sampling_intervals") if k != "half_year"] if not quick else ["weekly", "monthly"]
    errs = list(value(P, "meas_error_levels")) if not quick else ["small", "large"]
    ident = []
    for tx in taus:
        for it in ints:
            for er in errs:
                r = run_cell(P, Cj, seed, ews_cfg(P), dict(scen, interval=it, meas_error=er), dict(tau_x=tx), n=n_use, mc=mc)
                b = r["summary"]["ews"]["bifurcation"]
                ident.append(dict(tau_x=tx, interval=it, meas_error=er, sens=b["sens_before_jump"], lead=b["lead_to_jump_median"],
                                  fa_stable=r["summary"]["ews"]["stable"]["fa_per_py"], level_sens=r["summary"]["level_rule"]["bifurcation"]["sens_before_jump"]))
                if verbose:
                    print(f"  ident τx={tx} {it} {er}: sens {b['sens_before_jump']['mean']} lead {b['lead_to_jump_median']['mean']}  ({time.time()-t0:.0f}s)")
    out = dict(phase="grid", seed=seed, n=n_use, mc=mc, params_hash=params_hash(), axes=res, identifiability=ident, seconds=time.time() - t0)
    _dump(out, "grid.json")
    if verbose:
        print(f"[格點] 完成（{time.time() - t0:.0f} s）：results/grid.json")
    return out


def verify(seed, quick=False):
    rep = []
    P = M0.load("thresholds.json"); chk = M0.check(P, verbose=False)
    rep.append(("所有參數皆有 source 與 status", not chk["missing_source"] and not chk["bad_status"]))
    a = phase_default(seed, quick=quick, verbose=False)
    b = phase_default(seed, quick=quick, verbose=False)
    ja = json.dumps({k: v for k, v in a.items() if k != "seconds"}, sort_keys=True, default=_json_default)
    jb = json.dumps({k: v for k, v in b.items() if k != "seconds"}, sort_keys=True, default=_json_default)
    rep.append(("決定性：同 seed 連跑兩次結果逐位元相同", ja == jb))
    rep.append(("六道品質關卡全過", all(g["passed"] for g in a["gates"])))
    rep.append(("閾值於校準集鎖定（Q3）", [g for g in a["gates"] if g["gate"].startswith("Q3")][0]["passed"]))
    r = subprocess.run([sys.executable, "-m", "pytest", os.path.join(ROOT, "tests"), "-q", "-p", "no:cacheprovider"], capture_output=True, text=True, encoding="utf-8")
    rep.append((f"單元測試：{r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-200:]}", r.returncode == 0))
    print("\n=== 驗收清單（模型端）===")
    for name, ok in rep:
        print(f"  [{'通過' if ok else '未過'}] {name}")
    return 0 if all(ok for _, ok in rep) else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=str, default=None, choices=["calibrate", "default", "grid"])
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--seed", type=int, default=20260818)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.verify:
        sys.exit(verify(a.seed, quick=a.quick))
    {"calibrate": phase_calibrate, "default": phase_default, "grid": phase_grid}.get(a.phase, lambda *x, **k: ap.print_help())(a.seed, n=a.n, quick=a.quick)
