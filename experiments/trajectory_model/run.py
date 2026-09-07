# -*- coding: utf-8 -*-
"""單一入口：python run.py [--params params.json] [--out results] [--seeds 5] [--jobs 12] [--quick] [--figures-only]

流程：載入參數檔（檢核出處、輸出 assumptions.md）→ 逐變體校準（v1 §1 Δμ、v2 參 λ0/β、模組二係數）
      → v2 捌 分層混合檢核（pilot 世代不過就停止）→ (變體 × seed) 平行跑模組二～五
      → 全部結果（不挑）寫 results.json → 六張圖。
格點（模組六）：窗長 {21,42,60,90} × 去趨勢 3、分層 2 × 分群法 2、seed >= 5、
      tau 14/30/60、翻轉時程 3/6/12 個月（不可達則標 unreachable 留白）、ohare_ranges 示範、含退出的一組（僅模組四）。
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

# 平行是「多程序」，每個程序的 BLAS/OpenMP 只准用 1 條執行緒；否則 12 個程序 × 多執行緒的
# sklearn/numpy 會互相搶核心（第一次跑時每個 worker 只吃到 ~65% CPU）。必須在 import numpy 之前設。
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from cohort import (load_params, _val, make_cohort, calibrate_delta_mu, calibrate_hazard, flip_timing_summary,
                    stratum_type_mix, type_separability, provenance_report, static_auc, with_events)
from risk import fit_risk_coefficients, risk_score, stratify
from clustering import run_clustering, generator_labels
from prediction import run_prediction, _auc
from warning import run_warning, rolling_indicators
import figures as FG


# ------------------------------------------------------------------ 單一 job
def cohort_summary(C):
    f, ev = C["is_flip"], C["event"]
    cls = C["cls"][~f]
    return dict(n=int(C["n"]), event_rate=float(ev.mean()), event_rate_linear=float(ev[~f].mean()),
                event_rate_flip=float(ev[f].mean()), flip_share=float(f.mean()),
                threshold_rate=float((C["t_threshold"] >= 0).mean()),
                linear_class_shares=[float(v) for v in np.bincount(cls, minlength=3) / max(1, len(cls))],
                egfr0_mean_by_type={"flip": float(C["egfr0"][f].mean()), "linear": float(C["egfr0"][~f].mean())},
                flip_timing=flip_timing_summary(C), lam0=C["lam0"], beta=C["beta"], kappa=C["kappa"], delta_mu_median=C["delta_mu_median"],
                type_separability_auc_first180=type_separability(C, 180),
                type_separability_auc_pre_onset=type_separability(C, 180, pre_onset_only=True))


def run_job(args):
    res = _run_job(args)
    outdir = args[0].get("_jobdir")     # 每個 job 完成就落地一份（晚期崩潰不會把幾小時的結果一起帶走）
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        with open(os.path.join(outdir, f"{res['variant']}_{res['seed']}.json"), "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in res.items() if k != "fig"}, f, ensure_ascii=False, default=_json_default)
    return res


def _run_job(args):
    P, variant, seed, calib, want_fig = args
    t0 = time.time()
    C = make_cohort(P, seed, tau=variant["tau"], delta_mu_median=calib["delta_mu"]["delta_mu_median"],
                    lam0=calib["hazard"]["lambda0_per_day"], beta=calib["hazard"]["beta_per_10_egfr"],
                    kappa=calib["hazard"]["kappa"], dropout=variant.get("dropout", False), start_mode=variant.get("start_mode"),
                    linear_onset=variant.get("linear_onset"))
    res = dict(variant=variant["name"], seed=int(seed), cohort=cohort_summary(C))
    if variant.get("module4_only"):
        res["prediction"] = run_prediction(C, P, seed, dropout=variant.get("dropout", False))
        res["seconds"] = time.time() - t0
        return res

    score = risk_score(C, calib["risk_coefs"])
    res["risk"] = dict(auc_of_score=_auc(C["event"], score), stratum_flip_share={}, stratum_single_type_max={})
    rng = np.random.default_rng(seed + 7)
    res["clustering"] = {}
    for q in P["risk_score"]["quantiles"]:
        strata = stratify(score, q)
        shares, worst = stratum_type_mix(C["is_flip"], strata)
        res["risk"][f"stratum_flip_share"][f"q{q}"] = shares
        res["risk"][f"stratum_single_type_max"][f"q{q}"] = worst
        res["risk"][f"event_rate_by_q{q}"] = [float(C["event"][strata == s].mean()) for s in range(q)]
        if worst > _val(P["risk_score"]["stratum_single_type_max"]) and variant["name"] != "ohare_ranges":
            # v2 捌：任一層單一型別 > 90% → H1 無從檢驗 → 停止（示範變體例外，它就是要示範這件事）
            res["aborted"] = f"風險層內單一型別佔比 {worst:.2f} > 門檻，H1 無從檢驗（q{q}）"
            res["seconds"] = time.time() - t0
            return res
        for m in ("A", "B"):
            res["clustering"][f"q{q}_{m}"] = run_clustering(C, strata, m, P, rng)
    res["prediction"] = run_prediction(C, P, seed)
    res["warning"] = {}
    first_alarm_default = None
    fig_win = P["warning"]["windows_days"][min(1, len(P["warning"]["windows_days"]) - 1)]
    for win in P["warning"]["windows_days"]:
        for mode in P["warning"]["detrend_modes"]:
            r = run_warning(C, P, win, mode, rng)
            fa = r.pop("_first_alarm")
            if win == fig_win and mode == "linear":
                first_alarm_default = (win, mode, fa)
            res["warning"][f"w{win}_{mode}"] = r
    if want_fig and first_alarm_default is not None:
        res["fig"] = figure_data(C, P, res, first_alarm_default, score)
    else:
        for k in res["clustering"].values():
            for r in k:
                r.pop("mean_traj", None)
    res["seconds"] = time.time() - t0
    return res


def figure_data(C, P, res, fad, score):
    win, mode, first = fad
    te, tc, tt, f = C["t_event"], C["t_crit"], C["t_threshold"], C["is_flip"]
    cand = np.where(f & (te >= 0) & (tc > 200) & (first >= 0) & (first <= te))[0]
    if len(cand) == 0:
        cand = np.where(f & (te >= 0) & (tc > 0))[0]
    i = int(cand[len(cand) // 2]) if len(cand) else int(np.where(f)[0][0])
    AR, SD = rolling_indicators(C["X"][i:i + 1], win, mode, P["warning"]["gaussian_bw_frac"]["value"])
    fig1 = dict(idx=i, x=C["X"][i].tolist(), ar1=AR[0].tolist(), sd=SD[0].tolist(), window=win, detrend=mode,
                t_crit=int(tc[i]), t_threshold=int(tt[i]), t_event=int(te[i]), first_alarm=int(first[i]),
                t_onset=float(C["t_onset"][i]), threshold=float(C["scale"]["threshold_egfr"]))
    q5 = stratify(score, 5)
    g = generator_labels(C)
    names = {0: "線性-緩慢", 1: "線性-進行", 2: "線性-加速", 10: "翻轉型-未翻轉", 11: "翻轉型-已翻轉"}
    gen_share = [{names[k]: float((g[q5 == s] == k).mean()) for k in names} for s in range(5)]
    ev = te >= 0
    fig5 = dict(flip_event=(te - first)[f & ev & (first >= 0) & (first <= te)].astype(float).tolist(),
                linear_event=(te - first)[~f & ev & (first >= 0) & (first <= te)].astype(float).tolist(),
                flip_crit=(tc - first)[f & (tc >= 0) & (first >= 0)].astype(float).tolist(),
                perm_p=res["warning"][f"w{win}_{mode}"]["perm_test_lead_flip_vs_linear"]["p"],
                perm_diff=res["warning"][f"w{win}_{mode}"]["perm_test_lead_flip_vs_linear"]["diff"])
    return dict(fig1=fig1, gen_share=gen_share, fig5=fig5, setting=f"w{win}_{mode}")


# ------------------------------------------------------------------ 聚合（不挑：全部 seed 的平均與範圍）
def aggregate(objs):
    objs = [o for o in objs if o is not None]
    if not objs:
        return None
    o0 = objs[0]
    if isinstance(o0, dict):
        keys = []                                                       # 所有 seed 的 key 聯集（codex 稽核：只看第一個會靜默丟欄位）
        for o in objs:
            for k in (o or {}):
                if k not in keys:
                    keys.append(k)
        return {k: aggregate([o.get(k) if isinstance(o, dict) else None for o in objs]) for k in keys if not str(k).startswith("_") and k != "mean_traj"}
    if isinstance(o0, list):
        if all(isinstance(o, list) and len(o) == len(o0) for o in objs) and o0 and not isinstance(o0[0], str):
            return [aggregate([o[i] for o in objs]) for i in range(len(o0))]
        return o0
    if isinstance(o0, (bool, np.bool_)):
        return dict(mean=float(np.mean([bool(o) for o in objs])), any=bool(any(objs)))
    if isinstance(o0, (int, float, np.integer, np.floating)):
        v = np.array([float(o) for o in objs if o is not None], float)
        v = v[np.isfinite(v)]
        if len(v) == 0:
            return dict(mean=np.nan, min=np.nan, max=np.nan, n=0)
        return dict(mean=float(v.mean()), min=float(v.min()), max=float(v.max()), n=int(len(v)))
    return o0


def _json_default(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    if isinstance(o, np.ndarray): return o.tolist()
    if callable(o): return None
    return str(o)


def make_figures(results, figdir):
    """六張圖全部由 results.json 內容重繪（`--figures-only`），改圖不必重跑三小時。"""
    fd = results["figure_data"]
    FG.fig1_single_case(fd["fig1"], figdir)
    FG.fig2_strata_types(fd["clustering_q5_A"], fd["gen_share"], figdir, title="預設變體 seed 0")
    FG.fig3_static_vs_dynamic(results["aggregate"]["default"]["prediction"], figdir)
    FG.fig4_timing_shift(fd["timing"], figdir)
    FG.fig5_leadtime(fd["fig5"], figdir)
    FG.fig6_tau_scan(results["tau_scans"], figdir)
    FG.fig7_k_selection(fd["clustering_q5_A"], fd.get("clustering_q5_B"), figdir)


# ------------------------------------------------------------------ 主程式
def build_variants(P, quick):
    fl = P["flip"]
    tau0 = _val(fl["tau_days"]); tgt0 = _val(fl["flip_time_target_days"])
    V = [dict(name="default", tau=tau0, target=tgt0)]
    if quick:
        return V
    for tau in fl["tau_grid_days"]:
        if tau != tau0:
            V.append(dict(name=f"tau{tau}", tau=tau, target=tgt0))
    for tgt in fl["flip_time_grid_days"]:
        if tgt != tgt0:
            V.append(dict(name=f"fliptime{tgt}", tau=tau0, target=tgt))
    V.append(dict(name="linear_from_day0", tau=tau0, target=tgt0, linear_onset="day0"))
    V.append(dict(name="ohare_ranges", tau=tau0, target=tgt0, start_mode="ohare_ranges"))
    V.append(dict(name="dropout", tau=tau0, target=tgt0, dropout=True, module4_only=True))
    for m in _val(P["hazard"]["beta_sensitivity_multipliers"]):       # H2 增益對 β 的依賴（只跑模組四）
        if m != 1.0:
            V.append(dict(name=f"beta_x{m:g}", tau=tau0, target=tgt0, beta_mult=m, module4_only=True))
    return V


def calibrate_variant(P, v, cache):
    """v1 §1 Δμ → v2 參 λ0/β → 模組二係數（pilot 世代）→ v2 捌 分層混合檢核。
    Δμ 不可達的變體回傳 None（標 unreachable 留白，不用替代值）。"""
    key = (v["tau"], v["target"], v.get("start_mode"), v.get("beta_mult", 1.0), v.get("linear_onset"))
    if key in cache:
        return cache[key]
    print(f"\n=== 校準變體 {v['name']}（tau={v['tau']}，翻轉時程目標 {v['target']} 天{'，'+v['start_mode'] if v.get('start_mode') else ''}）===")
    dm = calibrate_delta_mu(P, tau=v["tau"], target_days=v["target"])
    if not dm["feasible"]:
        cache[key] = dict(delta_mu=dm, unreachable=True)
        return cache[key]
    Pv = P
    if v.get("beta_mult", 1.0) != 1.0:
        Pv = json.loads(json.dumps({k: x for k, x in P.items() if not k.startswith("_")}))
        Pv["hazard"]["beta_per_10_egfr"]["value"] = _val(P["hazard"]["beta_per_10_egfr"]) * v["beta_mult"]
    kfix = None
    if v.get("beta_mult", 1.0) != 1.0:                                  # β 敏感度只動 β 與 λ0：kappa 固定為 default 的值，X 與 U 才與 default 相同
        kfix = cache[(v["tau"], v["target"], None, 1.0, None)]["hazard"]["kappa"]
    hz = calibrate_hazard(Pv, dm["delta_mu_median"], tau=v["tau"], start_mode=v.get("start_mode"), linear_onset=v.get("linear_onset"), kappa_fixed=kfix)
    pilot = make_cohort(P, P["sim"]["seed"] + 1000, n=1500, tau=v["tau"], delta_mu_median=dm["delta_mu_median"],
                        lam0=hz["lambda0_per_day"], beta=hz["beta_per_10_egfr"], kappa=hz["kappa"], start_mode=v.get("start_mode"),
                        linear_onset=v.get("linear_onset"))
    coefs = P["risk_score"]["coefficients"]["value"] or fit_risk_coefficients(pilot)
    score = risk_score(pilot, coefs)
    mix = {}
    for q in P["risk_score"]["quantiles"]:
        shares, worst = stratum_type_mix(pilot["is_flip"], stratify(score, q))
        mix[f"q{q}"] = dict(flip_share_by_stratum=shares, single_type_max=worst)
    worst_all = max(m["single_type_max"] for m in mix.values())
    sep = type_separability(pilot, 180)
    print(f"[模組二] 風險分數係數（pilot 配適）：{coefs}")
    print(f"[v2 捌 檢核] pilot 各層單一型別最大佔比 {worst_all:.2f}（門檻 {_val(P['risk_score']['stratum_single_type_max'])}）；"
          f"前 180 天型別可分辨 AUC {sep:.3f}；靜態 C {static_auc(pilot):.3f}；事件率 {pilot['event'].mean():.3f}")
    calib = dict(delta_mu=dm, hazard=hz, risk_coefs=coefs, pilot_flip_timing=flip_timing_summary(pilot),
                 pilot_stratum_mix=mix, pilot_type_separability_auc_first180=sep, unreachable=False)
    if worst_all > _val(P["risk_score"]["stratum_single_type_max"]) and v.get("start_mode") != "ohare_ranges" and v.get("linear_onset") != "day0":
        print(f"※ 警告：風險層內單一型別佔比 {worst_all:.2f} > 門檻，H1 無從檢驗——停止（v2 捌）。")
        sys.exit(2)
    cache[key] = calib
    return calib


def run_gates(P, calib, n=4000, n_gate=12000):
    """v2 附錄參：三道閘門（在 default 變體、獨立 seed 的檢核世代上），任一未過即中止。回傳 list[dict]。"""
    v = calib["variant"]
    C = make_cohort(P, P["sim"]["seed"] + 2000, n=n, tau=v["tau"], delta_mu_median=calib["delta_mu"]["delta_mu_median"],
                    lam0=calib["hazard"]["lambda0_per_day"], beta=calib["hazard"]["beta_per_10_egfr"], kappa=calib["hazard"]["kappa"])
    score = risk_score(C, calib["risk_coefs"])
    worst = max(stratum_type_mix(C["is_flip"], stratify(score, q))[1] for q in P["risk_score"]["quantiles"])
    thr1 = _val(P["risk_score"]["stratum_single_type_max"])
    g1 = dict(gate="閘門一 風險層內單一型別佔比（全部層之最大值）", value=round(float(worst), 3),
              criterion=f"<= {thr1}", passed=bool(worst <= thr1), fail_means="H1 被設定成偽，x0 重疊未生效")
    # 閘門二：β=0 → 事件與 x(t) 無關。ΔC 在 n=1500 的抽樣雜訊約 ±0.02，n=12000 約 ±0.007（兩個方向都有）；
    # 依 codex 稽核改為「每個地標的 |ΔC| 都要 < 0.01」（帶號平均會讓正負抵消），並以 3 個閘門 seed 平均壓噪音，逐地標列印。
    Pq = dict(P); Pq["prediction"] = dict(P["prediction"], landmarks_days=[180, 365, 730])
    per_seed = []
    for gs in range(3):
        Cg = make_cohort(P, P["sim"]["seed"] + 3000 + gs, n=n_gate, tau=v["tau"], delta_mu_median=calib["delta_mu"]["delta_mu_median"],
                         lam0=calib["hazard"]["lambda0_per_day"], beta=0.0, kappa=calib["hazard"]["kappa"])
        Cg = with_events(Cg, calib["hazard"]["lambda0_per_day"] * 2.5, 0.0, P)
        r0 = run_prediction(Cg, Pq, 77 + gs, timing=False)
        per_seed.append({L: row["c_gain"] for L, row in r0["landmarks"].items()})
    gains = {L: float(np.mean([d[L] for d in per_seed])) for L in per_seed[0]}
    worst = float(max(abs(g) for g in gains.values()))
    g2 = dict(gate=f"閘門二 β=0 時動態相對靜態的 C 增益：max_L |ΔC_L|（n={n_gate}×3 seed 平均；各地標／各 seed 見 detail）", value=round(worst, 4),
              detail=dict(mean={L: round(g, 4) for L, g in gains.items()}, per_seed=[{L: round(g, 4) for L, g in d.items()} for d in per_seed]),
              criterion="< 0.01", passed=bool(worst < 0.01), fail_means="結局仍與 x(t) 耦合，洩漏未解")
    auc = type_separability(C, 180, pre_onset_only=True)
    g3 = dict(gate="閘門三 僅取定態期資料（起始日 >= 180 天者之前 180 天）之型別分類 AUROC", value=round(float(auc), 3),
              criterion="0.45–0.55", passed=bool(0.45 <= auc <= 0.55), fail_means="兩型在機制啟動前即可分，混淆未除")
    return [g1, g2, g3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "params.json"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--quick", action="store_true", help="小規模煙霧測試（n=400、1 seed、少量置換）")
    ap.add_argument("--figures-only", action="store_true", help="只用既有 results.json 重繪圖")
    a = ap.parse_args()

    if a.figures_only:
        with open(os.path.join(a.out, "results.json"), encoding="utf-8") as f:
            make_figures(json.load(f), os.path.join(a.out, "figures"))
        print("已重繪 figures/")
        return

    P = load_params(a.params)
    if a.n: P["sim"]["n"] = a.n
    if a.seeds: P["sim"]["n_seeds"] = a.seeds
    if a.quick:
        P["sim"].update(n=(P["sim"]["n"] if a.n else min(P["sim"]["n"], 400)), n_seeds=1)
        P["clustering"].update(n_bootstrap=20, n_permutation=10)
        P["warning"].update(windows_days=[21, 42], detrend_modes=["linear"], null_subjects=20, null_perms_per_subject=10,
                            n_permutation_leadtime=200)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "assumptions.md"), "w", encoding="utf-8") as f:     # v2 拾壹：附錄用
        f.write(provenance_report(P["_provenance"]["missing_source"], P["_provenance"]["derived"]) + "\n")
    P["_jobdir"] = os.path.join(a.out, "jobs")

    # τ 掃描曲線（v2 壹：獨立出圖）——三個 tau 都算，不論主分析用哪個
    tau_scans = {}
    for tau in P["flip"]["tau_grid_days"]:
        tau_scans[str(tau)] = calibrate_delta_mu(P, tau=tau, verbose=False)["scan"]

    variants = build_variants(P, a.quick)
    seeds = [P["sim"]["seed"] + i for i in range(P["sim"]["n_seeds"])]
    cache, jobs, calibrations, skipped = {}, [], {}, []
    gates = None
    for v in variants:
        calib = calibrate_variant(P, v, cache)
        calibrations[v["name"]] = dict(variant=v, **calib)
        if calib.get("unreachable"):
            skipped.append(v["name"])
            print(f"    → 變體 {v['name']} 標 unreachable，留白不跑")
            continue
        if v["name"] == "default":
            # v2 附錄參：三道閘門，任一未過即中止
            gates = run_gates(P, calibrations["default"], n=(1500 if a.quick else 4000), n_gate=(6000 if a.quick else 12000))
            print("\n=== 放行閘門（v2 附錄參）===")
            for g in gates:
                print(f"  [{'通過' if g['passed'] else '未過'}] {g['gate']}：{g['value']}（條件 {g['criterion']}）"
                      + (f"  detail={g['detail']}" if g.get("detail") else ""))
            with open(os.path.join(a.out, "assumptions.md"), "a", encoding="utf-8") as f:
                f.write("\n## 放行閘門（v2 附錄參）\n\n| 閘門 | 值 | 條件 | 結果 |\n|---|---|---|---|\n")
                for g in gates:
                    f.write(f"| {g['gate']} | {g['value']}{(' ' + str(g['detail'])) if g.get('detail') else ''} | {g['criterion']} | {'通過' if g['passed'] else '未過：' + g['fail_means']} |\n")
            if not all(g["passed"] for g in gates):
                print("※ 閘門未全過——中止，不跑完整格點（v2 附錄參）。")
                with open(os.path.join(a.out, "gates.json"), "w", encoding="utf-8") as f:
                    json.dump(dict(gates=gates, calibration=calibrations), f, ensure_ascii=False, indent=1, default=_json_default)
                sys.exit(3)
        for i, s in enumerate(seeds):
            jobs.append((P, v, s, calib, v["name"] == "default" and i == 0))
    print(f"\n共 {len(jobs)} 個 job（{len(variants) - len(skipped)} 變體 × {len(seeds)} seed；留白 {skipped}），平行 {a.jobs} 程序 …")

    t0 = time.time()
    per = {}
    if a.jobs > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=a.jobs) as ex:
            for r in ex.map(run_job, jobs):
                per.setdefault(r["variant"], []).append(r)
                print(f"  完成 {r['variant']} seed={r['seed']}（{r['seconds']:.0f} s）" + (f"  ※ 中止：{r['aborted']}" if r.get("aborted") else ""))
    else:
        for j in jobs:
            r = run_job(j); per.setdefault(r["variant"], []).append(r)
            print(f"  完成 {r['variant']} seed={r['seed']}（{r['seconds']:.0f} s）" + (f"  ※ 中止：{r['aborted']}" if r.get("aborted") else ""))

    fig_res = next((r for r in per.get("default", []) if "fig" in r), None)
    figdata = fig_res.pop("fig") if fig_res else None
    agg = {v: aggregate([{k: x for k, x in r.items() if k != "fig"} for r in rs if not r.get("aborted")]) for v, rs in per.items()}

    # 洩漏警訊（建置提示詞 模組四）：任何地標動態 C 指數增益 > 門檻就大聲說
    leak = [(v, L, row["c_gain"]["mean"]) for v, A in agg.items() if A and A.get("prediction") and "landmarks" in A["prediction"]
            for L, row in A["prediction"]["landmarks"].items() if row["c_gain"]["mean"] > P["prediction"]["leak_alert_c_gain"]["value"]]

    results = dict(meta=dict(date=time.strftime("%Y-%m-%d %H:%M"), seconds=time.time() - t0, quick=a.quick,
                             n=P["sim"]["n"], seeds=seeds, params_file=a.params, skipped_unreachable=skipped),
                   provenance=P["_provenance"], params={k: v for k, v in P.items() if not k.startswith("_")},
                   calibration=calibrations, gates=gates, tau_scans=tau_scans, aggregate=agg, per_seed=per, leak_alert=leak)
    if figdata:
        results["figure_setting"] = figdata["setting"]
        results["figure_data"] = dict(**figdata, clustering_q5_A=fig_res["clustering"]["q5_A"],
                                      clustering_q5_B=fig_res["clustering"]["q5_B"], timing=fig_res["prediction"]["timing"])
    with open(os.path.join(a.out, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1, default=_json_default)
    if figdata:
        make_figures(results, os.path.join(a.out, "figures"))
    print(f"\n完成，{time.time() - t0:.0f} s。結果：{a.out}/results.json、figures/、assumptions.md")
    if leak:
        print("※ 洩漏警訊：以下 (變體, 地標) 的動態 C 指數增益 > 0.05，請檢查資料流再解讀：", leak)
    for v, A in agg.items():
        if not A or "prediction" not in A:
            continue
        pr = A["prediction"]["landmarks"]
        print(f"[{v}] 事件率 {A['cohort']['event_rate']['mean']:.2f}；靜態 C {A['prediction']['static']['auc']['mean']:.3f}；"
              f"型別可分辨 AUC(前180天) {A['cohort']['type_separability_auc_first180']['mean']:.3f}；"
              + "；".join(f"L{L}: 靜態 {r['auc_static']['mean']:.3f} 動態 {r['auc_dynamic']['mean']:.3f} NRI(abs) {r['reclass_abs']['nri']['mean']:+.3f}" for L, r in pr.items()))
        for k, w in A.get("warning", {}).items():
            fl, li = w["by_type"]["flip"], w["by_type"]["linear"]
            print(f"    {k}: 提前期中位 翻轉 {fl['lead_to_event_days']['median']['mean']:.0f} / 爬升 {li['lead_to_event_days']['median']['mean']:.0f} 天，"
                  f"p={w['perm_test_lead_flip_vs_linear']['p']['mean']:.3f}；偵測 {fl['detection_rate']['mean']:.2f}/{li['detection_rate']['mean']:.2f}；"
                  f"偽警報/人年 {fl['false_alarms_per_person_year']['mean']:.2f}/{li['false_alarms_per_person_year']['mean']:.2f}；"
                  f"下降型/人年 {fl['downward']['episodes_per_person_year']['mean']:.2f}/{li['downward']['episodes_per_person_year']['mean']:.2f}；"
                  f"基線 AR1 {w['baseline_ar1']['flip']['mean']:.2f}/{w['baseline_ar1']['linear']['mean']:.2f}")


if __name__ == "__main__":
    main()
