# -*- coding: utf-8 -*-
"""模組四事後重算（不重跑模組三／五）。

為什麼可以：世代生成由 seed 與參數完全決定（make_cohort 不依賴模組執行順序），模組四的交叉驗證切分也由 seed 決定，
所以對 results.json 裡每個 job 用同一 seed 重生成世代、只重跑 run_prediction，結果與當時整跑等價。
用途：codex 稽核後模組四新增了 refit 對照、同風險集 NRI、Pipeline 內補值——這些只影響 prediction 區塊。
同時以新規則（max_L |ΔC_L|、3 seed）重跑閘門二並更新 gates 與 assumptions.md。

用法：python recompute_prediction.py --out results [--jobs 12]
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from cohort import load_params, make_cohort
from prediction import run_prediction
from run import aggregate, _json_default, run_gates, make_figures, _val


def one(args):
    P, variant, seed, calib = args
    C = make_cohort(P, seed, tau=variant["tau"], delta_mu_median=calib["delta_mu"]["delta_mu_median"],
                    lam0=calib["hazard"]["lambda0_per_day"], beta=calib["hazard"]["beta_per_10_egfr"],
                    kappa=calib["hazard"]["kappa"], dropout=variant.get("dropout", False), start_mode=variant.get("start_mode"),
                    linear_onset=variant.get("linear_onset"))
    return variant["name"], seed, run_prediction(C, P, seed, dropout=variant.get("dropout", False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = ap.parse_args()
    path = os.path.join(a.out, "results.json")
    R = json.load(open(path, encoding="utf-8"))
    P = load_params(R["meta"]["params_file"], verbose=False)
    t0 = time.time()
    tasks = []
    for vname, rs in R["per_seed"].items():
        calib = R["calibration"][vname]
        for r in rs:
            if r.get("aborted"):
                continue
            tasks.append((P, calib["variant"], r["seed"], calib))
    print(f"重算模組四：{len(tasks)} 個 job，平行 {a.jobs} …")
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for vname, seed, pred in ex.map(one, tasks):
            for r in R["per_seed"][vname]:
                if r["seed"] == seed:
                    r["prediction"] = pred
            print(f"  {vname} seed={seed} 完成")
    R["aggregate"] = {v: aggregate([{k: x for k, x in r.items() if k != "fig"} for r in rs if not r.get("aborted")]) for v, rs in R["per_seed"].items()}
    thr = P["prediction"]["leak_alert_c_gain"]["value"]
    R["leak_alert"] = [(v, L, row["c_gain"]["mean"]) for v, A in R["aggregate"].items() if A and A.get("prediction") and "landmarks" in A["prediction"]
                       for L, row in A["prediction"]["landmarks"].items() if row["c_gain"]["mean"] > thr]
    R["leak_alert_vs_refit"] = [(v, L, row["c_gain_vs_refit"]["mean"]) for v, A in R["aggregate"].items() if A and A.get("prediction") and "landmarks" in A["prediction"]
                                for L, row in A["prediction"]["landmarks"].items() if row["c_gain_vs_refit"]["mean"] > thr]
    # 閘門二新規則
    gates = run_gates(P, R["calibration"]["default"], n=4000, n_gate=12000)
    R["gates"] = gates
    with open(os.path.join(a.out, "assumptions.md"), encoding="utf-8") as f:
        txt = f.read()
    head = txt.split("\n## 放行閘門")[0]
    with open(os.path.join(a.out, "assumptions.md"), "w", encoding="utf-8") as f:
        f.write(head + "\n## 放行閘門（v2 附錄參；模組四事後重算時以新規則重跑）\n\n| 閘門 | 值 | 條件 | 結果 |\n|---|---|---|---|\n")
        for g in gates:
            f.write(f"| {g['gate']} | {g['value']}{(' ' + str(g['detail'])) if g.get('detail') else ''} | {g['criterion']} | {'通過' if g['passed'] else '未過：' + g['fail_means']} |\n")
    if "figure_data" in R:
        fig_res = next(r for r in R["per_seed"]["default"] if r["seed"] == R["meta"]["seeds"][0])
        R["figure_data"]["timing"] = fig_res["prediction"]["timing"]
    R["meta"]["module4_recomputed"] = dict(date=time.strftime("%Y-%m-%d %H:%M"),
        note="codex 稽核後：Pipeline 內補值、同 estimand 的 refit 對照（auc_static_refit / c_gain_vs_refit）、同風險集同門檻 NRI（reclass_same_riskset_*）；閘門二改 max_L|ΔC_L|、3 seed。世代與 CV 切分由 seed 決定，重算與整跑等價。")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(R, f, ensure_ascii=False, indent=1, default=_json_default)
    if "figure_data" in R:
        make_figures(R, os.path.join(a.out, "figures"))
    print(f"完成 {time.time() - t0:.0f} s；閘門：", [(g["gate"][:4], g["value"], g["passed"]) for g in gates])
    for g in gates:
        print(f"  [{'通過' if g['passed'] else '未過'}] {g['gate']}：{g['value']}（{g['criterion']}）" + (f" detail={g['detail']}" if g.get("detail") else ""))


if __name__ == "__main__":
    main()
