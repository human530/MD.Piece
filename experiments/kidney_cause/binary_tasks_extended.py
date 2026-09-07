# -*- coding: utf-8 -*-
"""擴充世代（NHANES 1999–2018，十個週期）上的感染／代謝二元任務。
    python binary_tasks_extended.py [--seed 20260830]

## 為什麼只跑 T2/T3/T4
免疫標籤來自 surplus sera ANA，只做過 1999–2004。2005 年後的受試者免疫狀態是**未知**而非陰性——
把他們當非免疫就是先前已修掉的標籤污染。因此免疫任務（T1）維持在 ANA 實測子樣本上，不使用本檔世代。

## 與 1999–2004 版的可比性
* 相同的納入條件、標籤定義、特徵政策、評估協定（5×5 重複巢狀 CV、門檻僅內層選）。
* 肌酸酐依 NHANES 官方注記逐週期校正（1999-2000、2005-2006 兩個週期需校正）。
* 新增**逐週期留一外測**：以「留出整個週期」模擬時間外推，這是小世代做不到、擴充後才可行的檢查。
* 樣本擴大會使 CI 收窄；AUROC 本身變動則需區分「真的更好」與「族群組成改變」——故一併報告逐週期結果。"""
import argparse
import json
import os
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

import numpy as np                                     # noqa: E402
from sklearn.impute import SimpleImputer               # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score   # noqa: E402

from nhanes_cohort import build_extended               # noqa: E402
import pipeline as PL                                  # noqa: E402
from binary_tasks import LABEL_ADJACENT, KIDNEY_CORE, kdigo, nested_binary   # noqa: E402

RESULTS = os.path.join(ROOT, "results")


def leave_one_cycle_out(d, feats, y, seed, model_key="HGB", n_boot=300):
    """留出整個週期作外測——擴充後才做得到的時間外推檢查。"""
    out = {}
    for cy in sorted(d["cycle"].unique()):
        te = (d["cycle"] == cy).to_numpy()
        if y[te].sum() < 5 or y[~te].sum() < 5:
            out[str(cy)] = dict(skipped=f"陽性不足（測試 {int(y[te].sum())}／訓練 {int(y[~te].sum())}）")
            continue
        Xtr, Xte = d.loc[~te, feats].to_numpy(float), d.loc[te, feats].to_numpy(float)
        imp = SimpleImputer(strategy="median").fit(Xtr)
        m = PL.models(seed)[model_key].fit(imp.transform(Xtr), y[~te])
        p = m.predict_proba(imp.transform(Xte))[:, 1]
        rng = np.random.default_rng(seed)
        n = te.sum()
        vals = [roc_auc_score(y[te][b], p[b]) for b in (rng.integers(0, n, n) for _ in range(n_boot))
                if 0 < y[te][b].sum() < n]
        out[str(cy)] = dict(n=int(n), n_pos=int(y[te].sum()), auroc=float(roc_auc_score(y[te], p)),
                            auprc=float(average_precision_score(y[te], p)), prevalence=float(y[te].mean()),
                            ci=[float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else [None, None])
    return out


def run(seed=20260830, verbose=True):
    P = json.load(open(os.path.join(ROOT, "params", "design.json"), encoding="utf-8"))
    E = build_extended(P, verbose=verbose)
    kd, feats = E["cohort"], E["features"]
    kd["kdigo_G"], kd["kdigo_A"], kd["kdigo_risk"] = kdigo(kd)

    tasks = {
        "T2_感染vs代謝": dict(df=kd[kd["cause"].isin(["感染性", "代謝性"])],
                              pos=lambda d: (d["cause"] == "感染性"), adj="infection"),
        "T3_感染vs其餘全部": dict(df=kd, pos=lambda d: d["lab_infection"].fillna(False).astype(bool), adj="infection"),
        "T4_代謝vs其餘": dict(df=kd[~kd["lab_infection"].fillna(False).astype(bool)],
                              pos=lambda d: d["lab_metabolic"].fillna(False).astype(bool), adj="metabolic"),
    }
    out = dict(seed=seed, created=time.strftime("%Y-%m-%dT%H:%M:%S"), cohort=E["counts"],
               cycles="NHANES 1999–2018（10 個週期）", n_features=len(feats),
               immune_excluded="免疫任務不在此檔——ANA 僅 1999–2004 做過，2005 年後免疫狀態未知而非陰性",
               kdigo_overall={k: {str(a): int(b) for a, b in kd[f"kdigo_{k}"].value_counts().items()}
                              for k in ("G", "A", "risk")},
               tasks={})
    for name, spec in tasks.items():
        d = spec["df"].copy()
        y = spec["pos"](d).astype(int).to_numpy()
        adj = LABEL_ADJACENT.get(spec["adj"], [])
        if verbose:
            print(f"[{name}] n={len(y)}、陽性 {int(y.sum())}（盛行率 {y.mean():.4f}）")
        variants = {"all": list(feats),
                    "leak_free": [f for f in feats if f not in adj],
                    "kidney_core": [f for f in KIDNEY_CORE if f in d.columns and f not in adj]}
        res = {}
        for vname, vf in variants.items():
            if verbose:
                print(f" ── {vname}（{len(vf)} 欄）")
            Xv = d[vf].to_numpy(float)
            res[vname] = {mk: nested_binary(Xv, y, seed, mk, verbose=verbose) for mk in ("LR", "HGB")}
            for r in res[vname].values():
                r["n_features"] = len(vf)
        loco = leave_one_cycle_out(d, variants["leak_free"], y, seed)
        head = res["leak_free"]["HGB"]
        out["tasks"][name] = dict(label_adjacent_removed=adj, variants=res, leave_one_cycle_out=loco,
                                  headline=dict(variant="leak_free", model="HGB", auroc=head["auroc"],
                                                auroc_ci=head["auroc_ci"], auprc=head["auprc"],
                                                auprc_baseline=head["auprc_baseline"]))
        if verbose:
            got = {c: round(v["auroc"], 3) for c, v in loco.items() if "auroc" in v}
            print(f" ── 留一週期外測（leak_free, HGB）：{got}")
        with open(os.path.join(RESULTS, "runs_log.jsonl"), "a", encoding="utf-8") as f:
            for vname, mm in res.items():
                for mk, r in mm.items():
                    f.write(json.dumps(dict(kind="binary_task_extended", task=name, variant=vname,
                                            at=out["created"], **r), ensure_ascii=False) + "\n")
    PL._dump(out, "binary_tasks_extended.json")
    if verbose:
        print("[完成] results/binary_tasks_extended.json")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260830)
    a = ap.parse_args()
    run(a.seed)
