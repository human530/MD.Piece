# -*- coding: utf-8 -*-
"""重新設定問題：把被標籤結構破壞的三類判別，拆成各自標籤乾淨的二元任務。
    python binary_tasks.py [--seed 20260830]

## 為什麼要改（這是問題設定的調整，不是調參）

三類設定有一個無解的兩難：
  * 用「全體腎損傷者」→ 感染 32／代謝 659 樣本夠，但未做 ANA 者被當成非免疫＝**免疫標籤污染**。
  * 用「ANA 實測子樣本」→ 免疫標籤乾淨，但感染只剩 **n=7**，該欄位的數字沒有意義。
兩者都無法同時成立，因為 NHANES 的 ANA 只做在隨機次樣本上。

正確的做法不是在三類上調參，而是**只問標籤成立的問題**：
  T1 免疫 vs 非免疫   ── 限 ANA 實測者（n=488）：兩邊都由實測 ANA 定義，標籤乾淨
  T2 感染 vs 代謝     ── 全體腎損傷（n=691）：兩邊都由血清學／問卷定義，與 ANA 無關
  T3 感染 vs 其餘全部  ── 全體腎損傷（n=2116，陽性 32）：極不平衡，主指標為 AUPRC
  T4 代謝 vs 其餘全部  ── 排除感染者後（n=2084）：避免與 T2 重複計數

## 誠實規則（與腎炎代理評估一致）
  * 5×5 重複分層巢狀交叉驗證；門檻只在內層選，外層只評估。
  * 病人層 bootstrap 95% CI；**極不平衡任務以 AUPRC 為主指標**，並列出盛行率作為 AUPRC 的隨機基準線。
  * 沒有未動用的保留集——三類的保留集已用掉一次，不得回收給新任務。此處全部是重複巢狀 CV，不是外部驗證。
  * 每個 run 追加至 results/runs_log.jsonl。"""
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
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,  # noqa: E402
                             brier_score_loss, roc_auc_score)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold    # noqa: E402

from nhanes_cohort import build                        # noqa: E402
import pipeline as PL                                  # noqa: E402

RESULTS = os.path.join(ROOT, "results")


def boot_ci(y, p, fn, n_boot, rng):
    n = len(y)
    vals = []
    for _ in range(n_boot):
        b = rng.integers(0, n, n)
        if 0 < y[b].sum() < n:
            vals.append(fn(y[b], p[b]))
    if not vals:
        return [None, None]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def nested_binary(X, y, seed, model_key, repeats=5, folds=5, n_boot=1000, verbose=True):
    """外層重複分層 K 折只評估；門檻由內層 CV 選（最大化平衡正確率）。回傳病人層彙總。"""
    y = np.asarray(y).astype(int)
    oof = np.zeros(len(y)); cnt = np.zeros(len(y)); thr_sel = []
    cv = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
    for tr, te in cv.split(X, y):
        m = PL.models(seed)[model_key].fit(X[tr], y[tr])
        # 內層選門檻（只用訓練折）
        inner = StratifiedKFold(folds, shuffle=True, random_state=seed)
        ip = np.zeros(len(tr))
        for itr, ite in inner.split(X[tr], y[tr]):
            mi = PL.models(seed)[model_key].fit(X[tr][itr], y[tr][itr])
            ip[ite] = mi.predict_proba(X[tr][ite])[:, 1]
        grid = np.quantile(ip, np.linspace(0.05, 0.95, 37))
        best = max(grid, key=lambda t: balanced_accuracy_score(y[tr], (ip >= t).astype(int)))
        thr_sel.append(float(best))
        oof[te] += m.predict_proba(X[te])[:, 1]; cnt[te] += 1
    p = oof / np.maximum(cnt, 1)
    rng = np.random.default_rng(seed)
    thr = float(np.median(thr_sel))
    pred = (p >= thr).astype(int)
    prev = float(y.mean())
    out = dict(
        model=model_key, n=int(len(y)), n_pos=int(y.sum()), prevalence=prev,
        auroc=float(roc_auc_score(y, p)), auroc_ci=boot_ci(y, p, roc_auc_score, n_boot, rng),
        auprc=float(average_precision_score(y, p)),
        auprc_ci=boot_ci(y, p, average_precision_score, n_boot, np.random.default_rng(seed + 1)),
        auprc_baseline=prev, auprc_lift=float(average_precision_score(y, p) / prev),
        balanced_accuracy=float(balanced_accuracy_score(y, pred)),
        sensitivity=float(pred[y == 1].mean()) if y.sum() else None,
        specificity=float(1 - pred[y == 0].mean()) if (y == 0).sum() else None,
        brier=float(brier_score_loss(y, p)), threshold_median_inner=thr,
        protocol=f"{repeats}×{folds} 重複分層巢狀 CV；門檻僅內層選；病人層 bootstrap {n_boot} 次")
    if verbose:
        print(f"  {model_key}: AUROC {out['auroc']:.3f} [{out['auroc_ci'][0]:.3f},{out['auroc_ci'][1]:.3f}]"
              f"｜AUPRC {out['auprc']:.3f}（基準 {prev:.3f}，提升 ×{out['auprc_lift']:.1f}）"
              f"｜平衡正確率 {out['balanced_accuracy']:.3f}")
    return out


# ── 標籤鄰近特徵（必須拔除才算誠實）：與標籤定義的生物學直接相連者
LABEL_ADJACENT = {
    "metabolic": ["LBXSGL", "LBXSOSSI"],                       # 血糖／滲透壓 ↔ HbA1c、糖尿病診斷
    "infection": ["LBXSATSI", "LBXSASSI", "LBXSGTSI", "LBXSTB"],  # 肝酶／膽紅素 ↔ B/C 肝血清學
    "immune": [],                                               # ANA 由 surplus sera 測定，常規檢驗中無直接下游
}

# ── 腎臟專屬指標組（使用者指定：以腎臟為對象的 biomarker 與檢驗標準）
KIDNEY_CORE = [
    "eGFR", "ACR", "URXUMA", "URXUCR",                          # 腎功能與白蛋白尿（KDIGO 兩軸）
    "LBXSCR", "LBXSBU", "SSCYPC",                               # 肌酸酐、尿素氮、胱蛋白酶抑制素 C
    "LBXSUA", "LBXSAL", "LBXSTP", "LBXSGB",                     # 尿酸、白蛋白、總蛋白、球蛋白
    "LBXSPH", "LBXSCA", "LBXPT21", "LBDVIDMS", "LBXSAPSI",      # CKD-MBD：磷、鈣、PTH、維生素D、ALP
    "LBXHGB", "LBXHCT", "LBXRDW", "LBXMCVSI", "LBXFER", "LBXSIR",  # 腎性貧血與鐵代謝
    "LBXSNASI", "LBXSKSI", "LBXSCLSI", "LBXSC3SI",              # 電解質與酸鹼（碳酸氫根）
    "LBXCRP", "NLR", "LBXWBCSI", "LBXLYPCT", "LBXPLTSI",        # 發炎與血球
    "LBXHCY", "age", "sex",
]


def kdigo(df):
    """KDIGO 2012 分期：G1–G5（eGFR）× A1–A3（ACR），回傳 (G, A, 風險類別)。

    缺值必須明確標為「未知」——np.select 的 default 會把 NaN 掃進最嚴重的一格
    （本專案 eGFR 缺值 708 人會被誤判為 G5，實際 G5 僅 26 人）。"""
    e, a_ = df["eGFR"].to_numpy(float), df["ACR"].to_numpy(float)
    g = np.select([np.isnan(e), e >= 90, e >= 60, e >= 45, e >= 30, e >= 15],
                  ["未知", "G1", "G2", "G3a", "G3b", "G4"], default="G5")
    a = np.select([np.isnan(a_), a_ < 30, a_ < 300], ["未知", "A1", "A2"], default="A3")
    order_g = {"G1": 0, "G2": 1, "G3a": 2, "G3b": 3, "G4": 4, "G5": 5}
    order_a = {"A1": 0, "A2": 1, "A3": 2}
    risk = []
    for gi, ai in zip(g, a):
        if gi == "未知" or ai == "未知":
            risk.append("無法分期")            # 任一軸缺值即不分期，不猜
            continue
        sc = order_g[gi] + order_a[ai]
        risk.append("低" if sc <= 1 else "中" if sc <= 2 else "高" if sc <= 4 else "極高")
    return g, a, np.array(risk, dtype=object)


def run(seed=20260830, verbose=True):
    P = json.load(open(os.path.join(ROOT, "params", "design.json"), encoding="utf-8"))
    C = build(P, verbose=False)
    kd, prim, feats = C["secondary"], C["primary"], C["features"]
    for d in (kd, prim):
        d["kdigo_G"], d["kdigo_A"], d["kdigo_risk"] = kdigo(d)
    if verbose:
        import collections
        print("[KDIGO] 全體腎損傷者分期：G", dict(collections.Counter(kd["kdigo_G"])),
              "｜A", dict(collections.Counter(kd["kdigo_A"])), "｜風險", dict(collections.Counter(kd["kdigo_risk"])))
    tasks = {
        "T1_免疫vs非免疫（ANA 實測者）": dict(
            df=prim, pos=lambda d: (d["lab_immune"] == 1), adj="immune",
            note="兩邊都由實測 ANA 定義——標籤乾淨；與 nephritis_proxy_repeated_eval 為同一問題之獨立實作，可互為檢核"),
        "T2_感染vs代謝（標籤皆乾淨）": dict(
            df=kd[kd["cause"].isin(["感染性", "代謝性"])], pos=lambda d: (d["cause"] == "感染性"), adj="infection",
            note="兩類皆由血清學／問卷定義，與 ANA 無關；這是三類設定被迫犧牲、但實際可解的問題"),
        "T3_感染vs其餘全部（極不平衡）": dict(
            df=kd, pos=lambda d: d["lab_infection"].fillna(False).astype(bool), adj="infection",
            note="全體腎損傷者；盛行率極低，主指標為 AUPRC 與其相對基準線的提升倍數"),
        "T4_代謝vs其餘（已排除感染者）": dict(
            df=kd[~kd["lab_infection"].fillna(False).astype(bool)], pos=lambda d: d["lab_metabolic"].fillna(False).astype(bool),
            adj="metabolic", note="排除感染者以免與 T2 重複計數；標籤由 HbA1c／糖尿病問卷定義"),
    }
    out = dict(seed=seed, created=time.strftime("%Y-%m-%dT%H:%M:%S"), n_features=len(feats),
               rationale="三類設定的標籤兩難無解（見檔頭）；改為只問標籤成立的二元問題。此為問題設定之調整，非調參。",
               no_holdout="三類保留集已用掉一次，不得回收給新任務；本檔全部為重複巢狀 CV，不是外部驗證。",
               tasks={})
    for name, spec in tasks.items():
        d = spec["df"]
        y = spec["pos"](d).astype(int).to_numpy()
        X = d[feats].to_numpy(float)
        if y.sum() < 10 or (y == 0).sum() < 10:
            out["tasks"][name] = dict(skipped=f"陽性 {int(y.sum())}／陰性 {int((y==0).sum())}——樣本不足，拒跑")
            if verbose:
                print(f"[{name}] 跳過：{out['tasks'][name]['skipped']}")
            continue
        if verbose:
            print(f"[{name}] n={len(y)}、陽性 {int(y.sum())}（盛行率 {y.mean():.3f}）")
        # 三個特徵集變體：全部／拔掉標籤鄰近（誠實主結果）／腎臟專屬指標組
        adj = LABEL_ADJACENT.get(spec["adj"], [])
        variants = {
            "all": [f for f in feats],
            "leak_free": [f for f in feats if f not in adj],
            "kidney_core": [f for f in KIDNEY_CORE if f in d.columns and f not in adj],
        }
        res = {}
        for vname, vfeats in variants.items():
            if verbose:
                tag = {"all": "全部特徵", "leak_free": f"拔除標籤鄰近{adj or '（無）'}", "kidney_core": "腎臟專屬指標組"}[vname]
                print(f" ── {tag}（{len(vfeats)} 欄）")
            Xv = d[vfeats].to_numpy(float)
            res[vname] = {mk: nested_binary(Xv, y, seed, mk, verbose=verbose) for mk in ("LR", "HGB")}
            for r in res[vname].values():
                r["n_features"] = len(vfeats)
        head = res["leak_free"]["HGB"]
        out["tasks"][name] = dict(note=spec["note"], label_adjacent_removed=adj, variants=res,
                                  headline=dict(variant="leak_free", model="HGB",
                                                auroc=head["auroc"], auroc_ci=head["auroc_ci"],
                                                auprc=head["auprc"], auprc_baseline=head["auprc_baseline"]))
        # KDIGO 分層描述（使用者指定之檢驗標準）
        out["tasks"][name]["kdigo_positive_mix"] = {k: int(v) for k, v in
                                                    d.loc[y == 1, "kdigo_risk"].value_counts().items()}
        with open(os.path.join(RESULTS, "runs_log.jsonl"), "a", encoding="utf-8") as f:
            for vname, mm in res.items():
                for mk, r in mm.items():
                    f.write(json.dumps(dict(kind="binary_task", task=name, variant=vname, at=out["created"], **r),
                                       ensure_ascii=False) + "\n")
    PL._dump(out, "binary_tasks.json")
    if verbose:
        print("[完成] results/binary_tasks.json")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260830)
    a = ap.parse_args()
    run(a.seed)
