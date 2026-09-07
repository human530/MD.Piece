# -*- coding: utf-8 -*-
"""腎病病因階層模型——真實 NHANES 1999–2004 資料。
    python pipeline.py [--seed 20260830]

Level 1  三大病因（免疫/感染/代謝）：有標籤者（免疫標籤來自 SSANA 隨機次樣本；感染/代謝來自全體）
Level 2  類內細項：免疫（特異自體抗體/IF 型態亞群）、感染（HBV vs HCV）——n 小者僅描述性，不做效能宣稱
Level 3  biomarker 歸因：LR 係數＋置換重要度＋逐標記單變量 AUC；供與文獻錨點對照（已知重現 vs 候選未知）

評估：重複分層 5 折（3 次重複）之 out-of-fold 機率；模型超參數固定事前（無內層調參＝無挑選）；
     一對其餘 AUROC＋bootstrap 95% CI、平衡正確率、混淆。測試協定事前鎖定，跑出多少報多少（AUC 目標不驅動事後調整）。
敏感度（事前列表）：排除血清葡萄糖（標籤鄰近）、排除重疊個案、僅 LR、逐週期留一。"""
import argparse
import json
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np                                    # noqa: E402
import pandas as pd                                   # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier   # noqa: E402
from sklearn.impute import SimpleImputer              # noqa: E402
from sklearn.linear_model import LogisticRegression   # noqa: E402
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score  # noqa: E402
from sklearn.model_selection import RepeatedStratifiedKFold   # noqa: E402
from sklearn.pipeline import make_pipeline            # noqa: E402
from sklearn.preprocessing import StandardScaler      # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402

from nhanes_cohort import build                       # noqa: E402
from provenance import summary as prov_summary        # noqa: E402

RESULTS = os.path.join(ROOT, "results")
CLASSES = ["免疫性", "感染性", "代謝性"]


def _jd(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    if isinstance(o, np.ndarray): return o.tolist()
    return str(o)


def _dump(obj, name):
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=_jd)


def models(seed):
    return dict(
        LR=make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(class_weight="balanced", max_iter=4000, random_state=seed)),
        HGB=HistGradientBoostingClassifier(random_state=seed, class_weight="balanced",
                                           max_depth=3, learning_rate=0.08, max_iter=300, l2_regularization=1.0))


def oof_proba(model_key, X, y, seed, n_rep=3, folds=5):
    """重複分層 K 折之 out-of-fold 機率（跨重複取平均）。超參數固定，無任何以測試表現為準的選擇。"""
    y = np.asarray(y)
    classes = np.array(sorted(set(y)))
    acc = np.zeros((len(y), len(classes))); cnt = np.zeros(len(y))
    cv = RepeatedStratifiedKFold(n_splits=folds, n_repeats=n_rep, random_state=seed)
    for tr, te in cv.split(X, y):
        m = models(seed)[model_key].fit(X[tr], y[tr])
        p = m.predict_proba(X[te])
        cols = {c: i for i, c in enumerate(m.classes_)}
        acc[te] += p[:, [cols[c] for c in classes]]
        cnt[te] += 1
    return acc / cnt[:, None], classes


def auc_ci(y_bin, p, n_boot, rng):
    ok = np.isfinite(p)
    y_bin, p = np.asarray(y_bin)[ok], np.asarray(p)[ok]
    if y_bin.sum() in (0, len(y_bin)):
        return dict(auc=None, lo=None, hi=None, n_pos=int(y_bin.sum()))
    a = roc_auc_score(y_bin, p)
    vals = []
    n = len(y_bin)
    for _ in range(n_boot):
        b = rng.integers(0, n, n)
        if 0 < y_bin[b].sum() < n:
            vals.append(roc_auc_score(y_bin[b], p[b]))
    v = np.array(vals)
    return dict(auc=float(a), lo=float(np.percentile(v, 2.5)), hi=float(np.percentile(v, 97.5)), n_pos=int(y_bin.sum()))


def level1(df, features, seed, nboot, tag, verbose=True):
    sub = df[df["cause"].isin(CLASSES)].copy()
    X = sub[features].to_numpy(float)
    y = sub["cause"].to_numpy()
    rng = np.random.default_rng(seed)
    out = dict(tag=tag, n=int(len(sub)), class_counts={c: int((y == c).sum()) for c in CLASSES}, models={})
    for mk in ("LR", "HGB"):
        proba, classes = oof_proba(mk, X, y, seed)
        pred = classes[proba.argmax(axis=1)]
        row = dict(balanced_accuracy=float(balanced_accuracy_score(y, pred)),
                   confusion=confusion_matrix(y, pred, labels=CLASSES).tolist(), labels=CLASSES,
                   ovr_auc={c: auc_ci((y == c).astype(int), proba[:, list(classes).index(c)], nboot, rng) for c in CLASSES})
        out["models"][mk] = row
        if verbose:
            aucs = "；".join(f"{c} {row['ovr_auc'][c]['auc']:.3f} [{row['ovr_auc'][c]['lo']:.3f},{row['ovr_auc'][c]['hi']:.3f}] (n={row['ovr_auc'][c]['n_pos']})"
                             for c in CLASSES if row["ovr_auc"][c]["auc"] is not None)
            print(f"[Level1:{tag}] {mk}: 平衡正確率 {row['balanced_accuracy']:.3f}｜一對其餘 AUROC {aucs}")
    out["oof"] = dict(seqn=sub["SEQN"].tolist(), y=y.tolist())
    return out, sub, X, y


def level3_attribution(sub, X, y, features, labels_zh, seed, nboot, verbose=True):
    """歸因（關聯，非因果）：HGB 置換重要度＋LR 係數＋逐標記單變量 AUC（一對其餘）。"""
    rng = np.random.default_rng(seed + 1)
    uni = {}
    for c in CLASSES:
        yb = (y == c).astype(int)
        rows = []
        for j, f in enumerate(features):
            v = X[:, j]
            r = auc_ci(yb[np.isfinite(v)], v[np.isfinite(v)], 200, rng)
            if r["auc"] is not None:
                a = r["auc"]
                rows.append(dict(feature=f, label=labels_zh.get(f, f), auc=max(a, 1 - a), direction=("高" if a >= 0.5 else "低"),
                                 lo=r["lo"], hi=r["hi"]))
        uni[c] = sorted(rows, key=lambda r: -r["auc"])[:15]
    imp_med = SimpleImputer(strategy="median").fit_transform(X)
    hgb = models(seed)["HGB"].fit(imp_med, y)
    pi = permutation_importance(hgb, imp_med, y, n_repeats=10, random_state=seed, scoring="balanced_accuracy")
    perm = sorted([dict(feature=f, label=labels_zh.get(f, f), importance=float(m), sd=float(s))
                   for f, m, s in zip(features, pi.importances_mean, pi.importances_std)], key=lambda r: -r["importance"])[:20]
    lr = models(seed)["LR"].fit(X, y)
    lrm = lr.named_steps["logisticregression"]
    coefs = {str(c): sorted([dict(feature=f, label=labels_zh.get(f, f), coef=float(w))
                             for f, w in zip(features, lrm.coef_[i])], key=lambda r: -abs(r["coef"]))[:15]
             for i, c in enumerate(lrm.classes_)}
    if verbose:
        for c in CLASSES:
            top = "、".join(f"{r['label']}({r['direction']},AUC {r['auc']:.2f})" for r in uni[c][:5])
            print(f"[Level3] {c} 前五單變量標記：{top}")
    return dict(univariate_top=uni, permutation_top=perm, lr_coef_top=coefs,
                note="關聯非因果；橫斷面資料無法區分『原因的標記』與『結果的標記』——此界線印在報告")


def level2_immune(sub_all_ana, verbose=True):
    """免疫類內部亞群（真實特異抗體標籤）：計數＋描述；n<15 之組不做效能宣稱。"""
    imm = sub_all_ana[sub_all_ana["cause"] == "免疫性"]
    g = {
        "SLE 相關（Sm/U1RNP/RiboP 任一+）": ((imm.get("SSSM") == 1) | (imm.get("SSU1RNP") == 1) | (imm.get("SSRIBOP") == 1)),
        "Ro/La（SSA/SSB 任一+）": ((imm.get("SSROSSA") == 1) | (imm.get("SSLASSB") == 1)),
        "硬皮病相關（TopoI/RNAPol 任一+）": ((imm.get("SSTOPOI") == 1) | (imm.get("SSRNAPOL") == 1)),
        "肌炎相關（Jo-1/PL/SRP/Mi-2 任一+）": ((imm.get("SSJO_1") == 1) | (imm.get("SSPL_7") == 1) |
                                              (imm.get("SSPL_12") == 1) | (imm.get("SSSRP") == 1) | (imm.get("SSMI_2") == 1)),
    }
    rows = {}
    for name, mask in g.items():
        m = mask.fillna(False) if hasattr(mask, "fillna") else pd.Series(False, index=imm.index)
        rows[name] = int(m.sum())
    rows["僅 ANA 型態陽性（無特異抗體）"] = int(len(imm) - int(pd.DataFrame({k: (v.fillna(False) if hasattr(v, 'fillna') else False) for k, v in g.items()}).any(axis=1).sum()))
    if verbose:
        print(f"[Level2 免疫] n={len(imm)}：{rows}（n<15 之亞群僅描述，不做效能宣稱）")
    return dict(n=int(len(imm)), subgroups=rows, caveat="亞群 n 皆小；此層為描述性，效能宣稱需臨床世代")


def level2_infection(kd, lab_meta, verbose=True):
    inf = kd[kd["cause"] == "感染性"]
    hb = lab_meta.get("hbsag_var"); hc = lab_meta.get("hcv_rna_var") or lab_meta.get("hcv_ab_var")
    rows = dict(HBV=int((inf[hb] == 1).sum()) if hb else None, HCV=int((inf[hc] == 1).sum()) if hc else None, n=int(len(inf)))
    if verbose:
        print(f"[Level2 感染] {rows}")
    return rows


def run(seed, verbose=True):
    P = json.load(open(os.path.join(ROOT, "params", "design.json"), encoding="utf-8"))
    nboot = 1000
    C = build(P, verbose=verbose)
    # Three-class analysis requires immune-label ascertainment for every participant.
    # Using the full kidney cohort would silently treat ANA-untested people as immune-negative.
    kd, all_kd, features = C["primary"], C["secondary"], C["features"]
    labels_zh = C["feature_labels"]

    # Level 1 主分析
    main, sub, X, y = level1(kd, features, seed, nboot, "主分析（ANA 實測、三類標籤可判定）", verbose)
    # 敏感度（事前列表）
    sens = {}
    feats_noglu = [f for f in features if f != "LBXSGL"]
    sens["排除血清葡萄糖"], *_ = level1(kd, feats_noglu, seed, 300, "sens:no_glucose", verbose)
    no_overlap = kd[~(((kd["lab_immune"] == 1) & (kd["lab_infection"] | kd["lab_metabolic"])) |
                      (kd["lab_infection"] & kd["lab_metabolic"]))]
    sens["排除重疊個案"], *_ = level1(no_overlap, features, seed, 300, "sens:no_overlap", verbose)
    cyc = {}
    for c in sorted(kd["cycle"].unique()):
        hold = kd[kd["cause"].isin(CLASSES)]
        tr, te = hold[hold["cycle"] != c], hold[hold["cycle"] == c]
        if len(te) < 30 or te["cause"].nunique() < 2:
            continue
        m = models(seed)["HGB"].fit(SimpleImputer(strategy="median").fit_transform(tr[features].to_numpy(float)), tr["cause"])
        p = m.predict_proba(SimpleImputer(strategy="median").fit(tr[features].to_numpy(float)).transform(te[features].to_numpy(float)))
        rngc = np.random.default_rng(seed)
        cyc[c] = {cl: auc_ci((te["cause"] == cl).astype(int).to_numpy(), p[:, list(m.classes_).index(cl)], 300, rngc)
                  for cl in CLASSES if cl in set(te["cause"])}
    sens["留一週期外測"] = cyc

    l3 = level3_attribution(sub, X, y, features, labels_zh, seed, nboot, verbose)
    l2i = level2_immune(kd[kd["in_ana_subsample"]], verbose)
    l2f = level2_infection(all_kd, C["counts"]["lab_meta"], verbose)

    # 預測輸出（供超音波佐證層以 SEQN≒patient_key 併接）
    proba, classes = oof_proba("HGB", X, y, seed)
    pd.DataFrame(dict(SEQN=sub["SEQN"].astype(int), cause_label=y,
                      **{f"p_{c}": proba[:, list(classes).index(c)] for c in CLASSES})) \
      .to_csv(os.path.join(RESULTS, "predictions.csv"), index=False, encoding="utf-8-sig")

    report = dict(
        seed=seed, date="2026-08-30",
        honesty=[
            "資料：NHANES 1999–2004 三週期（真實人類、公開下載，出處帳本 results/provenance.json 逐檔 SHA256）。",
            "標籤是共病代理（問卷診斷＋血清學＋surplus sera 自體抗體），不是切片病因；免疫標籤僅在隨機 1/3 次樣本可判定。",
            "橫斷面資料：所有歸因為『關聯』，不是因果；『導致』需要縱貫或介入證據。",
            "未使用調查權重（目的是判別不是盛行率推估）——列為限制。",
            "三類主分析中的感染類僅 n=7：不適合穩健效能宣稱；全腎損傷世代的感染亞型只做描述。",
            "AUC 目標 0.9 只驅動設計；本報告數字為鎖定協定下的實跑結果，未做任何事後調整。"],
        cohort=C["counts"], features=dict(n=len(features), cols=features, archive=C["archive"]),
        level1=main, sensitivity=sens, level2_immune=l2i, level2_infection=l2f, level3=l3,
        provenance=prov_summary())
    _dump(report, "report.json")
    if verbose:
        print("[完成] results/report.json、predictions.csv")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260830)
    a = ap.parse_args()
    run(a.seed)
