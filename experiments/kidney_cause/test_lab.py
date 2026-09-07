# -*- coding: utf-8 -*-
"""模型測試實驗室——「不斷測試」而不自欺的機制。
    python test_lab.py            # 跑整個 battery（只用開發集）
    python test_lab.py --final HGB_full   # 用鎖定保留集評「一個」最終設定，一生一次

防自欺三件套：
  1. **鎖定保留集**：第一次執行即以 seed 抽 25%（依病因分層）存 results/holdout_seqn.json；
     battery 永不觸碰；--final 只能跑一次（檔案存在即拒絕，除非 --force-reason 說明理由並留紀錄）。
  2. **全數記錄**：每個 run（設定、AUC、時間）追加到 results/runs_log.jsonl——報告呈現全部 run，不挑好看的。
  3. **標記研究表**：每個特徵 × 病因的單變量 AUC＋可得率＋逐週期覆蓋，全部列出（不是只列前十）。

battery 內容（開發集，重複分層 5 折 OOF）：
  A 標記全表：feature × class 單變量 AUC（95% CI）＋缺值率
  B 群組增量：base(生化+CBC+CRP+腎功能) → +脂質盤 → +重金屬 → +營養素 → +骨礦(VitD/PTH/BAP) → +CystatinC → full
  C 群組移除：full 減一群的 ΔAUC
  D 種子穩定性：full × 5 seeds 的 AUC 平均±SD
  E 留一週期外測：full"""
import argparse
import hashlib
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

import numpy as np                                    # noqa: E402
import pandas as pd                                   # noqa: E402
from sklearn.impute import SimpleImputer              # noqa: E402
from sklearn.metrics import balanced_accuracy_score   # noqa: E402

from nhanes_cohort import build                       # noqa: E402
import pipeline as PL                                 # noqa: E402

RESULTS = os.path.join(ROOT, "results")
CLS = PL.CLASSES
GROUPS = {
    "base": ["LBXSCR", "LBXSBU", "LBXSUA", "LBXSAL", "LBXSGL", "LBXSCH", "LBXSTR", "LBXSGTSI", "LBXSASSI", "LBXSATSI",
             "LBXSLDSI", "LBXSAPSI", "LBXSTB", "LBXSTP", "LBXSGB", "LBXSPH", "LBXSCA", "LBXSNASI", "LBXSKSI", "LBXSCLSI",
             "LBXSC3SI", "LBXSIR", "LBXSOSSI", "LBXWBCSI", "LBXLYPCT", "LBXMOPCT", "LBXNEPCT", "LBXEOPCT", "LBXBAPCT",
             "LBXRBCSI", "LBXHGB", "LBXHCT", "LBXMCVSI", "LBXMC", "LBXMCHSI", "LBXRDW", "LBXPLTSI", "LBXMPSI", "LBXCRP",
             "URXUMA", "URXUCR", "ACR", "eGFR", "NLR", "age", "sex"],
    "lipids": ["LBXTC", "LBDHDL", "LBDLDL", "LBXTR"],
    "metals": ["LBXBPB", "LBXBCD", "LBXTHG", "LBXCOT"],
    "nutrition": ["LBXFER", "LBXFOL", "LBXRBF", "LBXB12", "LBXHCY", "LBXMMA"],
    "bone_mineral": ["LBDVIDMS", "LBXPT21", "LBXBAP"],
    "cystatin": ["SSCYPC"],
}


def log_run(rec):
    os.makedirs(RESULTS, exist_ok=True)
    rec = dict(rec, at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    with open(os.path.join(RESULTS, "runs_log.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=PL._jd) + "\n")


def get_holdout(sub, seed):
    p = os.path.join(RESULTS, "holdout_seqn.json")
    if os.path.exists(p):
        doc = json.load(open(p, encoding="utf-8"))
        assert hashlib.sha256(json.dumps(sorted(doc["seqn"])).encode()).hexdigest()[:16] == doc["hash"], "保留集檔案被改動"
        return set(doc["seqn"])
    rng = np.random.default_rng(seed + 777)
    hold = []
    for c in CLS:
        ids = sub.loc[sub["cause"] == c, "SEQN"].to_numpy()
        hold += list(rng.choice(ids, size=max(1, int(round(0.25 * len(ids)))), replace=False))
    hold = sorted(int(x) for x in hold)
    doc = dict(note="鎖定保留集：battery 永不觸碰；--final 只能評一次", frac=0.25, seed=seed + 777,
               seqn=hold, hash=hashlib.sha256(json.dumps(sorted(hold)).encode()).hexdigest()[:16],
               created=time.strftime("%Y-%m-%dT%H:%M:%S"))
    json.dump(doc, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[holdout] 建立鎖定保留集 n={len(hold)}（免疫/感染/代謝 各 25%）→ holdout_seqn.json")
    return set(hold)


def eval_config(dev, feats, seed, nboot=300, tag=""):
    X = dev[feats].to_numpy(float); y = dev["cause"].to_numpy()
    proba, classes = PL.oof_proba("HGB", X, y, seed)
    pred = classes[proba.argmax(axis=1)]
    rng = np.random.default_rng(seed)
    aucs = {c: PL.auc_ci((y == c).astype(int), proba[:, list(classes).index(c)], nboot, rng) for c in CLS}
    row = dict(tag=tag, n=int(len(dev)), n_features=len(feats),
               balanced_accuracy=float(balanced_accuracy_score(y, pred)),
               ovr_auc=aucs, mean_auc=float(np.mean([a["auc"] for a in aucs.values() if a["auc"] is not None])))
    log_run(dict(kind="config", seed=seed, features_hash=hashlib.sha256(",".join(sorted(feats)).encode()).hexdigest()[:12], **row))
    return row


def battery(seed=20260830, verbose=True):
    P = json.load(open(os.path.join(ROOT, "params", "design.json"), encoding="utf-8"))
    C = build(P, verbose=verbose)
    kd = C["secondary"]; sub = kd[kd["cause"].isin(CLS)].copy()
    hold = get_holdout(sub, seed)
    dev = sub[~sub["SEQN"].isin(hold)].copy()
    print(f"[lab] 開發集 n={len(dev)}（保留集 n={len(sub) - len(dev)} 鎖定不碰）：" +
          "、".join(f"{c} {int((dev['cause'] == c).sum())}" for c in CLS))
    have = set(dev.columns)
    groups = {g: [f for f in fs if f in have] for g, fs in GROUPS.items()}

    # A 標記全表
    rng = np.random.default_rng(seed)
    marker_table = []
    all_feats = [f for fs in groups.values() for f in fs]
    for f in all_feats:
        v = dev[f].to_numpy(float)
        row = dict(feature=f, label=C["feature_labels"].get(f, f), missing=float(np.mean(~np.isfinite(v))),
                   group=[g for g, fs in groups.items() if f in fs][0],
                   by_cycle_avail={cy: float(dev.loc[dev["cycle"] == cy, f].notna().mean()) for cy in sorted(dev["cycle"].unique())})
        for c in CLS:
            yb = (dev["cause"] == c).astype(int).to_numpy()
            ok = np.isfinite(v)
            r = PL.auc_ci(yb[ok], v[ok], 200, rng) if ok.sum() > 30 else dict(auc=None, lo=None, hi=None, n_pos=0)
            a = r["auc"]
            row[c] = dict(auc=(max(a, 1 - a) if a is not None else None), direction=("高" if (a or 0.5) >= 0.5 else "低"),
                          lo=r["lo"], hi=r["hi"])
        marker_table.append(row)
    log_run(dict(kind="marker_table", n_markers=len(marker_table)))

    # B 群組增量
    order = ["base", "lipids", "metals", "nutrition", "bone_mineral", "cystatin"]
    inc = []
    feats = []
    for g in order:
        feats = feats + groups[g]
        inc.append(dict(step=("+" + g if g != "base" else "base"), **eval_config(dev, feats, seed, tag=f"inc:{g}")))
        if verbose:
            r = inc[-1]
            print(f"[B 增量] {r['step']:14s} 特徵 {r['n_features']:3d}｜免疫 {r['ovr_auc']['免疫性']['auc']:.3f}｜"
                  f"感染 {r['ovr_auc']['感染性']['auc']:.3f}｜代謝 {r['ovr_auc']['代謝性']['auc']:.3f}｜平均 {r['mean_auc']:.3f}")
    full_feats = feats

    # C 群組移除
    drops = []
    for g in order:
        rest = [f for f in full_feats if f not in groups[g]]
        r = eval_config(dev, rest, seed, tag=f"drop:{g}")
        drops.append(dict(dropped=g, **r))
        if verbose:
            print(f"[C 移除] full−{g:12s} 平均 AUC {r['mean_auc']:.3f}（Δ {r['mean_auc'] - inc[-1]['mean_auc']:+.3f}）")

    # D 種子穩定性
    seed_rows = [eval_config(dev, full_feats, s, nboot=100, tag=f"seed:{s}") for s in range(seed, seed + 5)]
    stab = {c: dict(mean=float(np.mean([r["ovr_auc"][c]["auc"] for r in seed_rows])),
                    sd=float(np.std([r["ovr_auc"][c]["auc"] for r in seed_rows]))) for c in CLS}
    if verbose:
        print("[D 穩定] " + "、".join(f"{c} {v['mean']:.3f}±{v['sd']:.3f}" for c, v in stab.items()))

    # E 留一週期外測（full）
    cyc_rows = {}
    for cy in sorted(dev["cycle"].unique()):
        tr, te = dev[dev["cycle"] != cy], dev[dev["cycle"] == cy]
        if len(te) < 30 or te["cause"].nunique() < 2:
            continue
        imp = SimpleImputer(strategy="median").fit(tr[full_feats].to_numpy(float))
        m = PL.models(seed)["HGB"].fit(imp.transform(tr[full_feats].to_numpy(float)), tr["cause"])
        p = m.predict_proba(imp.transform(te[full_feats].to_numpy(float)))
        rngc = np.random.default_rng(seed)
        cyc_rows[cy] = {c: PL.auc_ci((te["cause"] == c).astype(int).to_numpy(), p[:, list(m.classes_).index(c)], 200, rngc)
                        for c in CLS if c in set(te["cause"])}
    out = dict(seed=seed, dev_n=int(len(dev)), holdout_n=int(len(sub) - len(dev)),
               dev_counts={c: int((dev["cause"] == c).sum()) for c in CLS},
               marker_table=marker_table, incremental=inc, group_drop=drops, seed_stability=stab,
               loco=cyc_rows, groups={g: fs for g, fs in groups.items()},
               note="全部在開發集；保留集鎖定未觸碰。所有 run 見 runs_log.jsonl（不挑好看的）。")
    PL._dump(out, "lab_report.json")
    print("[lab] 完成 → results/lab_report.json")
    return out


def final_eval(config_name, seed=20260830, force_reason=None):
    p_done = os.path.join(RESULTS, "holdout_eval.json")
    if os.path.exists(p_done) and not force_reason:
        raise SystemExit("保留集已評過一次（holdout_eval.json 存在）。再評＝多重比較，需 --force-reason 說明並留紀錄。")
    P = json.load(open(os.path.join(ROOT, "params", "design.json"), encoding="utf-8"))
    C = build(P, verbose=False)
    kd = C["secondary"]; sub = kd[kd["cause"].isin(CLS)].copy()
    hold = get_holdout(sub, seed)
    dev, te = sub[~sub["SEQN"].isin(hold)], sub[sub["SEQN"].isin(hold)]
    have = set(sub.columns)
    groups = {g: [f for f in fs if f in have] for g, fs in GROUPS.items()}
    assert config_name == "HGB_full", "目前唯一登記的最終設定是 HGB_full（全部群組）"
    feats = [f for fs in groups.values() for f in fs]
    imp = SimpleImputer(strategy="median").fit(dev[feats].to_numpy(float))
    m = PL.models(seed)["HGB"].fit(imp.transform(dev[feats].to_numpy(float)), dev["cause"])
    p = m.predict_proba(imp.transform(te[feats].to_numpy(float)))
    rng = np.random.default_rng(seed)
    aucs = {c: PL.auc_ci((te["cause"] == c).astype(int).to_numpy(), p[:, list(m.classes_).index(c)], 1000, rng) for c in CLS}
    pred = np.asarray(m.classes_)[p.argmax(axis=1)]
    doc = dict(config=config_name, holdout_n=int(len(te)), counts={c: int((te["cause"] == c).sum()) for c in CLS},
               ovr_auc=aucs, balanced_accuracy=float(balanced_accuracy_score(te["cause"], pred)),
               evaluated_once_at=time.strftime("%Y-%m-%dT%H:%M:%S"), force_reason=force_reason)
    PL._dump(doc, "holdout_eval.json")
    log_run(dict(kind="FINAL_HOLDOUT", **{k: v for k, v in doc.items() if k != "ovr_auc"}))
    print("[final] 保留集（一生一次）：" + "、".join(
        f"{c} {aucs[c]['auc']:.3f} [{aucs[c]['lo']:.3f},{aucs[c]['hi']:.3f}] (n={aucs[c]['n_pos']})" for c in CLS))
    return doc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260830)
    ap.add_argument("--final", type=str, default=None)
    ap.add_argument("--force-reason", type=str, default=None)
    a = ap.parse_args()
    if a.final:
        final_eval(a.final, a.seed, a.force_reason)
    else:
        battery(a.seed)
