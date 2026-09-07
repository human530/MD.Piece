# -*- coding: utf-8 -*-
"""方向判定工具：丟入常規檢驗數值 → 輸出感染／代謝兩軸的「大概方向」。

    python direction.py train                 # 訓練並存成可攜參數（一次）
    python direction.py predict values.json   # 對新病人判定
    python direction.py demo                  # 用保留集真實病人示範

## 這個工具能做什麼、不能做什麼

| 軸 | 能否給方向 | 依據 |
|---|---|---|
| 感染方向 | ✅ 大概方向 | AUROC 0.79–0.82，十週期外測穩定 |
| 代謝方向 | ✅ 大概方向 | AUROC 0.82；**若你手上有 HbA1c，直接看它，不需要模型** |

**免疫軸已自本工具移除**（2026-09-03）：常規檢驗對免疫性腎損傷無訊號（AUROC 0.575，
60 標記僅「性別」通過 FDR），與其輸出「無法判定」佔版面，不如專注兩個有訊號的軸。
免疫軸待醫院切片資料（含 C3/C4/anti-dsDNA）到位後另建。

「大概方向」的意思：**排序有意義，但單一個人的判定可能錯**。
它告訴你「這組數值比較像哪個方向」，不是「這個人是什麼病」。

## 輸出格式

每個軸給三樣東西：校準機率、相對盛行率的倍數、三段判定（傾向／不確定／不傾向）。

另附「主要依據」：哪幾個數值把這個人推向該方向。這是「根據那些數值」的可追溯部分。
"""
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

import numpy as np                                          # noqa: E402
from sklearn.isotonic import IsotonicRegression             # noqa: E402
from sklearn.impute import SimpleImputer                    # noqa: E402
from sklearn.linear_model import LogisticRegression         # noqa: E402
from sklearn.metrics import roc_auc_score                   # noqa: E402
from sklearn.model_selection import StratifiedKFold         # noqa: E402
from sklearn.preprocessing import StandardScaler            # noqa: E402

from binary_tasks import LABEL_ADJACENT                     # noqa: E402
from nhanes_cohort import DERIVED, FEATURE_LABELS, build_extended   # noqa: E402

PARAMS = os.path.join(ROOT, "params", "direction_model.json")
LABEL = {**FEATURE_LABELS, **DERIVED, "age": "年齡", "sex": "性別"}
AXES = {
    "感染": dict(label="lab_infection", adjacent=LABEL_ADJACENT["infection"]),
    "代謝": dict(label="lab_metabolic", adjacent=LABEL_ADJACENT["metabolic"]),
}


def _fit(X, y, seed):
    imp = SimpleImputer(strategy="median").fit(X)
    sc = StandardScaler().fit(imp.transform(X))
    lr = LogisticRegression(class_weight="balanced", max_iter=4000,
                            random_state=seed).fit(sc.transform(imp.transform(X)), y)
    return dict(medians=imp.statistics_.tolist(), mean=sc.mean_.tolist(),
                scale=sc.scale_.tolist(), coef=lr.coef_[0].tolist(),
                intercept=float(lr.intercept_[0]))


def _raw(fp, x):
    xi = np.where(np.isnan(x), np.array(fp["medians"]), x)
    z = (xi - np.array(fp["mean"])) / np.array(fp["scale"])
    return float(1 / (1 + np.exp(-(z @ np.array(fp["coef"]) + fp["intercept"])))), z


def train(seed=20260830, folds=5):
    """每個軸：5 折交叉配適集成＋保序校準，門檻以盛行率倍數定義（事前指定）。"""
    P = json.load(open(os.path.join(ROOT, "params", "design.json"), encoding="utf-8"))
    E = build_extended(P, verbose=False)
    kd, feats = E["cohort"], E["features"]
    out = dict(seed=seed, n=int(len(kd)), axes={},
               scope="感染／代謝兩軸；免疫軸已移除（常規檢驗無訊號）")
    for name, spec in AXES.items():
        ff = [f for f in feats if f not in spec["adjacent"]]
        y = kd[spec["label"]].fillna(False).astype(bool).astype(int).to_numpy()
        X = kd[ff].to_numpy(float)
        oof, models = np.zeros(len(y)), []
        for tr, va in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y):
            fp = _fit(X[tr], y[tr], seed)
            oof[va] = np.array([_raw(fp, r)[0] for r in X[va]])
            models.append(fp)
        iso = IsotonicRegression(out_of_bounds="clip").fit(oof, y)
        cal = iso.predict(oof)
        prev = float(y.mean())
        # 三段門檻（事前指定，以盛行率倍數定義，不看結果調）：
        #   傾向   ＝ 校準機率 ≥ 2× 盛行率
        #   不傾向 ＝ 校準機率 ≤ 0.5× 盛行率
        t_hi, t_lo = 2.0 * prev, 0.5 * prev
        band = np.where(cal >= t_hi, "傾向", np.where(cal <= t_lo, "不傾向", "不確定"))
        out["axes"][name] = dict(
            features=ff, prevalence=prev, oof_auroc=float(roc_auc_score(y, oof)),
            t_high=t_hi, t_low=t_lo,
            band_distribution={b: int((band == b).sum()) for b in ("傾向", "不確定", "不傾向")},
            answered_sensitivity=float(((band == "傾向") & (y == 1)).sum() /
                                       max(((band != "不確定") & (y == 1)).sum(), 1)),
            answered_specificity=float(((band == "不傾向") & (y == 0)).sum() /
                                       max(((band != "不確定") & (y == 0)).sum(), 1)),
            ensemble=models, isotonic_x=iso.X_thresholds_.tolist(),
            isotonic_y=iso.y_thresholds_.tolist())
        a = out["axes"][name]
        print(f"[{name}] n={len(y)} 陽性 {y.sum()}（{prev:.3f}）｜OOF AUROC {a['oof_auroc']:.3f}"
              f"｜傾向/不確定/不傾向 = {a['band_distribution']}"
              f"｜作答者敏感度 {a['answered_sensitivity']:.2f} 特異度 {a['answered_specificity']:.2f}")
    json.dump(out, open(PARAMS, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"[存檔] {PARAMS}")
    return out


def predict(values, model=None, top_k=4):
    """values: dict{變數代碼: 數值}。缺的變數以訓練集中位數補，並在輸出標明。"""
    M = model or json.load(open(PARAMS, encoding="utf-8"))
    res = dict(input_n=len(values), axes={})
    for name, a in M["axes"].items():
        ff = a["features"]
        x = np.array([float(values.get(f, np.nan)) for f in ff])
        missing = [f for f, v in zip(ff, x) if np.isnan(v)]
        raws, contribs = [], np.zeros(len(ff))
        for fp in a["ensemble"]:
            r, z = _raw(fp, x)
            raws.append(r)
            contribs += z * np.array(fp["coef"])
        raw = float(np.mean(raws))
        cal = float(np.interp(raw, a["isotonic_x"], a["isotonic_y"]))
        band = ("傾向" if cal >= a["t_high"] else "不傾向" if cal <= a["t_low"] else "不確定")
        contribs /= len(a["ensemble"])
        order = np.argsort(-np.abs(contribs))
        drivers = [dict(var=ff[i], name=LABEL.get(ff[i], ff[i]),
                        value=None if np.isnan(x[i]) else float(x[i]),
                        push=("→推向" if contribs[i] > 0 else "←推離"),
                        weight=float(contribs[i]))
                   for i in order[:top_k] if not np.isnan(x[i])]
        res["axes"][name] = dict(
            probability=cal, prevalence=a["prevalence"],
            times_prevalence=cal / a["prevalence"], band=band,
            drivers=drivers, missing_filled_with_median=missing,
            n_missing=len(missing), n_features=len(ff))
    return res


def _bar(p, prev, width=10):
    k = int(round(min(p / (4 * prev), 1.0) * width))
    return "█" * k + "░" * (width - k)


def render(res):
    L = ["方向判定（大概方向，非診斷）", "─" * 62]
    for name, a in res["axes"].items():
        L.append(f"  {name}方向  {_bar(a['probability'], a['prevalence'])}  "
                 f"{a['probability']:.4f}（盛行率 ×{a['times_prevalence']:.1f}）→ **{a['band']}**")
        for d in a["drivers"]:
            L.append(f"        {d['push']} {d['name']}={d['value']:.3g}")
        if a["n_missing"]:
            L.append(f"        ⚠ 缺 {a['n_missing']}/{a['n_features']} 項，以中位數補入")
    return "\n".join(L)


def demo(n_each=2, seed=7):
    P = json.load(open(os.path.join(ROOT, "params", "design.json"), encoding="utf-8"))
    E = build_extended(P, verbose=False)
    kd = E["cohort"]
    M = json.load(open(PARAMS, encoding="utf-8"))
    hold = set(json.load(open(os.path.join(ROOT, "results", "final_holdout_seqn.json"),
                              encoding="utf-8"))["seqn"])
    te = kd[kd["SEQN"].isin(hold)]
    rng = np.random.default_rng(seed)
    allf = sorted({f for a in M["axes"].values() for f in a["features"]})
    for tag, mask in (("真實：B/C 肝陽性", te["lab_infection"].fillna(False).astype(bool)),
                      ("真實：糖尿病", te["lab_metabolic"].fillna(False).astype(bool)),
                      ("真實：兩者皆無", ~(te["lab_infection"].fillna(False).astype(bool) |
                                          te["lab_metabolic"].fillna(False).astype(bool)))):
        for i in rng.choice(te[mask].index, size=n_each, replace=False):
            vals = {f: te.loc[i, f] for f in allf if not np.isnan(te.loc[i, f])}
            print(f"\n══ {tag} ══")
            print(render(predict(vals, M)))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if cmd == "train":
        train()
    elif cmd == "predict":
        vals = json.load(open(sys.argv[2], encoding="utf-8"))
        print(render(predict(vals)))
    else:
        demo()
