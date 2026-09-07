# -*- coding: utf-8 -*-
"""直接看 biomarker——繞過標籤瓶頸的兩條路。
    python biomarker_direct.py [--seed 20260830]

## 為什麼要這樣做

聯合預測的 AUROC 受限於標籤品質（免疫 0.575）。但**標籤爛不代表 biomarker 沒訊號**：
單一標記的組間差異可能真實存在，只是不足以支撐聯合判別；而完全不用標籤的分型，
更可以問「資料裡本來就有幾群人」這個標籤管不到的問題。

## A. 單標記掃描（監督式，但只看單變量）

對免疫標籤（ANA 實測子樣本 n=554）逐標記做 Mann-Whitney U 雙尾檢定，
報告效果量（Cliff's δ）、單變量 AUC，並以 Benjamini-Hochberg 控制偽發現率。
**即使聯合 AUROC 只有 0.575，個別標記仍可能有真實差異**——這兩件事不矛盾。

## B. 無監督分型（完全不用標籤）

在腎損傷世代上只用 biomarker 分群，**分群過程完全不看任何標籤**，
分完再檢查各群的已知標籤盛行率。因為標籤沒參與分群，任何對齊都是真發現。

### 分群的致命陷阱與本檔的防範

分群演算法**永遠會給你分群**，即使資料裡根本沒有結構。故本檔強制三道檢查：

1. **穩定度**：自助重抽兩半、分別分群、以調整後蘭德指數（ARI）比對一致性。
2. **虛無對照**：把每一欄**各自獨立打亂**（保留邊際分布、摧毀欄間相關結構），
   跑完全相同的流程。若真實資料的穩定度 ≈ 打亂後的穩定度，**就是沒有結構**。
3. **雙尾檢定**：所有組間比較一律雙尾，不做單尾。

第 2 點是本檔的核心紀律，等同於分流模型的 M0 洩漏基準——
沒有這個對照，任何分群結果都只是演算法的產物。"""
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

import numpy as np                                          # noqa: E402
from scipy.stats import mannwhitneyu                        # noqa: E402
from sklearn.cluster import KMeans                          # noqa: E402
from sklearn.impute import SimpleImputer                    # noqa: E402
from sklearn.metrics import adjusted_rand_score, roc_auc_score  # noqa: E402
from sklearn.preprocessing import StandardScaler            # noqa: E402

from nhanes_cohort import build, build_extended             # noqa: E402
import pipeline as PL                                       # noqa: E402
from binary_tasks import LABEL_ADJACENT                     # noqa: E402

RESULTS = os.path.join(ROOT, "results")
K_RANGE = (2, 3, 4, 5, 6)
N_BOOT_STABILITY = 25


def bh_fdr(p):
    """Benjamini-Hochberg 校正後的 q 值。"""
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank, i in enumerate(order[::-1]):
        prev = min(prev, p[i] * n / (n - rank))
        q[i] = prev
    return q


def marker_scan(df, feats, y, verbose=True):
    """逐標記 Mann-Whitney U 雙尾＋Cliff's δ＋單變量 AUC＋BH-FDR。"""
    rows, pv = [], []
    for f in feats:
        v = df[f].to_numpy(float)
        a, b = v[(y == 1) & ~np.isnan(v)], v[(y == 0) & ~np.isnan(v)]
        if len(a) < 10 or len(b) < 10:
            continue
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        delta = 2 * u / (len(a) * len(b)) - 1            # Cliff's δ ∈ [-1, 1]
        mask = ~np.isnan(v)
        auc = float(roc_auc_score(y[mask], v[mask]))
        rows.append(dict(marker=f, n_pos=len(a), n_neg=len(b), p_two_sided=float(p),
                         cliffs_delta=float(delta), univariate_auc=auc,
                         auc_distance_from_chance=abs(auc - 0.5)))
        pv.append(p)
    if not rows:
        return []
    for r, q in zip(rows, bh_fdr(pv)):
        r["q_bh"] = float(q)
        r["significant_fdr05"] = bool(q < 0.05)
    rows.sort(key=lambda r: r["p_two_sided"])
    if verbose:
        sig = [r for r in rows if r["significant_fdr05"]]
        print(f"  掃描 {len(rows)} 個標記；FDR<0.05 者 {len(sig)} 個")
        for r in rows[:8]:
            star = "✱" if r["significant_fdr05"] else " "
            print(f"   {star} {r['marker']:12s} p={r['p_two_sided']:.2e}  q={r['q_bh']:.3f}  "
                  f"δ={r['cliffs_delta']:+.3f}  單變量AUC={r['univariate_auc']:.3f}")
    return rows


def _fit_labels(X, k, seed):
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)


def cluster_stability(X, k, seed, n_boot=N_BOOT_STABILITY):
    """自助重抽兩半分別分群，以重疊樣本的 ARI 衡量穩定度。"""
    rng = np.random.default_rng(seed)
    n = len(X)
    aris = []
    for _ in range(n_boot):
        idx = rng.permutation(n)
        h1, h2 = idx[: n // 2], idx[n // 2:]
        m1 = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X[h1])
        m2 = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X[h2])
        # 在全體上互相預測，比較兩個模型的分群是否一致
        aris.append(adjusted_rand_score(m1.predict(X), m2.predict(X)))
    return float(np.mean(aris)), float(np.std(aris))


def null_stability(X, k, seed, n_boot=N_BOOT_STABILITY):
    """虛無對照：每欄各自獨立打亂（保留邊際、摧毀相關），跑相同流程。"""
    rng = np.random.default_rng(seed + 777)
    Xp = np.column_stack([rng.permutation(X[:, j]) for j in range(X.shape[1])])
    return cluster_stability(Xp, k, seed, n_boot)


def run(seed=20260830, verbose=True):
    P = json.load(open(os.path.join(ROOT, "params", "design.json"), encoding="utf-8"))
    out = dict(seed=seed, created=time.strftime("%Y-%m-%dT%H:%M:%S"), scans={}, clustering={})

    # ══ A. 單標記掃描（免疫標籤，僅 ANA 實測者）
    if verbose:
        print("\n══ A. 單標記掃描（監督式單變量，雙尾＋BH-FDR）")
    C = build(P, verbose=False)
    feats = C["features"]
    ana, kd = C["primary"], C["secondary"]   # primary＝ANA 實測者（三類皆可判定）；secondary＝全腎損傷
    for tag, col in (("免疫（ANA 實測者）", "lab_immune"), ("感染", "lab_infection"),
                     ("代謝", "lab_metabolic")):
        sub = ana if "免疫" in tag else kd
        if col not in sub.columns:
            continue
        y = sub[col].fillna(False).astype(bool).to_numpy().astype(int)
        if y.sum() < 10:
            out["scans"][tag] = dict(skipped=f"陽性僅 {int(y.sum())} 例")
            continue
        adj = LABEL_ADJACENT.get({"lab_infection": "infection", "lab_metabolic": "metabolic",
                                  "lab_immune": "immune"}[col], [])
        ff = [f for f in feats if f in sub.columns and f not in adj]
        if verbose:
            print(f"\n [{tag}] n={len(y)} 陽性 {int(y.sum())}（已拔 {len(adj)} 個標籤鄰近欄）")
        out["scans"][tag] = dict(n=int(len(y)), n_pos=int(y.sum()),
                                 label_adjacent_removed=adj, markers=marker_scan(sub, ff, y, verbose))

    # ══ B. 無監督分型（完全不用標籤）
    if verbose:
        print("\n\n══ B. 無監督分型（分群過程不看任何標籤）")
    E = build_extended(P, verbose=False)
    ke, efeats = E["cohort"], E["features"]
    adj_all = set(LABEL_ADJACENT["infection"]) | set(LABEL_ADJACENT["metabolic"])
    cf = [f for f in efeats if f not in adj_all]
    X = StandardScaler().fit_transform(SimpleImputer(strategy="median")
                                       .fit_transform(ke[cf].to_numpy(float)))
    if verbose:
        print(f" 世代 n={len(X)}｜分群用 {len(cf)} 欄 biomarker（標籤全程未參與）")
        print(f"\n {'k':>2s}  {'真實穩定度(ARI)':>16s}  {'打亂後(虛無)':>14s}  {'差距':>8s}  判定")
    for k in K_RANGE:
        real_m, real_s = cluster_stability(X, k, seed)
        null_m, null_s = null_stability(X, k, seed)
        gap = real_m - null_m
        # 判定：真實穩定度須顯著高於虛無對照，否則所謂「分群」只是演算法產物
        has_structure = bool(gap > 0.10 and real_m > null_m + 2 * null_s)
        lab = _fit_labels(X, k, seed)
        sizes = [int((lab == c).sum()) for c in range(k)]
        enrich = {}
        for name, col in (("免疫", None), ("感染", "lab_infection"), ("代謝", "lab_metabolic")):
            if col and col in ke.columns:
                v = ke[col].fillna(False).astype(bool).to_numpy()
                enrich[name] = [round(float(v[lab == c].mean()), 4) for c in range(k)]
        out["clustering"][f"k={k}"] = dict(
            stability_real=real_m, stability_real_sd=real_s,
            stability_null=null_m, stability_null_sd=null_s, gap=gap,
            has_structure_beyond_null=has_structure,
            cluster_sizes=sizes, label_prevalence_per_cluster=enrich)
        if verbose:
            verdict = "✅ 超過虛無" if has_structure else "❌ 與打亂無異"
            print(f" {k:2d}  {real_m:16.3f}  {null_m:14.3f}  {gap:+8.3f}  {verdict}")
            if has_structure:
                print(f"     群大小 {sizes}")
                for nm, pv in enrich.items():
                    print(f"     {nm}盛行率/群 {pv}")

    out["conclusion"] = dict(
        marker_scan="見 scans——個別標記的顯著性與聯合判別力是兩件事",
        clustering=("找到超過虛無對照的結構"
                    if any(v.get("has_structure_beyond_null") for v in out["clustering"].values())
                    else "**所有 k 的分群穩定度都未超過打亂對照——biomarker 空間中沒有可靠的潛在分型**"))
    PL._dump(out, "biomarker_direct.json")
    if verbose:
        print(f"\n[結論] 分型：{out['conclusion']['clustering']}")
        print("[完成] results/biomarker_direct.json")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260830)
    run(ap.parse_args().seed)
