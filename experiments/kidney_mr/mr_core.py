# -*- coding: utf-8 -*-
"""兩樣本孟德爾隨機化核心——以基因為工具變數推論因果方向。

## 為什麼要這個

`kidney_cause` 的 ExWAS 證明橫斷面資料無法區分方向：
血鉛 OR 1.140↑ 但尿鉛 0.915↓（同一元素）、ACEI/ARB 1.119↑（那是**護腎藥**）。
統計調整解決不了，因為資料裡沒有時間。

MR 用**基因型**當工具變數。基因在受孕時隨機分配、且**不會被腎病改變**，
故若「使暴露較高的基因變異」也與腎功能較差相關，方向只能是 暴露→腎損傷。

## 三個工具變數假設（違反其一，結論即不成立）

| # | 假設 | 本檔如何檢查 |
|---|---|---|
| IV1 | 工具與**暴露**強相關 | F 統計量 >10；報告每個工具的 F |
| IV2 | 工具與**混淆**無關 | 無法直接檢驗——僅能靠設計與文獻論證 |
| IV3 | 工具**只透過暴露**影響結果（無多效性）| MR-Egger 截距、加權中位數、留一法 |

**IV3 是最常被違反的，而且本專案親身踩過**：先前的三酸甘油酯錨點被 MR 文獻推翻
（PMID 28754456），因為訊號完全由 GCKR 這個多效性變異驅動。故本檔一律同時報告
四種估計量，**四者方向不一致時判為多效性污染，不得宣稱因果**。

## 估計量

| 方法 | 對多效性的容忍 | 何時採信 |
|---|---|---|
| IVW（反變異數加權）| 無——假設全部工具有效 | 主要估計 |
| MR-Egger | 容許方向性多效性；**截距≠0 即為多效性證據** | 檢查用 |
| 加權中位數 | 容許 <50% 工具無效 | 穩健性 |
| 留一法 | 找出單一變異驅動的結果 | 穩健性 |

## LD 獨立性的近似

正規做法需 LD 參考面板做 clumping。本機無面板，改以**距離修剪**近似：
同一染色體上每 1 Mb 視窗只保留 P 值最小者。此為保守近似，**須在報告中標明**——
殘餘 LD 會使工具間相關，低估標準誤（即高估顯著性）。
"""
import numpy as np
from scipy import stats

MIN_F = 10.0                 # 弱工具門檻（慣例）
CLUMP_WINDOW_BP = 1_000_000  # 距離修剪視窗


def harmonise(expo, out):
    """對齊兩份 GWAS 的效應等位基因。

    expo/out 皆為 dict：rsid → dict(beta, se, ea, oa, eaf, p, n)
    回傳對齊後的陣列。**等位基因不相容者一律剔除，不猜測。**
    回文 SNP（A/T、C/G）因無法確定股別，一律剔除——這是保守選擇。
    """
    PALINDROMIC = {("A", "T"), ("T", "A"), ("C", "G"), ("G", "C")}
    rows, dropped = [], dict(missing_in_outcome=0, palindromic=0, allele_mismatch=0)
    for rs, e in expo.items():
        o = out.get(rs)
        if o is None:
            dropped["missing_in_outcome"] += 1
            continue
        ea, oa = e["ea"].upper(), e["oa"].upper()
        if (ea, oa) in PALINDROMIC:
            dropped["palindromic"] += 1
            continue
        oea, ooa = o["ea"].upper(), o["oa"].upper()
        if {ea, oa} != {oea, ooa}:
            dropped["allele_mismatch"] += 1
            continue
        bo = o["beta"] if oea == ea else -o["beta"]      # 效應等位基因對齊
        rows.append(dict(rsid=rs, beta_exp=e["beta"], se_exp=e["se"],
                         beta_out=bo, se_out=o["se"], ea=ea, oa=oa,
                         eaf=e.get("eaf"), n_exp=e.get("n")))
    return rows, dropped


def clump_by_distance(snps, window=CLUMP_WINDOW_BP):
    """距離修剪：同染色體每個視窗只留 P 最小者。**LD clumping 的保守近似。**

    snps 需含 chr、pos、p_exp。無位置資訊者原樣保留並標記。
    """
    have_pos = [s for s in snps if s.get("chr") and s.get("pos")]
    no_pos = [s for s in snps if not (s.get("chr") and s.get("pos"))]
    kept = []
    for c in sorted({s["chr"] for s in have_pos}):
        cand = sorted([s for s in have_pos if s["chr"] == c], key=lambda s: s["p_exp"])
        taken = []
        for s in cand:
            if all(abs(s["pos"] - t["pos"]) > window for t in taken):
                taken.append(s)
        kept += taken
    return kept + no_pos, dict(n_before=len(snps), n_after=len(kept) + len(no_pos),
                               n_without_position=len(no_pos))


def f_statistic(beta_exp, se_exp):
    """每個工具的 F＝(beta/se)²。F<10 為弱工具。"""
    return (np.asarray(beta_exp) / np.asarray(se_exp)) ** 2


def ivw(bx, by, sy):
    """反變異數加權——主要估計量。假設全部工具有效（無多效性）。"""
    bx, by, sy = map(np.asarray, (bx, by, sy))
    w = 1.0 / sy ** 2
    b = float(np.sum(w * bx * by) / np.sum(w * bx ** 2))
    se = float(np.sqrt(1.0 / np.sum(w * bx ** 2)))
    return dict(method="IVW", beta=b, se=se, ci=[b - 1.96 * se, b + 1.96 * se],
                p=float(2 * stats.norm.sf(abs(b / se))), n_snp=len(bx))


def mr_egger(bx, by, sy):
    """MR-Egger：容許方向性多效性。**截距≠0 即為多效性證據。**"""
    bx, by, sy = map(np.asarray, (bx, by, sy))
    if len(bx) < 3:
        return dict(method="MR-Egger", error="工具數 <3，無法估計")
    w = 1.0 / sy ** 2
    X = np.column_stack([np.ones(len(bx)), bx])
    W = np.diag(w)
    try:
        cov = np.linalg.inv(X.T @ W @ X)
    except np.linalg.LinAlgError:
        return dict(method="MR-Egger", error="設計矩陣退化")
    coef = cov @ (X.T @ W @ by)
    resid = by - X @ coef
    dof = max(len(bx) - 2, 1)
    sigma2 = float(resid @ W @ resid / dof)
    se = np.sqrt(np.diag(cov) * max(sigma2, 1.0))     # 不容許過度離散使 SE 縮小
    return dict(method="MR-Egger", beta=float(coef[1]), se=float(se[1]),
                ci=[float(coef[1] - 1.96 * se[1]), float(coef[1] + 1.96 * se[1])],
                p=float(2 * stats.t.sf(abs(coef[1] / se[1]), dof)),
                intercept=float(coef[0]), intercept_se=float(se[0]),
                intercept_p=float(2 * stats.t.sf(abs(coef[0] / se[0]), dof)),
                pleiotropy_detected=bool(2 * stats.t.sf(abs(coef[0] / se[0]), dof) < 0.05),
                n_snp=len(bx))


def weighted_median(bx, by, sy, n_boot=1000, seed=20260831):
    """加權中位數：容許最多 50% 工具無效。"""
    bx, by, sy = map(np.asarray, (bx, by, sy))
    if len(bx) < 3:
        return dict(method="加權中位數", error="工具數 <3")

    def _wm(b, w):
        o = np.argsort(b)
        b, w = b[o], w[o]
        cw = np.cumsum(w) - 0.5 * w
        cw /= np.sum(w)
        k = np.searchsorted(cw, 0.5)
        if k == 0:
            return float(b[0])
        if k >= len(b):
            return float(b[-1])
        return float(b[k - 1] + (b[k] - b[k - 1]) * (0.5 - cw[k - 1]) / (cw[k] - cw[k - 1]))

    ratio = by / bx
    w = (bx ** 2) / (sy ** 2)
    est = _wm(ratio, w)
    rng = np.random.default_rng(seed)
    boots = [_wm(rng.normal(by, sy) / bx, w) for _ in range(n_boot)]
    se = float(np.std(boots))
    return dict(method="加權中位數", beta=est, se=se,
                ci=[est - 1.96 * se, est + 1.96 * se],
                p=float(2 * stats.norm.sf(abs(est / se))) if se > 0 else None, n_snp=len(bx))


def leave_one_out(rsids, bx, by, sy):
    """留一法：找出由單一變異驅動的結果（多效性的典型徵象）。"""
    if len(bx) < 3:
        return []
    out = []
    for i, rs in enumerate(rsids):
        m = np.ones(len(bx), bool)
        m[i] = False
        r = ivw(np.asarray(bx)[m], np.asarray(by)[m], np.asarray(sy)[m])
        out.append(dict(excluded=rs, beta=r["beta"], p=r["p"]))
    return out


def run_mr(snps, exposure_name, outcome_name):
    """完整 MR：四種估計量＋弱工具檢查＋一致性判定。

    **一致性判定是本檔的核心紀律**：四種估計量方向不一致，或 MR-Egger 截距顯著，
    即判為多效性污染，`causal_claim_supported=False`——不得宣稱因果。
    """
    rsids = [s["rsid"] for s in snps]
    bx = np.array([s["beta_exp"] for s in snps])
    sx = np.array([s["se_exp"] for s in snps])
    by = np.array([s["beta_out"] for s in snps])
    sy = np.array([s["se_out"] for s in snps])
    F = f_statistic(bx, sx)
    res = dict(exposure=exposure_name, outcome=outcome_name, n_snp=len(snps),
               f_stat=dict(mean=float(F.mean()), min=float(F.min()),
                           n_weak=int((F < MIN_F).sum()),
                           weak_instrument_warning=bool((F < MIN_F).any())),
               estimates={})
    for fn in (ivw, mr_egger, weighted_median):
        r = fn(bx, by, sy)
        res["estimates"][r["method"]] = r
    res["leave_one_out"] = leave_one_out(rsids, bx, by, sy)

    # ── 一致性判定
    betas = [r["beta"] for r in res["estimates"].values() if "beta" in r]
    same_dir = bool(betas and (all(b > 0 for b in betas) or all(b < 0 for b in betas)))
    egger = res["estimates"].get("MR-Egger", {})
    pleio = bool(egger.get("pleiotropy_detected"))
    loo = res["leave_one_out"]
    loo_driven = bool(loo and res["estimates"]["IVW"]["p"] < 0.05 and
                      any(x["p"] > 0.05 for x in loo))
    res["consistency"] = dict(
        all_methods_same_direction=same_dir,
        egger_intercept_pleiotropy=pleio,
        single_snp_driven=loo_driven,
        causal_claim_supported=bool(same_dir and not pleio and not loo_driven
                                    and res["estimates"]["IVW"]["p"] < 0.05
                                    and not res["f_stat"]["weak_instrument_warning"]),
        note=("四估計量同向、Egger 截距不顯著、非單一變異驅動、無弱工具、IVW 顯著"
              "——五項全過才判為支持因果"))
    return res
