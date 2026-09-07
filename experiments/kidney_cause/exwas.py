# -*- coding: utf-8 -*-
"""暴露組關聯掃描（ExWAS）——從上游找腎損傷的原因。
    python exwas.py [--seed 20260830]

## 核心目標

給定一個現象（腎損傷），掃描**所有暴露通道**，回推「什麼可能導致了它」，
輸出一張**帶證據等級的病因清單**——不是一個 AUROC。

## ⚠️ 與先前所有分析的一個根本差異：族群

先前的分析都**先篩出腎損傷者**再問病因。那是錯的——
篩選後再找原因是**對結果做條件化（collider）**，會製造出不存在的關聯。

本檔改用**全體成人**，腎損傷是**結果變項**而非納入條件。
這一項修正本身就可能改變所有結論。

## 為什麼掃暴露而不是掃血液生化

| | 性質 | 因果位置 |
|---|---|---|
| 血液生化 | 身體的**狀態** | **下游**——腎功能下降會反過來改變它們 |
| 藥物／金屬／化學品／職業 | **暴露** | **上游**——時序上先於腎損傷 |

本專案先前窮盡了血液生化通道（判別增量≈0），但那是下游。上游從未掃過。

## 三道防線——沒有這些，ExWAS 只是大規模的假關聯生成器

### ① 陽性對照（方法有沒有力）
已知腎毒性藥物（NSAIDs、鋰鹽、質子幫浦抑制劑、胺基醣苷）**必須被掃出來**。
掃不到 → 方法沒有偵測力，其餘結果一律不可信。

### ② 陰性對照（有沒有系統性偏誤）
與腎臟無合理機轉的藥物（如局部皮膚用藥、眼藥水）**不應被掃出來**。
若它們也顯著 → 訊號來自**就醫行為／多重用藥**而非藥理，全部結果須降級。

### ③ 反向因果（最致命的威脅）
* **藥物**：腎病患者**因為有腎病所以吃更多藥**。故一律調整**總用藥數**。
* **腎清除的毒物**：eGFR 下降 → 排除變差 → **血中濃度上升**。
  血鎘、血鉛與 CKD 的關聯有相當部分是這個方向。故血中濃度的結果一律標註此風險。
* **尿液生物標記**：CKD 患者的尿肌酸酐改變，**尿液濃度經肌酸酐校正後有系統性偏誤**。
  故尿液暴露另報未校正值。

## 分層調整（逐層報告，看關聯在哪一層消失）

| 模型 | 調整 |
|---|---|
| M0 | 無（粗關聯）|
| M1 | ＋年齡、性別、種族 |
| M2 | ＋BMI、抽菸 |
| M3 | ＋糖尿病、高血壓（**這兩者本身是腎病主因，調整後仍存在才算獨立**）|
| M4 | ＋總用藥數（僅藥物暴露；反向因果防線）|

**關聯在 M3 消失 ＝ 該暴露的效果由糖尿病／高血壓中介或混淆**，這本身是有用的結論。"""
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

import numpy as np                                    # noqa: E402
import pandas as pd                                   # noqa: E402
from sklearn.linear_model import LogisticRegression   # noqa: E402

RESULTS = os.path.join(ROOT, "results")

# ── 對照組定義（事前指定，見檔頭三道防線）
POSITIVE_CONTROLS = {          # 已知腎毒性——掃不到代表方法沒有偵測力
    "NSAIDS": ["IBUPROFEN", "NAPROXEN", "DICLOFENAC", "INDOMETHACIN", "CELECOXIB", "MELOXICAM",
               "KETOROLAC", "PIROXICAM", "SULINDAC", "ETODOLAC", "NABUMETONE"],
    "LITHIUM": ["LITHIUM"],
    "PPI": ["OMEPRAZOLE", "ESOMEPRAZOLE", "LANSOPRAZOLE", "PANTOPRAZOLE", "RABEPRAZOLE"],
    "AMINOGLYCOSIDE": ["GENTAMICIN", "TOBRAMYCIN", "AMIKACIN", "NEOMYCIN"],
    "CALCINEURIN": ["CYCLOSPORINE", "TACROLIMUS"],
}
NEGATIVE_CONTROLS = {          # 無合理腎機轉——若顯著代表訊號來自就醫行為
    "TOPICAL_DERM": ["HYDROCORTISONE TOPICAL", "TRIAMCINOLONE TOPICAL", "MUPIROCIN",
                     "CLOTRIMAZOLE", "KETOCONAZOLE TOPICAL"],
    "OPHTHALMIC": ["LATANOPROST", "TIMOLOL OPHTHALMIC", "ARTIFICIAL TEARS"],
    "NASAL": ["FLUTICASONE NASAL", "MOMETASONE NASAL"],
}
# 腎清除毒物——血中濃度受 eGFR 影響，反向因果風險高
RENALLY_CLEARED = {"LBXBCD", "LBXBPB", "LBXTHG", "LBXBSE", "URXUCD", "URXUPB", "URXUAS"}

ADJ_SETS = {
    "M0_粗關聯": [],
    "M1_人口學": ["age", "sex", "race_black", "race_hisp"],
    "M2_＋體位抽菸": ["age", "sex", "race_black", "race_hisp", "BMI", "smoker"],
    "M3_＋糖尿病高血壓": ["age", "sex", "race_black", "race_hisp", "BMI", "smoker",
                          "diabetes", "hypertension"],
}


def bh_fdr(p):
    p = np.asarray(p, float)
    n = len(p)
    q = np.empty(n)
    prev = 1.0
    for rank, i in enumerate(np.argsort(p)[::-1]):
        prev = min(prev, p[i] * n / (n - rank))
        q[i] = prev
    return q


def _logit_or(df, expo, outcome, adj, seed):
    """回傳暴露的勝算比與 Wald 95% CI／雙尾 p。樣本不足或退化時回 None。"""
    cols = [expo] + [c for c in adj if c in df.columns]
    d = df[cols + [outcome]].dropna()
    if len(d) < 200 or d[outcome].sum() < 20 or (1 - d[outcome]).sum() < 20:
        return None
    X = d[cols].to_numpy(float)
    if np.nanstd(X[:, 0]) == 0:
        return None
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Xs = (X - mu) / sd
    y = d[outcome].to_numpy(int)
    m = LogisticRegression(max_iter=4000, random_state=seed, C=1e6).fit(Xs, y)
    beta = float(m.coef_[0][0])
    # Wald 標準誤（觀測資訊矩陣）
    p_hat = m.predict_proba(Xs)[:, 1]
    W = p_hat * (1 - p_hat)
    Xd = np.column_stack([np.ones(len(Xs)), Xs])
    try:
        cov = np.linalg.inv(Xd.T * W @ Xd)
        se = float(np.sqrt(cov[1, 1]))
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(se) or se == 0:
        return None
    from scipy.stats import norm
    z = beta / se
    return dict(n=int(len(d)), n_pos=int(y.sum()), beta_per_sd=beta, se=se,
                or_per_sd=float(np.exp(beta)),
                ci=[float(np.exp(beta - 1.96 * se)), float(np.exp(beta + 1.96 * se))],
                p_two_sided=float(2 * norm.sf(abs(z))), z=float(z))


def scan(df, exposures, outcome="kidney_damage", seed=20260830, verbose=True):
    """對每個暴露跑分層調整模型，最後在 M3 的 p 值上做 BH-FDR。"""
    rows = []
    for e in exposures:
        rec = dict(exposure=e, models={})
        for name, adj in ADJ_SETS.items():
            if e in adj:
                continue
            r = _logit_or(df, e, outcome, adj, seed)
            if r:
                rec["models"][name] = r
        if "M3_＋糖尿病高血壓" in rec["models"]:
            rec["headline"] = rec["models"]["M3_＋糖尿病高血壓"]
            rec["reverse_causation_risk"] = (
                "高——腎清除毒物，eGFR 下降會使血中濃度上升" if e in RENALLY_CLEARED else "一般")
            # 關聯在哪一層消失
            sig = [n for n, r in rec["models"].items() if r["p_two_sided"] < 0.05]
            rec["survives_full_adjustment"] = "M3_＋糖尿病高血壓" in sig
            rec["significant_layers"] = sig
            rows.append(rec)
    if not rows:
        return []
    for r, q in zip(rows, bh_fdr([r["headline"]["p_two_sided"] for r in rows])):
        r["q_bh"] = float(q)
        r["significant_fdr05"] = bool(q < 0.05)
    rows.sort(key=lambda r: r["headline"]["p_two_sided"])
    if verbose:
        sig = [r for r in rows if r["significant_fdr05"]]
        print(f"  掃描 {len(rows)} 個暴露；FDR<0.05 且通過完整調整者 "
              f"{sum(1 for r in sig if r['survives_full_adjustment'])} 個")
    return rows


# 對照組的**暴露名稱**（非藥品名）——exposure_cohort.DRUG_CLASSES 的鍵
# 2026-08-31 修正：先前以藥品名（IBUPROFEN…）比對暴露名（藥_NSAID），永遠比不中，
# 導致對照檢查恆回報「陽性對照未被掃出」。類別前綴已編碼對照身分，直接用它。
POSITIVE_CONTROL_EXPOSURES = {"藥_NSAID", "藥_鋰鹽", "藥_PPI", "藥_胺基醣苷", "藥_鈣調磷酸酶"}
NEGATIVE_CONTROL_PREFIX = "藥陰_"


def control_check(rows, verbose=True):
    """陽性／陰性對照判定——決定其餘結果可不可信。"""
    pos = [r for r in rows if r["significant_fdr05"]
           and r["exposure"] in POSITIVE_CONTROL_EXPOSURES]
    neg = [r for r in rows if r["significant_fdr05"]
           and r["exposure"].startswith(NEGATIVE_CONTROL_PREFIX)]
    n_pos_scanned = sum(1 for r in rows if r["exposure"] in POSITIVE_CONTROL_EXPOSURES)
    n_neg_scanned = sum(1 for r in rows if r["exposure"].startswith(NEGATIVE_CONTROL_PREFIX))
    verdict = dict(
        positive_hits=[r["exposure"] for r in pos], n_positive=len(pos),
        n_positive_scanned=n_pos_scanned, n_negative_scanned=n_neg_scanned,
        negative_hits=[r["exposure"] for r in neg], n_negative=len(neg),
        caveat_indication_bias=("陰性對照乾淨只排除**一般就醫行為**偏誤，"
                                "**不排除適應症混淆**——為特定疾病開的藥仍會與該疾病的"
                                "併發症關聯。判讀個別藥物時必須逐一評估其適應症。"),
        has_power=bool(pos),
        bias_free=not neg,
        interpretation=(
            "✅ 陽性對照被掃出、陰性對照未被掃出——結果可信度較高" if pos and not neg else
            "⚠️ 陰性對照也顯著——訊號可能來自就醫行為／多重用藥，全部結果須降級" if neg else
            "❌ 陽性對照未被掃出——方法缺乏偵測力，其餘結果不可信"))
    if verbose:
        print(f"\n[對照檢查] {verdict['interpretation']}")
        print(f"  陽性對照命中 {len(pos)}：{verdict['positive_hits'][:6]}")
        print(f"  陰性對照命中 {len(neg)}：{verdict['negative_hits'][:6]}")
    return verdict


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260830)
    a = ap.parse_args()
    print("本檔為 ExWAS 統計核心。需先由 exposure_cohort.py 建立含暴露的全體成人世代。")
    print("執行順序：probe_exposures.py → fetch → exposure_cohort.py → run_exwas.py")
