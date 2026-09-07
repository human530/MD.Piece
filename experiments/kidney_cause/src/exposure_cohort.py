# -*- coding: utf-8 -*-
"""全體成人暴露世代（NHANES 1999–2018）——ExWAS 專用。

## 與既有 build_extended() 的關鍵差異

`build_extended()` **先篩出腎損傷者**再問病因。找原因時那是錯的
（對結果條件化＝collider）。本檔保留**全體成人**，`kidney_damage` 是**結果變項**。

## 暴露通道

| 通道 | 變數 | 反向因果風險 |
|---|---|---|
| 處方藥物 | RXQ_RX 長格式 → 類別指標＋總用藥數 | **高**——腎病患者吃更多藥，故必調整總用藥數 |
| 血中金屬 | 鉛 LBXBPB、鎘 LBXBCD、汞 LBXTHG、硒 LBXBSE | **高**——腎清除，eGFR↓ 使血中濃度↑ |
| 體位 | BMXBMI | 一般 |
| 血壓 | BPXSY1–4／BPXDI1–4 取平均 | 一般（但高血壓同時是腎病主因與結果）|
| 抽菸 | SMQ020／SMQ040 | 一般 |

藥名變數跨週期不一致：2003 年後為 `RXDDRUG`，1999–2002 為 `RXD240B`。"""
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nhanes_cohort import (RAW, egfr_ckdepi2021, load_all, load_extended, _read)   # noqa: E402
from provenance import require_real                                                # noqa: E402

# ── 藥物類別（事前指定；陽性／陰性對照見 exwas.py）
DRUG_CLASSES = {
    # 陽性對照——已知腎毒性
    "藥_NSAID": ["IBUPROFEN", "NAPROXEN", "DICLOFENAC", "INDOMETHACIN", "CELECOXIB",
                 "MELOXICAM", "KETOROLAC", "PIROXICAM", "SULINDAC", "ETODOLAC", "NABUMETONE"],
    "藥_鋰鹽": ["LITHIUM"],
    "藥_PPI": ["OMEPRAZOLE", "ESOMEPRAZOLE", "LANSOPRAZOLE", "PANTOPRAZOLE", "RABEPRAZOLE"],
    "藥_胺基醣苷": ["GENTAMICIN", "TOBRAMYCIN", "AMIKACIN", "NEOMYCIN"],
    "藥_鈣調磷酸酶": ["CYCLOSPORINE", "TACROLIMUS"],
    # 陰性對照——無合理腎機轉
    "藥陰_外用皮膚": ["MUPIROCIN", "CLOTRIMAZOLE", "TERBINAFINE"],
    "藥陰_眼用": ["LATANOPROST", "TIMOLOL", "DORZOLAMIDE", "BRIMONIDINE"],
    "藥陰_鼻用": ["FLUTICASONE", "MOMETASONE", "BUDESONIDE"],
    # 其他常見類別（探索用）
    "藥_ACEI_ARB": ["LISINOPRIL", "ENALAPRIL", "RAMIPRIL", "LOSARTAN", "VALSARTAN",
                    "IRBESATAN", "IRBESARTAN", "OLMESARTAN", "BENAZEPRIL", "QUINAPRIL"],
    "藥_利尿劑": ["FUROSEMIDE", "HYDROCHLOROTHIAZIDE", "CHLORTHALIDONE", "SPIRONOLACTONE",
                  "TRIAMTERENE", "BUMETANIDE"],
    "藥_他汀": ["ATORVASTATIN", "SIMVASTATIN", "ROSUVASTATIN", "PRAVASTATIN", "LOVASTATIN"],
    "藥_雙胍": ["METFORMIN"],
    "藥_磺醯脲": ["GLIPIZIDE", "GLYBURIDE", "GLIMEPIRIDE"],
    "藥_胰島素": ["INSULIN"],
    "藥_別嘌醇": ["ALLOPURINOL"],
    "藥_抗生素_其他": ["AMOXICILLIN", "AZITHROMYCIN", "CIPROFLOXACIN", "LEVOFLOXACIN",
                       "TRIMETHOPRIM", "SULFAMETHOXAZOLE", "CEPHALEXIN", "DOXYCYCLINE"],
}
METALS = {"LBXBPB": "血鉛", "LBXBCD": "血鎘", "LBXTHG": "血汞", "LBXBSE": "血硒",
          "LBDBPB": "血鉛", "LBDBCD": "血鎘", "LBDTHG": "血汞", "LBDBSE": "血硒"}

# ── 化學品／尿液通道（除藥物與血金屬外的其餘暴露）
#   前綴規則：URX*＝尿液濃度、LBX*＝血清濃度；LBD*／URD* 多為偵測極限旗標，排除。
CHEM_CHANNELS = {
    "塑化劑":  dict(prefix=("URX",), matrix="urine", note="鄰苯二甲酸酯代謝物"),
    "PFAS":    dict(prefix=("LBX",), matrix="serum", note="全氟烷基物質"),
    "農藥":    dict(prefix=("URX",), matrix="urine", note="農藥代謝物"),
    "尿金屬":  dict(prefix=("URX",), matrix="urine", note="尿中重金屬"),
    "尿砷":    dict(prefix=("URX",), matrix="urine", note="尿砷（含物種分析）"),
}
# 調查權重變數——**絕不可當暴露**（子樣本權重，與結果無因果關係）
WEIGHT_PAT = ("WTS", "WTF", "WTM", "WTD", "WTI", "SDM", "SDD")
# 內建陰性對照：砷貝他因為海鮮來源的**無毒**砷形式。
# 若總砷關聯顯著而砷貝他因不顯著 → 支持真實毒性效應；若砷貝他因也顯著 → 是飲食混淆。
ARSENIC_NEGATIVE_CONTROL = "URXUAB"
# ── 結果定義所用之變數——**絕不可當暴露**（通道消耗規則）
#   kidney_damage = (eGFR<60) | (ACR>=30)
#   eGFR ← LBXSCR；ACR ← URXUMA / (URXUCR/100)
#   2026-08-31 稽核：URXUCR 曾被納入暴露清單，且其 _percr 版本＝常數 100（完全退化）。
OUTCOME_SOURCE_VARS = {"LBXSCR", "LBXSCR_raw", "URXUMA", "URXUCR", "URDUMA", "URDUCR",
                       "eGFR", "ACR", "kidney_damage"}
DRUG_NAME_VARS = ["RXDDRUG", "RXD240B", "RXDDRGID"]
DRUG_DAYS_VARS = ["RXDDAYS", "RXD260"]


def _usable(col):
    """可用暴露的判準。

    **2026-08-31 修正**：先前用 `nunique > 8` 排除退化欄，但那會把**所有二元暴露**
    （17 個藥物類別，nunique=2）一併砍掉——導致陽性對照根本沒被掃描，
    對照檢查回報「方法無偵測力」，差點把自己的 bug 報成研究結論。

    正確判準分兩種：
      * 常數欄（nunique<2）一律排除——`URXUCR_percr`＝100 即屬此類
      * 二元欄：少數類至少 50 例，否則統計上無意義
      * 連續欄：相異值 >8（避免把類別編碼當連續處理）
    """
    v = col.dropna()
    if len(v) < 500 or v.nunique() < 2:
        return False
    if v.nunique() == 2:                       # 二元暴露（藥物類別等）
        return int(v.value_counts().min()) >= 50
    return v.nunique() > 8


def _files_by_channel(man, channel):
    return [k for k, v in man.get("files", {}).items() if v.get("channel") == channel]


def _load_channel(man, channel, cols_filter=None, verbose=True):
    """讀某通道全部週期，直向合併（各週期欄位不同者取聯集）。"""
    frames = []
    for key in _files_by_channel(man, channel):
        p = os.path.join(RAW, key)
        if not os.path.exists(p):
            continue
        require_real(p)
        try:
            f = _read(key)
        except Exception as e:
            if verbose:
                print(f"  [跳過] {key}：{e}")
            continue
        if cols_filter:
            keep = [c for c in f.columns if c == "SEQN" or cols_filter(c)]
            if len(keep) <= 1:
                continue
            f = f[keep]
        frames.append(f)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True, sort=False)


def build_drug_features(man, verbose=True):
    """RXQ_RX 長格式 → 每人的類別指標＋總用藥數。"""
    rx = _load_channel(man, "藥物", verbose=verbose)
    if rx is None:
        return None
    name_var = next((v for v in DRUG_NAME_VARS if v in rx.columns), None)
    if name_var is None:
        if verbose:
            print("  [藥物] 找不到藥名變數——跳過")
        return None
    rx = rx[["SEQN", name_var]].dropna()
    rx[name_var] = rx[name_var].astype(str).str.upper().str.strip()
    rx = rx[~rx[name_var].isin({"", "NAN", "55555", "77777", "99999"})]
    if verbose:
        print(f"  [藥物] 處方紀錄 {len(rx):,} 筆｜{rx['SEQN'].nunique():,} 人｜"
              f"相異藥名 {rx[name_var].nunique():,}｜藥名變數 {name_var}")
    out = pd.DataFrame({"SEQN": rx["SEQN"].unique()})
    for cls, names in DRUG_CLASSES.items():
        pat = "|".join(re.escape(n) for n in names)
        hit = rx.loc[rx[name_var].str.contains(pat, na=False, regex=True), "SEQN"].unique()
        out[cls] = out["SEQN"].isin(hit).astype(int)
        if verbose and out[cls].sum():
            print(f"     {cls:16s} {out[cls].sum():5d} 人")
    cnt = rx.groupby("SEQN")[name_var].nunique().rename("藥_總用藥數").reset_index()
    return out.merge(cnt, on="SEQN", how="left")


def build(P=None, verbose=True):
    """全體成人＋暴露。回傳 dict(cohort, exposures, covariates, counts)。"""
    import json
    man = json.load(open(os.path.join(ROOT, "params", "manifest.json"), encoding="utf-8"))
    base, _ = load_all(verbose=False)
    ext = load_extended(verbose=False)
    df = pd.concat([base, ext], ignore_index=True, sort=False)
    age_min = (P or {}).get("population", {}).get("value", {}).get("age_min", 20)
    df = df[df["age"] >= age_min].copy()

    # ── 結果變項（**不作為納入條件**）
    female = df["sex"] == 2
    df["eGFR"] = egfr_ckdepi2021(df["LBXSCR"].to_numpy(float), df["age"].to_numpy(float),
                                 female.to_numpy())
    df["ACR"] = df["URXUMA"] / (df["URXUCR"] / 100.0)
    df["kidney_damage"] = ((df["eGFR"] < 60) | (df["ACR"] >= 30)).astype(float)
    df.loc[df["eGFR"].isna() & df["ACR"].isna(), "kidney_damage"] = np.nan
    if verbose:
        print(f"\n[暴露世代] 成人 {len(df):,}｜腎損傷 {int(df['kidney_damage'].sum()):,}"
              f"（{df['kidney_damage'].mean():.1%}）｜結果可判定 {int(df['kidney_damage'].notna().sum()):,}")

    exposures, covars = [], []

    # ── 藥物
    if verbose:
        print("\n[通道] 藥物")
    drugs = build_drug_features(man, verbose)
    if drugs is not None:
        df = df.merge(drugs, on="SEQN", how="left")
        for c in drugs.columns:
            if c != "SEQN" and c != "藥_總用藥數":
                df[c] = df[c].fillna(0)
                exposures.append(c)
        df["藥_總用藥數"] = df["藥_總用藥數"].fillna(0)
        covars.append("藥_總用藥數")

    # ── 血中金屬
    if verbose:
        print("\n[通道] 血中金屬")
    met = _load_channel(man, "血金屬", cols_filter=lambda c: c in METALS, verbose=verbose)
    if met is not None:
        met = met.drop_duplicates("SEQN")
        # 別名統一（LBD* → LBX*）
        for src, dst in (("LBDBPB", "LBXBPB"), ("LBDBCD", "LBXBCD"),
                         ("LBDTHG", "LBXTHG"), ("LBDBSE", "LBXBSE")):
            if src in met.columns:
                met[dst] = met[dst].fillna(met[src]) if dst in met.columns else met[src]
                met = met.drop(columns=[src])
        df = df.merge(met, on="SEQN", how="left", suffixes=("", "_m"))
        for c in ("LBXBPB", "LBXBCD", "LBXTHG", "LBXBSE"):
            if c in df.columns and df[c].notna().sum() > 500:
                exposures.append(c)
                if verbose:
                    print(f"     {METALS[c]:6s}({c}) 有值 {int(df[c].notna().sum()):,} 人")

    # ── 化學品／尿液通道（塑化劑、PFAS、農藥、尿金屬、尿砷）
    urinary = []
    for ch, spec in CHEM_CHANNELS.items():
        if verbose:
            print(f"\n[通道] {ch}（{spec['note']}）")
        f = _load_channel(man, ch,
                          cols_filter=lambda c, p=spec["prefix"]: (
                              c.startswith(p) and not c.startswith(WEIGHT_PAT)),
                          verbose=verbose)
        if f is None:
            if verbose:
                print("     無可用檔案")
            continue
        f = f.drop_duplicates("SEQN")
        cols = [c for c in f.columns if c != "SEQN"]
        df = df.merge(f, on="SEQN", how="left", suffixes=("", f"_{ch}"))
        added = 0
        for c in cols:
            if c in df.columns and _usable(df[c]):
                exposures.append(c)
                added += 1
                if spec["matrix"] == "urine":
                    urinary.append(c)
        if verbose:
            print(f"     納入 {added} 個變數（有值 >500 人且非二元）")

    # 結果來源變數一律剔除（通道消耗規則）——必須在建校正版之前做
    urinary = [c for c in urinary if c not in OUTCOME_SOURCE_VARS]
    exposures = [c for c in exposures if c not in OUTCOME_SOURCE_VARS]

    # 尿液暴露的肌酸酐校正——**同時保留未校正值**
    # CKD 患者尿肌酸酐本身改變，校正後有系統性偏誤；兩者並列讓讀者看得到差異。
    if urinary and "URXUCR" in df.columns:
        ucr = df["URXUCR"].replace(0, np.nan) / 100.0
        for c in list(urinary):
            df[f"{c}_percr"] = df[c] / ucr
            exposures.append(f"{c}_percr")
        if verbose:
            print(f"\n[尿液校正] {len(urinary)} 個尿液暴露另建肌酸酐校正版（_percr）"
                  f"——未校正與校正版並列，兩者在 CKD 中偏誤方向不同")

    # ── 共變項：體位、血壓、抽菸
    for ch, want, mk in (("體位", lambda c: c == "BMXBMI", None),
                         ("血壓", lambda c: c.startswith(("BPXSY", "BPXDI")), None),
                         ("抽菸", lambda c: c in ("SMQ020", "SMQ040"), None)):
        f = _load_channel(man, ch, cols_filter=want, verbose=verbose)
        if f is not None:
            df = df.merge(f.drop_duplicates("SEQN"), on="SEQN", how="left", suffixes=("", "_c"))
    if "BMXBMI" in df.columns:
        df["BMI"] = df["BMXBMI"]
        covars.append("BMI")
    sy = [c for c in df.columns if c.startswith("BPXSY")]
    di = [c for c in df.columns if c.startswith("BPXDI")]
    if sy and di:
        df["SBP"] = df[sy].replace(0, np.nan).mean(axis=1)
        df["DBP"] = df[di].replace(0, np.nan).mean(axis=1)
        df["hypertension"] = ((df["SBP"] >= 140) | (df["DBP"] >= 90)).astype(float)
        covars += ["hypertension"]
    if "SMQ020" in df.columns:
        df["smoker"] = (df["SMQ020"] == 1).astype(float)
        covars.append("smoker")
    df["diabetes"] = ((df.get("DIQ010") == 1) | (df.get("LBXGH", pd.Series(np.nan, index=df.index)) >= 6.5)).astype(float)
    covars.append("diabetes")
    for c in ("race_black", "race_hisp"):
        if c not in df.columns:
            df[c] = 0.0          # DEMO 未載入種族變數時以 0 佔位，並於報告標明
    covars += ["age", "sex", "race_black", "race_hisp"]

    exposures = [e for e in dict.fromkeys(exposures)                       # 去重、保序
                 if not e.startswith(WEIGHT_PAT)                           # 權重防呆（雙保險）
                 and e not in OUTCOME_SOURCE_VARS                          # 結果來源防呆（雙保險）
                 and e.replace("_percr", "") not in OUTCOME_SOURCE_VARS    # 其校正版亦然
                 and _usable(df[e])]                                       # 退化欄防呆（見 _usable）
    counts = dict(n_adults=int(len(df)), n_outcome_known=int(df["kidney_damage"].notna().sum()),
                  n_kidney_damage=int(df["kidney_damage"].sum()),
                  n_exposures=len(exposures), exposures=exposures, covariates=covars,
                  urinary_exposures=urinary,
                  arsenic_negative_control=(ARSENIC_NEGATIVE_CONTROL
                                            if ARSENIC_NEGATIVE_CONTROL in exposures else None),
                  n_by_exposure={e: int(df[e].notna().sum()) for e in exposures})
    if verbose:
        print(f"\n[完成] 暴露 {len(exposures)} 個｜共變項 {len(covars)} 個")
    return dict(cohort=df, exposures=exposures, covariates=covars, counts=counts)
