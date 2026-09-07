# -*- coding: utf-8 -*-
"""NHANES 1999–2004 三週期世代建構：合併 → 腎損傷族群 → 三大病因標籤 → 特徵/封存集分離。

誠實邊界（印在每份輸出）：
  * 標籤是「共病代理」（問卷診斷＋血清學＋surplus sera 自體抗體），不是切片病因。
  * 免疫標籤只在 SSANA 次樣本（1999–2004 surplus sera，n≈4,532）可判定——
    主世代因此限定為「SSANA 次樣本 ∩ 腎損傷」，三類標籤皆可判定；
    次世代（感染 vs 代謝，不含免疫）用全部三週期擴大樣本。
  * 每個輸入檔皆經 provenance.require_real()（雜湊驗證）——零自製資料。

變數名以候選清單在執行期自省；找不到即 raise 並列出該檔實際欄位（不猜、不填）。"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from provenance import require_real  # noqa: E402

RAW = os.path.join(ROOT, "data", "raw")

CYCLES = {  # 檔名 → 週期
    "1999-2000": dict(demo="DEMO.xpt", biochem="LAB18.xpt", cbc="LAB25.xpt", crp="LAB11.xpt",
                      acr="LAB16.xpt", hba1c="LAB10.xpt", hep="LAB02.xpt", diq="DIQ.xpt"),
    "2001-2002": dict(demo="DEMO_B.xpt", biochem="L40_B.xpt", cbc="L25_B.xpt", crp="L11_B.xpt",
                      acr="L16_B.xpt", hba1c="L10_B.xpt", hep="L02_B.xpt", diq="DIQ_B.xpt"),
    "2003-2004": dict(demo="DEMO_C.xpt", biochem="L40_C.xpt", cbc="L25_C.xpt", crp="L11_C.xpt",
                      acr="L16_C.xpt", hba1c="L10_C.xpt", hep="L02_C.xpt", diq="DIQ_C.xpt"),
}
# 第二批（腎臟相關指數全景）：脂質盤／重金屬（腎小管毒性）／營養素／維生素 D（CKD-MBD）／PTH 在 L11_C
EXTRA_FILES = {
    "1999-2000": ["LAB13.xpt", "LAB13AM.xpt", "LAB06.xpt"],
    "2001-2002": ["L13_B.xpt", "L13AM_B.xpt", "L06_B.xpt", "VID_B.xpt"],
    "2003-2004": ["L13_C.xpt", "L13AM_C.xpt", "L06BMT_C.xpt", "L06NB_C.xpt", "L06MH_C.xpt", "L06TFR_C.xpt", "VID_C.xpt"],
}
# ── 2005–2018 七個週期（僅供不需要 ANA 的任務：感染／代謝）
# 檔名皆以「HTTP 200 且內容開頭為 HEADER RECORD」實測驗證——
# 注意 /Nchs/Nhanes/YYYY-YYYY/ 這個路徑對任何檔名都回 200＋HTML 錯誤頁，不可用於探測存在性。
CYCLES_EXT = {
    "2005-2006": dict(demo="DEMO_D.xpt", biochem="BIOPRO_D.xpt", cbc="CBC_D.xpt", acr="ALB_CR_D.xpt",
                      hba1c="GHB_D.xpt", diq="DIQ_D.xpt", hepb="HEPBD_D.xpt", hepc="HEPC_D.xpt", crp="CRP_D.xpt"),
    "2007-2008": dict(demo="DEMO_E.xpt", biochem="BIOPRO_E.xpt", cbc="CBC_E.xpt", acr="ALB_CR_E.xpt",
                      hba1c="GHB_E.xpt", diq="DIQ_E.xpt", hepb="HEPBD_E.xpt", hepc="HEPC_E.xpt", crp="CRP_E.xpt"),
    "2009-2010": dict(demo="DEMO_F.xpt", biochem="BIOPRO_F.xpt", cbc="CBC_F.xpt", acr="ALB_CR_F.xpt",
                      hba1c="GHB_F.xpt", diq="DIQ_F.xpt", hepb="HEPBD_F.xpt", hepc="HEPC_F.xpt", crp="CRP_F.xpt"),
    "2011-2012": dict(demo="DEMO_G.xpt", biochem="BIOPRO_G.xpt", cbc="CBC_G.xpt", acr="ALB_CR_G.xpt",
                      hba1c="GHB_G.xpt", diq="DIQ_G.xpt", hepb="HEPBD_G.xpt", hepc="HEPC_G.xpt"),
    "2013-2014": dict(demo="DEMO_H.xpt", biochem="BIOPRO_H.xpt", cbc="CBC_H.xpt", acr="ALB_CR_H.xpt",
                      hba1c="GHB_H.xpt", diq="DIQ_H.xpt", hepb="HEPBD_H.xpt", hepc="HEPC_H.xpt"),
    "2015-2016": dict(demo="DEMO_I.xpt", biochem="BIOPRO_I.xpt", cbc="CBC_I.xpt", acr="ALB_CR_I.xpt",
                      hba1c="GHB_I.xpt", diq="DIQ_I.xpt", hepb="HEPBD_I.xpt", hepc="HEPC_I.xpt", crp="HSCRP_I.xpt"),
    "2017-2018": dict(demo="DEMO_J.xpt", biochem="BIOPRO_J.xpt", cbc="CBC_J.xpt", acr="ALB_CR_J.xpt",
                      hba1c="GHB_J.xpt", diq="DIQ_J.xpt", hepb="HEPBD_J.xpt", hepc="HEPC_J.xpt", crp="HSCRP_J.xpt"),
}
EXTRA_FILES_EXT = {   # 脂質盤在新週期拆成三個檔
    "2005-2006": ["TCHOL_D.xpt", "HDL_D.xpt", "TRIGLY_D.xpt"],
    "2007-2008": ["TCHOL_E.xpt", "HDL_E.xpt", "TRIGLY_E.xpt"],
    "2009-2010": ["TCHOL_F.xpt", "HDL_F.xpt", "TRIGLY_F.xpt"],
    "2011-2012": ["TCHOL_G.xpt", "HDL_G.xpt", "TRIGLY_G.xpt"],
    "2013-2014": ["TCHOL_H.xpt", "HDL_H.xpt", "TRIGLY_H.xpt"],
    "2015-2016": ["TCHOL_I.xpt", "HDL_I.xpt", "TRIGLY_I.xpt"],
    "2017-2018": ["TCHOL_J.xpt", "HDL_J.xpt", "TRIGLY_J.xpt"],
}
# 肌酸酐標準化校正係數（NHANES 官方分析注記，逐週期查證；未列者不需校正）
SCR_CALIBRATION = {
    "1999-2000": (-0.184, 0.960),
    "2005-2006": (-0.016, 0.978),    # BIOPRO_D 文件明載「Correction ... is highly recommended」
}

ANA_FILES = ["SSANA_A.xpt", "SSANA2_A.xpt"]
CYSTATIN_FILES = ["SSCYST_A.xpt", "SSCYST_B.xpt"]           # surplus sera 1999-2002，跨週期以 SEQN 併

# 變數候選（NHANES SAS 名；執行期驗證存在性）
CAND = dict(
    age=["RIDAGEYR"], sex=["RIAGENDR"],
    scr=["LBXSCR"], hba1c=["LBXGH"], diq=["DIQ010"],
    uma=["URXUMA"], ucr=["URXUCR"],
    hbsag=["LBDHBG", "LBXHBG"], hbcab=["LBXHBC"],
    hcv_ab=["LBXHCV", "LBDHCV"], hcv_rna=["LBXHCR", "LBDHCR", "SSHCV", "LBXHCVRNA"],
)

# 特徵通道（封存集除外的「全部」常規檢驗）：由檔案欄位動態決定，這裡列「已知語意」的中文名對照
FEATURE_LABELS = {
    "LBXSCR": "血清肌酸酐", "LBXSBU": "尿素氮 BUN", "LBXSUA": "尿酸", "LBXSAL": "血清白蛋白",
    "LBXSGL": "血清葡萄糖（隨機）", "LBXSCH": "總膽固醇", "LBXSTR": "三酸甘油酯",
    "LBXSGTSI": "GGT", "LBXSASSI": "AST", "LBXSATSI": "ALT", "LBXSLDSI": "LDH", "LBXSAPSI": "鹼性磷酸酶",
    "LBXSTB": "總膽紅素", "LBXSTP": "總蛋白", "LBXSGB": "球蛋白", "LBXSPH": "磷", "LBXSCA": "鈣",
    "LBXSNASI": "鈉", "LBXSKSI": "鉀", "LBXSCLSI": "氯", "LBXSC3SI": "碳酸氫根", "LBXSIR": "鐵", "LBXSOSSI": "滲透壓",
    "LBXWBCSI": "白血球", "LBXLYPCT": "淋巴球 %", "LBXMOPCT": "單核球 %", "LBXNEPCT": "嗜中性球 %",
    "LBXEOPCT": "嗜酸性球 %", "LBXBAPCT": "嗜鹼性球 %", "LBXRBCSI": "紅血球", "LBXHGB": "血色素",
    "LBXHCT": "血比容", "LBXMCVSI": "MCV", "LBXMC": "MCHC", "LBXMCHSI": "MCH", "LBXRDW": "RDW",
    "LBXPLTSI": "血小板", "LBXMPSI": "平均血小板體積", "LBXCRP": "CRP",
    "URXUMA": "尿白蛋白", "URXUCR": "尿肌酸酐",
    # ── 第二批：腎臟相關指數全景
    "LBXTC": "總膽固醇（脂質盤）", "LBDHDL": "HDL 膽固醇", "LBXHDD": "HDL 膽固醇（直接法）",
    "LBDLDL": "LDL 膽固醇", "LBXTR": "三酸甘油酯（空腹）",
    "LBXBPB": "血鉛", "LBXBCD": "血鎘", "LBXTHG": "血汞", "LBXCOT": "血清可丁尼（菸暴露）",
    "LBXFER": "鐵蛋白", "LBXFOL": "血清葉酸", "LBXRBF": "紅血球葉酸", "LBXB12": "維生素 B12",
    "LBXHCY": "同半胱胺酸", "LBXMMA": "甲基丙二酸",
    "LBDVIDMS": "25-羥維生素 D（LC-MS）", "LBXPT21": "副甲狀腺素 PTH", "LBXBAP": "骨鹼性磷酸酶 BAP",
    "SSCYPC": "胱蛋白酶抑制素 C（Cystatin C）",
}
# 衍生特徵
DERIVED = dict(ACR="尿白蛋白/肌酸酐比（mg/g）", eGFR="估計腎絲球過濾率（CKD-EPI 2021）", NLR="嗜中性球/淋巴球比")


# 跨週期變數別名：同一檢驗在不同週期的 SAS 名不同，讀檔即統一
#   LBDSCR  = 血清肌酸酐（2001-2002）；其餘週期為 LBXSCR
#   （2026-08-30 稽核發現：漏了這個別名會讓整個 2001-2002 週期的肌酸酐被靜默丟棄，
#     導致 708 人無 eGFR、753 人無法 KDIGO 分期——缺值看似是資料特性，實為合併缺陷）
ALIASES = {"LBDSCR": "LBXSCR"}


def _read(key):
    p = os.path.join(RAW, key)
    require_real(p)
    df = pd.read_sas(p, format="xport")
    df.columns = [c.upper() for c in df.columns]
    for src, dst in ALIASES.items():
        if src in df.columns and dst not in df.columns:
            df = df.rename(columns={src: dst})
    return df


def _pick(df, cands, where, required=True):
    for c in cands:
        if c in df.columns:
            return c
    if required:
        raise RuntimeError(f"{where} 找不到候選變數 {cands}；實際欄位：{sorted(df.columns)[:40]}…")
    return None


def egfr_ckdepi2021(scr_mgdl, age, is_female):
    """CKD-EPI 2021（無種族係數）。scr 需為標準化 mg/dL。"""
    k = np.where(is_female, 0.7, 0.9)
    a = np.where(is_female, -0.241, -0.302)
    r = scr_mgdl / k
    egfr = 142.0 * np.minimum(r, 1) ** a * np.maximum(r, 1) ** (-1.200) * (0.9938 ** age)
    return egfr * np.where(is_female, 1.012, 1.0)


def load_all(verbose=True):
    """回傳 (df 全體合併, meta)。血清肌酸酐依 NHANES 分析指引：1999-2000 需校正（standard = -0.184 + 0.960×SCr），
    2001-2004 不需（校正式為 NHANES 官方分析注記；敏感度分析含未校正版）。"""
    rows = []
    for cyc, files in CYCLES.items():
        demo = _read(files["demo"])
        d = demo[["SEQN", _pick(demo, CAND["age"], files["demo"]), _pick(demo, CAND["sex"], files["demo"])]].copy()
        d.columns = ["SEQN", "age", "sex"]
        d["cycle"] = cyc
        for role in ("biochem", "cbc", "crp", "acr", "hba1c", "hep", "diq"):
            f = _read(files[role])
            keep = [c for c in f.columns if c == "SEQN" or c in FEATURE_LABELS or
                    any(c in CAND[k] for k in ("hba1c", "diq", "hbsag", "hbcab", "hcv_ab", "hcv_rna"))]
            d = d.merge(f[keep].drop_duplicates("SEQN"), on="SEQN", how="left", suffixes=("", f"_{role}"))
        for extra in EXTRA_FILES.get(cyc, []):
            f = _read(extra)
            keep = [c for c in f.columns if c == "SEQN" or c in FEATURE_LABELS]
            if len(keep) > 1:
                d = d.merge(f[keep].drop_duplicates("SEQN"), on="SEQN", how="left", suffixes=("", "_x"))
        if cyc in SCR_CALIBRATION and "LBXSCR" in d.columns:
            a, b = SCR_CALIBRATION[cyc]
            d["LBXSCR_raw"] = d["LBXSCR"]
            d["LBXSCR"] = a + b * d["LBXSCR"]                   # 肌酸酐標準化校正（逐週期依官方注記）
        rows.append(d)
        if verbose:
            print(f"[cohort] {cyc}: n={len(d)}")
    df = pd.concat(rows, ignore_index=True)

    # Cystatin C（surplus 1999-2002）：兩檔各覆蓋一段，SEQN 併接
    for cf in CYSTATIN_FILES:
        cy = _read(cf)
        keep = [c for c in cy.columns if c in ("SEQN", "SSCYPC")]
        if "SSCYPC" in keep:
            df = df.merge(cy[keep].drop_duplicates("SEQN").rename(columns={"SSCYPC": f"SSCYPC_{cf[:8]}"}), on="SEQN", how="left")
    cys_cols = [c for c in df.columns if c.startswith("SSCYPC_")]
    if cys_cols:
        df["SSCYPC"] = df[cys_cols].bfill(axis=1).iloc[:, 0]
        df = df.drop(columns=cys_cols)
    # HDL 名稱跨週期不同（99-01 LBDHDL、03-04 LBXHDD）→ 合併為單欄
    if "LBXHDD" in df.columns:
        df["LBDHDL"] = df.get("LBDHDL", pd.Series(np.nan, index=df.index)).fillna(df["LBXHDD"])
        df = df.drop(columns=["LBXHDD"])
    ana = _read(ANA_FILES[0])
    ana_cols = [c for c in ana.columns if c != "SEQN"]
    df = df.merge(ana.drop_duplicates("SEQN"), on="SEQN", how="left", indicator="in_ana")
    df["in_ana_subsample"] = (df["in_ana"] == "both")
    df = df.drop(columns=["in_ana"])
    if verbose:
        print(f"[cohort] 合併：n={len(df)}；SSANA 次樣本 {int(df['in_ana_subsample'].sum())}")
    return df, dict(ana_cols=ana_cols)


def load_extended(verbose=True):
    """2005–2018 七個週期。這些週期沒有 surplus sera ANA，故免疫狀態一律未知。"""
    rows = []
    for cyc, files in CYCLES_EXT.items():
        demo = _read(files["demo"])
        d = demo[["SEQN", "RIDAGEYR", "RIAGENDR"]].copy()
        d.columns = ["SEQN", "age", "sex"]
        d["cycle"] = cyc
        for role, fn in files.items():
            if role == "demo":
                continue
            f = _read(fn)
            keep = [c for c in f.columns if c == "SEQN" or c in FEATURE_LABELS or
                    c in ("LBXGH", "DIQ010", "LBDHBG", "LBXHBC", "LBXHCR", "LBDHCV", "LBXHCV", "LBDHCI")]
            if len(keep) > 1:
                d = d.merge(f[keep].drop_duplicates("SEQN"), on="SEQN", how="left", suffixes=("", "_x"))
        for extra in EXTRA_FILES_EXT.get(cyc, []):
            f = _read(extra)
            keep = [c for c in f.columns if c == "SEQN" or c in FEATURE_LABELS]
            if len(keep) > 1:
                d = d.merge(f[keep].drop_duplicates("SEQN"), on="SEQN", how="left", suffixes=("", "_x"))
        if cyc in SCR_CALIBRATION and "LBXSCR" in d.columns:
            a, b = SCR_CALIBRATION[cyc]
            d["LBXSCR_raw"] = d["LBXSCR"]
            d["LBXSCR"] = a + b * d["LBXSCR"]
        rows.append(d)
        if verbose:
            print(f"[cohort-ext] {cyc}: n={len(d)}")
    df = pd.concat(rows, ignore_index=True)
    df["in_ana_subsample"] = False
    return df


def build_extended(P, verbose=True):
    """1999–2018 全週期世代，**僅供不需要 ANA 的任務**（感染／代謝）。

    2005 年後沒有 ANA，免疫狀態是未知而非陰性——本函式回傳的世代不含可用的免疫標籤，
    任何免疫任務都必須改用 build() 的 primary（ANA 實測子樣本）。"""
    base, _ = load_all(verbose=verbose)
    ext = load_extended(verbose=verbose)
    df = pd.concat([base, ext], ignore_index=True, sort=False)
    df = df[df["age"] >= P["population"]["value"]["age_min"]].copy()
    female = df["sex"] == 2
    df["eGFR"] = egfr_ckdepi2021(df["LBXSCR"].to_numpy(float), df["age"].to_numpy(float), female.to_numpy())
    df["ACR"] = df["URXUMA"] / (df["URXUCR"] / 100.0)
    df["kidney_damage"] = (df["eGFR"] < 60) | (df["ACR"] >= 30)
    kd = df[df["kidney_damage"].fillna(False)].copy()
    hbsag = _pick(kd, CAND["hbsag"], "hep", required=False)
    hcv_rna = _pick(kd, CAND["hcv_rna"], "hep", required=False)
    hcv_ab = _pick(kd, CAND["hcv_ab"], "hep", required=False)
    kd["lab_metabolic"] = (kd["DIQ010"] == 1) | (kd["LBXGH"] >= 6.5)
    inf = pd.Series(False, index=kd.index)
    if hbsag:
        inf |= kd[hbsag] == 1
    if hcv_rna:
        inf |= kd[hcv_rna] == 1
    kd["lab_infection"] = inf
    kd["cause"] = np.where(kd["lab_infection"], "感染性",
                           np.where(kd["lab_metabolic"].fillna(False), "代謝性", "其他/未歸類"))
    kd["NLR"] = kd["LBXNEPCT"] / kd["LBXLYPCT"].replace(0, np.nan)
    archive = set(["LBXGH", "DIQ010"] + [c for c in kd.columns if c.startswith("SS")] +
                  [v for v in (hbsag, hcv_rna, hcv_ab) if v] +
                  [c for c in kd.columns if c.startswith(("LBXHB", "LBDHB", "LBXHC", "LBDHC", "LBXHA", "LBXHD", "LBDHD"))])
    feats = [c for c in kd.columns if c in FEATURE_LABELS and c not in archive] + ["ACR", "eGFR", "NLR", "age", "sex"]
    if verbose:
        print(f"[cohort-ext] 全週期成人 {len(df)}；腎損傷 {len(kd)}；"
              f"感染 {int(kd['lab_infection'].sum())}、代謝 {int(kd['lab_metabolic'].fillna(False).sum())}")
        print(f"[cohort-ext] 逐週期腎損傷：{kd['cycle'].value_counts().sort_index().to_dict()}")
    return dict(cohort=kd, features=feats, archive=sorted(archive),
                counts=dict(n=len(kd), infection=int(kd["lab_infection"].sum()),
                            metabolic=int(kd["lab_metabolic"].fillna(False).sum()),
                            by_cycle={str(k): int(v) for k, v in kd["cycle"].value_counts().sort_index().items()}),
                feature_labels={**FEATURE_LABELS, **DERIVED, "age": "年齡", "sex": "性別"})


def build(P, verbose=True):
    df, meta = load_all(verbose=verbose)
    pop = P["population"]["value"]
    df = df[df["age"] >= pop["age_min"]].copy()

    # 腎損傷定義
    female = df["sex"] == 2
    df["eGFR"] = egfr_ckdepi2021(df["LBXSCR"].to_numpy(float), df["age"].to_numpy(float), female.to_numpy())
    df["ACR"] = df["URXUMA"] / (df["URXUCR"] / 100.0)           # mg/L ÷ (mg/dL→mg/L 係數) → mg/g
    df["kidney_damage"] = (df["eGFR"] < 60) | (df["ACR"] >= 30)
    kd = df[df["kidney_damage"].fillna(False)].copy()
    if verbose:
        print(f"[cohort] 成人 {len(df)}；腎損傷（eGFR<60 或 ACR≥30）{len(kd)}")

    # ── 三大病因標籤（共病代理；操作型定義見 design.json）
    hbsag = _pick(kd, CAND["hbsag"], "hep", required=False)
    hcv_rna = _pick(kd, CAND["hcv_rna"], "hep", required=False)
    hcv_ab = _pick(kd, CAND["hcv_ab"], "hep", required=False)
    lab_meta = dict(hbsag_var=hbsag, hcv_rna_var=hcv_rna, hcv_ab_var=hcv_ab)
    kd["lab_metabolic"] = (kd["DIQ010"] == 1) | (kd["LBXGH"] >= 6.5)
    inf = pd.Series(False, index=kd.index)
    if hbsag:
        inf |= kd[hbsag] == 1
    if hcv_rna:
        inf |= kd[hcv_rna] == 1
    elif hcv_ab:
        inf |= kd[hcv_ab] == 1
        lab_meta["note_hcv"] = "無 RNA 變數，以抗體陽性代理（列敏感度）"
    kd["lab_infection"] = inf
    # 免疫：SSANA 次樣本內，SSTOT ≥3（該檔自身的 3+/4+ 陽性規則，對應滴度 ≥1:80 確認）或任一特異抗體陽性
    sp_cols = [c for c in meta["ana_cols"] if c.startswith("SS") and c not in
               ("SSTOT", "SSNUC", "SSCYT", "WTANA6YR") and not c.startswith(("SS8", "SS16", "SS32", "SS64", "SS12"))
               and not c.startswith(("SSNU", "SSCY", "SSMI"))]
    sp_pos = (kd[sp_cols] == 1).any(axis=1) if sp_cols else pd.Series(False, index=kd.index)
    kd["lab_immune"] = np.where(kd["in_ana_subsample"], ((kd["SSTOT"] >= 3) | sp_pos), np.nan)

    # 歸類（順位 免疫 > 感染 > 代謝；重疊另計）
    def assign(r):
        if r["in_ana_subsample"] and r["lab_immune"] == 1:
            return "免疫性"
        if r["lab_infection"]:
            return "感染性"
        if r["lab_metabolic"]:
            return "代謝性"
        return "其他/未歸類"
    kd["cause"] = kd.apply(assign, axis=1)
    overlap = dict(
        immune_and_infection=int(((kd["lab_immune"] == 1) & kd["lab_infection"]).sum()),
        immune_and_metabolic=int(((kd["lab_immune"] == 1) & kd["lab_metabolic"]).sum()),
        infection_and_metabolic=int((kd["lab_infection"] & kd["lab_metabolic"]).sum()))

    # ── 特徵／封存集分離（通道消耗規則）
    archive = set(["LBXGH", "DIQ010"] + [c for c in kd.columns if c.startswith("SS")] +
                  [v for v in (hbsag, hcv_rna, hcv_ab) if v] +
                  [c for c in kd.columns if c.startswith(("LBXHB", "LBDHB", "LBXHC", "LBDHC", "LBXHA", "LBXHD", "LBDHD"))])
    feat_cols = [c for c in kd.columns if c in FEATURE_LABELS and c not in archive]
    kd["NLR"] = kd["LBXNEPCT"] / kd["LBXLYPCT"].replace(0, np.nan)
    features = feat_cols + ["ACR", "eGFR", "NLR", "age", "sex"]

    primary = kd[kd["in_ana_subsample"]].copy()                 # 三類皆可判定
    secondary = kd.copy()                                        # 感染 vs 代謝（免疫未知者不進 Level1 三類）
    counts = dict(primary={c: int((primary["cause"] == c).sum()) for c in primary["cause"].unique()},
                  secondary_all={c: int((kd["cause"] == c).sum()) for c in kd["cause"].unique()},
                  overlap=overlap, n_kidney_damage=len(kd), n_primary=len(primary), lab_meta=lab_meta,
                  archive_n=len(archive), feature_n=len(features))
    if verbose:
        print(f"[cohort] 主世代（SSANA∩腎損傷）n={len(primary)}：{counts['primary']}")
        print(f"[cohort] 全腎損傷 n={len(kd)}：{counts['secondary_all']}；重疊 {overlap}")
        print(f"[cohort] 特徵 {len(features)} 欄；封存 {len(archive)} 欄（標籤來源，不得為特徵）")
    return dict(primary=primary, secondary=secondary, features=features, archive=sorted(archive),
                counts=counts, feature_labels={**FEATURE_LABELS, **DERIVED, "age": "年齡", "sex": "性別"})
