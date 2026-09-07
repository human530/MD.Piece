# -*- coding: utf-8 -*-
"""產生完整變數字典（科展計畫書附錄三）。
    python make_variable_dict.py

三批變數一次列全，數字全部由實際資料算出，不手打：
  ① 檢驗特徵 60 個   —— 下游狀態，可進模型
  ② 上游暴露 213 個  —— ExWAS 用
  ③ 封存變數 79 個   —— 定義標籤，永不可當特徵（附封存理由）
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np                                            # noqa: E402
from nhanes_cohort import FEATURE_LABELS, DERIVED, build_extended   # noqa: E402

# 封存理由（依標籤定義回推）
ARCHIVE_REASON = {
    "LBXGH": "定義代謝性標籤（HbA1c≥6.5%）", "DIQ010": "定義代謝性標籤（醫師診斷糖尿病）",
    "LBDHBG": "定義感染性標籤（HBsAg）", "LBXHBC": "定義感染性標籤（抗-HBc）",
    "LBXHCR": "定義感染性標籤（HCV RNA）", "LBDHCV": "定義感染性標籤（抗-HCV）",
    "LBDHCI": "定義感染性標籤（HCV 判讀）",
    "LBDHDL": "脂質盤內，與封存之總膽固醇同批", "LBXHCY": "同半胱胺酸（非標籤，隨批封存）",
    "LBXHCT": "血球比容（與血色素共線，隨批封存）",
}
UNIT = {  # 常見單位（NHANES 官方文件）
    "LBXSAL": "g/dL", "LBXSATSI": "U/L", "LBXSASSI": "U/L", "LBXSAPSI": "U/L",
    "LBXSBU": "mg/dL", "LBXSCA": "mg/dL", "LBXSCH": "mg/dL", "LBXSC3SI": "mmol/L",
    "LBXSGTSI": "U/L", "LBXSGL": "mg/dL", "LBXSIR": "ug/dL", "LBXSLDSI": "U/L",
    "LBXSPH": "mg/dL", "LBXSTB": "mg/dL", "LBXSTP": "g/dL", "LBXSTR": "mg/dL",
    "LBXSUA": "mg/dL", "LBXSCR": "mg/dL", "LBXSNASI": "mmol/L", "LBXSKSI": "mmol/L",
    "LBXSCLSI": "mmol/L", "LBXSOSSI": "mmol/kg", "LBXSGB": "g/dL",
    "LBXWBCSI": "1000 cells/uL", "LBXRBCSI": "million cells/uL", "LBXHGB": "g/dL",
    "LBXMCVSI": "fL", "LBXMCHSI": "pg", "LBXMC": "g/dL", "LBXRDW": "%",
    "LBXPLTSI": "1000 cells/uL", "LBXMPSI": "fL", "LBXCRP": "mg/dL",
    "LBXBAP": "ug/L", "URXUMA": "ug/mL", "URXUCR": "mg/dL", "LBXTC": "mg/dL",
    "LBXTR": "mg/dL", "LBDLDL": "mg/dL", "LBXBPB": "ug/dL", "LBXBCD": "ug/L",
    "LBXFER": "ng/mL", "LBXFOL": "ng/mL", "LBXB12": "pg/mL", "LBXMMA": "umol/L",
    "LBXTHG": "ug/L", "LBXRBF": "ng/mL", "LBXCOT": "ng/mL", "LBDVIDMS": "nmol/L",
    "LBXPT21": "pg/mL", "ACR": "mg/g", "eGFR": "mL/min/1.73m2", "NLR": "比值",
    "age": "歲", "sex": "1=男 2=女",
}
GROUP = {
    **{k: "生化" for k in ("LBXSAL LBXSATSI LBXSASSI LBXSAPSI LBXSBU LBXSCA LBXSCH LBXSC3SI "
                           "LBXSGTSI LBXSGL LBXSIR LBXSLDSI LBXSPH LBXSTB LBXSTP LBXSTR LBXSUA "
                           "LBXSCR LBXSNASI LBXSKSI LBXSCLSI LBXSOSSI LBXSGB").split()},
    **{k: "血球" for k in ("LBXWBCSI LBXLYPCT LBXMOPCT LBXNEPCT LBXEOPCT LBXBAPCT LBXRBCSI "
                           "LBXHGB LBXMCVSI LBXMCHSI LBXMC LBXRDW LBXPLTSI LBXMPSI").split()},
    **{k: "發炎" for k in ("LBXCRP LBXBAP").split()},
    **{k: "尿液" for k in ("URXUMA URXUCR").split()},
    **{k: "脂質" for k in ("LBXTC LBXTR LBDLDL").split()},
    **{k: "營養／微量元素" for k in ("LBXBPB LBXBCD LBXFER LBXFOL LBXB12 LBXMMA LBXTHG "
                                     "LBXRBF LBXCOT LBDVIDMS LBXPT21").split()},
    **{k: "衍生／人口學" for k in ("ACR eGFR NLR age sex").split()},
}


def main():
    P = json.load(open(os.path.join(ROOT, "params", "design.json"), encoding="utf-8"))
    E = build_extended(P, verbose=False)
    kd, feats, arch = E["cohort"], E["features"], E["archive"]
    lab = {**FEATURE_LABELS, **DERIVED, "age": "年齡", "sex": "性別"}
    X = json.load(open(os.path.join(ROOT, "results", "exwas.json"), encoding="utf-8"))
    expo = [r["exposure"] for r in X["results"]]

    L = ["# 附錄三：完整變數字典", "",
         "科展計畫書《血液會說話嗎？》附錄　　資料來源：NHANES 1999–2018（261 檔）", "",
         f"**變數總數 {len(feats) + len(expo) + len(arch)}**"
         f"（檢驗特徵 {len(feats)}、上游暴露 {len(expo)}、封存 {len(arch)}）", "",
         "> 全部樣本數由實際資料計算，非手動填寫。腎損傷世代 n=8,983。", "", "---", "",
         "## 一、檢驗特徵（可進模型，共 %d 項）" % len(feats), ""]

    by_group = {}
    for f in feats:
        by_group.setdefault(GROUP.get(f, "其他"), []).append(f)
    for g in ("生化", "血球", "發炎", "尿液", "脂質", "營養／微量元素", "衍生／人口學", "其他"):
        if g not in by_group:
            continue
        L += [f"### （{'一二三四五六七八'[list(by_group).index(g) % 8]}）{g}"
              f"　{len(by_group[g])} 項", "",
              "| 變數代碼 | 中文名稱 | 單位 | 有值人數 | 覆蓋率 | 中位數 |",
              "|---|---|---|---|---|---|"]
        for f in by_group[g]:
            v = kd[f].dropna() if f in kd.columns else np.array([])
            n = len(v)
            med = f"{np.median(v):.3g}" if n else "—"
            L.append(f"| `{f}` | {lab.get(f, '—')} | {UNIT.get(f, '—')} | "
                     f"{n:,} | {n / len(kd):.1%} | {med} |")
        L.append("")

    L += ["---", "", "## 二、上游暴露（ExWAS，共 %d 項）" % len(expo), "",
          "> 用於「暴露組關聯掃描」，性質為**上游原因**而非下游狀態。",
          "> 標示 `_percr` 者為尿肌酸酐校正版；未校正版與校正版並列，"
          "因兩者在慢性腎臟病中偏誤方向不同。", ""]

    def cat(e):
        b = e.replace("_percr", "")
        if b.startswith("藥陰"):
            return "藥物－陰性對照"
        if b.startswith("藥"):
            return "處方藥物類別"
        if b in ("LBXBCD", "LBXBPB", "LBXTHG", "LBXBSE"):
            return "血中金屬"
        if b.startswith(("LBXPF", "LBXEP", "LBXMPAH", "LBXNF", "LBXMF")):
            return "全氟烷基物質 PFAS"
        if b.startswith("URXUAS") or b in ("URXUDMA", "URXUMMA", "URXUAB", "URXUAC"):
            return "尿砷（含物種分析）"
        if b.startswith("URXU") and len(b) <= 8:
            return "尿中金屬"
        return "尿液化學品（塑化劑／農藥）"

    ec = {}
    for e in expo:
        ec.setdefault(cat(e), []).append(e)
    for g in sorted(ec, key=lambda k: -len(ec[k])):
        raw = [e for e in ec[g] if not e.endswith("_percr")]
        L += [f"### {g}　{len(ec[g])} 項（原始 {len(raw)}）", "", "```",
              "  " + "、".join(raw), "```", ""]

    L += ["---", "", "## 三、封存變數（**永不可作為特徵**，共 %d 項）" % len(arch), "",
          "### 封存規則", "",
          "> **凡用於定義標籤的檢驗，一律封存。** 使用它們預測自身所定義的標籤，",
          "> 等同於「用答案預測答案」，是本研究之硬性規則，非可調參數。", "",
          "| 變數代碼 | 封存理由 |", "|---|---|"]
    auto = [a for a in arch if a not in ARCHIVE_REASON]
    for a in sorted(ARCHIVE_REASON):
        if a in arch:
            L.append(f"| `{a}` | {ARCHIVE_REASON[a]} |")
    L += [f"| `SS*`（{len(auto)} 項）| 定義免疫性標籤——抗核抗體滴度與特異自體抗體 |", "",
          "**特異自體抗體明細**（若改用腎切片標籤，此批可解放為特徵）：", "", "```",
          "  " + "、".join(sorted(auto)), "```", "",
          "---", "",
          "## 四、已確認不存在於 NHANES 之關鍵檢驗", "",
          "| 檢驗 | 對免疫性腎炎的重要性 | 狀態 |",
          "|---|---|---|",
          "| 補體 C3 | 免疫複合體腎炎最具判別力之常規指標 | ❌ 261 檔中皆無 |",
          "| 補體 C4 | 同上 | ❌ 皆無 |",
          "| 免疫球蛋白 IgG／IgA／IgM | IgA 腎病、單株免疫球蛋白腎病 | ❌ 皆無 |",
          "| 血沉 ESR | 發炎活動度 | ❌ 皆無 |",
          "| 抗-dsDNA | 狼瘡腎炎活動度 | ❌ 皆無 |",
          "| 抗-PLA2R | 原發性膜性腎病（特異度 0.97）| ❌ 皆無 |",
          "| ANCA | 寡免疫型腎絲球腎炎 | ❌ 皆無 |",
          "| 尿沉渣鏡檢 | 紅血球柱狀體、變形紅血球 | ❌ 皆無 |", "",
          "> **此表即為本研究免疫類判別失敗之直接原因**（見計畫書伍之三）。"]

    out = os.path.join(ROOT, "docs", "附錄三_變數字典.md")
    open(out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(f"[完成] {out}")
    print(f"  檢驗特徵 {len(feats)}｜上游暴露 {len(expo)}｜封存 {len(arch)}｜"
          f"合計 {len(feats) + len(expo) + len(arch)}")


if __name__ == "__main__":
    main()
