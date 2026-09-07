# -*- coding: utf-8 -*-
"""歸因對照表的單一真相來源：特徵 → 文獻證據等級標籤。

由 docs/verified_anchors_2026-08-30.json（逐條經 PubMed 摘要原文比對）與早期 literature_anchors.json 匯總。
report.md 與離線 UI 共用此檔，避免兩處各寫一份而不一致。

等級語彙（顯示於報告與介面）：
  causal_pos   因果級證據支持（MR／RCT 陽性）
  causal_neg   因果級證據反對（MR 陰性／RCT 陰性）——重要：關聯存在但介入無效或無因果
  observational 僅觀察性關聯
  reverse      反向因果警告（腎功能→標記，而非標記→病因）
  adjacent     標籤鄰近（與標籤定義的生物學直接相連，不算獨立發現）
  conflict     與文獻方向相反（本資料 vs 文獻）
  candidate    候選（本資料之發現，待確認）——預設值
"""

LEVELS = {
    "causal_pos": "因果級（支持）",
    "causal_neg": "因果級（反對）",
    "observational": "僅觀察性",
    "reverse": "反向因果警告",
    "adjacent": "標籤鄰近",
    "conflict": "與文獻方向相反",
    "candidate": "候選（待確認）",
    "artifact": "量測偏斜／假訊號",
}

# key = "病因|特徵"；value = (等級, 說明)
ANCHORS = {
    # ── 代謝性
    "代謝性|LBXSTR": ("observational", "TG：MR 顯示表面關聯由 GCKR 單一多效性變異驅動，排除後無因果（PMID 28754456）；"
                                       "既有觀察性錨點 PMID 27537361/30300472 下修為僅觀察性"),
    "代謝性|LBXTR": ("observational", "同上（空腹 TG）"),
    "代謝性|LBDLDL": ("causal_neg", "LDL 與 eGFR／CKD／白蛋白尿 MR 皆無因果關聯（PMID 28754456）"),
    "代謝性|LBDHDL": ("causal_pos", "HDL 每高 17 mg/dL：eGFR 高 0.8%、eGFR<60 OR 0.85（MR，PMID 28754456）；"
                                     "惟反向 MR 亦顯示 eGFR↑→HDL↑（PMID 35856088），方向需並陳"),
    "代謝性|LBXSUA": ("reverse", "尿酸：觀察性關聯已知（PMID 26342044/26935413），但降尿酸 RCT 未減緩 eGFR 下降（PMID 32579811）；"
                                  "且反向 MR 顯示 eGFR↑→尿酸↓（PMID 35856088）"),
    "代謝性|LBXSGTSI": ("observational", "GGT 對糖尿病腎病之病因專一性（前瞻世代，PMID 27537361）"),
    "代謝性|LBXSGL": ("adjacent", "標籤含 HbA1c／糖尿病診斷，血糖與其高度相關——敏感度分析已排除重跑"),
    "代謝性|LBXSOSSI": ("adjacent", "滲透壓由血糖／尿素／鈉決定，與標籤生物化學直接相連（無專屬文獻錨點）"),
    "代謝性|LBXBPB": ("causal_pos", "血鉛：一般族群 MR 無效，糖尿病層 IVW beta −0.034（p=0.013）（PMID 34661687）；"
                                     "工具變數 GWAS n=5,433 偏小，作者自陳待確認"),
    "代謝性|LBDVIDMS": ("causal_neg", "維生素D：VITAL-DKD RCT 補充 5 年對 eGFR 無效（PMID 31703120）；MR 亦無因果（PMID 39403589）"),

    # ── 感染性
    "感染性|LBXFER": ("observational", "慢性 B 肝 33.3%／C 肝 43.0% 有鐵蛋白或 TfSat 升高（PMID 22297603）——"
                                        "本資料候選發現 (a) 由「候選」升格為「已知重現」"),
    "感染性|LBXSIR": ("observational", "同上（血清鐵）"),
    "感染性|LBXSATSI": ("adjacent", "標籤＝B/C 肝血清學；ALT 為肝炎直接下游，非獨立發現。"
                                     "另反向 MR：eGFR↑→ALT↑（PMID 35856088）"),
    "感染性|LBXSASSI": ("adjacent", "同上（AST）"),
    "感染性|LBXSGTSI": ("adjacent", "同上（GGT）"),
    "感染性|LBXSGB": ("observational", "慢性病毒性肝炎之多株高球蛋白血症為既知現象；本資料 3.65 vs 3.10–3.20 與之一致，"
                                        "惟無「用於病因鑑別」之效能文獻（候選 c 部分支持）"),

    # ── 免疫性
    "免疫性|sex": ("observational", "自體免疫之女性優勢（本資料 69% 女性）"),
    "免疫性|NLR": ("conflict", "統合分析：NLR↑ 診斷狼瘡腎炎 AUC 0.81（PMID 39052098）；"
                                "本資料免疫組 NLR 反而較低（2.06 vs 2.24–2.28）——方向相反，佐證「ANA 陽性共存 ≠ SLE／狼瘡腎炎」"),
    "免疫性|LBXLYPCT": ("reverse", "反向 MR：eGFR↑→淋巴球%↑（PMID 35856088），部分為腎功能之後果"),
    "免疫性|LBXFER": ("candidate", "鐵狀態 MR 因果僅限 IgA 腎病（PMID 38999730），本資料免疫標籤為 ANA 陽性共存，不可直接套用"),
    "免疫性|LBXSGL": ("adjacent", "組成效應：其餘類以糖尿病為大宗，非免疫疾病降血糖"),
    "免疫性|LBXSTR": ("adjacent", "組成效應（同上）"),

    # ── 量測偏斜造成的假訊號（已解剖，不列為發現）
    "*|LBXTHG": ("artifact", "血汞：1999-2002 僅量特定子群（可得率 0.11/0.10，2003-04 才 0.97），"
                              "感染組僅少數人有值且 78% 為女性；魚類攝取／族裔與 HBV/HCV 盛行率共變——量測子集偏斜之混淆候選"),
    "*|LBXCOT": ("candidate", "可丁尼（菸暴露）：可得率 44%，缺值非隨機；菸暴露與 DN 之前瞻世代效果量極小，"
                               "本資料訊號宜先過『誰被量到』這關"),

    # ── 不分病因之通用警告
    "*|LBXSPH": ("causal_pos", "磷↑→eGFR↓、BUN↑（MR，PMID 39403589）；不專一於任一病因"),
    "*|LBXSCA": ("causal_pos", "鈣↑→UACR↑（MR，PMID 39403589）；限白蛋白尿，非過濾功能"),
    "*|LBXHCY": ("causal_neg", "HOST RCT：降 Hcy 25.8% 但死亡率 HR 1.04、至透析時間 P=.38（PMID 17848650）——結果標記非驅動因子"),
    "*|LBXPT21": ("candidate", "PTH 因遺傳率 z<4 被排除於 MR 之外（PMID 39403589）——無因果證據可用，不得宣稱"),
    "*|LBXHGB": ("reverse", "血色素為 eGFR 之非線性因果下游（PMID 35856088）"),
}


def lookup(cause, feature):
    """回傳 (level_key, level_zh, note)。先找病因專屬，再找通用，最後 candidate。"""
    for key in (f"{cause}|{feature}", f"*|{feature}"):
        if key in ANCHORS:
            lv, note = ANCHORS[key]
            return lv, LEVELS[lv], note
    return "candidate", LEVELS["candidate"], "本資料之發現，待文獻與前瞻確認"


def as_dict():
    """供離線 UI 匯出（前端不重寫一份）。"""
    return {k: dict(level=v[0], level_zh=LEVELS[v[0]], note=v[1]) for k, v in ANCHORS.items()}
