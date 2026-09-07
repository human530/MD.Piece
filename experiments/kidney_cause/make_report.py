# -*- coding: utf-8 -*-
"""由 results/*.json 產生 report.md（誠實界線、證據等級歸因表、免疫標籤修正、全景擴充測試）。

歸因對照表的等級與說明來自 src/anchor_map.py（單一真相來源，離線 UI 共用同一份）。"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from anchor_map import lookup as anchor_lookup   # noqa: E402

CLS = ("免疫性", "感染性", "代謝性")


def f3(v):
    return "—" if v is None else f"{v:.3f}"


def load(name):
    p = os.path.join(ROOT, "results", name)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def main():
    R = load("report.json")
    prox = load("nephritis_proxy_repeated_eval.json")
    lab = load("lab_report.json")
    hold = load("holdout_eval.json")
    L = ["# 腎病病因階層模型（免疫／感染／代謝）——NHANES 1999–2004 真實資料分析報告\n",
         f"seed {R['seed']}・{R['date']}（歸因表更新於 2026-08-30）\n",
         "## 誠實界線（先讀這段）\n"]
    L += [f"- {h}" for h in R["honesty"]]
    L.append("- **免疫標籤效度警語**：『免疫性』＝ANA 強陽性（或特異抗體＋）∩ 腎損傷之共存，一般族群 ANA 陽性率約一成，"
             "其中多數與腎病無因果關係——此標籤是三類中最弱的，效能與歸因都要在這個前提下讀。\n")

    # ── 免疫代理標籤的修正（最重要的一段，放在最前面）
    if prox:
        hgb = prox["evaluations"]["HGB"]["patient_pooled"]
        lr = prox["evaluations"]["LR"]["patient_pooled"]
        co, pr_ = prox["cohort"], prox["protocol"]
        L.append("## ⚠ 免疫代理標籤的修正（推翻本檔 Level 1 的免疫欄位）\n")
        L.append(f"下方 Level 1 的三類模型把「**未做 ANA 者**」一併當成非免疫，屬標籤污染——"
                 f"NHANES 只有 surplus sera 隨機次樣本做過 ANA，其餘人的免疫狀態是**未知**而非陰性。")
        L.append(f"另行以「只納入實際做過 ANA 且有腎損傷者」重建二元代理（n={co['n']}，陽性 {co['positive']}、陰性 {co['negative']}），"
                 f"以 {pr_['repeats']}×{pr_['outer_folds']} 重複巢狀交叉驗證（門檻只在內層選）評估：\n")
        L.append("| 模型 | AUROC（95% CI，病人層 bootstrap） | AUPRC | 平衡正確率 |")
        L.append("|---|---|---|---|")
        for mk, m in (("LR", lr), ("HGB", hgb)):
            ci = m["auroc_ci95_patient_bootstrap"]
            L.append(f"| {mk} | {m['auroc']:.3f}（{ci[0]:.3f}–{ci[1]:.3f}） | {m['auprc']:.3f} | {m['balanced_accuracy']:.3f} |")
        L.append("")
        L.append(f"**結論：修正標籤污染後，免疫代理的判別力只有 AUROC {hgb['auroc']:.3f}——接近隨機。**"
                 "下方 Level 1 表中的「免疫性」欄位（0.81）是污染造成的樂觀值，**不應引用**。"
                 "感染性／代謝性標籤不受此影響（全體皆有肝炎血清學與糖尿病問卷）。")
        art = prox.get("artifact", {})
        if art:
            L.append(f"研究用模型：`{art.get('path')}`（SHA256 `{art.get('sha256','')[:32]}…`）——ANA／特異抗體陽性之代理，"
                     "非活檢確診腎炎模型，不得用於臨床診斷。\n")

    C = R["cohort"]
    L.append("## 世代（全部真實、可逐檔驗證）\n")
    L.append(f"- 三週期合併 31,126 人 → 成人 15,332 → 腎損傷 {C['n_kidney_damage']}（eGFR<60 或 ACR≥30）")
    L.append(f"- 標籤分布（全腎損傷）：{C['secondary_all']}；重疊 {C['overlap']}（歸類順位 免疫>感染>代謝）")
    L.append(f"- SSANA 次樣本 ∩ 腎損傷 n={C['n_primary']}；特徵 {R['features']['n']} 欄、封存 {C['archive_n']} 欄（標籤來源不得為特徵）")
    L.append(f"- 出處帳本：{R['provenance']['n_files']} 檔（URL＋SHA256）見 results/provenance.json\n")

    L.append("## Level 1　三大病因（一對其餘 AUROC，重複分層 5 折 OOF，bootstrap 95% CI）\n")
    L.append("| 分析 | 模型 | 免疫性 ⚠ | 感染性 | 代謝性 | 平衡正確率 |")
    L.append("|---|---|---|---|---|---|")

    def row(tag, block):
        for mk in ("LR", "HGB"):
            m = block["models"][mk]
            cells = []
            for c in CLS:
                a = m["ovr_auc"][c]
                cells.append(f"{f3(a['auc'])} [{f3(a['lo'])},{f3(a['hi'])}] (n={a['n_pos']})" if a["auc"] is not None else "—")
            L.append(f"| {tag} | {mk} | {cells[0]} | {cells[1]} | {cells[2]} | {m['balanced_accuracy']:.3f} |")

    cc = R["level1"].get("class_counts", {})
    row(f"主分析（ANA 實測子樣本 n={R['level1'].get('n')}）", R["level1"])
    row("敏感度：排除血清葡萄糖", R["sensitivity"]["排除血清葡萄糖"])
    row("敏感度：排除重疊個案", R["sensitivity"]["排除重疊個案"])
    L.append("")
    L.append(f"**樣本量與取捨（必讀）**：主分析已改為只用「實際做過 ANA」的子樣本（各類 n：{cc}），"
             "這樣免疫標籤才乾淨；代價是**感染類只剩極少數個案**，其 AUROC 的信賴區間極寬、數字不穩定，不應單獨引用。")
    L.append("若改用全體腎損傷者（n=782；感染 32、代謝 659），感染／代謝樣本較大，但免疫欄位會回到被污染的樂觀值——"
             "**兩種切法各有無法同時解決的缺陷，這正是 NHANES 做三類病因判別的天花板**："
             "唯一有乾淨免疫標籤的子樣本裡幾乎沒有感染個案。\n")
    main_auc = [R["level1"]["models"]["HGB"]["ovr_auc"][c]["auc"] for c in CLS]
    L.append("**對 AUC≥0.9 目標的照實回答**：主分析（HGB）" + "／".join(f3(v) for v in main_auc) + "——**未達 0.9**；"
             "且免疫欄位在修正標籤後掉到 0.584。到 0.9 的正路是更強的標籤（切片病因、臨床診斷碼）與更多感染／免疫樣本，"
             "不是在這份資料上調參。\n")
    if hold:
        L.append(f"**鎖定保留集（一生一次，已於 {hold['evaluated_once_at']} 使用，n={hold['holdout_n']}）**："
                 + "、".join(f"{c} {f3(hold['ovr_auc'][c]['auc'])}"
                             f"[{f3(hold['ovr_auc'][c]['lo'])},{f3(hold['ovr_auc'][c]['hi'])}]"
                             f"(n={hold['ovr_auc'][c]['n_pos']})" for c in CLS)
                 + f"；平衡正確率 {f3(hold['balanced_accuracy'])}。此檔存在後不得再評估。\n")
    cyc = R["sensitivity"].get("留一週期外測", {})
    if cyc:
        L.append("**留一週期外測（時間外推）**：")
        for c, d in sorted(cyc.items()):
            cells = "；".join(f"{k} {f3(v['auc'])} [{f3(v['lo'])},{f3(v['hi'])}]" for k, v in d.items() if v["auc"] is not None)
            L.append(f"- 留出 {c}：{cells}")
        L.append("")

    # ── 全景擴充與疊代測試
    if lab:
        L.append("## 腎臟相關指數全景擴充與疊代測試（開發集，保留集未觸碰）\n")
        L.append(f"- 開發集 n={lab['dev_n']}（{lab['dev_counts']}）；保留集 n={lab['holdout_n']} 分層鎖定")
        L.append("- ⚠ 本節疊代測試跑在**全體腎損傷者**（含未做 ANA 者）之上，其免疫欄位仍帶標籤污染的樂觀性；"
                 "此處要看的是「加特徵群有沒有增量」，不是絕對數字。")
        L.append("\n| 加入的特徵群 | 特徵數 | 免疫性 | 感染性 | 代謝性 | 三類平均 |")
        L.append("|---|---|---|---|---|---|")
        for r in lab["incremental"]:
            L.append(f"| {r['step']} | {r['n_features']} | " +
                     " | ".join(f3(r["ovr_auc"][c]["auc"]) for c in CLS) + f" | **{f3(r['mean_auc'])}** |")
        drops = "；".join(f"−{d['dropped']} → {f3(d['mean_auc'])}" for d in lab["group_drop"])
        L.append(f"\n- 群組移除：{drops}——只有拿掉 base 會崩，其餘每群 ≤0.003")
        L.append("- 五種子穩定性：" + "、".join(f"{c} {f3(v['mean'])}±{f3(v['sd'])}" for c, v in lab["seed_stability"].items()))
        L.append("\n**結論：把「所有腎臟相關指數」（脂質盤／重金屬／營養素／骨礦／Cystatin C，18 個新標記）加進來，"
                 "對病因判別的增量幾乎為零**——常規生化＋CBC 已攜帶大部分可判別訊號。"
                 "這些標記的價值在歸因與機轉解讀，不在推高 AUC。\n")
        for n_ in lab.get("artifact_notes", []):
            L.append(f"> ⚠ **{n_['marker']}**：{n_['claim']} → **{n_['verdict']}**。{n_['dissection']}\n")

    # ── 重新設定的二元任務（問題設定調整，非調參）
    bt = load("binary_tasks.json")
    if bt:
        L.append("## 重新設定的二元任務（2026-08-30 調整）\n")
        L.append(bt["rationale"])
        L.append(f"\n{bt['no_holdout']}\n")
        L.append("每個任務跑三個特徵集：**全部特徵**／**拔除標籤鄰近**（誠實主結果）／**腎臟專屬指標組**"
                 "（eGFR・ACR・肌酸酐・BUN・Cystatin C・尿酸・白蛋白／球蛋白・CKD-MBD 磷鈣PTH維生素D・"
                 "腎性貧血與鐵代謝・電解質酸鹼・發炎血球）。\n")
        L.append("| 任務 | n（陽性） | 特徵集 | 模型 | AUROC（95% CI） | AUPRC（基準／提升） | 平衡正確率 |")
        L.append("|---|---|---|---|---|---|---|")
        vzh = {"all": "全部", "leak_free": "**拔除標籤鄰近**", "kidney_core": "腎臟專屬"}
        for tname, t in bt["tasks"].items():
            if "skipped" in t:
                L.append(f"| {tname} | — | — | — | — | — | 跳過：{t['skipped']} |")
                continue
            first = True
            for vname in ("all", "leak_free", "kidney_core"):
                for mk in ("LR", "HGB"):
                    r = t["variants"][vname][mk]
                    head = f"{tname} | {r['n']}（{r['n_pos']}）" if first else " | "
                    first = False
                    L.append(f"| {head} | {vzh[vname]} | {mk} | {r['auroc']:.3f}"
                             f" [{r['auroc_ci'][0]:.3f},{r['auroc_ci'][1]:.3f}] | "
                             f"{r['auprc']:.3f}（{r['auprc_baseline']:.3f}／×{r['auprc_lift']:.1f}） | "
                             f"{r['balanced_accuracy']:.3f} |")
        L.append("")
        for tname, t in bt["tasks"].items():
            if "skipped" in t:
                continue
            h = t["headline"]
            adj = t.get("label_adjacent_removed") or []
            L.append(f"- **{tname}**：主結果（拔除標籤鄰近 {adj if adj else '（無）'}，{h['model']}）"
                     f" AUROC **{h['auroc']:.3f}**（{h['auroc_ci'][0]:.3f}–{h['auroc_ci'][1]:.3f}）、"
                     f"AUPRC {h['auprc']:.3f}（隨機基準 {h['auprc_baseline']:.3f}）。{t['note']}"
                     f" 陽性個案之 KDIGO 風險組成：{t.get('kdigo_positive_mix', {})}")
        L.append("\n**讀法**：極不平衡任務（T3 盛行率 1.6%）看 AUROC 會過度樂觀，應看 AUPRC 相對其基準線的提升倍數；"
                 "「全部特徵」與「拔除標籤鄰近」的落差即為標籤洩漏的量化值。\n")

    # ── Level 3 歸因（帶證據等級）
    L.append("## Level 3　biomarker 歸因（關聯，非因果）\n")
    L.append("證據等級語彙：**因果級（支持）**＝MR／RCT 陽性；**因果級（反對）**＝MR 陰性或介入 RCT 陰性（關聯在、因果不在）；"
             "**僅觀察性**；**反向因果警告**＝腎功能→標記而非標記→病因；**標籤鄰近**＝與標籤定義的生物學直接相連；"
             "**與文獻方向相反**；**候選**＝本資料之發現待確認。\n")
    L.append("錨點逐條經 PubMed 摘要原文比對（deep-research 的查證代理兩度撞用量上限，改由主迴圈人工核對），"
             "全表見 `docs/verified_anchors_2026-08-30.json`。\n")
    for c in CLS:
        L.append(f"**{c}**（單變量 AUC 前十，方向＝該類相對其餘）\n")
        L.append("| 標記 | 方向 | AUC | 95% CI | 證據等級 | 說明 |")
        L.append("|---|---|---|---|---|---|")
        for r in R["level3"]["univariate_top"][c][:10]:
            lv, lv_zh, note = anchor_lookup(c, r["feature"])
            L.append(f"| {r['label']} | {r['direction']} | {r['auc']:.3f} | [{f3(r['lo'])},{f3(r['hi'])}] | **{lv_zh}** | {note} |")
        L.append("")
    L.append("補充讀法：\n")
    L.append("1. **組成效應**：免疫性的「血糖低／TG 低」是因為其餘類以糖尿病為大宗，不是免疫疾病讓血糖變低。")
    L.append("2. **反向因果**：反向 MR（PMID 35856088）顯示 eGFR↑→淋巴球%↑／HDL↑／ALT↑，eGFR↑→尿酸↓／TG↓——"
             "本表若干組間差異有一部分是腎功能本身的後果，而非病因專一訊號。")
    L.append("3. **本資料三個候選發現的查證結果**：(a) 感染組鐵蛋白偏高＝**已知重現**（慢性 B 肝 33%／C 肝 43% 有鐵指標升高，"
             "PMID 22297603）；(b) 滲透壓×糖尿病腎病＝**標籤鄰近**（滲透壓由血糖決定），非獨立發現；"
             "(c) 球蛋白區分感染性＝生物學一致（多株高球蛋白血症）但查無鑑別效能文獻，維持候選。")
    L.append("4. 置換重要度與 LR 係數完整表在 report.json。\n")

    L.append("## Level 2　類內細項\n")
    li = R["level2_immune"]
    L.append(f"- 免疫（n={li['n']}）：{li['subgroups']}——{li['caveat']}")
    lf = R["level2_infection"]
    L.append(f"- 感染（n={lf['n']}）：HBV {lf['HBV']}、HCV {lf['HCV']}（標籤層資訊，僅供分流示意）\n")

    L.append("## 超音波佐證層\n")
    L.append("公開世界無上萬張可下載腎臟超音波；`us_validation/` 已定義交換格式與一致性分析（κ、逐類混淆、分歧解剖），"
             "醫院影像到位即可與 results/predictions.csv 以個案鍵併接執行；到位前該層拒跑、不產生任何假數字。\n")
    open(os.path.join(ROOT, "results", "report.md"), "w", encoding="utf-8").write("\n".join(L))
    print("[report] results/report.md")


if __name__ == "__main__":
    main()
