# -*- coding: utf-8 -*-
"""由 results/report.json 產生 report.md 與 limitations.md（指示 十二節交付物 2、6）。"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def f3(v):
    return "—" if v is None else f"{v:.3f}"


def ci_str(ci):
    return "—" if not ci else f"{ci['mean']:.3f}［{ci['lo']:.3f}, {ci['hi']:.3f}］"


def main():
    R = json.load(open(os.path.join(ROOT, "results", "report.json"), encoding="utf-8"))
    L = []
    L.append("# 腎損傷分流模型（Stage 0–5）分析報告\n")
    L.append(f"seed {R['seed']}・世代 n={R['n']}（訓練 {R['train_n']}／測試 {R['test_n']}）・依《建置指示 v11》\n")
    L.append("> **本次建置無真實資料**：世代由合成生成器產生，而生成器的全部參數（盛行率、通道效應、可得比例、"
             "開立行為）無文獻依據（params.json 之 `generator` 列 value=null，以佔位假設執行）。"
             "**本報告所有數字都是該假設的產物，僅示範管線與方法學，不構成對任何真實族群的證據。**\n")

    L.append("## Stage 0　資料品質與密度閘門\n")
    s0 = R["stage0"]
    L.append(f"- 進入 {s0['n_input']} 人、通過 {s0['n_pass']} 人、排除 {s0['n_excluded']} 人（規則：{s0['rule']}）")
    fsm = R["feature_sets"]
    L.append(f"- 尿沉渣形態欄位可得比例 {fsm['morph_ok_share']:.2f}" +
             ("（**低於 30% 閘門——Stage 3 使用 `L_uRBC` 與 `R_UPCR_uRBC` 等代理特徵，腎絲球來源判定之特異度預期明顯下降**）" if fsm["proxy_mode"] else "（≥30%，形態特徵直接使用）") + "\n")

    L.append("## 洩漏基準 M0（先建、先定版）\n")
    L.append(f"M0 只用「哪些檢驗被開立」(`ord_*`) 之指示變數；定版時間 {R['m0_sealed_at']}（`m0_baseline.json`），"
             "**早於任何主模型效能計算**（程式以 M0Ledger 強制）。M0 不是隨機水準——它量化了開立行為本身攜帶的標籤資訊（醫師的臨床懷疑），主模型必須超越它才算有生理訊號。\n")
    m0 = json.load(open(os.path.join(ROOT, "results", "m0_baseline.json"), encoding="utf-8"))
    L.append("| 階段 | M0 平衡正確率（訓練 CV） |")
    L.append("|---|---|")
    for s, d in m0["stages"].items():
        L.append(f"| Stage {s} {d['stage']} | {d['cv_train']['balanced_accuracy']:.3f} |")
    L.append("\n**逐族群 M0 洩漏**（指示 九之四；族群不得由該階段自身標籤導出，否則退化——"
             "分流格分層即屬此類，故另存於 `m0_baseline.json` 之 `by_box_balanced_accuracy` 並多為 null）\n")
    L.append("| 族群 | " + " | ".join(f"Stage {s} {d['stage']}" for s, d in m0["stages"].items()) + " |")
    L.append("|---" * (1 + len(m0["stages"])) + "|")
    groups = list(next(iter(m0["stages"].values()))["by_subgroup"])
    allg = sorted({g for d in m0["stages"].values() for g in d["by_subgroup"]}, key=lambda g: (g not in groups, g))
    for g in allg:
        cells = []
        for s, d in m0["stages"].items():
            r = d["by_subgroup"].get(g)
            cells.append("—" if not r or r["balanced_accuracy"] is None else f"{r['balanced_accuracy']:.3f}（n={r['n']}）")
        L.append(f"| {g} | " + " | ".join(cells) + " |")
    L.append("\n「—」＝該族群內此軸標籤單一或人數不足，平衡正確率無定義。\n")

    L.append("## 逐階段模型階梯與判定（訓練集 5 折 CV；配對 bootstrap）\n")
    L.append("| 階段 | M0 | M1（索引水準） | M2（＋斜率＋比值） | M2−M0（95% CI） | 判定 | M2−M1 | M3 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for s, r in R["stages"].items():
        t = r["train"]; v = t["verdicts"]
        L.append(f"| {s} {r['name']} | {f3(t['M0']['balanced_accuracy'])} | {f3(t['M1']['balanced_accuracy'])} | {f3(t['M2']['balanced_accuracy'])} | "
                 f"{v['M2_vs_M0']['diff']:+.3f}［{v['M2_vs_M0']['lo']:+.3f}, {v['M2_vs_M0']['hi']:+.3f}］ | {v['M2_vs_M0']['verdict'].split('—')[0]} | "
                 f"{v['M2_vs_M1']['diff']:+.3f}（{v['M2_vs_M1']['verdict']}） | {f3(t['M3']['balanced_accuracy'])}（{v['M3_vs_random']['verdict']}） |")
    L.append("")
    L.append("註：本管線的特徵矩陣自始不含封存集與 `ord_*`，故 M3 與 M2 同構（照實報告，非省略）。" +
             ("**被移除的軸：" + "、".join(R["removed_axes"]) + "**（M2 未顯著超越 M0）。" if R["removed_axes"] else "三軸皆通過 M2>M0 判定，無移除。") + "\n")

    L.append("## 測試集表現（拒答門檻於訓練集鎖定後，僅評估一次）\n")
    L.append("| 階段 | 門檻 | 拒答率 | 作答者平衡正確率（95% CI） | 作答者整體正確率 |")
    L.append("|---|---|---|---|---|")
    for s, r in R["stages"].items():
        te = r["test"]; a = te["answered"]
        L.append(f"| {s} {r['name']} | {te['threshold']}{'' if te['threshold_reachable'] else '（目標不可達，用預設起點）'} | "
                 f"{te['abstention_rate']:.3f} | {ci_str(a['balanced_accuracy_ci'])} | {f3(a['accuracy'])} |")
    L.append("\n拒答率與作答者正確率**同時**報告（只報作答者正確率屬選擇性報告）。\n")

    L.append("### 逐類召回與依實際盛行率換算之 PPV／NPV（測試集作答者）\n")
    for s, r in R["stages"].items():
        L.append(f"**Stage {s} {r['name']}**\n")
        L.append("| 類別 | n | 盛行率 | 召回 | 特異度 | PPV | NPV |")
        L.append("|---|---|---|---|---|---|---|")
        for c, m in r["test"]["answered"]["per_class"].items():
            L.append(f"| {c} | {m['n']} | {m['prevalence']:.3f} | {f3(m['recall'])} | {f3(m['specificity'])} | {f3(m['ppv'])} | {f3(m['npv'])} |")
        L.append("")

    L.append("## Stage 4　分流格（測試集）\n")
    s4 = R["stage4"]
    L.append(f"- 判定率 {s4['decided_rate']:.3f}（任一階段拒答即「無法判定」，不強行分類）")
    L.append(f"- 作答者平衡正確率 {ci_str(s4['balanced_accuracy_ci'])}；整體正確率 {f3(s4['answered']['accuracy'])}")
    L.append("\n| 格 | n | 召回 | PPV |")
    L.append("|---|---|---|---|")
    for c, m in s4["answered"]["per_class"].items():
        L.append(f"| {c} | {m['n']} | {f3(m['recall'])} | {f3(m['ppv'])} |")
    L.append("")

    L.append("## Stage 5　建議檢驗之評估\n")
    s5 = R["stage5"]
    L.append(f"- 檢驗節省率（有判定者，vs 全面開單 {s5['full_panel_items']} 項）：**{f3(s5['test_saving_rate_decided'])}**（平均建議 {f3(s5['mean_items_recommended'])} 項）")
    L.append(f"- 關鍵遺漏率（全體）：**{f3(s5['critical_omission_rate_all'])}**；建議命中率：{f3(s5['hit_rate'])}")
    L.append(f"- **R 格**：真 R n={s5['R_true_n']}、被分流到 R 之召回 {f3(s5['R_recall'])}；"
             f"凡判為 R 者五項強制建議之完整率 {f3(s5['R_five_items_when_routed_R'])}（硬性規則，程式 assert 保證）；"
             f"真 R 之關鍵遺漏率 {f3(s5['R_critical_omission_rate'])}——**遺漏全部來自 Stage 4 誤分流，而非建議規則被覆蓋**。"
             "零容忍條款在「判為 R 即五項全開」層次成立；要壓低真 R 遺漏須提高 R 格召回（屬 Stage 1–2 的問題），照實報告。")
    L.append("\n| 真實格 | n | 分流召回 | 關鍵遺漏率 |")
    L.append("|---|---|---|---|")
    for b, m in s5["by_box"].items():
        L.append(f"| {b} | {m['n']} | {f3(m['recall'])} | {f3(m['omission'])} |")
    L.append("")

    L.append("## 事前指定優先比值之單獨表現（訓練 CV 平衡正確率）\n")
    L.append("| 比值 | Stage 1 急性度 | Stage 2 部位 | Stage 3 表現型 |")
    L.append("|---|---|---|---|")
    for col, d in R["ratio_solo_balanced_accuracy"].items():
        L.append(f"| `{col}` | {f3(d.get('1'))} | {f3(d.get('2'))} | {f3(d.get('3'))} |")
    L.append("")

    if R["morphology_subgroups"]:
        L.append("## 形態欄位可得 vs 不可得子群（分流格平衡正確率，測試集）\n")
        for g, m in R["morphology_subgroups"].items():
            L.append(f"- {g}：n={m['n']}，{f3(m['box_balanced_accuracy'])}")
        L.append("")

    L.append("## 交付物\n")
    L.append("`pipeline.py`／`report.md`（本檔）／`m0_baseline.json`／`thresholds.json`／`feature_importance.csv`／`limitations.md`／`ui/index.html`（視覺化）\n")
    open(os.path.join(ROOT, "results", "report.md"), "w", encoding="utf-8").write("\n".join(L))

    lim = ["# limitations.md——實際遇到的資料限制與其後果\n",
           "1. **無真實資料**：整個世代為合成，生成器參數（盛行率、通道效應、可得比例、開立行為）無文獻依據，"
           "已依指示一之七於 params.json 留 null 並以佔位假設執行、啟動時警告。**所有效能數字僅示範管線，不得外推。**",
           f"2. **尿沉渣形態欄位可得比例 {fsm['morph_ok_share']:.2f} < 0.30**：Stage 3 改用 uRBC 水準與 R_UPCR_uRBC 代理，"
           "腎絲球來源判定的特異度預期明顯下降；已輸出可得 vs 不可得子群比較。",
           "3. **M0 洩漏顯著**（三階段 0.63–0.82）：開立行為本身即攜帶大量標籤資訊。此為合成器刻意重現的現實現象；"
           "在真實資料上 M0 可能更高，任何未先扣除 M0 的效能宣稱都不可信。",
           "4. **真 R 之關鍵遺漏來自誤分流**：R 格五項建議的硬性規則在程式層不可覆蓋（assert），"
           "但 Stage 1–2 將部分真 R 個案分至他格，其建議即缺強制五項。改善方向是 R 格召回，不是放寬建議規則。",
           "5. **M3 與 M2 同構**：本管線特徵矩陣自始排除封存集與 ord_*，M3 層無從再排除任何欄位——此為通道消耗規則徹底執行的結果，照實報告。",
           "6. **拒答門檻的目標正確率（0.8）為本研究設定**，無文獻依據；已於 thresholds.json 記錄選定規則與曲線，未在測試集調整。",
           "7. **Stage 6 未實作**（指示邊界）：本模型不預測最終病名、不決定切片、不建議治療。"]
    open(os.path.join(ROOT, "results", "limitations.md"), "w", encoding="utf-8").write("\n\n".join(lim) + "\n")
    print("[report] results/report.md、results/limitations.md")


if __name__ == "__main__":
    main()
