# -*- coding: utf-8 -*-
"""科展計畫書四張圖（兩軸版：感染／代謝）——全部由 results/*.json 與 params/*.json 生成，數字不手打。
    python make_figures.py [複製目的資料夾]

圖一　世代建立流程（兩軸標籤、通道消耗規則、每軸特徵數）
圖二　感染／代謝兩軸之判別能力（AUROC 三→十週期、AUPRC 提升、留一週期外測、方向判定三段分布）
圖三　反向因果的直接證據（同一元素血中 vs 尿中；護腎藥 vs 禁忌藥方向相反）
圖四　被攔截的方法學陷阱（兩軸可量化者；六項全覽）

2026-09-07：免疫軸自最終模型移除後重製；圖中不再呈現免疫軸。
"""
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import matplotlib                                   # noqa: E402
matplotlib.use("Agg")
import numpy as np                                  # noqa: E402
import matplotlib.pyplot as plt                     # noqa: E402
from matplotlib.lines import Line2D                 # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch   # noqa: E402

# 中文字型（Windows 內建）——找不到就退回預設並警告。JhengHei 缺 ≥／−，圖中一律用 ≧ 與「降」。
for fam in ("Microsoft JhengHei", "Microsoft YaHei", "SimHei", "PMingLiU"):
    try:
        matplotlib.font_manager.findfont(fam, fallback_to_default=False)
        plt.rcParams["font.family"] = fam
        break
    except Exception:
        continue
else:
    print("⚠️ 找不到中文字型，圖中中文可能顯示為方框")
plt.rcParams["axes.unicode_minus"] = False
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)
INK, SUB = "#1a1a1a", "#5a5a5a"
INF, INF2, MET = "#c62828", "#ad1457", "#2e7d32"      # 感染軸紅系、代謝軸綠
NEU, WARN, BAD = "#1565c0", "#ef6c00", "#c62828"
NAME = {"LBXSATSI": "ALT", "LBXSASSI": "AST", "LBXSGTSI": "GGT", "LBXSTB": "總膽紅素",
        "LBXSGL": "血糖", "LBXSOSSI": "滲透壓"}
TASKS = (("T2_", "感染 vs 代謝", INF2), ("T3_", "感染 vs 其餘", INF), ("T4_", "代謝 vs 其餘", MET))


def _load(name):
    return json.load(open(os.path.join(ROOT, "results", name), encoding="utf-8"))


def _param(name):
    return json.load(open(os.path.join(ROOT, "params", name), encoding="utf-8"))


def _task(bundle, prefix):
    return next(v for k, v in bundle["tasks"].items() if k.startswith(prefix))


def _save(fig, name):
    fig.savefig(os.path.join(FIG, name), dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig1_cohort():
    """圖一：世代建立流程（兩軸）。"""
    c = _load("exwas.json")["cohort"]
    B = _load("binary_tasks_extended.json")
    D = _param("direction_model.json")
    n_adult, n_known, n_kd = c["n_adults"], c["n_outcome_known"], c["n_kidney_damage"]
    n_inf, n_met = B["cohort"]["infection"], B["cohort"]["metabolic"]
    adj_inf = [NAME.get(v, v) for v in _task(B, "T3_")["label_adjacent_removed"]]
    adj_met = [NAME.get(v, v) for v in _task(B, "T4_")["label_adjacent_removed"]]
    f_inf, f_met = len(D["axes"]["感染"]["features"]), len(D["axes"]["代謝"]["features"])

    fig, ax = plt.subplots(figsize=(8.4, 8.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10.6); ax.axis("off")
    boxes = [
        (5.0, 9.7, 6.6, "NHANES 1999–2018　十個週期、261 個公開檔（290 MB）", "#eceff1"),
        (5.0, 8.3, 5.4, f"成人（≧20 歲）\nn = {n_adult:,}", "#e3f2fd"),
        (5.0, 6.9, 5.4, f"腎功能結果可判定\nn = {n_known:,}", "#e3f2fd"),
        (5.0, 5.5, 5.8, f"腎損傷（eGFR<60 或 ACR≧30）\nn = {n_kd:,}　（{n_kd/n_known:.1%}）", "#fff8e1"),
    ]
    for x, y, w, txt, col in boxes:
        ax.add_patch(FancyBboxPatch((x - w / 2, y - .42), w, .84, boxstyle="round,pad=0.06",
                                    fc=col, ec=SUB, lw=1.1))
        ax.text(x, y, txt, ha="center", va="center", fontsize=10.5, color=INK, linespacing=1.5)
    for y0, y1 in ((9.28, 8.74), (7.88, 7.34), (6.48, 5.94)):
        ax.add_patch(FancyArrowPatch((5, y0), (5, y1), arrowstyle="-|>", mutation_scale=14,
                                     lw=1.2, color=SUB))
    ax.text(8.3, 7.6, f"排除結果無法判定\n{n_adult - n_known:,} 人", ha="center", va="center",
            fontsize=9, color=SUB, style="italic")

    lab = [(3.1, "感染性", n_inf, "#ffcdd2", INF, "HBsAg＋ 或 HCV RNA＋", f"{n_inf/n_kd:.1%}"),
           (6.9, "代謝性", n_met, "#c8e6c9", MET, "醫師診斷糖尿病 或 HbA1c≧6.5%", f"{n_met/n_kd:.1%}")]
    for x, name, n, col, ec, sub, pct in lab:
        ax.add_patch(FancyBboxPatch((x - 1.5, 3.55), 3.0, 1.05, boxstyle="round,pad=0.06",
                                    fc=col, ec=ec, lw=1.4))
        ax.text(x, 4.25, name, ha="center", fontsize=11.5, color=INK, weight="bold")
        ax.text(x, 3.85, f"n = {n:,}（盛行率 {pct}）", ha="center", fontsize=10.2, color=INK)
        ax.text(x, 3.22, sub, ha="center", fontsize=8.4, color=SUB)
        ax.add_patch(FancyArrowPatch((5, 5.06), (x, 4.62), arrowstyle="-|>",
                                     mutation_scale=12, lw=1.1, color=SUB,
                                     connectionstyle="arc3,rad=0.12" if x < 5 else "arc3,rad=-0.12"))
    ax.text(5, 2.86, "兩個標籤各自獨立判定（可同時成立），不合併為互斥分類",
            ha="center", fontsize=8.8, color=SUB, style="italic")

    ax.add_patch(FancyBboxPatch((0.7, 1.05), 8.6, 1.5, boxstyle="round,pad=0.08",
                                fc="#fafafa", ec=BAD, lw=1.4, ls="--"))
    ax.text(5, 2.24, "通道消耗規則：定義標籤的檢驗一律封存（79 項），永不作為特徵",
            ha="center", fontsize=10, color=BAD, weight="bold")
    ax.text(5, 1.78, f"可用特徵 {B['n_features']} 項，再拔除各軸「標籤鄰近」項：",
            ha="center", fontsize=9.4, color=INK)
    ax.text(5, 1.36, f"感染軸 {f_inf} 項（拔除 {'・'.join(adj_inf)}）　│　"
                     f"代謝軸 {f_met} 項（拔除 {'・'.join(adj_met)}）",
            ha="center", fontsize=9.4, color=INK)
    ax.text(5, .45, "圖一　世代建立流程（感染／代謝兩軸）", ha="center", fontsize=12.5,
            weight="bold", color=INK)
    _save(fig, "圖一_世代建立流程.png")


def fig2_discrimination():
    """圖二：兩軸判別能力——AUROC（三→十週期）、AUPRC 提升、留一週期外測、方向判定三段分布。"""
    B, S, D = _load("binary_tasks_extended.json"), _load("binary_tasks.json"), _param("direction_model.json")
    rows = []
    for pre, name, col in TASKS:
        r10 = _task(B, pre)["variants"]["leak_free"]["HGB"]
        r3 = _task(S, pre)["variants"]["leak_free"]["HGB"]
        rows.append(dict(name=name, col=col, auc=r10["auroc"], ci=r10["auroc_ci"], auc3=r3["auroc"],
                         lift=r10["auprc_lift"], prev=r10["prevalence"], n=r10["n"], n_pos=r10["n_pos"],
                         loco=_task(B, pre)["leave_one_cycle_out"]))

    fig, ((a1, a2), (a3, a4)) = plt.subplots(2, 2, figsize=(13.2, 8.9),
                                             gridspec_kw={"width_ratios": [1.35, 1], "height_ratios": [1, .95]})
    fig.subplots_adjust(wspace=.32, hspace=.62)
    y = list(range(len(rows)))[::-1]

    # (a) AUROC：十週期實心＋CI；三週期空心，箭頭指向十週期（小樣本樂觀被擴充樣本修正）
    for yi, r in zip(y, rows):
        a1.plot(r["ci"], [yi, yi], color=r["col"], lw=2.8, solid_capstyle="round", alpha=.75)
        a1.plot(r["auc"], yi, "o", color=r["col"], ms=9.5, zorder=4)
        a1.text(r["ci"][1] + .012, yi, f"{r['auc']:.3f}", va="center", fontsize=10, color=r["col"], weight="bold")
        a1.plot(r["auc3"], yi + .33, "o", mfc="white", mec=r["col"], ms=7.5, mew=1.6, zorder=4)
        a1.annotate("", xy=(r["auc"], yi + .1), xytext=(r["auc3"], yi + .3),
                    arrowprops=dict(arrowstyle="->", color=r["col"], lw=1, alpha=.6))
        a1.text(r["auc3"], yi + .5, f"三週期 {r['auc3']:.3f}", ha="center", va="bottom",
                fontsize=8.2, color=SUB)
    a1.axvline(.5, color=SUB, ls=":", lw=1.2)
    a1.text(.5, .03, "隨機猜測", ha="center", va="bottom", fontsize=8.5, color=SUB, transform=a1.get_xaxis_transform())
    a1.axvline(.9, color=WARN, ls="--", lw=1.3)
    a1.text(.9, .03, "原始目標 0.90", ha="center", va="bottom", fontsize=8.5, color=WARN, transform=a1.get_xaxis_transform())
    a1.set_yticks(y)
    a1.set_yticklabels([f"{r['name']}\n（n={r['n']:,}，陽性 {r['n_pos']:,}）" for r in rows], fontsize=9.8)
    a1.set_xlim(.44, .98); a1.set_ylim(-.6, len(rows) - .1)
    a1.set_xlabel("AUROC（十週期，梯度提升，leak_free；95% 信賴區間）", fontsize=10)
    a1.set_title("(a) 判別能力：三週期估計（空心）經十週期重測（實心）後下修", fontsize=10.8, weight="bold", loc="left")
    a1.grid(axis="x", alpha=.25); a1.spines[["top", "right"]].set_visible(False)

    # (b) AUPRC 提升倍數
    a2.barh(y, [r["lift"] for r in rows], color=[r["col"] for r in rows], alpha=.8, height=.55)
    for yi, r in zip(y, rows):
        a2.text(r["lift"] + .12, yi, f"×{r['lift']:.1f}", va="center", fontsize=10, color=r["col"], weight="bold")
    a2.axvline(1, color=SUB, ls=":", lw=1.4)
    a2.text(1, .97, "無提升", ha="center", va="top", fontsize=8.5, color=SUB, transform=a2.get_xaxis_transform())
    a2.set_yticks(y); a2.set_yticklabels([f"{r['name']}\n（盛行率 {r['prev']:.1%}）" for r in rows], fontsize=9.8)
    a2.set_xlim(0, max(r["lift"] for r in rows) * 1.25); a2.set_ylim(-.6, len(rows) - .1)
    a2.set_xlabel("AUPRC 相對盛行率基準之提升倍數（跨門檻平均）", fontsize=10)
    a2.set_title("(b) 相對盛行率之提升", fontsize=10.8, weight="bold", loc="left")
    a2.grid(axis="x", alpha=.25); a2.spines[["top", "right"]].set_visible(False)

    # (c) 留一週期外測
    cycles = sorted(rows[0]["loco"].keys())
    xs = np.arange(len(cycles))
    for r in rows:
        v = [r["loco"][c]["auroc"] for c in cycles]
        med = float(np.median(v))
        a3.plot(xs, v, "-o", color=r["col"], lw=1.6, ms=5, alpha=.85, label=f"{r['name']}（中位 {med:.3f}，虛線）")
        a3.axhline(med, color=r["col"], ls=":", lw=1)
    a3.set_xticks(xs); a3.set_xticklabels([f"{c[2:4]}–{c[7:9]}" for c in cycles], fontsize=8.5)
    a3.set_xlim(-.4, len(cycles) - .6)
    a3.set_ylim(.6, .92); a3.set_ylabel("留出該週期時之 AUROC", fontsize=9.5)
    a3.set_xlabel("留出的 NHANES 週期（以其餘九週期訓練）", fontsize=9.5)
    t3 = next(r for r in rows if r["name"] == "感染 vs 其餘")
    npos = [t3["loco"][c]["n_pos"] for c in cycles]
    a3.set_title(f"(c) 時間外推：留一週期外測（感染任務單週期陽性僅 {min(npos)}–{max(npos)} 例）",
                 fontsize=10.8, weight="bold", loc="left")
    a3.legend(fontsize=8.2, loc="lower left", ncol=1, frameon=False)
    a3.grid(alpha=.25); a3.spines[["top", "right"]].set_visible(False)

    # (d) 方向判定工具：三段分布
    bands = ("傾向", "不確定", "不傾向")
    axes_ = (("感染", INF), ("代謝", MET))
    for yi, (nm, col) in zip((1, 0), axes_):
        a = D["axes"][nm]; bd = a["band_distribution"]; tot = sum(bd.values())
        left = 0
        for b, fc, al in zip(bands, (col, "#9e9e9e", col), (.9, .3, .38)):
            v = bd[b] / tot
            a4.barh(yi, v, left=left, color=fc, alpha=al, height=.55, edgecolor="white")
            if v >= .1:
                a4.text(left + v / 2, yi, f"{b}\n{v:.0%}", ha="center", va="center", fontsize=8.6, color=INK)
            left += v
        a4.text(1.02, yi, f"OOF AUROC {a['oof_auroc']:.3f}\n傾向／不確定／不傾向＝{bd['傾向']:,}／{bd['不確定']:,}／{bd['不傾向']:,}\n"
                          f"作答者敏感度 {a['answered_sensitivity']:.2f}／特異度 {a['answered_specificity']:.2f}",
                va="center", fontsize=8.2, color=SUB, transform=a4.get_yaxis_transform())
    a4.set_yticks([1, 0]); a4.set_yticklabels([f"{nm}軸\n（盛行率 {D['axes'][nm]['prevalence']:.1%}）" for nm, _ in axes_], fontsize=9.8)
    a4.set_xlim(0, 1); a4.set_ylim(-.6, 1.6)
    a4.set_xticks([0, .25, .5, .75, 1]); a4.set_xticklabels(["0", "25%", "50%", "75%", "100%"], fontsize=8.5)
    a4.set_xlabel(f"腎損傷者 n={D['n']:,} 之三段判定分布（傾向 ≧2× 盛行率；不傾向 ≦0.5× 盛行率）", fontsize=9.2)
    a4.set_title("(d) 方向判定工具（5 折交叉配適邏輯迴歸＋保序校準）", fontsize=10.8, weight="bold", loc="left")
    a4.spines[["top", "right"]].set_visible(False)

    fig.suptitle("圖二　感染／代謝兩軸之判別能力與方向判定工具", fontsize=13, weight="bold", y=.985)
    _save(fig, "圖二_三大病因判別能力.png")


def fig3_reverse():
    """圖三：反向因果的直接證據——血中 vs 尿中；顯著尿液暴露方向；護腎藥 vs 禁忌藥。"""
    X = _load("exwas.json")
    R = {r["exposure"]: r for r in X["results"]}
    pairs = [("鉛", "LBXBPB", "URXUPB"), ("鎘", "LBXBCD", "URXUCD")]
    uri = [r for r in X["results"] if r["significant_fdr05"] and r["exposure"].startswith("URX")]
    n_prot = sum(1 for r in uri if r["headline"]["or_per_sd"] < 1)

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15.6, 4.6), gridspec_kw={"width_ratios": [1.2, .8, 1.25]})
    fig.subplots_adjust(wspace=.55)

    def forest(ax, items):
        ypos, labels = [], []
        for yv, lab, key, col in items:
            r = R[key]; h = r["headline"]
            ax.plot(h["ci"], [yv, yv], color=col, lw=2.6, alpha=.75, solid_capstyle="round")
            ax.plot(h["or_per_sd"], yv, "o", color=col, ms=9, zorder=3)
            star = "＊" if r["significant_fdr05"] else ""
            ax.text(h["ci"][1] + .012, yv, f"{h['or_per_sd']:.3f}{star}", va="center", fontsize=10, color=col, weight="bold")
            ypos.append(yv); labels.append(lab)
        ax.axvline(1, color=SUB, ls=":", lw=1.4)
        ax.text(1, .985, "無關聯", ha="center", va="top", fontsize=8.5, color=SUB, transform=ax.get_xaxis_transform())
        ax.set_yticks(ypos); ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel("勝算比（每增加 1 標準差；主調整層）", fontsize=9.8)
        ax.grid(axis="x", alpha=.25); ax.spines[["top", "right"]].set_visible(False)

    items = []
    for i, (zh, b, u) in enumerate(pairs):
        a1.axhspan(i * 2.6 - .55, i * 2.6 + 1.55, color="#000000", alpha=.035, zorder=0)
        items += [(i * 2.6 + 1, f"{zh}－血中", b, BAD), (i * 2.6, f"{zh}－尿中", u, NEU)]
    forest(a1, items)
    a1.set_title("(a) 同一元素，血中「有害」而尿中「保護」", fontsize=11, weight="bold", loc="left")
    a1.text(.5, -.26, "＊ 通過偽發現率校正", transform=a1.transAxes, ha="center", fontsize=8.5, color=SUB)

    a2.bar(["看似保護\n(OR<1)", "看似有害\n(OR>1)"], [n_prot, len(uri) - n_prot], color=[NEU, BAD], alpha=.8, width=.5)
    for i, v in enumerate([n_prot, len(uri) - n_prot]):
        a2.text(i, v + .18, str(v), ha="center", fontsize=13, weight="bold", color=INK)
    a2.set_ylabel("顯著的尿液暴露個數", fontsize=10)
    a2.set_ylim(0, max(n_prot, 1) * 1.35)
    a2.set_title("(b) 顯著尿液暴露之方向", fontsize=11, weight="bold", loc="left")
    a2.grid(axis="y", alpha=.25); a2.spines[["top", "right"]].set_visible(False)
    a2.text(.5, -.30, "無毒理學說法可解釋多種金屬同時保護腎臟；\n排泄生理學可完全解釋（腎功能↓→尿中濃度↓）",
            transform=a2.transAxes, ha="center", fontsize=8.6, color=BAD)

    forest(a3, [(2, "ACEI／ARB（護腎藥）", "藥_ACEI_ARB", BAD),
                (1, "雙胍（腎功能差即停用）", "藥_雙胍", NEU),
                (0, "NSAID（教科書腎毒物）", "藥_NSAID", SUB)])
    a3.set_ylim(-.6, 2.6)
    a3.set_title("(c) 處方常規決定方向，非藥物毒性", fontsize=11, weight="bold", loc="left")
    a3.text(.5, -.26, "護腎藥因腎病而開始、雙胍因腎病而停止；成藥 NSAID 不在處方資料中",
            transform=a3.transAxes, ha="center", fontsize=8.6, color=BAD)

    fig.suptitle("圖三　反向因果的直接證據——關聯方向由腎排泄能力與處方常規決定，非毒性作用",
                 fontsize=12.5, weight="bold", y=1.07)
    _save(fig, "圖三_反向因果證據.png")


def fig4_traps():
    """圖四：被攔截的方法學陷阱——(a) 兩軸上可量化者（虛報 vs 實際）；(b) 六項全覽。"""
    B, S = _load("binary_tasks_extended.json"), _load("binary_tasks.json")
    t4, t3, t2 = _task(B, "T4_"), _task(B, "T3_"), _task(B, "T2_")
    traps = [
        ("代謝：血糖未拔除\n（標籤下游洩漏）", t4["variants"]["all"]["HGB"]["auroc"], t4["variants"]["leak_free"]["HGB"]["auroc"], MET),
        ("感染：肝酶未拔除\n（標籤下游洩漏）", t3["variants"]["all"]["HGB"]["auroc"], t3["variants"]["leak_free"]["HGB"]["auroc"], INF),
        ("感染：小樣本樂觀\n（三週期→十週期）", _task(S, "T2_")["variants"]["leak_free"]["HGB"]["auroc"], t2["variants"]["leak_free"]["HGB"]["auroc"], INF2),
    ]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.6, 4.8), gridspec_kw={"width_ratios": [1, 1.15]})
    fig.subplots_adjust(wspace=.25)
    x = np.arange(len(traps)); w = .34
    for i, (nm, hi, lo, col) in enumerate(traps):
        a1.bar(i - w / 2, hi, w, color=col, alpha=.35, hatch="//", edgecolor=col)
        a1.bar(i + w / 2, lo, w, color=col, alpha=.9)
        a1.text(i - w / 2, hi + .012, f"{hi:.3f}", ha="center", fontsize=9.5, color=col, weight="bold")
        a1.text(i + w / 2, lo + .012, f"{lo:.3f}", ha="center", fontsize=9.5, color=col, weight="bold")
        a1.text(i, max(hi, lo) + .045, f"降 {hi-lo:.3f}", ha="center", fontsize=9.5, color=SUB, style="italic", weight="bold")
    a1.axhline(.9, color=WARN, ls="--", lw=1.3)
    a1.text(2.95, .906, "原始目標\n0.90", fontsize=8.5, color=WARN, ha="right", va="bottom")
    a1.set_xticks(x); a1.set_xticklabels([t[0] for t in traps], fontsize=9.2)
    a1.set_xlim(-.6, 3.0)
    a1.set_ylabel("AUROC", fontsize=10.5); a1.set_ylim(.5, 1.02)
    a1.legend(handles=[Patch(fc="#bdbdbd", alpha=.6, hatch="//", ec="#757575", label="若未攔截（虛報值）"),
                       Patch(fc="#616161", label="實際值")],
              fontsize=9, loc="upper center", bbox_to_anchor=(.5, -.2), ncol=2, frameon=False)
    a1.set_title("(a) 兩軸上可量化的三項——每次修正都使結果變差", fontsize=10.8, weight="bold", loc="left", pad=12)
    a1.grid(axis="y", alpha=.25); a1.spines[["top", "right"]].set_visible(False)

    a2.axis("off")
    rows = [("① 未受檢者當作陰性", "標籤虛報 0.812（實 0.584）", "標籤稽核"),
            ("② 標籤下游未拔除", f"代謝虛報 {traps[0][1]:.3f}（實 {traps[0][2]:.3f}）", "洩漏量化"),
            ("③ 變數別名漏列", "整週期肌酸酐靜默丟棄", "缺值型態檢查"),
            ("④ 預設值吞噬缺值", "最重分期虛報 734 例（實 40）", "分布合理性檢查"),
            ("⑤ 二元變數被誤砍", "陽性對照未被掃描", "★ 陽性對照"),
            ("⑥ 只驗顯著不驗方向", "保護方向誤報為毒性", "方向一致性檢查")]
    a2.text(.02, 1.0, "(b) 六項陷阱與其攔截機制", fontsize=10.8, weight="bold", transform=a2.transAxes, va="top")
    for i, (name, harm, guard) in enumerate(rows):
        yy = .845 - i * .137
        hl = i == 4
        a2.add_patch(FancyBboxPatch((.02, yy - .052), .96, .105, boxstyle="round,pad=0.012",
                                    fc="#fff3e0" if hl else "#fafafa", ec=BAD if hl else "#dddddd",
                                    lw=1.5 if hl else .9, transform=a2.transAxes, clip_on=False))
        a2.text(.05, yy, name, fontsize=9.6, transform=a2.transAxes, va="center",
                weight="bold" if hl else "normal", color=INK)
        a2.text(.40, yy, harm, fontsize=8.8, transform=a2.transAxes, va="center", color=BAD)
        a2.text(.77, yy, guard, fontsize=9, transform=a2.transAxes, va="center",
                color=BAD if hl else MET, weight="bold" if hl else "normal")
    a2.text(.5, .012, "★ 第 ⑤ 項偽裝成乾淨的陰性結論；攔截它的不是研究者的謹慎，而是事前設置的陽性對照",
            fontsize=8.8, transform=a2.transAxes, ha="center", color=BAD, style="italic")
    fig.suptitle("圖四　被攔截的方法學陷阱——每一次修正都使結果變差", fontsize=12.5, weight="bold", y=1.05)
    _save(fig, "圖四_方法學陷阱.png")


if __name__ == "__main__":
    fig1_cohort(); print("[完成] 圖一_世代建立流程.png")
    fig2_discrimination(); print("[完成] 圖二_三大病因判別能力.png")
    fig3_reverse(); print("[完成] 圖三_反向因果證據.png")
    fig4_traps(); print("[完成] 圖四_方法學陷阱.png")
    print(f"→ {FIG}")
    if len(sys.argv) > 1:
        dst = sys.argv[1]
        assert os.path.isdir(dst), dst
        for f in os.listdir(FIG):
            if f.endswith(".png"):
                shutil.copy(os.path.join(FIG, f), os.path.join(dst, f))
        print(f"→ 已複製至 {dst}")
