# -*- coding: utf-8 -*-
"""圖六：判別力 ≠ 決策效用——每 1000 名腎損傷者的換算（icon array）＋ 全驗 vs 模型作業點。
    python make_decision_figure.py [複製目的資料夾]

輸入只有三個數，其餘全部由程式推導：
  盛行率   = results/final_model.json → development.auprc_baseline（開發集，0.01885）
  敏感度／特異度 = 0.803／0.563：開發集最佳可行作業點，出處 docs/FINAL_MODEL_STATUS.md §3 表（ROC 懸崖前最後一點）
推導值（444／15.1／3.7／PPV 0.034／×1.8）與 FINAL_MODEL_STATUS.md §4 及計畫書伍之七一致，程式內以 assert 對照。
"""
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from make_figures import FIG, INF, INK, MET, SUB, WARN, _save, np, plt     # noqa: E402
from matplotlib.patches import Patch, Rectangle                             # noqa: E402

SENS, SPEC = 0.803, 0.563           # docs/FINAL_MODEL_STATUS.md §3（開發集 ROC 可行性表）
N = 1000
TP_C, FN_C, FP_C, TN_C = INF, "#7f0000", "#ffcc80", "#e0e0e0"


def main():
    F = json.load(open(os.path.join(ROOT, "results", "final_model.json"), encoding="utf-8"))
    prev = F["development"]["auprc_baseline"]
    pos = N * prev
    tp, fn = pos * SENS, pos * (1 - SENS)
    neg = N - pos
    fp, tn = neg * (1 - SPEC), neg * SPEC
    flagged, unflagged = tp + fp, fn + tn
    ppv, lift = tp / flagged, (tp / flagged) / prev
    assert round(pos, 1) == 18.9 and round(flagged) == 444 and round(tp, 1) == 15.1 and round(fn, 1) == 3.7, (pos, flagged, tp, fn)
    assert round(ppv, 3) == 0.034 and round(lift, 1) == 1.8, (ppv, lift)

    fig = plt.figure(figsize=(14, 7.4))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.45, 1], hspace=.9, wspace=.28, left=.05, right=.98, top=.82, bottom=.14)
    a = fig.add_subplot(gs[:, 0])
    bx = [fig.add_subplot(gs[i, 1]) for i in range(3)]

    # ── (a) icon array：40 欄 × 25 列 = 1000 人，先排「建議送驗」區塊再排「不建議」
    cols, rows = 40, 25
    n_tp, n_fn, n_fp = int(round(tp)), int(round(fn)), int(round(fp))
    n_flag = int(round(flagged))
    n_tn = N - n_tp - n_fn - n_fp
    assert n_tp + n_fp == n_flag and n_tp + n_fn + n_fp + n_tn == N
    kinds = [TP_C] * n_tp + [FP_C] * n_fp + [FN_C] * n_fn + [TN_C] * n_tn
    for k, colr in enumerate(kinds):
        r, c = divmod(k, cols)
        a.add_patch(Rectangle((c, rows - 1 - r), .82, .82, fc=colr, ec="white", lw=.4))
        if colr == FN_C:
            a.text(c + .41, rows - 1 - r + .41, "×", ha="center", va="center", fontsize=7, color="white", weight="bold")
    # 兩區塊的分界（第 n_flag 格之後）
    rb, cb = divmod(n_flag, cols)
    a.plot([0, cb, cb, cols], [rows - rb, rows - rb, rows - rb - 1, rows - rb - 1], color=INK, lw=1.6, zorder=5)
    a.set_xlim(-.3, cols + .3); a.set_ylim(-.5, rows + .3); a.set_aspect("equal"); a.axis("off")
    a.text(0, rows + .55, f"建議送驗 {flagged:.0f} 人（{flagged/N:.0%}）", fontsize=10.5, weight="bold", color=INK, va="bottom")
    a.text(cols, rows - rb - 1.5, f"不建議送驗 {unflagged:.0f} 人（{unflagged/N:.0%}）", fontsize=10.5, weight="bold",
           color=INK, ha="right", va="top", bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=.85))
    a.set_title(f"(a) 每 1000 名腎損傷者（真有 B／C 肝 {pos:.1f} 人）：模型於作業點的分派結果",
                fontsize=11, weight="bold", loc="left", pad=22)
    a.legend(handles=[Patch(fc=TP_C, label=f"抓到的真陽性 {tp:.1f} 人"),
                      Patch(fc=FN_C, label=f"漏掉的真陽性 {fn:.1f} 人（{fn/pos:.0%}，標 ×）"),
                      Patch(fc=FP_C, label=f"假陽性（白驗）{fp:.1f} 人"),
                      Patch(fc=TN_C, label=f"真陰性 {tn:.1f} 人")],
             loc="upper center", bbox_to_anchor=(.5, -.02), ncol=2, fontsize=9, frameon=False)

    # ── (b) 全驗 vs 模型作業點：三個指標
    strat = ["全驗\n（不用模型）", f"模型作業點\n（敏感度 {SENS:.3f}／特異度 {SPEC:.3f}）"]
    metrics = [("送驗人數", [N, flagged], SUB, "人", f"省 {1-flagged/N:.0%} 的檢驗"),
               ("抓到的病例", [pos, tp], INF, "人", f"每抓到 1 例需驗 {1/prev:.0f} → {1/ppv:.0f} 人（PPV {prev:.1%} → {ppv:.1%}，×{lift:.1f}）"),
               ("漏掉的病例", [0, fn], FN_C, "人", f"代價：漏掉 {fn/pos:.0%} 的病例")]
    for ax, (name, vals, colr, unit, note) in zip(bx, metrics):
        y = [1, 0]
        ax.barh(y, vals, color=colr, alpha=.75, height=.55)
        for yi, v in zip(y, vals):
            ax.text(v + max(vals) * .02, yi, f"{v:,.1f} {unit}" if v % 1 else f"{v:,.0f} {unit}", va="center", fontsize=9.6, color=INK, weight="bold")
        ax.set_yticks(y); ax.set_yticklabels(strat, fontsize=9)
        ax.set_xlim(0, max(vals) * 1.45); ax.set_xticks([])
        ax.set_title(f"{name}　—　{note}", fontsize=9.6, weight="bold", loc="left", color=colr if colr != SUB else INK)
        ax.spines[["top", "right", "bottom"]].set_visible(False)
    bx[0].text(0, 1.38, "(b) 兩種策略的比較", transform=bx[0].transAxes, fontsize=11, weight="bold", va="bottom")

    fig.suptitle(f"圖六　判別力 ≠ 決策效用——AUROC 0.815 的模型，在作業點上 PPV 僅 {ppv:.3f}（盛行率 ×{lift:.1f}）",
                 fontsize=12.5, weight="bold", y=.97)
    fig.text(.5, .03, "肝炎血清學便宜、無創、無風險：對這種檢驗，「全驗」比「省 56% 但漏 20%」更合理——"
                      "即使模型有判別力，在此情境下不改變決策；此即最終工具改採「方向」而非「決策」之理由",
             ha="center", fontsize=9.3, color=INF)
    fig.text(.5, .005, f"盛行率取開發集 {prev:.4f}（results/final_model.json）；作業點為開發集 ROC 懸崖前最後可行點（docs/FINAL_MODEL_STATUS.md §3）；其餘數字由此推導",
             ha="center", fontsize=8, color=SUB)
    _save(fig, "圖六_每千人決策換算.png")
    print("[完成] 圖六_每千人決策換算.png")
    print(f"  盛行率 {prev:.4f} → {pos:.1f}/1000；送驗 {flagged:.1f}；抓到 {tp:.1f}；漏 {fn:.1f}；PPV {ppv:.3f}（×{lift:.2f}）")
    if len(sys.argv) > 1:
        shutil.copy(os.path.join(FIG, "圖六_每千人決策換算.png"), os.path.join(sys.argv[1], "圖六_每千人決策換算.png"))
        print(f"→ 已複製至 {sys.argv[1]}")


if __name__ == "__main__":
    main()
