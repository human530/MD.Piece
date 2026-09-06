# -*- coding: utf-8 -*-
"""圖五：方向判定工具對保留集一名真實病人（B／C 肝陽性）的實際輸出，畫成圖。
    python make_demo_figure.py [複製目的資料夾]

數字全部來自 direction.predict(params/direction_demo_patient.json)——與 direction.py demo 的文字輸出同一病人、同一參數。
"""
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from make_figures import FIG, INF, INK, MET, SUB, _save, np, plt          # noqa: E402（沿用字型與配色）
from matplotlib.patches import FancyBboxPatch                              # noqa: E402
from direction import predict                                              # noqa: E402

GREY, CAP = "#78909c", 4.4      # 推離的顏色；橫軸上限（盛行率倍數；direction.render 的文字長條以 ×4 封頂，此處留邊讓 ×4.0 的標記不被切掉）


def main():
    vals = json.load(open(os.path.join(ROOT, "params", "direction_demo_patient.json"), encoding="utf-8"))
    res = predict(vals)
    exp = json.load(open(os.path.join(ROOT, "params", "direction_demo_expected.json"), encoding="utf-8"))
    for ax_name, e in exp.items():          # 與已存的參考答案一致才畫（回歸檢查）
        a = res["axes"][ax_name]
        assert abs(a["probability"] - e["cal"]) < 1e-5 and a["band"] == e["band"] and a["n_missing"] == e["miss"], (ax_name, a)

    axes = (("感染", INF), ("代謝", MET))
    fig, grid = plt.subplots(2, 2, figsize=(13, 7.2), gridspec_kw={"width_ratios": [1.25, 1], "height_ratios": [1, 1]})
    fig.subplots_adjust(hspace=.75, wspace=.38)

    for row, (name, col) in enumerate(axes):
        a = res["axes"][name]
        prev, p, mult, band = a["prevalence"], a["probability"], a["times_prevalence"], a["band"]
        g, d = grid[row]

        # ── 左：盛行率倍數量尺，三段區間＋此病人的位置
        for x0, x1, fc, al, lab in ((0, .5, col, .35, "不傾向"), (.5, 2, "#9e9e9e", .3, "不確定"), (2, CAP, col, .9, "傾向")):
            g.axvspan(x0, x1, color=fc, alpha=al, lw=0)
            g.text((x0 + x1) / 2, .86, lab, ha="center", va="center", fontsize=9.5,
                   color="white" if lab == "傾向" else INK, weight="bold")
        xm = min(mult, CAP - .15)
        g.plot([xm, xm], [0, .62], color=INK, lw=2.4, solid_capstyle="round", zorder=5)
        g.plot(xm, .62, "v", color=INK, ms=12, zorder=6)
        g.annotate(f"校準機率 {p:.4f}\n＝盛行率 {prev:.1%} 的 ×{mult:.1f}",
                   xy=(xm, .3), xytext=(xm - .15 if mult > 2 else xm + .15, .30),
                   ha="right" if mult > 2 else "left", va="center", fontsize=9.8, color=INK,
                   bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=col, lw=1.2))
        g.set_xlim(0, CAP); g.set_ylim(0, 1)
        g.set_yticks([])
        g.set_xticks([0, .5, 1, 2, 3, 4])
        g.set_xticklabels(["0", f"×0.5\n({.5*prev:.1%})", f"×1\n({prev:.1%})", f"×2\n({2*prev:.1%})", "×3", "×4"], fontsize=8.6)
        g.set_xlabel("校準機率／該軸盛行率（括號＝對應機率）；門檻事前指定：傾向 ≧×2、不傾向 ≦×0.5", fontsize=8.8)
        g.spines[["top", "right", "left"]].set_visible(False)
        g.set_title(f"{name}方向  →  判定「{band}」", fontsize=12, weight="bold", loc="left", color=col)

        # ── 右：主要依據（前四項推動因子，Σ 各折模型之 z×係數）
        drv = a["drivers"]
        w = np.array([x["weight"] for x in drv])
        y = np.arange(len(drv))[::-1]
        d.barh(y, w, color=[col if v > 0 else GREY for v in w], alpha=.85, height=.55)
        d.axvline(0, color=INK, lw=1)
        for yi, x, v in zip(y, drv, w):
            side = 1 if v > 0 else -1
            d.text(v + side * .02 * max(abs(w)), yi, x["push"], va="center", ha="left" if v > 0 else "right",
                   fontsize=9, color=col if v > 0 else GREY, weight="bold")
        d.set_yticks(y); d.set_yticklabels([f"{x['name']} = {x['value']:.3g}" for x in drv], fontsize=9.8)
        lim = max(abs(w)) * 1.6
        d.set_xlim(-lim, lim)
        d.set_xlabel("對此軸邏輯值的貢獻（標準化值 × 係數，5 折平均）", fontsize=8.8)
        d.set_title(f"主要依據（{name}軸前四項）", fontsize=11, weight="bold", loc="left")
        d.grid(axis="x", alpha=.25); d.spines[["top", "right"]].set_visible(False)
        d.text(1, 1.03, f"缺 {a['n_missing']}/{a['n_features']} 項以訓練集中位數補入", transform=d.transAxes,
               ha="right", va="bottom", fontsize=8.4, color=SUB, style="italic")

    fig.suptitle("圖五　方向判定工具之實際輸出——鎖定保留集一名 B／C 肝陽性真實病人（輸入 45 項常規檢驗值）",
                 fontsize=12.5, weight="bold", y=.995)
    fig.text(.5, .005, "輸出為「大概方向」而非診斷：兩軸各自獨立，皆附可追溯的推動因子；離線網頁 direction.html 對同一病人輸出至小數第六位一致",
             ha="center", fontsize=9, color=SUB)
    _save(fig, "圖五_方向判定示範.png")
    print("[完成] 圖五_方向判定示範.png")
    print("  " + " | ".join(f"{n} {res['axes'][n]['probability']:.4f} ×{res['axes'][n]['times_prevalence']:.1f} {res['axes'][n]['band']}" for n, _ in axes))
    if len(sys.argv) > 1:
        shutil.copy(os.path.join(FIG, "圖五_方向判定示範.png"), os.path.join(sys.argv[1], "圖五_方向判定示範.png"))
        print(f"→ 已複製至 {sys.argv[1]}")


if __name__ == "__main__":
    main()
