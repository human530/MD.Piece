# -*- coding: utf-8 -*-
"""圖表：由 results/*.json 產生五張圖（PNG 300 dpi ＋ PDF）。
    python figures.py
圖 A 模型階梯與洩漏基準｜圖 B 拒答曲線｜圖 C 分流格混淆矩陣｜圖 D Stage 5 建議評估｜圖 E M0 逐族群洩漏"""
import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

plt.rcParams.update({"font.sans-serif": ["Noto Sans CJK TC", "Noto Sans TC", "Microsoft JhengHei", "DejaVu Sans"],
                     "axes.unicode_minus": False, "font.size": 9, "axes.linewidth": 0.8})
G = ["0.0", "0.35", "0.62", "0.85"]
FIGS = os.path.join(ROOT, "results", "figs")
STG = {"1": "急性度", "2": "部位", "3": "表現型"}


def save(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    for ext, kw in (("png", dict(dpi=300)), ("pdf", {})):
        fig.savefig(os.path.join(FIGS, f"{name}.{ext}"), bbox_inches="tight", **kw)
    plt.close(fig)
    print(f"[figs] {name}.png / .pdf")


def load():
    res = os.path.join(ROOT, "results")
    return (json.load(open(os.path.join(res, "report.json"), encoding="utf-8")),
            json.load(open(os.path.join(res, "m0_baseline.json"), encoding="utf-8")),
            json.load(open(os.path.join(res, "thresholds.json"), encoding="utf-8")))


def figA(R):
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    stages = list(R["stages"]); layers = ["M0", "M1", "M2", "M3"]
    w = 0.19
    for j, L in enumerate(layers):
        vals = [R["stages"][s]["train"][L]["balanced_accuracy"] for s in stages]
        x = np.arange(len(stages)) + (j - 1.5) * w
        ax.bar(x, vals, width=w * 0.92, color=("white" if L == "M0" else G[min(j, 3)]), edgecolor="0.0", lw=0.9,
               hatch=("///" if L == "M0" else None), label=L)
        for xi, v in zip(x, vals):
            ax.text(xi, v + 0.008, f"{v:.2f}", ha="center", fontsize=7)
    ax.axhline(0.5, color="0.4", lw=0.8, ls=(0, (4, 3)))
    ax.text(-0.45, 0.508, "二類隨機水準", fontsize=7, color="0.35", ha="left")
    labels = []
    for st in stages:
        v = R["stages"][st]["train"]["verdicts"]["M2_vs_M0"]
        labels.append(f"Stage {st} {STG[st]}\nM2−M0 {v['diff']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}]　{v['verdict'].split('——')[0]}")
    ax.set_xticks(range(len(stages))); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("平衡正確率（訓練集 5 折 CV）"); ax.set_ylim(0.4, 1.0)
    ax.legend(ncol=4, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.14))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    save(fig, "figA_ladder")


def figB(R, TH):
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    for i, (s, curve) in enumerate(TH["curves"].items()):
        t = [r["threshold"] for r in curve]
        ba = [r["balanced_accuracy"] for r in curve]
        cov = [r["coverage"] for r in curve]
        ax.plot(t, ba, "-", color=G[i], lw=1.6, label=f"Stage {s} {STG[s]}：作答者平衡正確率")
        ax.plot(t, cov, "--", color=G[i], lw=1.0, label=f"Stage {s} {STG[s]}：作答率")
        th = TH["thresholds"][s]["threshold"]
        ax.axvline(th, color=G[i], lw=0.8, ls=(0, (1, 2)))
        ax.plot([th], [R["stages"][s]["test"]["answered"]["balanced_accuracy"]], "o", ms=6, mfc="white", mec=G[i], mew=1.4)
    ax.set_xlabel("信心門檻（訓練集選定後鎖定；圓點＝測試集實測）"); ax.set_ylabel("比例")
    ax.set_ylim(0.4, 1.02)
    ax.legend(fontsize=7, frameon=False, ncol=2, loc="lower left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    save(fig, "figB_abstention")


def figC(R):
    cm = np.array(R["stage4"]["answered"]["confusion"], float); lab = R["stage4"]["answered"]["labels"]
    row = cm.sum(axis=1, keepdims=True); frac = np.divide(cm, np.where(row == 0, 1, row))
    fig, ax = plt.subplots(figsize=(5.0, 4.4))
    im = ax.imshow(frac, cmap="Greys", vmin=0, vmax=1)
    for i in range(len(lab)):
        for j in range(len(lab)):
            if cm[i, j]:
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8,
                        color="white" if frac[i, j] > 0.55 else "0.1")
    ax.set_xticks(range(len(lab))); ax.set_xticklabels(lab); ax.set_yticks(range(len(lab))); ax.set_yticklabels(lab)
    ax.set_xlabel("預測分流格"); ax.set_ylabel("真實分流格")
    ax.set_title(f"判定率 {R['stage4']['decided_rate']:.2f}｜作答者平衡正確率 {R['stage4']['answered']['balanced_accuracy']:.3f}", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8, label="列內比例")
    save(fig, "figC_box_confusion")


def figD(R):
    s5 = R["stage5"]; by = s5["by_box"]
    boxes = list(by)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.2), gridspec_kw=dict(width_ratios=[1.25, 1]))
    x = np.arange(len(boxes))
    a1.bar(x - 0.2, [by[b]["recall"] or 0 for b in boxes], width=0.38, color=G[2], edgecolor="0.0", lw=0.8, label="分流召回")
    a1.bar(x + 0.2, [by[b]["omission"] or 0 for b in boxes], width=0.38, color="white", edgecolor="0.0", lw=0.8, hatch="///", label="關鍵遺漏率")
    for i, b in enumerate(boxes):
        a1.text(i, 1.03, f"n={by[b]['n']}", ha="center", fontsize=7, color="0.35")
    a1.set_xticks(x); a1.set_xticklabels(boxes); a1.set_ylim(0, 1.22); a1.set_ylabel("比例")
    a1.legend(fontsize=8, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    a1.set_xlabel("真實分流格（A2／E 之強制建議為空，故遺漏率 0）")
    ks = [("檢驗\n節省率", s5["test_saving_rate_decided"]), ("關鍵\n遺漏率", s5["critical_omission_rate_all"]),
          ("建議\n命中率", s5["hit_rate"]), ("判為 R 者\n五項完整", s5["R_five_items_when_routed_R"])]
    a2.bar(range(len(ks)), [k[1] or 0 for k in ks], width=0.6, color=[G[1], "white", G[2], G[3]],
           edgecolor="0.0", lw=0.8, hatch=[None, "///", None, None])
    for i, k in enumerate(ks):
        a2.text(i, (k[1] or 0) + 0.02, f"{k[1]:.2f}" if k[1] is not None else "—", ha="center", fontsize=8)
    a2.set_xticks(range(len(ks))); a2.set_xticklabels([k[0] for k in ks], fontsize=8); a2.set_ylim(0, 1.15)
    a2.set_title("Stage 5 整體", fontsize=9)
    for ax in (a1, a2):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    save(fig, "figD_stage5")


def figE(M0):
    stages = list(M0["stages"])
    groups = [g for g in M0["stages"][stages[0]]["by_subgroup"]
              if any(M0["stages"][s]["by_subgroup"].get(g, {}).get("balanced_accuracy") is not None for s in stages)]
    fig, ax = plt.subplots(figsize=(8.4, 3.2))
    w = 0.26
    for j, s in enumerate(stages):
        d = M0["stages"][s]["by_subgroup"]
        vals = [(d.get(g) or {}).get("balanced_accuracy") or np.nan for g in groups]
        x = np.arange(len(groups)) + (j - 1) * w
        ax.bar(x, vals, width=w * 0.9, color=G[j], edgecolor="0.0", lw=0.8,
               label=f"Stage {s} {STG[s]}（整體 {M0['stages'][s]['cv_train']['balanced_accuracy']:.2f}）")
        for xi, v in zip(x, vals):
            if np.isfinite(v):
                ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=6.5)
    ax.axhline(0.5, color="0.4", lw=0.8, ls=(0, (4, 3)))
    ax.set_xticks(range(len(groups))); ax.set_xticklabels(groups, fontsize=8, rotation=12, ha="right")
    ax.set_ylabel("M0 平衡正確率"); ax.set_ylim(0.4, 1.05)
    ax.set_title("只用「哪些檢驗被開立」就能達到的水準——洩漏下限，逐族群（空缺＝該族群內標籤單一，無定義）", fontsize=9)
    ax.legend(fontsize=7.5, frameon=False, ncol=3, loc="lower left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    save(fig, "figE_m0_by_subgroup")


if __name__ == "__main__":
    R, M0, TH = load()
    figA(R); figB(R, TH); figC(R); figD(R); figE(M0)
    print("[figs] 完成 → results/figs/（生成器參數無文獻依據，數字不得對外引用）")
