# -*- coding: utf-8 -*-
"""圖：A 世代流程｜B Level1 AUROC 森林圖（含 0.9 目標線）｜C 各類 biomarker 歸因｜D 混淆矩陣。
    python figures.py"""
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
                     "axes.unicode_minus": False, "font.size": 9})
FIGS = os.path.join(ROOT, "results", "figs")
CLS = ["免疫性", "感染性", "代謝性"]
G = ["0.15", "0.45", "0.72"]


def save(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    for ext, kw in (("png", dict(dpi=300)), ("pdf", {})):
        fig.savefig(os.path.join(FIGS, f"{name}.{ext}"), bbox_inches="tight", **kw)
    plt.close(fig)
    print(f"[figs] {name}")


def figA(R):
    C = R["cohort"]
    steps = [("NHANES 1999–2004 合併", 31126), ("成人 ≥20", 15332), ("腎損傷 eGFR<60 或 ACR≥30", C["n_kidney_damage"]),
             ("有病因標籤（三類）", sum(v for k, v in C["secondary_all"].items() if k != "其他/未歸類"))]
    fig, ax = plt.subplots(figsize=(7.4, 3.0))
    y = np.arange(len(steps))[::-1]
    vals = [s[1] for s in steps]
    ax.barh(y, vals, height=0.55, color=G[2], edgecolor="0.1")
    for yi, (lab, v) in zip(y, steps):
        ax.text(v * 1.02, yi, f"{v:,}", va="center", fontsize=9)
        ax.text(30, yi, lab, va="center", fontsize=9)
    lab3 = "、".join(f"{k} {v}" for k, v in C["secondary_all"].items() if k != "其他/未歸類")
    ax.set_xscale("log"); ax.set_yticks([]); ax.set_xlabel("人數（對數尺度）")
    ax.set_title(f"世代流程（真實資料）：{lab3}；其他/未歸類 {C['secondary_all'].get('其他/未歸類', 0)} 照實列出", fontsize=9)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    save(fig, "figA_cohort_flow")


def figB(R):
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    blocks = [("主分析 LR", R["level1"]["models"]["LR"]), ("主分析 HGB", R["level1"]["models"]["HGB"]),
              ("排除重疊 HGB", R["sensitivity"]["排除重疊個案"]["models"]["HGB"]),
              ("排除血糖 HGB", R["sensitivity"]["排除血清葡萄糖"]["models"]["HGB"])]
    yy = 0; ylabels = []
    for bi, (tag, m) in enumerate(blocks):
        for ci, c in enumerate(CLS):
            a = m["ovr_auc"][c]
            if a["auc"] is None:
                continue
            ax.plot([a["lo"], a["hi"]], [yy, yy], color=G[ci % 3], lw=2)
            ax.plot([a["auc"]], [yy], "o", ms=6, color=G[ci % 3], mec="0.1")
            ylabels.append((yy, f"{tag}｜{c} (n={a['n_pos']})"))
            yy += 1
        yy += 0.6
    ax.axvline(0.9, color="0.1", lw=1.1, ls=(0, (5, 3)))
    ax.text(0.902, yy - 1, "設計目標 0.9", fontsize=8, rotation=90, va="top")
    ax.axvline(0.5, color="0.75", lw=0.8)
    ax.set_yticks([p for p, _ in ylabels]); ax.set_yticklabels([l for _, l in ylabels], fontsize=7.5)
    ax.set_xlabel("一對其餘 AUROC（點＝實測，線＝bootstrap 95% CI）"); ax.set_xlim(0.45, 1.0)
    ax.invert_yaxis()
    ax.set_title("主分析未達 0.9（照實）；排除重疊為樂觀敏感度、非主結果", fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    save(fig, "figB_auroc_forest")


def figC(R):
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
    for ax, c in zip(axes, CLS):
        rows = R["level3"]["univariate_top"][c][:8][::-1]
        y = np.arange(len(rows))
        v = [r["auc"] for r in rows]
        col = ["white" if r["direction"] == "低" else G[1] for r in rows]
        ax.barh(y, v, height=0.6, color=col, edgecolor="0.1", hatch=["//" if r["direction"] == "低" else None for r in rows])
        ax.set_yticks(y); ax.set_yticklabels([f"{r['label']}（{r['direction']}）" for r in rows], fontsize=8)
        ax.set_xlim(0.5, 0.85); ax.set_xlabel("單變量 AUC")
        ax.set_title(c, fontsize=10)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("各病因的驅動標記（灰=該類較高，斜線=該類較低；關聯非因果，組成效應見報告）", fontsize=9, y=1.03)
    fig.tight_layout()
    save(fig, "figC_biomarkers")


def figD(R):
    m = R["level1"]["models"]["HGB"]
    cm = np.array(m["confusion"], float)
    row = cm.sum(axis=1, keepdims=True)
    frac = cm / np.where(row == 0, 1, row)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(frac, cmap="Greys", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=10,
                    color="white" if frac[i, j] > 0.55 else "0.1")
    ax.set_xticks(range(3)); ax.set_xticklabels(CLS); ax.set_yticks(range(3)); ax.set_yticklabels(CLS)
    ax.set_xlabel("預測"); ax.set_ylabel("真實（代理標籤）")
    ax.set_title(f"HGB OOF 混淆（平衡正確率 {m['balanced_accuracy']:.3f}）", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8, label="列內比例")
    save(fig, "figD_confusion")


if __name__ == "__main__":
    R = json.load(open(os.path.join(ROOT, "results", "report.json"), encoding="utf-8"))
    figA(R); figB(R); figC(R); figD(R)
    print("[figs] 完成（標籤為共病代理；關聯非因果）")
