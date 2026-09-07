# -*- coding: utf-8 -*-
"""視覺化（依 docs/視覺化說明_v1.md 四層）：
  圖B 位能地景（機制層，無資料）      figB_landscape
  圖A 機制展示廊（資料層，6 欄×3 列）  figA_gallery
  圖C 警報操作特性（效能層）          figC_operating
  圖D 可辨識地圖（邊界層）            figD_identifiability——需 results/grid.json，缺檔則跳過標「待補」
繪圖規則：灰階四階、類別以線型區分；Noto Sans CJK 前置 DejaVu 後備；圖內不放標題（圖說在文件）；
代表個案＝發作日最接近該機制中位數者（不得挑最漂亮）；縱軸不放大；PNG 300 dpi＋PDF。
資料：校準 seed 族鎖閾 → 測試 seed 族一個資料集（與正式流程同規則；n 可由 --n 縮小）。
    python figures.py [--seed 20260818] [--n 1500] [--only A|B|C|D]"""
import argparse
import json
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402

import m0_params as M0                    # noqa: E402
from m0_params import value               # noqa: E402
from m1_generator import simulate_cohort, observe, MECHS, lower_state  # noqa: E402
from m5_ews import score_cohort, lock_threshold, first_alarm, rolling_indicators  # noqa: E402
from seeding import module_rng            # noqa: E402
from run import ews_cfg, _stop_days       # noqa: E402

plt.rcParams.update({
    "font.sans-serif": ["Noto Sans CJK TC", "Noto Sans TC", "Microsoft JhengHei", "DejaVu Sans"],
    "axes.unicode_minus": False, "font.size": 9, "axes.linewidth": 0.8,
})
G4 = ["0.0", "0.35", "0.6", "0.82"]                                   # 灰階四階
MECH_ORDER = ("bifurcation", "stochastic_escape", "continuous_deterioration", "exogenous_shock", "noise_amplification", "stable")
MECH_ZH = {"bifurcation": "分岔翻轉", "stochastic_escape": "隨機越障", "continuous_deterioration": "連續惡化",
           "exogenous_shock": "外生衝擊", "noise_amplification": "雜訊放大", "stable": "穩定未發作"}
FIGS = os.path.join(ROOT, "results", "figs")


def save(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    for ext, kw in (("png", dict(dpi=300)), ("pdf", {})):
        fig.savefig(os.path.join(FIGS, f"{name}.{ext}"), bbox_inches="tight", **kw)
    plt.close(fig)
    print(f"[figs] {name}.png / .pdf")


# ------------------------------------------------------------------ 圖B 位能地景
def figB(P):
    mu_c = float(value(P, "mu_c"))
    stages = [("全劑量", -0.9), ("減藥中", -0.3), ("接近臨界", 0.30), ("跨過臨界", 0.55)]   # μ = μ_i − g(t) 示意
    xs = np.linspace(-1.8, 1.8, 400)
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.6), sharey=True)
    for ax, (lab, mu) in zip(axes, stages):
        V = xs ** 4 / 4 - xs ** 2 / 2 - mu * xs
        ax.plot(xs, V, color=G4[0], lw=1.4)
        # 球：緩解側谷底（存在時），否則活動側谷底
        r = np.roots([1.0, 0.0, -1.0, -mu]); r = np.sort(r[np.abs(r.imag) < 1e-9].real)
        ball = r[0] if (mu < mu_c and len(r) == 3) else r[-1]
        ax.plot([ball], [ball ** 4 / 4 - ball ** 2 / 2 - mu * ball], "o", ms=9, color=G4[1], mec=G4[0], mew=0.8)
        pos = (0.03, 0.97, "left") if mu < 0 else (0.97, 0.97, "right")
        ax.annotate(f"{lab}\nμ = {mu:+.2f}" + ("（> μc）" if mu > mu_c else ""), xy=pos[:2],
                    xycoords="axes fraction", fontsize=8, va="top", ha=pos[2])
        ax.set_xlabel("疾病活動度 x", fontsize=8)
        ax.set_xticks([-1, 0, 1]); ax.set_yticks([])
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("位能（谷底越淺，回復越慢）", fontsize=8)
    save(fig, "figB_landscape")


# ------------------------------------------------------------------ 資料（圖A／圖C 共用：校準鎖閾＋一個測試資料集）
def make_data(P, Cj, seed, n):
    cfg = ews_cfg(P)
    fam = value(P, "seed_families")
    out = {}
    for tag, s in (("cal", seed + fam["calibration"]), ("test", seed + fam["test"])):
        C = simulate_cohort(P, Cj, s, n=n)
        y = observe(C, P, Cj["observation_scenarios"]["default"], module_rng(s, "obs", 0))
        stop, fu = _stop_days(C)
        sc = score_cohort(y, C["run_in"], C["T"], cfg["window_days"], cfg["min_obs"], cfg["detrend"],
                          cfg["bw_frac"], cfg["eval_every"], cfg["joint"], stop_at=stop)
        out[tag] = dict(C=C, y=y, sc=sc, stop=stop, fu=fu)
    lk = lock_threshold(out["cal"]["sc"]["S"], out["cal"]["C"]["mech"] == "stable", out["cal"]["fu"], cfg["budget"])
    t_alarm = first_alarm(out["test"]["sc"]["S"], out["test"]["sc"]["eval_times"], lk["threshold"])
    return out["test"], lk, t_alarm, cfg


# ------------------------------------------------------------------ 圖A 機制展示廊
def _representative(C, mech):
    """發作日最接近該機制發作日中位數者；多數未發作的機制取第一個未發作者（不挑訊號）。"""
    idx = np.where(C["mech"] == mech)[0]
    ev = idx[C["t_event"][idx] >= 0]
    if len(ev) >= max(3, len(idx) // 2):
        med = np.median(C["t_event"][ev])
        return ev[np.argmin(np.abs(C["t_event"][ev] - med))]
    ne = idx[C["t_event"][idx] < 0]
    return (ne if len(ne) else idx)[0]


def figA(P, test, cfg):
    C, y = test["C"], test["y"]
    run_in, T = C["run_in"], C["T"]
    eval_times = np.arange(run_in, T, cfg["eval_every"])
    tgrid = np.arange(T)
    fig, axes = plt.subplots(3, 6, figsize=(13, 5.4), sharex=True)
    for c, mech in enumerate(MECH_ORDER):
        i = _representative(C, mech)
        ok = np.isfinite(y[i]); t_i, y_i = tgrid[ok], y[i][ok].astype(float)
        ar1, sd = rolling_indicators(t_i, y_i, eval_times, cfg["window_days"], cfg["min_obs"], cfg["detrend"], cfg["bw_frac"])
        rows = ((y_i, t_i, "觀測序列"), (ar1, eval_times, "滾動 AR(1)"), (sd, eval_times, "滾動 SD"))
        rate = float(C["event_24m"][C["mech"] == mech].mean())
        for r, (v, t, lab) in enumerate(rows):
            ax = axes[r, c]
            days = (np.asarray(t) - run_in)
            if r == 0:
                ax.plot(days, v, ".", ms=1.6, color=G4[1])
            else:
                ax.plot(days, v, "-", lw=1.1, color=G4[0])
            ax.axvline(0, color=G4[2], lw=0.9, ls=(0, (2, 2)))                       # 減藥起始（點虛線）
            if C["t_event"][i] >= 0:
                ax.axvline(C["t_event"][i] - run_in, color=G4[0], lw=1.0)            # 發作日（實線）
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if c == 0:
                ax.set_ylabel(lab, fontsize=8)
            if r == 0:
                ax.annotate(MECH_ZH[mech], xy=(0.5, 1.04), xycoords="axes fraction", ha="center", fontsize=9)
                ax.annotate(f"發作率 {rate:.2f}", xy=(0.03, 0.03), xycoords="axes fraction", fontsize=7, color=G4[1])
            if r == 2:
                ax.set_xlabel("減藥後天數", fontsize=8)
    for r in range(0, 3):                                                             # 同列同刻度（不得放大差異）
        lo = min(a.get_ylim()[0] for a in axes[r]); hi = max(a.get_ylim()[1] for a in axes[r])
        for a in axes[r]:
            a.set_ylim(lo, hi)
    fig.subplots_adjust(hspace=0.12, wspace=0.3)
    save(fig, "figA_gallery")


# ------------------------------------------------------------------ 圖C 警報操作特性
def figC(P, test, lk, t_alarm):
    C = test["C"]; run_in = C["run_in"]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw=dict(width_ratios=[1.1, 1]))
    # 左：固定假警報負擔下，各機制的（發作前）警報比例；未發作機制＝觸發（假警報）比例
    PRE = ("bifurcation", "stochastic_escape", "continuous_deterioration", "exogenous_shock")
    xs, vals, labels, ns, hatches = [], [], [], [], []
    for k, mech in enumerate(MECH_ORDER):
        sel = C["mech"] == mech
        ev = sel & (C["t_event"] >= 0)
        if mech in PRE and ev.sum() >= 3:
            pre = ((t_alarm >= 0) & (t_alarm < C["t_event"]))[ev]
            vals.append(float(pre.mean())); ns.append(int(ev.sum())); hatches.append("///")
        else:
            vals.append(float((t_alarm[sel] >= 0).mean())); ns.append(int(sel.sum())); hatches.append(None)
        labels.append(MECH_ZH[mech]); xs.append(k)
    for x, v, h in zip(xs, vals, hatches):
        axL.bar([x], [v], width=0.62, color=G4[3] if h is None else "white", edgecolor=G4[0], lw=0.7, hatch=h)
        axL.annotate(f"n={ns[x]}", xy=(x, v), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=6.5, color=G4[1])
    fa = float((t_alarm[C["mech"] == "stable"] >= 0).mean())
    axL.axhline(fa, color=G4[0], lw=0.9, ls=(0, (4, 2)))
    axL.annotate(f"穩定機制觸發比例 {fa:.2f}\n（達成負擔 {lk.get('achieved_fa_per_py'):.2f}/病人年）",
                 xy=(0.02, 0.97), xycoords="axes fraction", va="top", fontsize=7.5)
    from matplotlib.patches import Patch
    axL.legend(handles=[Patch(facecolor="white", edgecolor=G4[0], hatch="///", label="發作前警報比例（有發作者）"),
                        Patch(facecolor=G4[3], edgecolor=G4[0], label="觸發比例（無發作機制＝假警報）")],
               loc="upper right", fontsize=6.5, frameon=False)
    axL.set_xticks(xs); axL.set_xticklabels(labels, fontsize=7.5)
    axL.set_ylabel("警報比例", fontsize=8); axL.set_ylim(0, 1)
    # 右：首次警報 → 發作 的提前期分布（有發作的機制）
    data, dl = [], []
    for mech in ("bifurcation", "stochastic_escape", "continuous_deterioration", "exogenous_shock"):
        sel = (C["mech"] == mech) & (C["t_event"] >= 0) & (t_alarm >= 0) & (t_alarm < C["t_event"])
        lead = (C["t_event"] - t_alarm)[sel]
        if len(lead) >= 3:
            data.append(lead); dl.append(f"{MECH_ZH[mech]}\n(n={len(lead)})")
    if data:
        bp = axR.boxplot(data, tick_labels=dl, widths=0.5, patch_artist=True,
                         medianprops=dict(color=G4[0], lw=1.2), boxprops=dict(facecolor=G4[3], color=G4[0]),
                         whiskerprops=dict(color=G4[0]), capprops=dict(color=G4[0]),
                         flierprops=dict(marker=".", ms=3, mfc=G4[1], mec=G4[1]))
        axR.set_ylabel("提前期（發作日 − 首次警報日，天）", fontsize=8)
        axR.tick_params(axis="x", labelsize=7)
    else:
        axR.annotate("無足夠發作前警報個案", xy=(0.5, 0.5), xycoords="axes fraction", ha="center")
    for ax in (axL, axR):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.subplots_adjust(wspace=0.28)
    save(fig, "figC_operating")


# ------------------------------------------------------------------ 圖D 可辨識地圖
def figD(P):
    path = os.path.join(ROOT, "results", "grid.json")
    if not os.path.exists(path):
        print("[figs] 圖D 待補：results/grid.json 尚未產生（python run.py --phase grid）")
        return
    ident = json.load(open(path, encoding="utf-8"))["identifiability"]
    taus = sorted({r["tau_x"] for r in ident})
    ints = [k for k in value(P, "sampling_intervals") if any(r["interval"] == k for r in ident)]
    errs = [k for k in value(P, "meas_error_levels") if any(r["meas_error"] == k for r in ident)]
    lk = json.load(open(os.path.join(ROOT, "results", "locked.json"), encoding="utf-8"))
    b_sens = lk["biopsy"]["sensitivity"] if lk.get("biopsy", {}).get("reachable") else None
    fig, axes = plt.subplots(1, len(taus), figsize=(2.9 * len(taus) + 1.2, 3.0), squeeze=False)
    Ms = []
    for a, tx in enumerate(taus):
        M = np.full((len(errs), len(ints)), np.nan)
        for r in ident:
            if r["tau_x"] == tx and r["sens"]["mean"] is not None:
                M[errs.index(r["meas_error"]), ints.index(r["interval"])] = r["sens"]["mean"]
        Ms.append(M)
        ax = axes[0, a]
        im = ax.imshow(M, cmap="Greys", vmin=0, vmax=1, aspect="auto", origin="upper")
        for i in range(len(errs)):
            for j in range(len(ints)):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if M[i, j] > 0.55 else G4[0])
        if b_sens is not None and np.isfinite(M).any() and np.nanmax(M) >= b_sens:
            ax.contour(M, levels=[b_sens], colors=[G4[0]], linewidths=1.2, linestyles="dashed")
        ax.set_xticks(range(len(ints))); ax.set_xticklabels([f"{value(P,'sampling_intervals')[k]} 天" for k in ints], fontsize=7)
        ax.set_yticks(range(len(errs))); ax.set_yticklabels([f"{value(P,'meas_error_levels')[k]:.2f}" for k in errs], fontsize=7)
        ax.set_xlabel("取樣間隔", fontsize=8)
        ax.annotate(f"τx = {tx} 天", xy=(0.5, 1.05), xycoords="axes fraction", ha="center", fontsize=9)
    axes[0, 0].set_ylabel("量測誤差 SD（x 單位）", fontsize=8)
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, pad=0.015)
    cb.set_label("跳轉前警報敏感度（固定負擔）", fontsize=8)
    note = "" if (b_sens is None or any(np.isfinite(M).any() and np.nanmax(M) >= b_sens for M in Ms)) else \
        f"（全圖低於切片臂敏感度 {b_sens:.2f}，無等值線）"
    if note:
        fig.text(0.01, 0.01, note, fontsize=7, color=G4[1])
    save(fig, "figD_identifiability")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260818)
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--only", type=str, default=None, choices=list("ABCD"))
    a = ap.parse_args()
    P = M0.load("thresholds.json"); Cj = M0.load("cohort.json")
    if value(P, "hazard")["lambda0_per_day"] is None:
        raise SystemExit("先跑 python run.py --phase calibrate")
    if a.only in (None, "B"):
        figB(P)
    if a.only in (None, "A", "C"):
        test, lk, t_alarm, cfg = make_data(P, Cj, a.seed, a.n)
        if a.only in (None, "A"):
            figA(P, test, cfg)
        if a.only in (None, "C"):
            figC(P, test, lk, t_alarm)
    if a.only in (None, "D"):
        figD(P)
    print("[figs] 完成 → results/figs/（原型參數含佔位值——不得對外引用）")
