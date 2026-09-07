# -*- coding: utf-8 -*-
"""五張圖（建置提示詞 輸出規格）：黑白灰、可讀字型、座標軸有單位。"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.style.use("grayscale")
plt.rcParams.update({"font.size": 10, "axes.grid": False, "figure.dpi": 130,
                     "font.family": ["Microsoft JhengHei", "DejaVu Sans"],
                     "axes.unicode_minus": False})
GREYS = ["0.15", "0.45", "0.7", "0.85", "0.3", "0.6"]
HATCH = ["", "//", "..", "xx", "\\\\", "++"]


def _save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    p = os.path.join(outdir, name)
    fig.tight_layout(); fig.savefig(p); plt.close(fig)
    return p


def fig1_single_case(fd, outdir):
    """單一翻轉型個案：eGFR(t)、滾動 AR(1)、滾動 SD，標出漂移起始／臨界日／門檻日／事件日／首次警報。"""
    x, ar, sd = np.asarray(fd["x"]), np.asarray(fd["ar1"]), np.asarray(fd["sd"])
    win, T = fd["window"], len(x)
    tt = np.arange(T); tw = np.arange(len(ar)) + win - 1
    fig, ax = plt.subplots(3, 1, figsize=(7.5, 6.5), sharex=True)
    ax[0].plot(tt, x, color="0.2", lw=0.8); ax[0].axhline(fd["threshold"], color="0.5", ls=":", lw=1)
    ax[0].set_ylabel("eGFR（mL/min/1.73m²；虛線=門檻 15）")
    ax[1].plot(tw, ar, color="0.2", lw=0.8); ax[1].set_ylabel(f"滾動 AR(1)（窗 {win} 天）")
    ax[2].plot(tw, sd, color="0.2", lw=0.8); ax[2].set_ylabel(f"滾動 SD（mL/min/1.73m²；窗 {win} 天）")
    ax[2].set_xlabel("時間（天）")
    for a in ax:
        if fd.get("t_onset", -1) >= 0: a.axvline(fd["t_onset"], color="0.7", ls=":", lw=1)
        if fd["t_crit"] >= 0: a.axvline(fd["t_crit"], color="0.1", ls="--", lw=1)
        if fd.get("t_threshold", -1) >= 0: a.axvline(fd["t_threshold"], color="0.4", ls="--", lw=0.8)
        if fd["t_event"] >= 0: a.axvline(fd["t_event"], color="0.1", ls="-", lw=1.2)
        if fd["first_alarm"] >= 0: a.axvline(fd["first_alarm"], color="0.55", ls="-.", lw=1)
    ax[0].set_title(f"翻轉型個案 #{fd['idx']}：漂移起始 {fd.get('t_onset', float('nan')):.0f}、--臨界日 {fd['t_crit']}、門檻日 {fd.get('t_threshold', -1)}、"
                    f"—事件日 {fd['t_event']}、-·-首次警報 {fd['first_alarm']}（去趨勢：{fd['detrend']}）", fontsize=8.5)
    return _save(fig, outdir, "fig1_flip_case_indicators.png")


def fig6_tau_scan(scans, outdir):
    """v2 壹：Δμ 掃描曲線——各 tau 下 median(t_threshold − t_crit) 對 Δμ 中位；tau ≤ 30 時預警窗口塌縮。"""
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    for j, (tau, rows) in enumerate(sorted(scans.items(), key=lambda kv: float(kv[0]))):
        m = [r["delta_mu_median"] for r in rows]
        d = [r["median_threshold_minus_crit_days"] if r["median_threshold_minus_crit_days"] is not None else np.nan for r in rows]
        fr = [r["frac_crossing"] for r in rows]
        ax[0].plot(m, d, marker="osd^"[j % 4], color=GREYS[j], label=f"τ = {tau} 天")
        ax[1].plot(m, fr, marker="osd^"[j % 4], color=GREYS[j], label=f"τ = {tau} 天")
    ax[0].axhline(180, color="k", ls=":", lw=0.8); ax[0].axhline(90, color="0.5", ls=":", lw=0.8)
    ax[0].set_xscale("log"); ax[0].set_xlabel("Δμ 中位（漂移量）"); ax[0].set_ylabel("t_threshold − t_crit 中位（天）")
    ax[0].set_title("鬆弛時間 τ 決定預警窗口是否存在（點線＝180／90 天）", fontsize=9); ax[0].legend(fontsize=8)
    ax[1].set_xscale("log"); ax[1].set_xlabel("Δμ 中位"); ax[1].set_ylabel("跨過 μc 且到達門檻的比例"); ax[1].legend(fontsize=8)
    # 圖說載明實測值（v2 附錄肆）
    mx = {}
    for tau, rows in scans.items():
        vals = [r["median_threshold_minus_crit_days"] for r in rows if r["median_threshold_minus_crit_days"] is not None and np.isfinite(r["median_threshold_minus_crit_days"])]
        mx[float(tau)] = max(vals) if vals else float("nan")
    parts = "、".join(f"τ={int(t)} 最長 {mx[t]:.0f} 天" for t in sorted(mx) if t < 60)
    ax[1].set_title(f"實測：{parts}（皆不足六個月）", fontsize=9)
    fig.suptitle("τ ≤ 30 時預警窗口塌縮：雜訊誘發的提早逃逸使臨界日與門檻日塌縮在一起", fontsize=9)
    return _save(fig, outdir, "fig6_tau_scan.png")


def fig2_strata_types(cl_res, gen_share, outdir, title=""):
    """各風險層的軌跡型態分布：左＝分群結果的群比例（方法 A、五分位），右＝生成器型別組成（對照）。"""
    S = len(cl_res)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
    for s, r in enumerate(cl_res):
        bottom = 0
        for k, sh in enumerate(r["shares"]):
            ax[0].bar(s, sh, bottom=bottom, color=GREYS[k % 6], hatch=HATCH[k % 6], edgecolor="k", lw=0.5)
            bottom += sh
        ax[0].text(s, 1.02, f"K={r['K']}\nARI={r.get('ari_vs_generator', np.nan):.2f}", ha="center", fontsize=7)
    ax[0].set_xlabel("風險層（五分位，0=最低）"); ax[0].set_ylabel("群比例"); ax[0].set_ylim(0, 1.15)
    ax[0].set_title("分群結果（方法 A）", fontsize=9)
    names = list(gen_share[0].keys())
    for s in range(S):
        bottom = 0
        for k, nm in enumerate(names):
            ax[1].bar(s, gen_share[s][nm], bottom=bottom, color=GREYS[k % 6], hatch=HATCH[k % 6], edgecolor="k", lw=0.5,
                      label=nm if s == 0 else None)
            bottom += gen_share[s][nm]
    ax[1].set_xlabel("風險層（五分位）"); ax[1].set_ylabel("比例"); ax[1].set_title("生成器真值（對照，非發現）", fontsize=9)
    ax[1].legend(fontsize=7, loc="upper right")
    fig.suptitle(title, fontsize=9)
    return _save(fig, outdir, "fig2_strata_trajectory_types.png")


def fig3_static_vs_dynamic(agg_pred, outdir):
    """C 指數（靜態 vs 動態，多 seed 均值與範圍）與淨重新分類（兩種閾值）。"""
    Ls = sorted(agg_pred["landmarks"].keys(), key=int)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    xs = np.arange(len(Ls))
    for j, (key, lab, mk) in enumerate((("auc_static", "靜態", "o"), ("auc_dynamic", "動態（地標）", "s"))):
        m = [agg_pred["landmarks"][L][key]["mean"] for L in Ls]
        lo = [agg_pred["landmarks"][L][key]["min"] for L in Ls]; hi = [agg_pred["landmarks"][L][key]["max"] for L in Ls]
        ax[0].errorbar(xs + 0.08 * j, m, yerr=[np.subtract(m, lo), np.subtract(hi, m)], fmt=mk, color=GREYS[j], capsize=3, label=lab)
    ax[0].set_xticks(xs); ax[0].set_xticklabels([f"{L} 天" for L in Ls]); ax[0].set_ylabel("C 指數（AUC）"); ax[0].legend(fontsize=8)
    ax[0].set_title("鑑別力（誤差棒＝各 seed 範圍）", fontsize=9)
    for j, rule in enumerate(("abs", "rel")):
        nri = [agg_pred["landmarks"][L][f"reclass_{rule}"]["nri"]["mean"] for L in Ls]
        mi = [agg_pred["landmarks"][L][f"reclass_{rule}"]["moved_in"]["mean"] for L in Ls]
        mo = [agg_pred["landmarks"][L][f"reclass_{rule}"]["moved_out"]["mean"] for L in Ls]
        ax[1].bar(xs + (j - 0.5) * 0.35, mi, 0.35, color=GREYS[j], hatch=HATCH[j], edgecolor="k", lw=0.5, label=f"移入高風險（{rule}）")
        ax[1].bar(xs + (j - 0.5) * 0.35, [-v for v in mo], 0.35, color="1.0", hatch=HATCH[j], edgecolor="k", lw=0.5, label=f"移出高風險（{rule}）")
        for xi, v, top in zip(xs, nri, mi):
            ax[1].text(xi + (j - 0.5) * 0.35, top + 0.008, f"NRI\n{v:+.2f}", ha="center", va="bottom", fontsize=6.5)
    ax[1].axhline(0, color="k", lw=0.6); ax[1].set_xticks(xs); ax[1].set_xticklabels([f"{L} 天" for L in Ls])
    ax[1].set_ylim(top=ax[1].get_ylim()[1] * 1.15)
    ax[1].set_ylabel("風險集內比例（上=移入，下=移出）"); ax[1].legend(fontsize=6.5, loc="lower right")
    ax[1].set_title("淨重新分類（abs=事件率閾值，rel=前 20%）", fontsize=9)
    return _save(fig, outdir, "fig3_static_vs_dynamic.png")


def fig4_timing_shift(timing, outdir):
    """介入時點位移：直方圖＋累積分布（決定書 §8），兩種閾值各一列。"""
    fig, ax = plt.subplots(2, 2, figsize=(9, 6))
    for i, rule in enumerate(("abs", "rel")):
        t = timing[rule]; Ls = np.array(t["landmarks"])
        hb, hd = np.array(t["shift_days_both"]["hist"], float), np.array(t["first_flag_hist_dynamic_only"], float)
        ax[i, 0].bar(Ls - 20, hb, 40, color="0.3", edgecolor="k", lw=0.4, label=f"靜態也高（n={t['n_both']}）：位移=動態首次判定日")
        ax[i, 0].bar(Ls + 20, hd, 40, color="0.8", hatch="//", edgecolor="k", lw=0.4, label=f"僅動態高（n={t['n_dynamic_only']}）：新增判定日")
        ax[i, 0].set_xlabel("天"); ax[i, 0].set_ylabel("人數"); ax[i, 0].legend(fontsize=7)
        ax[i, 0].set_title(f"閾值 {rule}；靜態高但動態從未判定 n={t['n_static_only']}", fontsize=9)
        for h, c, lab in ((hb, "0.3", "靜態也高"), (hd, "0.7", "僅動態高")):
            if h.sum() > 0:
                ax[i, 1].step(Ls, np.cumsum(h) / h.sum(), where="post", color=c, label=lab)
        ax[i, 1].set_xlabel("動態首次判定日（天）"); ax[i, 1].set_ylabel("累積比例"); ax[i, 1].set_ylim(0, 1.02); ax[i, 1].legend(fontsize=7)
    return _save(fig, outdir, "fig4_intervention_timing_shift.png")


def fig5_leadtime(lead, outdir):
    """翻轉型 vs 爬升型的提前期（到事件日）分布；翻轉型另附到臨界日。"""
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    data = [np.asarray(lead["flip_event"]), np.asarray(lead["linear_event"])]
    bp = ax[0].boxplot([d[np.isfinite(d)] for d in data], labels=["翻轉型", "爬升型"], patch_artist=True, widths=0.5)
    for p, c in zip(bp["boxes"], ["0.4", "0.85"]):
        p.set_facecolor(c)
    ax[0].axhline(0, color="k", lw=0.6, ls=":")
    ax[0].set_ylabel("提前期＝事件日 − 首次警報日（天）")
    ax[0].set_title(f"到事件日；雙尾置換 p={lead['perm_p']:.3f}（中位差 {lead['perm_diff']:+.0f} 天）", fontsize=9)
    fc = np.asarray(lead["flip_crit"]); fc = fc[np.isfinite(fc)]
    if len(fc):
        ax[1].hist(fc, bins=30, color="0.5", edgecolor="k", lw=0.4)
    ax[1].axvline(0, color="k", lw=0.8, ls="--")
    ax[1].set_xlabel("臨界日 − 首次警報日（天；>0 = 臨界日前就警報）"); ax[1].set_ylabel("人數")
    ax[1].set_title("翻轉型：到已知臨界日", fontsize=9)
    return _save(fig, outdir, "fig5_leadtime_flip_vs_linear.png")


def fig7_k_selection(clA, clB, outdir):
    """v2 附錄壹：每一風險層（五分位）的 BIC 曲線與穩定度曲線同圖雙軸；標出雙準則選定的 K、
    Hennig 門檻 0.60／0.75、與 k_ceiling_hit。"""
    sets = [("方法 A（GMM）", clA)] + ([("方法 B（k-means）", clB)] if clB else [])
    S = len(clA)
    fig, axes = plt.subplots(len(sets), S, figsize=(2.6 * S, 2.6 * len(sets) + 0.6), squeeze=False)
    for r, (name, cl) in enumerate(sets):
        for s, row in enumerate(cl):
            ax = axes[r, s]
            ks = sorted(int(k) for k in row["bic_by_k"])
            bic = [row["bic_by_k"][str(k)] if str(k) in row["bic_by_k"] else row["bic_by_k"][k] for k in ks]
            stab = [row["stability_by_k"][str(k)] if str(k) in row["stability_by_k"] else row["stability_by_k"][k] for k in ks]
            ax.plot(ks, bic, "o-", color="0.15", ms=3, lw=1)
            ax.set_ylabel("BIC" if s == 0 else "")
            ax2 = ax.twinx()
            ax2.plot(ks, stab, "s--", color="0.55", ms=3, lw=1)
            ax2.set_ylim(0, 1.05)
            ax2.axhline(0.60, color="0.7", ls=":", lw=0.8); ax2.axhline(0.75, color="0.7", ls=":", lw=0.8)
            if s == S - 1: ax2.set_ylabel("穩定度（虛線）")
            else: ax2.set_yticklabels([])
            ax.axvline(row["K"], color="0.3", lw=1.2, alpha=0.6)
            tag = ("撞頂 " if row.get("k_ceiling_hit") else "") + ("連續 " if row.get("continuous_heterogeneity") else "")
            ax.set_title(f"{name} 層{row['stratum']}：K={row['K']} 真{row.get('n_true_labels', '?')} {tag}", fontsize=7.5)
            ax.set_xlabel("K"); ax.tick_params(labelsize=7); ax2.tick_params(labelsize=7)
    fig.suptitle("群數選取（雙準則：BIC 改善 < 5% 範圍 且 穩定度 ≥ 0.60 的最小 K）；「撞頂」= BIC 到 k_max 仍單調改善", fontsize=9)
    return _save(fig, outdir, "fig7_k_selection_curves.png")
