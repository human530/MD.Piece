# -*- coding: utf-8 -*-
"""特徵建構（指示 三節）。三種讀法，全部**以全世代參數標準化**（指示 一之二：禁止逐人標準化）。

  L0_<通道>  索引窗內（|day_from_index| ≤ method.index_window_days）最接近索引日之水準 → 全世代 z
             （稽核修正 2026-08-30：原「該通道最近一次有量到」會把索引後最長 364 天的追蹤值當基線，
             使 M1 混入未來資訊並膨脹其表現；改為事前鎖定之索引窗，窗內無值即 NaN 交折內補值）
  L_<通道>   各成套對數值之中位數 → 全世代 z
  S_<通道>   對數值對天數之線性斜率，年化 → 除以全世代標準差（依指示原文不置中）
  R_<A>_<B> 兩通道水準之差（＝對數比值）→ 全世代 z

標準化參數（median / sd）由整個世代一次算出並存檔，之後對任何個案都套用同一組值——
`transform_one()` 提供單一個案的轉換，與矩陣版共用同一組參數（checks 以此驗證無逐人標準化）。
缺值一律 NaN 往下傳（不填 0），由模型管線內的補值處理。"""
import numpy as np
import pandas as pd

from params_io import value

EPS = 1e-9


def _to_model_scale(P, ch, v):
    return np.log(np.maximum(v, EPS)) if ch in value(P, "log_channels") else v


def patient_summaries(P, long_df, channels):
    """每人每通道的：索引值、中位數、年化斜率（皆為模型尺度；不足者 NaN）。"""
    min_k = int(value(P, "method.slope_min_measurements"))
    ids = np.sort(long_df["patient_id"].unique())
    idx_pos = {p: i for i, p in enumerate(ids)}
    n = len(ids)
    out = {k: {ch: np.full(n, np.nan) for ch in channels} for k in ("index", "median", "slope")}
    d_all = long_df["day_from_index"].to_numpy(float)
    pid_all = long_df["patient_id"].to_numpy()
    for ch in channels:
        v = _to_model_scale(P, ch, long_df[ch].to_numpy(float))
        ok = np.isfinite(v)
        df = pd.DataFrame(dict(pid=pid_all[ok], day=d_all[ok], v=v[ok]))
        if df.empty:
            continue
        med = df.groupby("pid")["v"].median()
        for p, m in med.items():
            out["median"][ch][idx_pos[p]] = m
        win = float(value(P, "method.index_window_days"))
        dfw = df[df["day"].abs() <= win].copy()                  # 索引窗外不得當基線
        dfw["absd"] = dfw["day"].abs()
        first = dfw.sort_values(["pid", "absd"]).groupby("pid").first()
        for p, m in first["v"].items():
            out["index"][ch][idx_pos[p]] = m
        g = df.groupby("pid")
        cnt, span = g["v"].size(), g["day"].max() - g["day"].min()
        eligible = cnt[(cnt >= min_k) & (span > 0)].index
        if len(eligible):
            sub = df[df["pid"].isin(eligible)]
            for p, gg in sub.groupby("pid"):
                x = gg["day"].to_numpy(); y = gg["v"].to_numpy()
                b = np.polyfit(x, y, 1)[0] * 365.0                     # 年化
                out["slope"][ch][idx_pos[p]] = b
    return ids, out


def fit_transform(P, long_df, channels, ratios):
    """回傳 (features DataFrame, params dict)。params 內含全世代 median/sd，供 transform_one 與介面共用。"""
    ids, S = patient_summaries(P, long_df, channels)
    cols, params = {}, dict(level_index={}, level_median={}, slope={}, ratio={}, channels=list(channels),
                            ratios=[list(r) for r in ratios], log_channels=list(value(P, "log_channels")))

    def z(vec, store, key, center=True):
        med = float(np.nanmedian(vec)) if np.isfinite(vec).any() else 0.0
        sd = float(np.nanstd(vec)) if np.isfinite(vec).any() else 1.0
        sd = sd if sd > EPS else 1.0
        store[key] = dict(center=med if center else 0.0, sd=sd)
        return (vec - (med if center else 0.0)) / sd

    for ch in channels:
        cols[f"L0_{ch}"] = z(S["index"][ch], params["level_index"], ch)
        cols[f"L_{ch}"] = z(S["median"][ch], params["level_median"], ch)
        cols[f"S_{ch}"] = z(S["slope"][ch], params["slope"], ch, center=False)   # 指示三節：只以 SD 標準化
    for a, b in ratios:
        cols[f"R_{a}_{b}"] = z(S["median"][a] - S["median"][b], params["ratio"], f"{a}_{b}")
    return pd.DataFrame(cols, index=ids), params


def transform_one(P, panels, params):
    """單一個案：panels = [{'day': d, 'Cr': v, ...}, ...]（缺值省略或 None）。回傳與矩陣版同名的特徵 dict。
    使用的標準化參數與整個世代完全相同——這是「禁止逐人標準化」的可執行證明。"""
    min_k = int(value(P, "method.slope_min_measurements"))
    ch_list, out = params["channels"], {}
    raw_med, raw_idx, raw_slope = {}, {}, {}
    for ch in ch_list:
        pts = [(float(p["day"]), float(p[ch])) for p in panels if p.get(ch) is not None and np.isfinite(float(p.get(ch, np.nan)))]
        if not pts:
            raw_med[ch] = raw_idx[ch] = raw_slope[ch] = np.nan
            continue
        days = np.array([q[0] for q in pts]); vals = _to_model_scale(P, ch, np.array([q[1] for q in pts]))
        raw_med[ch] = float(np.median(vals))
        win = float(value(P, "method.index_window_days"))
        inw = np.abs(days) <= win
        raw_idx[ch] = float(vals[inw][np.argmin(np.abs(days[inw]))]) if inw.any() else np.nan
        raw_slope[ch] = float(np.polyfit(days, vals, 1)[0] * 365.0) if (len(pts) >= min_k and days.max() > days.min()) else np.nan
    for ch in ch_list:
        pi, pm, ps = params["level_index"][ch], params["level_median"][ch], params["slope"][ch]
        out[f"L0_{ch}"] = (raw_idx[ch] - pi["center"]) / pi["sd"]
        out[f"L_{ch}"] = (raw_med[ch] - pm["center"]) / pm["sd"]
        out[f"S_{ch}"] = raw_slope[ch] / ps["sd"]
    for a, b in params["ratios"]:
        pr = params["ratio"][f"{a}_{b}"]
        out[f"R_{a}_{b}"] = (raw_med[a] - raw_med[b] - pr["center"]) / pr["sd"]
    return out
