# -*- coding: utf-8 -*-
"""指示 十一節的必要自我檢查。每一條都是執行期斷言，不通過即 raise（建置失敗）。"""
import numpy as np

from features import transform_one
from params_io import value


def check_archive_absent(P, feature_frames, long_df, pat_df):
    """1. 封存集所有欄位都不在任何特徵矩陣中（通道消耗規則）。"""
    arch = value(P, "channels_archive")
    for name, X in feature_frames.items():
        for a in arch:
            bad = [c for c in X.columns if a.lower() in c.lower()]
            assert not bad, f"{name} 含封存集欄位：{bad}"
    assert not [c for c in long_df.columns if c in arch], "縱向表含封存集數值欄位"
    assert all(f"ord_{a}" in pat_df.columns for a in arch), "封存集之開立指示變數缺漏（M0 需要）"
    return True


def check_no_per_patient_scaling(P, long_df, X, fparams, patient_ids, n_probe=25, rng=None):
    """2. 沒有任何逐人標準化：把單一病人抽出來單獨轉換，結果必須與整批一致（同一組全世代參數）。"""
    rng = rng or np.random.default_rng(0)
    probe = rng.choice(patient_ids, size=min(n_probe, len(patient_ids)), replace=False)
    for pid in probe:
        rows = long_df[long_df["patient_id"] == pid]
        panels = [{"day": r["day_from_index"], **{c: (None if not np.isfinite(r[c]) else float(r[c])) for c in fparams["channels"]}}
                  for _, r in rows.iterrows()]
        one = transform_one(P, panels, fparams)
        for k, v in one.items():
            batch = X.loc[pid, k]
            if np.isfinite(v) and np.isfinite(batch):
                assert abs(v - batch) < 1e-6, f"病人 {pid} 特徵 {k} 單獨轉換 {v} ≠ 批次 {batch}（疑似逐人標準化）"
    return True


def check_m0_sealed_before(ledger):
    """3. M0 已存檔且時間戳早於任何主模型效能計算。"""
    return ledger.require_sealed()


def check_missing_is_nan(long_df, channels):
    """4. 未開立的通道為 NaN，不是 0。"""
    for ch in channels:
        v = long_df[ch].to_numpy(float)
        assert np.isnan(v).any(), f"{ch} 完全沒有缺值——缺值機制可能被填補掉了"
        nz = v[np.isfinite(v)]
        if ch not in ("uRBC_dys", "uRBC_acan", "uRBC_cast"):
            assert not (nz == 0).any(), f"{ch} 出現 0 值，疑似以 0 代替未開立"
    return True


def check_stage_features_L1(P, X_cols_by_stage):
    """5. Stage 1–3 的特徵全部來自 L1 集。"""
    L1 = set(value(P, "channels_L1")); L2 = set(value(P, "channels_L2")); arch = set(value(P, "channels_archive"))
    for stage, cols in X_cols_by_stage.items():
        for c in cols:
            body = c.split("_", 1)[1]
            chans = [body] if body in L1 else body.split("_")
            for ch in chans:
                assert ch not in L2 and ch not in arch, f"Stage {stage} 的特徵 {c} 使用了非 L1 通道 {ch}"
                assert ch in L1, f"Stage {stage} 的特徵 {c} 通道 {ch} 不在 L1 集"
    return True


def check_report_completeness(report):
    """6/7. 每個階段都有回報拒答率；主要指標包含平衡正確率。"""
    for stage, r in report["stages"].items():
        assert "abstention_rate" in r["test"], f"Stage {stage} 未回報拒答率"
        assert "balanced_accuracy" in r["test"]["answered"], f"Stage {stage} 未回報平衡正確率"
    return True


def check_determinism(run_a, run_b):
    """8. 隨機種子已固定，重跑結果一致。"""
    assert run_a == run_b, "重跑結果不一致——種子未完全固定"
    return True
