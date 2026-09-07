# -*- coding: utf-8 -*-
"""腎損傷分流模型 Stage 0–5——完整可重現管線（建置指示 v11）。

    python pipeline.py [--seed 20260828] [--n 4000] [--quick]

流程（順序為方法學要求，不得調換）：
  資料 → Stage 0 閘門 → 特徵（全世代標準化）→ 訓練/測試分割
  → 【M0 三階段全部建好並定版存檔】→ 主模型階梯 M1/M2/M3 → 層間判定（配對 bootstrap）
  → 拒答門檻於訓練集鎖定 → 測試集評估一次 → Stage 4 分流格 → Stage 5 建議與評估
  → 指示 十一節八項 assert → report.md / *.json / feature_importance.csv / limitations.md
Stage 6 不實作。"""
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

import numpy as np                                   # noqa: E402
import pandas as pd                                  # noqa: E402

import params_io as PIO                              # noqa: E402
from params_io import value                          # noqa: E402
from seeding import module_rng                       # noqa: E402
import datagen                                       # noqa: E402
import features as F                                 # noqa: E402
import models as M                                   # noqa: E402
import boxes as BX                                   # noqa: E402
import checks as CK                                  # noqa: E402

RESULTS = os.path.join(ROOT, "results")
STAGES = {"1": ("y_acute", "急性度"), "2": ("y_site", "部位"), "3": ("y_pheno", "表現型")}


def _jd(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    if isinstance(o, np.ndarray): return o.tolist()
    return str(o)


def _dump(obj, name):
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=_jd)


def stage_feature_sets(P, X, morph_share_by_col):
    """各層×各階段特徵欄位（指示 六節；Stage 1–3 只用 L1）。
    形態欄位可得（逐欄計算自三個形態欄，取最小值過閘）<30% 時，**只有 Stage 3** 走代理路徑
    （指示 七節只授權 Stage 3；稽核修正：舊版連 Stage 1–2 一併剔除形態特徵，超出授權範圍）。"""
    L1 = value(P, "channels_L1")
    morph = ["uRBC_dys", "uRBC_acan", "uRBC_cast"]
    gate = float(value(P, "method.morphology_availability_gate"))
    share_min = float(min(morph_share_by_col.values()))
    proxy = share_min < gate
    l0 = [f"L0_{c}" for c in L1]
    l_all = [f"L_{c}" for c in L1] + [f"S_{c}" for c in L1] + [c for c in X.columns if c.startswith("R_")]
    drop = {f"{p}_{m}" for p in ("L0", "L", "S") for m in morph}
    stages = {}
    for st in ("1", "2", "3"):
        use_proxy = proxy and st == "3"
        stages[st] = dict(M1=[c for c in l0 if not (use_proxy and c in drop)],
                          M2=[c for c in l_all if not (use_proxy and c in drop)])
        stages[st]["M3"] = stages[st]["M2"]
    return dict(stages=stages, proxy_mode=bool(proxy), morph_ok_share=share_min,
                morph_share_by_col={k: float(v) for k, v in morph_share_by_col.items()})


def m0_subgroups(pat_df, tr_ids, stage, morph_measured):
    """M0 逐族群報告用的族群（指示 九之四）。

    **族群不得由該階段自身的標籤導出**——例如用「分流格」分層時，格內該軸標籤幾乎單一，
    平衡正確率無定義（退化）。故採用：資料密度、形態欄位可得性、以及**其他軸**的類別。"""
    d = pat_df.set_index("patient_id").loc[tr_ids]
    q = d["n_panels"].to_numpy()
    mm = morph_measured.loc[tr_ids].to_numpy()                 # 由 long_df 三個形態欄計算，非生成器旗標
    sub = {"成套數 1–2": q <= 2, "成套數 3–4": (q >= 3) & (q <= 4), "成套數 ≥5": q >= 5,
           "形態欄位可得": mm, "形態欄位不可得": ~mm}
    other = {"1": "y_site", "2": "y_acute", "3": "y_site"}[stage]
    for v in sorted(map(str, d[other].unique())):
        lab = {"0": "非快速惡化", "1": "快速惡化"}.get(v, v) if other == "y_acute" else v
        sub[f"{lab}"] = d[other].astype(str).to_numpy() == v
    return sub


def run(seed, n, quick=False, verbose=True):
    P = PIO.load()
    chk = PIO.check(P, verbose=verbose)
    assert not chk["bad"], f"參數檔錯誤：{chk['bad']}"
    folds = int(value(P, "method.cv_folds")); nboot = 200 if quick else int(value(P, "method.bootstrap_n"))

    # ── 資料（先做決定性自檢：同 seed 兩次模擬逐位元相同——指示 十一之八的資料層檢查）
    da = datagen.simulate(P, seed=seed, n=120); db = datagen.simulate(P, seed=seed, n=120)
    CK.check_determinism((da[0].to_json(), da[1].to_json()), (db[0].to_json(), db[1].to_json()))
    long_df, pat_df = datagen.simulate(P, seed=seed, n=n)
    gen_truth = pat_df[["patient_id", "morph_available", "archetype"]].copy()   # 生成器內部真相，僅供對照，不進契約
    pat_df = pat_df.drop(columns=["morph_available", "archetype"])              # 契約欄位以外一律移除（真實資料不會有）
    arch = value(P, "channels_archive"); L1 = value(P, "channels_L1"); L2 = value(P, "channels_L2")

    # ── Stage 0：資料品質與密度閘門
    req = value(P, "method.stage0_required_channels"); min_p = int(value(P, "method.stage0_min_panels"))
    has_req = long_df.groupby("patient_id")[req].apply(lambda g: g.notna().any().all())
    ok_ids = set(pat_df.loc[(pat_df["n_panels"] >= min_p), "patient_id"]) & set(has_req[has_req].index)
    stage0 = dict(n_input=int(len(pat_df)), n_pass=int(len(ok_ids)), n_excluded=int(len(pat_df) - len(ok_ids)),
                  rule=f"至少 {min_p} 次成套且 {req} 至少一次有值")
    pat_df = pat_df[pat_df["patient_id"].isin(ok_ids)].reset_index(drop=True)
    long_df = long_df[long_df["patient_id"].isin(ok_ids)].reset_index(drop=True)

    # ── 特徵（全世代標準化）＋ 契約斷言
    CK.check_missing_is_nan(long_df, L1 + [c for c in L2 if long_df[c].notna().any()])
    X_all, fparams = F.fit_transform(P, long_df, L1 + L2, value(P, "priority_ratios"))
    X_all = X_all.loc[pat_df["patient_id"].to_numpy()]
    # 形態欄位可得比例：依指示 七節，由三個形態欄逐欄計算（病人層級「至少一次非缺值」），不用任何生成器旗標
    morph_cols = ["uRBC_dys", "uRBC_acan", "uRBC_cast"]
    morph_by_col = {c: float(long_df.groupby("patient_id")[c].apply(lambda g: g.notna().any())
                             .reindex(pat_df["patient_id"]).fillna(False).mean()) for c in morph_cols}
    morph_measured = long_df.groupby("patient_id")[morph_cols].apply(lambda g: g.notna().any().any())                             .reindex(pat_df["patient_id"]).fillna(False)
    morph_ok = float(min(morph_by_col.values()))
    fs = stage_feature_sets(P, X_all, morph_by_col)
    ord_cols = [f"ord_{a}" for a in arch] + [f"ord_{t}" for t in L2]
    X_ord = pat_df.set_index("patient_id")[ord_cols].astype(float)
    if verbose:
        print(f"[Stage 0] 進 {stage0['n_input']}、過 {stage0['n_pass']}；形態欄位可得 {morph_ok:.2f}"
              f"（{'<' if fs['proxy_mode'] else '≥'} 閘門 {value(P, 'method.morphology_availability_gate')}"
              f"{'——Stage 3 走代理特徵，特異度預期明顯下降' if fs['proxy_mode'] else ''}）")

    # ── 訓練／測試分割（門檻與層間判定在訓練集；測試集只評一次）
    r_split = module_rng(seed, "split")
    ids = pat_df["patient_id"].to_numpy()
    y_box_all = pat_df.set_index("patient_id")["y_box"]
    test_frac = float(value(P, "method.test_split"))
    test_mask = np.zeros(len(ids), bool)
    for b in np.unique(y_box_all.values):                      # 依分流格分層抽測試集
        bi = np.where(y_box_all.values == b)[0]
        test_mask[r_split.choice(bi, size=max(1, int(round(test_frac * len(bi)))), replace=False)] = True
    tr_ids, te_ids = ids[~test_mask], ids[test_mask]
    Y = {s: pat_df.set_index("patient_id")[col] for s, (col, _) in STAGES.items()}

    # ── 【先建 M0，全部定版存檔之後才允許任何主模型效能計算】（指示 一之四）
    ledger = M.M0Ledger(os.path.join(RESULTS, "m0_baseline.json"))
    m0_proba, m0_pred = {}, {}
    for s, (col, zh) in STAGES.items():
        ytr = Y[s].loc[tr_ids].to_numpy()
        proba, classes = M.cv_proba(X_ord.loc[tr_ids].to_numpy(), ytr, folds, seed, ledger=ledger, m0=True)
        pred = classes[proba.argmax(axis=1)]
        met = M.class_metrics(ytr, pred, classes)
        # 逐族群之 M0 洩漏（指示 九之四：不可只報整體——某些族群的洩漏會遠高於平均）
        by_sub = {}
        for gname, gmask in m0_subgroups(pat_df, tr_ids, s, morph_measured).items():
            if gmask.sum() >= 30 and len(np.unique(ytr[gmask])) > 1:
                by_sub[gname] = dict(n=int(gmask.sum()), balanced_accuracy=float(M.balanced_accuracy_score(ytr[gmask], pred[gmask])))
            else:
                by_sub[gname] = dict(n=int(gmask.sum()), balanced_accuracy=None,
                                     note="退化：族群內該軸標籤單一或人數不足，平衡正確率無定義")
        yb = y_box_all.loc[tr_ids].to_numpy()
        by_box = {str(b): (float(M.balanced_accuracy_score(ytr[yb == b], pred[yb == b])) if len(np.unique(ytr[yb == b])) > 1 else None)
                  for b in np.unique(yb)}
        ledger.record(s, dict(stage=zh, cv_train=met, by_subgroup=by_sub, by_box_balanced_accuracy=by_box,
                              by_box_note="分流格由標籤導出，多數格內該軸標籤單一故為 null（退化）；有意義的族群見 by_subgroup",
                              features="ord_* 指示變數（哪些檢驗被開立）", n_features=len(ord_cols)))
        m0_proba[s], m0_pred[s] = proba, pred
        if verbose:
            print(f"[M0] Stage {s} {zh}：平衡正確率 {met['balanced_accuracy']:.3f}（只用開立行為——這是洩漏下限）")
    m0_doc = ledger.seal(extra=dict(seed=seed, n=int(len(ids)), train_n=int(len(tr_ids))))

    # ── 主模型階梯 M1 → M2 → M3（訓練集 CV）＋層間判定
    rng_boot = module_rng(seed, "bootstrap")
    stages_report, removed_axes, ladder_preds, used_cols = {}, [], {}, {}
    for s, (col, zh) in STAGES.items():
        ytr = Y[s].loc[tr_ids].to_numpy()
        preds, metrics, aucless = {}, {}, {}
        for layer in ("M1", "M2", "M3"):
            cols = fs["stages"][s][layer]
            used_cols[f"Stage{s}_{layer}"] = list(cols)       # 供封存集/L1 斷言逐層核對（稽核修正：舊版九個 frame 是同一份 M2 的別名）
            Xtr = X_all.loc[tr_ids, cols].to_numpy()          # M3＝M2 排除封存集與 ord_*（本管線兩者本就不在特徵中，故 M3≡M2，照實報告）
            proba, classes = M.cv_proba(Xtr, ytr, folds, seed, ledger=ledger)
            pred = classes[proba.argmax(axis=1)]
            preds[layer] = pred
            metrics[layer] = M.class_metrics(ytr, pred, classes)
            aucless[layer] = dict(proba=proba, classes=classes)
        cmp_20 = M.paired_diff_ci(ytr, preds["M2"], m0_pred[s], nboot, rng_boot)
        cmp_21 = M.paired_diff_ci(ytr, preds["M2"], preds["M1"], nboot, rng_boot)
        rand = np.full(len(ytr), pd.Series(ytr).mode()[0])
        cmp_3r = M.paired_diff_ci(ytr, preds["M3"], rand, nboot, rng_boot)
        verdicts = dict(
            M2_vs_M0=dict(**cmp_20, verdict=("通過" if cmp_20["significant"] else "未通過——該軸自管線移除，該軸不具超越開立行為之生理訊號")),
            M2_vs_M1=dict(**cmp_21, verdict=("有增額價值" if cmp_21["significant"] else "無增額價值（不移除，照實報告）")),
            M3_vs_random=dict(**cmp_3r, verdict=("有獨立增益" if cmp_3r["significant"] else "無獨立增益（照實報告）")))
        if not cmp_20["significant"]:
            removed_axes.append(s)
        stages_report[s] = dict(name=zh, label=col, train=dict(
            M0=dict(balanced_accuracy=ledger.stages[s]["cv_train"]["balanced_accuracy"]),
            **{L: dict(balanced_accuracy=metrics[L]["balanced_accuracy"], accuracy=metrics[L]["accuracy"]) for L in ("M1", "M2", "M3")},
            verdicts=verdicts), ladder=aucless)
        ladder_preds[s] = preds
        if verbose:
            print(f"[階梯] Stage {s} {zh}：M0 {ledger.stages[s]['cv_train']['balanced_accuracy']:.3f} | "
                  f"M1 {metrics['M1']['balanced_accuracy']:.3f} | M2 {metrics['M2']['balanced_accuracy']:.3f} | "
                  f"M2−M0 {cmp_20['diff']:+.3f} [{cmp_20['lo']:+.3f},{cmp_20['hi']:+.3f}] {verdicts['M2_vs_M0']['verdict'][:4]}")

    # ── 拒答門檻：訓練集上選定並鎖定
    grid = value(P, "method.abstain_grid"); target = float(value(P, "method.abstain_target_accuracy"))
    thresholds = {}
    for s in STAGES:
        curve = M.coverage_accuracy_curve(stages_report[s]["ladder"]["M2"]["proba"], stages_report[s]["ladder"]["M2"]["classes"], Y[s].loc[tr_ids].to_numpy(), grid)
        pick = M.pick_threshold(curve, target, float(value(P, "method.abstain_grid_start")))
        thresholds[s] = dict(**pick, curve=curve)
    _dump(dict(note="各階段拒答門檻，於訓練集依作答率—正確率曲線選定後鎖定；測試集不得調整",
               rule=f"最小門檻使訓練集作答者平衡正確率 ≥ {target}（不可達→0.6 並標記）",
               thresholds={s: {k: v for k, v in t.items() if k != "curve"} for s, t in thresholds.items()},
               curves={s: t["curve"] for s, t in thresholds.items()}), "thresholds.json")

    # ── 測試集：訓練全量配適 → 只評估一次
    final_models, te_out = {}, {}
    for s, (col, zh) in STAGES.items():
        removed = s in removed_axes
        cols = fs["stages"][s]["M2"]
        model = M.fit_full(X_all.loc[tr_ids, cols].to_numpy(), Y[s].loc[tr_ids].to_numpy(), seed, ledger=ledger)
        final_models[s] = dict(model=model, cols=cols)
        proba = model.predict_proba(X_all.loc[te_ids, cols].to_numpy())
        classes = model.classes_
        pred, answered, conf = M.apply_abstention(proba, classes, thresholds[s]["threshold"])
        if removed:
            # 判定規則：M2 未顯著超越 M0 → **該階段自管線移除**（指示 六節、十四節）。
            # 移除的意義是下游不得再用它：該軸一律輸出「無法判定」，Stage 4 因此無法成格；
            # 但其他軸照跑（逐階段停止，不是整案停止）。下方仍計算該軸的測試效能供照實報告。
            pred_routing = np.full(len(pred), BX.UNDECIDED, dtype=object)
        else:
            pred_routing = pred
        yte = Y[s].loc[te_ids].to_numpy()
        ans_met = M.class_metrics(yte[answered], np.asarray(pred, dtype=object)[answered], classes) if answered.sum() else None
        ci = M.bootstrap_ci(yte[answered], np.asarray(pred, dtype=object)[answered], classes, nboot, rng_boot) if answered.sum() else None
        te_out[s] = dict(pred=pred, pred_routing=pred_routing, answered=answered, proba=proba, classes=classes)
        stages_report[s]["removed"] = removed
        stages_report[s]["test"] = dict(threshold=thresholds[s]["threshold"], threshold_reachable=thresholds[s]["reachable"],
                                        abstention_rate=float(1 - answered.mean()), answered_rate=float(answered.mean()),
                                        answered=dict(**(ans_met or {}), balanced_accuracy_ci=ci))
        if verbose and answered.sum():
            print(f"[測試] Stage {s} {zh}{'（已移除，僅供參考）' if removed else ''}：拒答率 {1-answered.mean():.3f}；"
                  f"作答者平衡正確率 {ans_met['balanced_accuracy']:.3f} [{ci['lo']:.3f},{ci['hi']:.3f}]")

    # ── Stage 4 分流格（測試集）
    yb_te = y_box_all.loc[te_ids].to_numpy()
    box_pred = BX.route(te_out["1"]["pred_routing"], te_out["2"]["pred_routing"], te_out["3"]["pred_routing"])
    decided = box_pred != BX.UNDECIDED
    box_classes = np.array(sorted(set(map(str, yb_te))))
    box_met = M.class_metrics(yb_te[decided], box_pred[decided], box_classes) if decided.any() else None
    box_ci = M.bootstrap_ci(yb_te[decided], box_pred[decided], box_classes, nboot, rng_boot) if decided.any() else None

    # ── Stage 5 建議與評估
    recs = BX.recommend(P, box_pred)
    stage5 = BX.evaluate_recommendations(P, yb_te, box_pred, recs)
    if (box_pred == "R").any():                               # R 格硬性規則之獨立驗算（由 box_pred 與 recs 重算，非建構時斷言）
        assert stage5["R_five_items_when_routed_R"] == 1.0, "R 格硬性規則遭覆蓋——凡判為 R 者五項必須全開（建置失敗）"
    if verbose:
        print(f"[Stage 4] 判定率 {decided.mean():.3f}；作答者平衡正確率 {box_met['balanced_accuracy']:.3f} [{box_ci['lo']:.3f},{box_ci['hi']:.3f}]")
        print(f"[Stage 5] 檢驗節省率 {stage5['test_saving_rate_decided']:.3f}；關鍵遺漏率(全) {stage5['critical_omission_rate_all']:.3f}；"
              f"R 格遺漏 {stage5['R_critical_omission_rate']:.3f}（真 R n={stage5['R_true_n']}，召回 {stage5['R_recall']:.3f}）；命中率 {stage5['hit_rate']:.3f}")

    # ── 指示 十一節 assert（其餘六條散在流程中，這裡補齊並統整）
    frames = {k: X_all[cols] for k, cols in used_cols.items()}    # 實際訓練用到的每一個 Stage×層（稽核修正）
    CK.check_archive_absent(P, frames, long_df, pat_df)
    CK.check_no_per_patient_scaling(P, long_df, X_all, fparams, ids, rng=module_rng(seed, "cv"))
    CK.check_stage_features_L1(P, used_cols)
    CK.check_m0_sealed_before(ledger)                             # 事後驗序：sealed_at 必早於第一次主模型效能計算

    # ── 特徵重要度（供「這項推測由哪些指標支持」）
    imp_rows = []
    for s, fm in final_models.items():
        lr = fm["model"].named_steps["lr"]
        for ci_, cls in enumerate(lr.classes_):
            for c, w in zip(fm["cols"], lr.coef_[ci_ if lr.coef_.shape[0] > 1 else 0]):
                imp_rows.append(dict(stage=s, stage_name=STAGES[s][1], klass=str(cls), feature=c, coef=float(w)))
            if lr.coef_.shape[0] == 1:
                break
    pd.DataFrame(imp_rows).to_csv(os.path.join(RESULTS, "feature_importance.csv"), index=False, encoding="utf-8-sig")

    # 優先比值單獨表現（指示 三節：報告中單獨列出）
    ratio_solo = {}
    for a, b in value(P, "priority_ratios"):
        col = f"R_{a}_{b}"
        for s in STAGES:
            ytr = Y[s].loc[tr_ids].to_numpy()
            Xr = X_all.loc[tr_ids, [col]].to_numpy()
            proba, classes = M.cv_proba(Xr, ytr, folds, seed, ledger=ledger)
            ratio_solo.setdefault(col, {})[s] = float(M.balanced_accuracy_score(ytr, classes[proba.argmax(axis=1)]))

    # 形態可得 vs 不可得子群（指示 七節）
    morph_sub = {}
    if fs["proxy_mode"]:
        mv = morph_measured.loc[te_ids].to_numpy()
        for gname, gmask in (("morph_available", mv), ("morph_missing", ~mv)):
            m = gmask & decided
            morph_sub[gname] = dict(n=int(gmask.sum()),
                                    box_balanced_accuracy=float(M.balanced_accuracy_score(yb_te[m], box_pred[m])) if m.sum() and len(set(yb_te[m])) > 1 else None)

    # ── 模型參數匯出（供離線 UI 個案分流：與 Python 完全同一組參數）
    models_doc = dict(stages={}, thresholds={s: thresholds[s]["threshold"] for s in STAGES},
                      stage_names={s: STAGES[s][1] for s in STAGES},
                      fparams=fparams, feature_cols=sorted({c for st in fs["stages"].values() for c in st["M2"]}),
                      index_window=int(value(P, "method.index_window_days")),
                      slope_min=int(value(P, "method.slope_min_measurements")),
                      recommendations=value(P, "stage5_recommendations"),
                      demo_cases=[])
    for s, fm in final_models.items():
        imp, sc_, lr = fm["model"].named_steps["imp"], fm["model"].named_steps["sc"], fm["model"].named_steps["lr"]
        models_doc["stages"][s] = dict(cols=fm["cols"], impute_median=[float(v) for v in imp.statistics_],
                                       scale_mean=[float(v) for v in sc_.mean_], scale_sd=[float(v) for v in sc_.scale_],
                                       coef=[[float(w) for w in row] for row in lr.coef_], intercept=[float(b) for b in lr.intercept_],
                                       classes=[str(c) for c in lr.classes_])
    seen = set()
    for pid in te_ids:
        b = str(y_box_all.loc[pid])
        if b in seen:
            continue
        seen.add(b)
        rows_p = long_df[long_df["patient_id"] == pid].sort_values("day_from_index")
        panels = [dict(day=int(r["day_from_index"]), **{c: (None if not np.isfinite(r[c]) else round(float(r[c]), 3)) for c in L1})
                  for _, r in rows_p.iterrows()][:4]
        pr = pat_df.set_index("patient_id").loc[pid]
        models_doc["demo_cases"].append(dict(box=b, acute=int(pr["y_acute"]), site=str(pr["y_site"]), pheno=str(pr["y_pheno"]), panels=panels))
    _dump(models_doc, "models.json")

    report = dict(seed=seed, n=int(len(ids)), stage0=stage0, feature_sets={k: v for k, v in fs.items() if k != "ladder"},
                  placeholders=chk["placeholders"], m0_sealed_at=m0_doc["sealed_at"], removed_axes=removed_axes,
                  stages={s: {k: v for k, v in r.items() if k != "ladder"} for s, r in stages_report.items()},
                  stage4=dict(decided_rate=float(decided.mean()), answered=box_met, balanced_accuracy_ci=box_ci),
                  stage5=stage5, ratio_solo_balanced_accuracy=ratio_solo, morphology_subgroups=morph_sub,
                  test_n=int(len(te_ids)), train_n=int(len(tr_ids)))
    CK.check_report_completeness(report)
    _dump(report, "report.json")
    return report, dict(P=P, X_all=X_all, fparams=fparams, final_models=final_models, thresholds=thresholds,
                        pat_df=pat_df, long_df=long_df, te_ids=te_ids, tr_ids=tr_ids, te_out=te_out,
                        box_pred=box_pred, recs=recs, fs=fs, y_box_all=y_box_all)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    P0 = PIO.load()
    seed = a.seed if a.seed is not None else int(value(P0, "method.seed"))
    n = a.n if a.n is not None else (800 if a.quick else int(value(P0, "method.n_patients")))
    run(seed, n, quick=a.quick)
    print("[完成] results/report.json（report.md 由 make_report.py 產生）")
