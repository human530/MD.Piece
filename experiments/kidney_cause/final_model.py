# -*- coding: utf-8 -*-
"""最終模型：腎損傷者的感染性病因分診。
    python final_model.py [--seed 20260830]

## 這個模型做什麼

輸入常規血液／尿液檢驗 → 輸出**是否建議開立肝炎血清學檢驗**。

三種輸出，不是二分：`建議檢驗` ／ `不建議` ／ **`無法判定`**（拒答）。
拒答是設計的一部分——模型在把握不足時明說，而非硬猜。
報告時**拒答率與作答者正確率一律並列**，只報後者是誤導。

## 為什麼只做這一件事

本專案測過三個病因軸，只有這一軸站得住：

| 軸 | 結論 | 依據 |
|---|---|---|
| 感染 | ✅ 可用 | AUROC 0.79–0.82；AUPRC 為盛行率基準的 6.7 倍；十週期外測全部 >0.67 |
| 代謝 | ⚠️ 不納入 | 0.816 但無增益價值——臨床上本來就知道誰有糖尿病 |
| 免疫 | ❌ 不可用 | 0.575；60 個標記單標記掃描僅「性別」過 FDR（q=0.004），無生理訊號 |

## 事前登錄（寫在跑之前）

* **模型**：邏輯迴歸（LR）。理由不是它分數高，而是 `TRAINING_SUMMARY.md` §5.1 已證明
  HGB 在本任務的優勢（小世代 +0.060）在樣本擴大後歸零（−0.002）＝過擬合產物。
  同分時取較簡單、係數可讀、可攜出的那個。HGB 一併報告以供對照。
* **校準**：`class_weight="balanced"` 會使輸出機率嚴重高估（LR 的 Brier 0.149 遠高於
  盛行率 0.019）。故最終模型加保序回歸（isotonic）校準，**只在內層折上配適**。
* **作業點**：篩檢用途，事前指定**作答者敏感度 ≥ 0.80**——漏掉一例感染的代價
  高於多開一張肝炎血清學單（該檢驗便宜、無創、無風險）。
* **拒答帶**：雙門檻只在**相異值**上搜尋（防保序回歸並列值造成退化區間），
  且以**分半法樣本外驗證**——A 折選、B 折驗，兩邊都滿足條件才算可行解。
* **成功判準（四項全過才算成立）**：作答者敏感度 ≥ 0.80、**特異度 ≥ 0.50**、
  拒答率 ≤ 30%、保留集 AUROC 之 CI 下界 > 0.70。
  未達成則照實報告，**不放寬條件、不退回單門檻、不調參**。
  找不到可行作業點時直接輸出 `status="INFEASIBLE"`——這是有效的研究結論，不是失敗。

> v1 曾以較寬的條件（無特異度下限、分位數選門檻、未做樣本外驗證）通過 2/3 項，
> 但保留集特異度僅 0.241（標記幾乎所有人）、拒答率 57.3%。該結果已證實為
> 實作缺陷所致，修正紀錄見下方程式碼註解。

## 關於保留集的誠實聲明

`results/holdout_eval.json` 記錄的既有保留集是為**三類問題**建的
（`test_lab.py` 以 `cause` 三分類擬合），且僅涵蓋 1999–2004。該問題已因
結構性不可解而作廢，故其保留集對本模型不適用，本檔另立保留集。

**但必須說明其限制**：1999–2018 的資料在本專案中已被反覆分析，
現在才切出的保留集**不是未曾窺視的測試集**。真正接近外部驗證的證據是
**留一週期外測**（`binary_tasks_extended.py`），本檔一併重報該結果，
並以它而非保留集作為主要的類化證據。"""
import argparse
import hashlib
import json
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np                                              # noqa: E402
from sklearn.isotonic import IsotonicRegression                 # noqa: E402
from sklearn.impute import SimpleImputer                        # noqa: E402
from sklearn.linear_model import LogisticRegression             # noqa: E402
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,   # noqa: E402
                             brier_score_loss, roc_auc_score)
from sklearn.model_selection import StratifiedKFold                             # noqa: E402
from sklearn.preprocessing import StandardScaler                # noqa: E402

from nhanes_cohort import build_extended                        # noqa: E402
import pipeline as PL                                           # noqa: E402
from binary_tasks import LABEL_ADJACENT                         # noqa: E402

RESULTS = os.path.join(ROOT, "results")

# ── 事前登錄常數，不得於見到結果後修改
MIN_SENSITIVITY = 0.80        # 作答者敏感度下限（篩檢用途）
MIN_SPECIFICITY = 0.50        # 作業者特異度下限——v2 新增，見下
MAX_ABSTAIN = 0.30            # 拒答率上限
MIN_HOLDOUT_CI_LOWER = 0.70   # 保留集 AUROC 之 CI 下界
MIN_BAND_MASS_OUTSIDE = 0.70  # 帶外樣本比例下限（＝拒答率上限的另一種表述，用於並列值防呆）
HOLDOUT_FRAC = 0.25
N_DEPLOY_FOLDS = 5            # 部署用交叉配適集成的折數

# ── v2 修正紀錄（2026-08-31）
# v1 的作業點無法轉移：開發集拒答 24.9% → 保留集 57.3%，特異度 0.451 → 0.241。
# 三個實作缺陷，逐一修正：
#   ① 並列值：保序回歸輸出階梯函數，大量樣本共用同一數值。v1 以分位數選門檻，
#      選出帶寬 0.001 的退化區間，保留集 57% 樣本恰落在單一並列值上全部被拒答。
#      → v2 改為只在**相異值**上搜尋，並強制帶外樣本 ≥70%。
#   ② 分數分布不匹配：v1 選門檻用 OOF 機率，部署卻用全開發集重配適的模型，
#      後者分數系統性更極端，套用同一門檻即失準。
#      → v2 部署端改用**交叉配適集成**（5 折模型取平均），與 OOF 同分布。
#   ③ 門檻在選它的那份資料上驗證：v1 未做樣本外確認。
#      → v2 以分半法（A 折選、B 折驗）只保留兩邊都滿足條件者。
# 另新增特異度下限 0.50——v1 保留集特異度 0.241 等於標記幾乎所有人，
# 這種規則即使敏感度 1.000 也沒有臨床意義，故列為硬條件。


def _fit(Xtr, ytr, seed):
    """插補→標準化→加權 LR。回傳 (predict_fn, 可攜出參數)。"""
    imp = SimpleImputer(strategy="median").fit(Xtr)
    sc = StandardScaler().fit(imp.transform(Xtr))
    lr = LogisticRegression(class_weight="balanced", max_iter=4000,
                            random_state=seed).fit(sc.transform(imp.transform(Xtr)), ytr)
    return (lambda X: lr.predict_proba(sc.transform(imp.transform(X)))[:, 1],
            dict(medians=imp.statistics_.tolist(), mean=sc.mean_.tolist(),
                 scale=sc.scale_.tolist(), coef=lr.coef_[0].tolist(),
                 intercept=float(lr.intercept_[0])))


def _band_metrics(p, y, lo, hi):
    """給定雙門檻，回傳 (拒答率, 敏感度, 特異度, 平衡正確率)；無法計算時回 None。"""
    ans = (p <= lo) | (p >= hi)
    if ans.sum() < 30:
        return None
    ya, pa = y[ans], (p[ans] >= hi).astype(int)
    if ya.sum() == 0 or (1 - ya).sum() == 0:
        return None
    return (float(1 - ans.mean()), float(pa[ya == 1].mean()), float(1 - pa[ya == 0].mean()),
            float(balanced_accuracy_score(ya, pa)))


def pick_bands(p, y, seed, min_sens=MIN_SENSITIVITY, min_spec=MIN_SPECIFICITY,
               max_abstain=MAX_ABSTAIN):
    """雙門檻選擇——v2：只搜相異值（防並列退化）、分半驗證（防選它的那份資料上才成立）。

    在 A 折上搜尋滿足全部條件者，逐一拿到 B 折驗證；只保留**兩邊都滿足**的候選，
    取其中 B 折平衡正確率最高者。回傳 (t_low, t_high, 診斷)。"""
    uniq = np.unique(p)
    grid = uniq if len(uniq) <= 80 else uniq[np.linspace(0, len(uniq) - 1, 80).astype(int)]
    folds = list(StratifiedKFold(2, shuffle=True, random_state=seed).split(p.reshape(-1, 1), y))
    tried, feasible_both = 0, []
    for ia, ib in folds:                                   # 兩個方向都做（A選B驗、B選A驗）
        pa_, ya_, pb_, yb_ = p[ia], y[ia], p[ib], y[ib]
        for i, lo in enumerate(grid):
            for hi in grid[i:]:
                tried += 1
                ma = _band_metrics(pa_, ya_, lo, hi)
                if ma is None:
                    continue
                ab_a, se_a, sp_a, _ = ma
                if ab_a > max_abstain or se_a < min_sens or sp_a < min_spec:
                    continue
                mb = _band_metrics(pb_, yb_, lo, hi)       # 樣本外確認
                if mb is None:
                    continue
                ab_b, se_b, sp_b, bacc_b = mb
                if ab_b > max_abstain or se_b < min_sens or sp_b < min_spec:
                    continue
                feasible_both.append((bacc_b, float(lo), float(hi), ab_b, se_b, sp_b))
    if not feasible_both:
        return None, None, dict(feasible=False, n_tried=tried,
                                note="全部條件（敏感度≥%.2f、特異度≥%.2f、拒答≤%.0f%%）"
                                     "在分半驗證下無可行雙門檻——不退回單門檻，照實回報失敗"
                                     % (min_sens, min_spec, max_abstain * 100))
    feasible_both.sort(reverse=True)
    bacc, lo, hi, ab, se, sp = feasible_both[0]
    return lo, hi, dict(feasible=True, n_tried=tried, n_feasible=len(feasible_both),
                        holdin_verified=dict(abstain_rate=ab, sensitivity=se, specificity=sp,
                                             balanced_accuracy=bacc),
                        band_mass_outside=float(((p <= lo) | (p >= hi)).mean()))


def evaluate(y, p, t_low, t_high):
    """含拒答的評估：拒答率與作答者指標並列。"""
    ans = (p <= t_low) | (p >= t_high)
    ya, pa = y[ans], (p[ans] >= t_high).astype(int)
    out = dict(n=int(len(y)), abstain_rate=float(1 - ans.mean()), n_answered=int(ans.sum()),
               auroc_all=float(roc_auc_score(y, p)), auprc_all=float(average_precision_score(y, p)),
               auprc_baseline=float(y.mean()), brier=float(brier_score_loss(y, p)))
    if ya.sum() and (1 - ya).sum():
        out.update(answered_sensitivity=float(pa[ya == 1].mean()),
                   answered_specificity=float(1 - pa[ya == 0].mean()),
                   answered_balanced_accuracy=float(balanced_accuracy_score(ya, pa)),
                   answered_ppv=float(ya[pa == 1].mean()) if pa.sum() else None,
                   answered_npv=float(1 - ya[pa == 0].mean()) if (1 - pa).sum() else None)
    return out


def run(seed=20260830, verbose=True):
    P = json.load(open(os.path.join(ROOT, "params", "design.json"), encoding="utf-8"))
    E = build_extended(P, verbose=verbose)
    kd, feats = E["cohort"], E["features"]
    adj = set(LABEL_ADJACENT["infection"])
    ff = [f for f in feats if f not in adj]
    y = kd["lab_infection"].fillna(False).astype(bool).astype(int).to_numpy()
    X = kd[ff].to_numpy(float)
    seqn = kd["SEQN"].to_numpy()

    if verbose:
        print(f"\n[最終模型] n={len(y)}｜陽性 {y.sum()}（盛行率 {y.mean():.4f}）｜特徵 {len(ff)} 欄")
        print(f"           已拔除標籤鄰近 {sorted(adj)}")

    # ── 保留集（新立；理由與限制見檔頭）
    hp = os.path.join(RESULTS, "final_holdout_seqn.json")
    if os.path.exists(hp):
        doc = json.load(open(hp, encoding="utf-8"))
        assert hashlib.sha256(json.dumps(sorted(doc["seqn"])).encode()).hexdigest()[:16] == doc["hash"], \
            "保留集檔案被改動——拒絕繼續"
        hold = set(doc["seqn"])
    else:
        rng = np.random.default_rng(seed + 4242)
        hold = set()
        for cls in (0, 1):                               # 依結果分層
            ids = seqn[y == cls]
            hold |= set(int(v) for v in rng.choice(ids, size=int(round(HOLDOUT_FRAC * len(ids))),
                                                   replace=False))
        hold = sorted(hold)
        json.dump(dict(note="最終模型專用保留集；開發階段永不觸碰",
                       caveat="1999–2018 已於本專案反覆分析，此非未曾窺視之測試集；"
                              "主要類化證據為留一週期外測",
                       frac=HOLDOUT_FRAC, seed=seed + 4242, seqn=hold,
                       hash=hashlib.sha256(json.dumps(sorted(hold)).encode()).hexdigest()[:16],
                       created=time.strftime("%Y-%m-%dT%H:%M:%S")),
                  open(hp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        hold = set(hold)
    te_mask = np.isin(seqn, list(hold))
    Xd, yd, Xt, yt = X[~te_mask], y[~te_mask], X[te_mask], y[te_mask]
    if verbose:
        print(f"[分割] 開發 {len(yd)}（陽性 {yd.sum()}）／保留 {len(yt)}（陽性 {yt.sum()}）")

    # ── 部署用交叉配適集成：OOF 分數與部署分數同分布（v2 修正 ②）
    if verbose:
        print(f"\n[訓練] 開發集 {N_DEPLOY_FOLDS} 折交叉配適集成（部署端用同一批折模型取平均）")
    oof_raw = np.zeros(len(yd))
    fold_models, fold_params = [], []
    for tr, va in StratifiedKFold(N_DEPLOY_FOLDS, shuffle=True, random_state=seed).split(Xd, yd):
        f, pr = _fit(Xd[tr], yd[tr], seed)
        oof_raw[va] = f(Xd[va])
        fold_models.append(f)
        fold_params.append(pr)

    if verbose:
        print("[校準] 保序回歸（內層 5 折配適，不用同一筆資料既校準又評估）")
    oof_cal = np.zeros(len(yd))
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(oof_raw.reshape(-1, 1), yd):
        iso = IsotonicRegression(out_of_bounds="clip").fit(oof_raw[tr], yd[tr])
        oof_cal[va] = iso.predict(oof_raw[va])

    t_low, t_high, band_info = pick_bands(oof_cal, yd, seed)
    if not band_info.get("feasible"):
        if verbose:
            print(f"\n❌ [作業點] {band_info['note']}")
            print(f"   已試 {band_info['n_tried']} 組門檻組合。**不放寬條件、不退回單門檻。**")
        PL._dump(dict(version="v2", status="INFEASIBLE", created=time.strftime("%Y-%m-%dT%H:%M:%S"),
                      prereg=dict(min_sensitivity=MIN_SENSITIVITY, min_specificity=MIN_SPECIFICITY,
                                  max_abstain=MAX_ABSTAIN),
                      band_search=band_info,
                      conclusion="在全部事前條件下無可行作業點——模型的判別力不足以同時滿足"
                                 "敏感度、特異度與拒答率三項要求。照實回報，不放寬條件。"),
                     "final_model.json")
        return dict(status="INFEASIBLE", band_search=band_info)
    if verbose:
        hv = band_info["holdin_verified"]
        print(f"[作業點] 拒答帶 [{t_low:.5f}, {t_high:.5f})　可行解 {band_info['n_feasible']} 組"
              f"（已通過分半樣本外驗證）")
        print(f"          驗證折：拒答 {hv['abstain_rate']:.1%}｜敏感度 {hv['sensitivity']:.3f}"
              f"｜特異度 {hv['specificity']:.3f}")

    dev_metrics = evaluate(yd, oof_cal, t_low, t_high)
    if verbose:
        print(f"[開發集] AUROC {dev_metrics['auroc_all']:.3f}｜校準後 Brier {dev_metrics['brier']:.4f}"
              f"｜拒答 {dev_metrics['abstain_rate']:.1%}"
              f"｜作答者敏感度 {dev_metrics.get('answered_sensitivity', float('nan')):.3f}"
              f"／特異度 {dev_metrics.get('answered_specificity', float('nan')):.3f}")

    # ── 保留集：以「折模型平均」評分，與 OOF 同分布（v2 修正 ②）
    iso_full = IsotonicRegression(out_of_bounds="clip").fit(oof_raw, yd)
    p_te = iso_full.predict(np.mean([f(Xt) for f in fold_models], axis=0))
    params = dict(ensemble_folds=fold_params, n_folds=N_DEPLOY_FOLDS,
                  isotonic_x=iso_full.X_thresholds_.tolist(),
                  isotonic_y=iso_full.y_thresholds_.tolist(),
                  usage="對新樣本：各折模型各算一次機率取平均 → 以 isotonic_x/y 線性內插校準 → 比對拒答帶")
    rng = np.random.default_rng(seed)
    boots = [roc_auc_score(yt[b], p_te[b]) for b in
             (rng.integers(0, len(yt), len(yt)) for _ in range(1000)) if 0 < yt[b].sum() < len(b)]
    ho = evaluate(yt, p_te, t_low, t_high)
    ho["auroc_ci"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    if verbose:
        print(f"\n[保留集｜一次性] AUROC {ho['auroc_all']:.3f} "
              f"[{ho['auroc_ci'][0]:.3f}, {ho['auroc_ci'][1]:.3f}]"
              f"｜AUPRC {ho['auprc_all']:.3f}（基準 {ho['auprc_baseline']:.4f}）")
        print(f"                拒答 {ho['abstain_rate']:.1%}"
              f"｜作答者敏感度 {ho.get('answered_sensitivity', float('nan')):.3f}"
              f"／特異度 {ho.get('answered_specificity', float('nan')):.3f}"
              f"｜PPV {ho.get('answered_ppv') if ho.get('answered_ppv') is None else round(ho['answered_ppv'], 3)}")

    # ── 事前判準判定（程式比對，不容事後詮釋）
    verdict = dict(
        sensitivity_ok=bool(ho.get("answered_sensitivity", 0) >= MIN_SENSITIVITY),
        specificity_ok=bool(ho.get("answered_specificity", 0) >= MIN_SPECIFICITY),
        abstain_ok=bool(ho["abstain_rate"] <= MAX_ABSTAIN),
        holdout_ci_ok=bool(ho["auroc_ci"][0] > MIN_HOLDOUT_CI_LOWER))
    verdict["all_met"] = bool(all(verdict.values()))

    # ── 係數（可讀性；標準化尺度）
    mean_coef = np.mean([fp["coef"] for fp in fold_params], axis=0)
    coefs = sorted(zip(ff, mean_coef.tolist()), key=lambda t: -abs(t[1]))[:15]

    out = dict(
        model="腎損傷者感染性病因分診（LR 交叉配適集成＋保序校準＋拒答帶）", version="v2",
        seed=seed,
        holdout_evaluation_number=2,
        holdout_note="**這是同一保留集的第二次評估**。v1 因作業點實作缺陷失敗（並列值退化、分數分布不匹配、門檻未經樣本外驗證），v2 修正後重評。第二次評估不具「一生一次」的統計保證，須計入多重比較。",
        created=time.strftime("%Y-%m-%dT%H:%M:%S"),
        cohort=dict(source="NHANES 1999–2018（10 週期，131 檔真實檔案）", n=int(len(y)),
                    n_pos=int(y.sum()), prevalence=float(y.mean()),
                    n_dev=int(len(yd)), n_holdout=int(len(yt))),
        features=ff, n_features=len(ff), label_adjacent_removed=sorted(adj),
        prereg=dict(min_sensitivity=MIN_SENSITIVITY, min_specificity=MIN_SPECIFICITY,
                    max_abstain=MAX_ABSTAIN,
                    min_holdout_ci_lower=MIN_HOLDOUT_CI_LOWER,
                    model_choice_reason="HGB 於本任務的優勢經證實為過擬合（見 TRAINING_SUMMARY §5.1）"),
        operating_point=dict(t_low=t_low, t_high=t_high, **band_info),
        development=dev_metrics, holdout=ho, verdict=verdict,
        top_coefficients=[dict(feature=f, coef_standardized=round(c, 4)) for f, c in coefs],
        portable_params=params,
        holdout_caveat="1999–2018 已於本專案反覆分析，此保留集非未曾窺視之測試集；"
                       "主要類化證據應以留一週期外測為準（binary_tasks_extended.json）",
        not_included=dict(
            免疫="不納入——AUROC 0.575，60 標記單標記掃描僅『性別』過 FDR（q=0.004），無生理訊號",
            代謝="不納入——0.816 但無增益價值，臨床上已知誰有糖尿病"),
        disclaimers=["標籤為血清學共病代理，非腎切片確診",
                     "輸出為『是否建議開立肝炎血清學』，非診斷",
                     "模型抓的是『此人可能有 B/C 肝』，不等於『肝炎導致腎損傷』",
                     "不構成臨床診斷工具"])
    PL._dump(out, "final_model.json")
    with open(os.path.join(RESULTS, "runs_log.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(kind="FINAL_MODEL_V2", holdout_eval_n=2, at=out["created"], **verdict,
                                holdout_auroc=ho["auroc_all"], holdout_ci=ho["auroc_ci"],
                                abstain=ho["abstain_rate"]), ensure_ascii=False) + "\n")
    if verbose:
        print("\n" + "═" * 70)
        for k, v in verdict.items():
            print(f"  {'✅' if v else '❌'}  {k}")
        print(f"\n事前判準整體：{'✅ 全數達成' if verdict['all_met'] else '❌ 未全數達成——照實報告，不調參'}")
        print("═" * 70)
        print("\n前 8 大係數（標準化尺度）：")
        for f_, c in coefs[:8]:
            print(f"  {f_:12s} {c:+.4f}")
        print("\n[完成] results/final_model.json（含可攜出參數，可於任何語言重建）")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260830)
    run(ap.parse_args().seed)
