# -*- coding: utf-8 -*-
"""三個病因模型的結合實驗（NHANES 1999–2018）。
    python combine_tasks.py [--seed 20260830]

## 事前登錄（寫在跑之前，不得於見到結果後修改）

**假說 H**：三個病因模型的輸出彼此含有資訊，結合後判別力提升，可達 AUROC ≥ 0.85。

**反假說 H0（本檔作者的預期）**：增益 ≈ 0。三個模型讀的是**同一批 60 欄特徵**，
結合只是同一份資訊的非線性重組，不是新的資訊通道。真正能加資訊的是新檢驗項目。

**成功判準（事前釘死）**：`leak_free` 特徵集下 AUROC ≥ 0.850 **且** bootstrap CI 下界 > 0.800。
未達成則照實報告，**不調參、不換特徵集、不改判準**。

## 四個結合方式

| 代號 | 做法 | 為什麼合理 |
|---|---|---|
| C1 | T3 感染 vs 其餘 ＋「代謝機率」當額外特徵 | 看起來像代謝的人，感染先驗較低 |
| C2 | T4 代謝 vs 其餘 ＋「感染機率」當額外特徵 | 反向同理 |
| C3 | 新任務：**任一可辨識病因**（感染 or 代謝）vs 無 | 陽性 3,282（非 169），樣本穩定性大幅改善 |
| C4 | C1＋C2 雙向堆疊後的平均效能 | 綜合判定是否達 0.85 |

## 不做什麼，以及為什麼

* **T2（感染 vs 代謝）不做堆疊**——該任務的兩個類別就是那兩個標籤，
  把「代謝機率」當特徵等於把結果當輸入，**循環論證**。
* **三類同時判別不重跑**——那正是 `pipeline.py` 的 Level 1，已證明在 NHANES 上
  結構性無解（用全體則免疫標籤污染、用 ANA 次樣本則感染僅 n=7）。重跑不會有新資訊。
* **免疫不納入結合**——擴充世代（2005 年後）沒有 ANA，那些人的免疫狀態是
  **未知而非陰性**。硬納入就是先前已修掉的標籤污染。

## 洩漏防範

堆疊特徵一律使用 **out-of-fold** 機率：預測病人 i 的輔助機率時，
產生該機率的模型未看過病人 i。仍存在標準堆疊的殘餘風險（產生 OOF 的模型
看過了後續會落在訓練折的其他病人），故 **C1/C2 的增益若小於 0.01 一律視為雜訊**。"""
import argparse
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

import numpy as np                                      # noqa: E402
from sklearn.model_selection import RepeatedStratifiedKFold   # noqa: E402

from nhanes_cohort import build_extended                # noqa: E402
import pipeline as PL                                   # noqa: E402
from binary_tasks import LABEL_ADJACENT, nested_binary  # noqa: E402

RESULTS = os.path.join(ROOT, "results")

TARGET_AUROC = 0.850          # 事前判準，不得修改
TARGET_CI_LOWER = 0.800
NOISE_FLOOR = 0.010           # 增益低於此值視為雜訊


def oof_prob(X, y, seed, model_key="HGB", repeats=3, folds=5):
    """重複分層 K 折之 out-of-fold 機率——預測病人 i 時模型未看過病人 i。"""
    X, y = np.asarray(X, float), np.asarray(y).astype(int)
    acc, cnt = np.zeros(len(y)), np.zeros(len(y))
    for tr, te in RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats,
                                          random_state=seed).split(X, y):
        m = PL.models(seed)[model_key].fit(X[tr], y[tr])
        acc[te] += m.predict_proba(X[te])[:, 1]
        cnt[te] += 1
    return acc / np.maximum(cnt, 1)


def run(seed=20260830, verbose=True):
    P = json.load(open(os.path.join(ROOT, "params", "design.json"), encoding="utf-8"))
    E = build_extended(P, verbose=verbose)
    kd, feats = E["cohort"], E["features"]

    inf = kd["lab_infection"].fillna(False).astype(bool).to_numpy()
    met = kd["lab_metabolic"].fillna(False).astype(bool).to_numpy()
    # leak_free：兩組標籤鄰近特徵都拔（結合任務同時涉及兩個標籤）
    adj = set(LABEL_ADJACENT["infection"]) | set(LABEL_ADJACENT["metabolic"])
    lf = [f for f in feats if f not in adj]
    X = kd[lf].to_numpy(float)
    if verbose:
        print(f"\n[結合實驗] n={len(kd)}｜特徵 {len(lf)} 欄（已拔除兩組標籤鄰近共 {len(adj)} 欄）")
        print(f"           感染 {inf.sum()}｜代謝 {met.sum()}｜任一 {(inf | met).sum()}")

    out = dict(seed=seed, created=time.strftime("%Y-%m-%dT%H:%M:%S"),
               cycles="NHANES 1999–2018（10 週期）", n=len(kd), n_features=len(lf),
               prereg=dict(target_auroc=TARGET_AUROC, target_ci_lower=TARGET_CI_LOWER,
                           noise_floor=NOISE_FLOOR,
                           hypothesis="三模型輸出互含資訊，結合後 AUROC ≥ 0.85",
                           author_expectation="增益≈0——三者讀同一批特徵，結合非新資訊通道"),
               baselines={}, combos={})

    # ── 基線（同特徵集，無堆疊）
    if verbose:
        print("\n── 基線（無堆疊，同一 leak_free 特徵集）")
    for tag, y in (("感染vs其餘", inf), ("代謝vs其餘", met)):
        if verbose:
            print(f"  [{tag}]")
        out["baselines"][tag] = {mk: nested_binary(X, y.astype(int), seed, mk, verbose=verbose)
                                 for mk in ("LR", "HGB")}

    # ── C1／C2：以對方的 OOF 機率當額外特徵
    p_met = oof_prob(X, met, seed)
    p_inf = oof_prob(X, inf, seed)
    for tag, y, extra, ename in (("C1_感染＋代謝機率", inf, p_met, "P(代謝)"),
                                 ("C2_代謝＋感染機率", met, p_inf, "P(感染)")):
        Xs = np.column_stack([X, extra])
        if verbose:
            print(f"\n── {tag}（額外特徵：{ename}）")
        out["combos"][tag] = {mk: nested_binary(Xs, y.astype(int), seed, mk, verbose=verbose)
                              for mk in ("LR", "HGB")}
        out["combos"][tag]["extra_feature"] = ename

    # ── C3：任一可辨識病因
    any_cause = (inf | met).astype(int)
    if verbose:
        print("\n── C3_任一可辨識病因（感染 or 代謝）vs 無")
    out["combos"]["C3_任一可辨識病因"] = {
        mk: nested_binary(X, any_cause, seed, mk, verbose=verbose) for mk in ("LR", "HGB")}

    # ── 事前判準判定（純程式比對，不容事後詮釋）
    verdict = {}
    for name, mm in out["combos"].items():
        best = max((r for k, r in mm.items() if isinstance(r, dict) and "auroc" in r),
                   key=lambda r: r["auroc"])
        base = out["baselines"].get("感染vs其餘" if "感染" in name else "代謝vs其餘")
        gain = None
        if base and not name.startswith("C3"):
            gain = best["auroc"] - max(r["auroc"] for r in base.values())
        verdict[name] = dict(
            best_model=best["model"], auroc=best["auroc"], ci=best["auroc_ci"],
            gain_vs_baseline=gain,
            gain_is_noise=(abs(gain) < NOISE_FLOOR) if gain is not None else None,
            meets_target=bool(best["auroc"] >= TARGET_AUROC and best["auroc_ci"][0] > TARGET_CI_LOWER))
    out["verdict"] = verdict
    out["conclusion"] = ("達成事前判準" if any(v["meets_target"] for v in verdict.values())
                         else "未達成事前判準（AUROC≥0.850 且 CI 下界>0.800）")

    PL._dump(out, "combine_tasks.json")
    with open(os.path.join(RESULTS, "runs_log.jsonl"), "a", encoding="utf-8") as f:
        for grp in ("baselines", "combos"):
            for name, mm in out[grp].items():
                for mk, r in mm.items():
                    if isinstance(r, dict) and "auroc" in r:
                        f.write(json.dumps(dict(kind="combine_task", group=grp, task=name,
                                                at=out["created"], **r), ensure_ascii=False) + "\n")
    if verbose:
        print("\n" + "═" * 72)
        for name, v in verdict.items():
            g = "—" if v["gain_vs_baseline"] is None else f"{v['gain_vs_baseline']:+.3f}"
            noise = "（雜訊層級）" if v["gain_is_noise"] else ""
            mark = "✅ 達標" if v["meets_target"] else "❌ 未達"
            print(f"{mark}  {name:22s} AUROC {v['auroc']:.3f} "
                  f"[{v['ci'][0]:.3f},{v['ci'][1]:.3f}]  增益 {g}{noise}")
        print(f"\n事前判準（AUROC≥{TARGET_AUROC} 且 CI 下界>{TARGET_CI_LOWER}）：{out['conclusion']}")
        print("═" * 72)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260830)
    run(ap.parse_args().seed)
