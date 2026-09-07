# -*- coding: utf-8 -*-
"""模型階梯、拒答、指標與 bootstrap（指示 六、八、九節）。

模型固定為可解釋線性模型：SimpleImputer(median) → StandardScaler → LogisticRegression(l2, class_weight=balanced)。
（補值器為必要之增補：缺值以 NaN 表示，線性模型無法直接吃 NaN；補值在 CV 折內進行，不外洩。）

M0Ledger 實作「M0 必須先建、先定版」的**程式層強制**（指示 一之四），機制有二：
  (1) 執行期硬擋：cv_proba/fit_full 接收 ledger；非 M0 呼叫發生在 seal() 之前即 raise（main_metric_event）。
  (2) 事後驗序：ledger 記錄第一次主模型效能計算的時戳 first_main_at，
      checks.check_m0_sealed_before 斷言 sealed_at < first_main_at。
（稽核 2026-08-30：舊版僅在 seal() 之後呼叫一次 require_sealed()，結構上不可能失敗——已依發現改為上述雙重機制。）"""
import json
import os
import time

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_model(seed=0):
    # 指示要求 penalty='l2'：sklearn 1.8 起以預設值（即 l2 正則）表達，避免棄用警告，語意不變
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler()),
                     ("lr", LogisticRegression(class_weight="balanced", max_iter=4000, random_state=seed))])


def cv_proba(X, y, folds, seed, ledger=None, m0=False):
    """5 折分層交叉驗證之 out-of-fold 機率。ledger 給定且非 M0 呼叫時，會在 M0 未定版前直接 raise（指示 一之四）。"""
    if ledger is not None and not m0:
        ledger.main_metric_event()
    y = np.asarray(y)
    cv = StratifiedKFold(folds, shuffle=True, random_state=int(seed))
    model = make_model(seed)
    proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")
    classes = np.unique(y)
    return proba, classes


def fit_full(X, y, seed, ledger=None):
    if ledger is not None:
        ledger.main_metric_event()
    m = make_model(seed).fit(X, np.asarray(y))
    return m


# ------------------------------------------------------------------ 指標
def class_metrics(y_true, y_pred, classes):
    """整體正確率、平衡正確率、逐類召回率，以及依本族群實際盛行率換算之 PPV／NPV（指示 九之一、九之二）。"""
    y_true = np.asarray(y_true).astype(str); y_pred = np.asarray(y_pred).astype(str)
    classes = np.asarray(classes).astype(str)
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    per = {}
    for i, c in enumerate(classes):
        pos = y_true == c
        prev = float(pos.mean())
        sens = float((y_pred[pos] == c).mean()) if pos.any() else np.nan
        spec = float((y_pred[~pos] != c).mean()) if (~pos).any() else np.nan
        den_p = sens * prev + (1 - spec) * (1 - prev)
        den_n = (1 - sens) * prev + spec * (1 - prev)
        per[str(c)] = dict(n=int(pos.sum()), prevalence=prev, recall=sens, specificity=spec,
                           ppv=float(sens * prev / den_p) if den_p > 0 else None,
                           npv=float(spec * (1 - prev) / den_n) if den_n > 0 else None)
    return dict(accuracy=float(accuracy_score(y_true, y_pred)),
                balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
                per_class=per, confusion=cm.tolist(), labels=[str(c) for c in classes], n=int(len(y_true)))


def bootstrap_ci(y_true, y_pred, classes, n_boot, rng, stat="balanced_accuracy"):
    y_true = np.asarray(y_true).astype(str); y_pred = np.asarray(y_pred).astype(str); n = len(y_true)
    vals = []
    for _ in range(n_boot):
        b = rng.integers(0, n, n)
        yt, yp = y_true[b], y_pred[b]
        if len(np.unique(yt)) < 2:
            continue
        vals.append(balanced_accuracy_score(yt, yp) if stat == "balanced_accuracy" else accuracy_score(yt, yp))
    v = np.array(vals)
    return dict(mean=float(v.mean()), lo=float(np.percentile(v, 2.5)), hi=float(np.percentile(v, 97.5)), n_boot=int(len(v)))


def paired_diff_ci(y_true, pred_a, pred_b, n_boot, rng):
    """配對 bootstrap：balanced accuracy(A) − balanced accuracy(B)；CI 下界 > 0 即為顯著超越。"""
    y_true = np.asarray(y_true).astype(str); pa = np.asarray(pred_a).astype(str); pb = np.asarray(pred_b).astype(str); n = len(y_true)
    d = []
    for _ in range(n_boot):
        b = rng.integers(0, n, n)
        yt = y_true[b]
        if len(np.unique(yt)) < 2:
            continue
        d.append(balanced_accuracy_score(yt, pa[b]) - balanced_accuracy_score(yt, pb[b]))
    d = np.array(d)
    lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
    return dict(diff=float(d.mean()), lo=lo, hi=hi, significant=bool(lo > 0), n_boot=int(len(d)))


# ------------------------------------------------------------------ 拒答（指示 八節）
def coverage_accuracy_curve(proba, classes, y_true, grid):
    """作答率—正確率曲線（在訓練資料上計算）。"""
    conf = proba.max(axis=1); pred = np.asarray(classes)[proba.argmax(axis=1)]
    y_true = np.asarray(y_true)
    rows = []
    for t in grid:
        ans = conf >= t
        rows.append(dict(threshold=float(t), coverage=float(ans.mean()),
                         balanced_accuracy=float(balanced_accuracy_score(y_true[ans], pred[ans])) if ans.sum() > 0 and len(np.unique(y_true[ans])) > 1 else None,
                         accuracy=float(accuracy_score(y_true[ans], pred[ans])) if ans.sum() > 0 else None,
                         n_answered=int(ans.sum())))
    return rows


def pick_threshold(curve, target, default_start):
    """事前鎖定之規則：取使訓練集『作答者平衡正確率 ≥ target』之最小門檻；不可達則用起點並標記 unreachable。"""
    for row in curve:
        if row["balanced_accuracy"] is not None and row["balanced_accuracy"] >= target:
            return dict(threshold=row["threshold"], reachable=True, train_balanced_accuracy=row["balanced_accuracy"],
                        train_coverage=row["coverage"], rule=f"最小門檻使訓練集作答者平衡正確率 ≥ {target}")
    return dict(threshold=float(default_start), reachable=False, train_balanced_accuracy=curve[0]["balanced_accuracy"],
                train_coverage=curve[0]["coverage"], rule=f"目標 {target} 不可達，改用預設起點 {default_start}（已標記）")


def apply_abstention(proba, classes, threshold):
    conf = proba.max(axis=1)
    pred = np.asarray(classes, dtype=object)[proba.argmax(axis=1)]
    answered = conf >= threshold
    out = np.where(answered, pred, "無法判定")
    return out, answered, conf


# ------------------------------------------------------------------ M0 定版帳本（指示 一之四）
class M0Ledger:
    def __init__(self, path):
        self.path = path
        self.stages = {}
        self.sealed_at = None
        self.first_main_at = None                     # 第一次主模型效能計算的時戳（由 main_metric_event 記錄）

    def main_metric_event(self):
        """主模型效能計算的進入點必須經過這裡：M0 未定版即 raise；已定版則記錄第一次時戳。"""
        if not self.sealed_at:
            raise RuntimeError("違反指示 一之四：主模型效能計算發生在 M0 定版之前")
        if self.first_main_at is None:
            self.first_main_at = time.time()

    def record(self, stage, payload):
        if self.sealed_at:
            raise RuntimeError("M0 已定版，不得再修改")
        self.stages[str(stage)] = payload

    def seal(self, extra=None):
        self.sealed_at = time.time()
        doc = dict(sealed_at_unix=self.sealed_at, sealed_at=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.sealed_at)),
                   note="M0＝只用 ord_*（哪些檢驗被開立）之洩漏基準。本檔於任何主模型效能計算之前寫入並定版。",
                   stages=self.stages, **(extra or {}))
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=1)
        return doc

    def require_sealed(self):
        if not self.sealed_at or not os.path.exists(self.path):
            raise RuntimeError("違反指示 一之四：M0 尚未定版存檔，不得計算或印出任何主模型效能數字")
        if os.path.getmtime(self.path) > time.time() + 1:
            raise RuntimeError("m0_baseline.json 時間戳異常")
        return True
