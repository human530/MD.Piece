# -*- coding: utf-8 -*-
"""Stage 4 分流格與 Stage 5 建議及其評估（指示 五、十節）。

Stage 4：由三階段的判定組合成六格；任一階段拒答 → 「無法判定」，不強行分類、不往下給建議。
Stage 5：依格給建議。**R 格為硬性規則**：凡判為 R 者，五項一律同時開立，不得由模型機率覆蓋（以 assert 保證）。"""
import numpy as np

from datagen import box
from params_io import value

UNDECIDED = "無法判定"


def route(acute_pred, site_pred, pheno_pred):
    """三階段判定 → 分流格；任一階段拒答即停止。"""
    out = []
    for a, s, p in zip(acute_pred, site_pred, pheno_pred):
        if UNDECIDED in (str(a), str(s), str(p)):
            out.append(UNDECIDED)
        else:
            out.append(box(int(a) == 1, str(s), str(p)))
    return np.array(out, dtype=object)


def recommend(P, box_pred):
    """每格之建議清單；R 格五項為硬性規則。"""
    R = value(P, "stage5_recommendations")
    recs = []
    for b in box_pred:
        if b == UNDECIDED:
            recs.append([])                                     # 不給建議，交回醫師
        else:
            recs.append(list(R[str(b)]["mandatory"]))
    # R 格硬性規則的驗證不在此處——這裡斷言自己剛建的清單是套套邏輯（稽核發現）。
    # 獨立驗算：evaluate_recommendations 由 box_pred 與 recs 重新計算 R_five_items_when_routed_R，
    # pipeline 對該量斷言 == 1.0。
    return recs


def evaluate_recommendations(P, box_true, box_pred, recs):
    """檢驗節省率、關鍵遺漏率（R 格零容忍）、建議命中率（指示 十節）。"""
    R = value(P, "stage5_recommendations")
    full = int(value(P, "full_panel_items"))
    box_true = np.asarray(box_true, dtype=object); box_pred = np.asarray(box_pred, dtype=object)
    decided = box_pred != UNDECIDED
    sizes = np.array([len(r) for r in recs], float)
    saving_decided = float(1 - (sizes[decided] / full).mean()) if decided.any() else None

    def miss(i):
        need = set(R[str(box_true[i])]["mandatory"])
        return bool(need - set(recs[i]))

    missed = np.array([miss(i) for i in range(len(recs))])
    true_R = box_true == "R"
    pred_R = box_pred == "R"
    hits = []
    for i in range(len(recs)):
        if recs[i]:
            need = set(R[str(box_true[i])]["mandatory"])
            hits.append(len(set(recs[i]) & need) / len(recs[i]))
    return dict(
        n=int(len(recs)), decided_rate=float(decided.mean()), undecided_rate=float((~decided).mean()),
        test_saving_rate_decided=saving_decided,
        test_saving_rate_all=float(1 - (sizes / full).mean()),
        mean_items_recommended=float(sizes[decided].mean()) if decided.any() else None, full_panel_items=full,
        critical_omission_rate_all=float(missed.mean()),
        critical_omission_rate_decided=float(missed[decided].mean()) if decided.any() else None,
        R_true_n=int(true_R.sum()),
        R_recall=float(pred_R[true_R].mean()) if true_R.any() else None,
        R_critical_omission_rate=float(missed[true_R].mean()) if true_R.any() else None,
        R_five_items_when_routed_R=float(np.mean([set(R["R"]["mandatory"]) <= set(recs[i]) for i in np.where(pred_R)[0]])) if pred_R.any() else None,
        hit_rate=float(np.mean(hits)) if hits else None,
        by_box={str(b): dict(n=int((box_true == b).sum()),
                             recall=float((box_pred[box_true == b] == b).mean()) if (box_true == b).any() else None,
                             omission=float(missed[box_true == b].mean()) if (box_true == b).any() else None)
                for b in sorted(set(map(str, box_true)))})
