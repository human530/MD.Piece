# -*- coding: utf-8 -*-
"""以病例级 multiple-instance pooling 改良甲状腺超音波开发模型。"""
import json
import time

import numpy as np
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from run import ROOT, SEED, bootstrap_auc
from thyroid_pathology import extract_features, fit_model, load_batch


def pool_cases(df, image_features, mode):
    features, target, patients = [], [], []
    for patient, indices in df.groupby("patient", sort=True).groups.items():
        values = image_features[np.asarray(list(indices))]
        if mode == "mean":
            pooled = values.mean(axis=0)
        elif mode == "max":
            pooled = values.max(axis=0)
        elif mode == "mean_max":
            pooled = np.concatenate([values.mean(axis=0), values.max(axis=0)])
        else:
            raise ValueError(f"未知 pooling：{mode}")
        features.append(pooled)
        target.append(int(df.loc[list(indices)[0], "target"]))
        patients.append(patient)
    return np.stack(features), np.asarray(target), patients


def evaluate(features, target, name):
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.full(len(target), np.nan)
    folds = []
    for fold, (train, test) in enumerate(splitter.split(features, target), 1):
        model = fit_model(features[train], target[train], np.ones(len(train)))
        oof[test] = model.predict_proba(features[test])[:, 1]
        folds.append({"fold": fold, "patients": len(test), "auc": float(roc_auc_score(target[test], oof[test]))})
    return {
        "variant": name,
        "patients": len(target),
        "auc": float(roc_auc_score(target, oof)),
        "auc_ci_95": bootstrap_auc(target, oof),
        "balanced_accuracy_at_0_5": float(balanced_accuracy_score(target, oof >= 0.5)),
        "brier": float(brier_score_loss(target, oof)),
        "folds": folds,
    }


def main():
    df = load_batch("batch1")
    image_features = extract_features(df, "batch1")
    runs = []
    for mode in ["mean", "max", "mean_max"]:
        features, target, _ = pool_cases(df, image_features, mode)
        runs.append(evaluate(features, target, f"case_{mode}_pooling"))
    result = {
        "dataset": "thyroid pathology ultrasound batch 1",
        "method": "frozen ResNet-18 embeddings aggregated before case-label training",
        "runs": runs,
        "claim_boundary": "development only; Batch 2 remains locked",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    results = ROOT / "results"
    (results / "thyroid_mil_development.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (results / "runs_log.jsonl").open("a", encoding="utf-8") as log:
        for run in runs:
            log.write(json.dumps({"dataset": "thyroid pathology ultrasound batch 1", **run, "at": result["created_at"]}, ensure_ascii=False) + "\n")
    for run in runs:
        print(f"{run['variant']}: AUROC {run['auc']:.3f} [{run['auc_ci_95'][0]:.3f}, {run['auc_ci_95'][1]:.3f}]")


if __name__ == "__main__":
    main()
