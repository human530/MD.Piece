# -*- coding: utf-8 -*-
"""Repeated evaluation and packaging for the immune-associated kidney-damage proxy.

This is not a nephritis diagnostic model: NHANES has no biopsy-confirmed nephritis
label. The target is ANA or specific-autoantibody positivity among kidney-damage
participants who were actually included in the ANA surplus-sera study.
"""
import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
from nhanes_cohort import build  # noqa: E402

RESULTS = os.path.join(ROOT, "results")
MODELS = os.path.join(ROOT, "models")
DEFAULT_SEED = 20260830


def make_estimator(name, seed):
    if name == "LR":
        return make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=4000, random_state=seed),
        )
    if name == "HGB":
        return HistGradientBoostingClassifier(
            class_weight="balanced", max_depth=3, learning_rate=0.05,
            max_iter=250, l2_regularization=1.0, random_state=seed,
        )
    raise ValueError(f"Unknown model: {name}")


def _metrics(y, probability, threshold):
    pred = probability >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "auroc": float(roc_auc_score(y, probability)),
        "auprc": float(average_precision_score(y, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "sensitivity": float(tp / (tp + fn)),
        "specificity": float(tn / (tn + fp)),
        "brier": float(brier_score_loss(y, probability)),
        "threshold": float(threshold),
    }


def choose_threshold(y, probability):
    candidates = np.unique(np.r_[0.5, np.quantile(probability, np.linspace(0.05, 0.95, 91))])
    scores = np.array([balanced_accuracy_score(y, probability >= t) for t in candidates])
    best = candidates[scores == scores.max()]
    return float(best[np.argmin(np.abs(best - 0.5))])


def inner_oof_threshold(name, X, y, seed, folds=4):
    probability = np.zeros(len(y), dtype=float)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (train, valid) in enumerate(cv.split(X, y)):
        model = make_estimator(name, seed + fold).fit(X[train], y[train])
        probability[valid] = model.predict_proba(X[valid])[:, 1]
    return choose_threshold(y, probability), probability


def bootstrap_auc(y, probability, seed, n_boot=2000):
    rng = np.random.default_rng(seed)
    negative = np.flatnonzero(y == 0)
    positive = np.flatnonzero(y == 1)
    values = []
    for _ in range(n_boot):
        sample = np.r_[rng.choice(negative, len(negative), replace=True),
                       rng.choice(positive, len(positive), replace=True)]
        values.append(roc_auc_score(y[sample], probability[sample]))
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def repeated_nested_evaluation(X, y, seqn, names=("LR", "HGB"), seed=DEFAULT_SEED,
                               repeats=5, outer_folds=5):
    cv = RepeatedStratifiedKFold(
        n_splits=outer_folds, n_repeats=repeats, random_state=seed,
    )
    splits = list(cv.split(X, y))
    output = {}
    for name in names:
        probability_sum = np.zeros(len(y), dtype=float)
        probability_count = np.zeros(len(y), dtype=int)
        rows = []
        for split_index, (train, test) in enumerate(splits):
            split_seed = seed + split_index
            threshold, _ = inner_oof_threshold(name, X[train], y[train], split_seed)
            model = make_estimator(name, split_seed).fit(X[train], y[train])
            probability = model.predict_proba(X[test])[:, 1]
            probability_sum[test] += probability
            probability_count[test] += 1
            rows.append({
                "split": split_index,
                "repeat": split_index // outer_folds,
                "fold": split_index % outer_folds,
                "n_test": int(len(test)),
                "positive_test": int(y[test].sum()),
                **_metrics(y[test], probability, threshold),
            })
        pooled_probability = probability_sum / probability_count
        pooled_threshold, _ = inner_oof_threshold(name, X, y, seed + 10000, folds=5)
        pooled = _metrics(y, pooled_probability, pooled_threshold)
        pooled["auroc_ci95_patient_bootstrap"] = bootstrap_auc(y, pooled_probability, seed)
        metric_names = ["auroc", "auprc", "balanced_accuracy", "sensitivity", "specificity", "brier"]
        output[name] = {
            "split_metrics": rows,
            "split_summary": {
                key: {"mean": float(np.mean([r[key] for r in rows])),
                      "sd": float(np.std([r[key] for r in rows], ddof=1))}
                for key in metric_names
            },
            "patient_pooled": pooled,
            "predictions": [
                {"SEQN": int(patient), "target": int(target), "probability": float(probability)}
                for patient, target, probability in zip(seqn, y, pooled_probability)
            ],
        }
    return output


def validate_input(frame, features, allow_extra=False):
    missing = sorted(set(features) - set(frame.columns))
    extra = sorted(set(frame.columns) - set(features))
    if missing:
        raise ValueError(f"Missing model features: {missing}")
    if extra and not allow_extra:
        raise ValueError(f"Unexpected model features: {extra}")
    return frame.loc[:, features].apply(pd.to_numeric, errors="raise").to_numpy(float)


def predict_bundle(bundle, frame, allow_extra=False):
    X = validate_input(frame, bundle["features"], allow_extra=allow_extra)
    probability = bundle["estimator"].predict_proba(X)[:, 1]
    return pd.DataFrame({
        "probability_immune_proxy": probability,
        "positive_at_locked_threshold": probability >= bundle["threshold"],
    }, index=frame.index)


def run(seed=DEFAULT_SEED, repeats=5, outer_folds=5):
    with open(os.path.join(ROOT, "params", "design.json"), encoding="utf-8") as f:
        design = json.load(f)
    cohort = build(design, verbose=False)
    data = cohort["primary"].copy()
    assert data["lab_immune"].notna().all(), "ANA-untested participants entered the proxy cohort"
    features = cohort["features"]
    X = data[features].to_numpy(float)
    y = data["lab_immune"].astype(int).to_numpy()
    seqn = data["SEQN"].astype(int).to_numpy()

    evaluations = repeated_nested_evaluation(
        X, y, seqn, seed=seed, repeats=repeats, outer_folds=outer_folds,
    )
    best = max(evaluations, key=lambda name: evaluations[name]["patient_pooled"]["auroc"])
    threshold, threshold_oof = inner_oof_threshold(best, X, y, seed + 20000, folds=5)
    estimator = make_estimator(best, seed).fit(X, y)
    created = datetime.now(timezone.utc).isoformat()
    training_hash = hashlib.sha256(json.dumps(sorted(seqn.tolist())).encode()).hexdigest()
    bundle = {
        "artifact_version": 1,
        "model_name": best,
        "target": "ANA_or_specific_autoantibody_positive_among_ANA-tested_adults_with_kidney_damage",
        "clinical_status": "research proxy; not biopsy-confirmed nephritis and not for clinical diagnosis",
        "features": features,
        "class_order": [0, 1],
        "threshold": threshold,
        "estimator": estimator,
        "sklearn_version": __import__("sklearn").__version__,
        "training_n": int(len(y)),
        "training_positive": int(y.sum()),
        "training_seqn_sha256": training_hash,
        "created_utc": created,
    }
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(MODELS, exist_ok=True)
    model_path = os.path.join(MODELS, "immune_kidney_proxy.joblib")
    joblib.dump(bundle, model_path)
    model_sha = hashlib.sha256(open(model_path, "rb").read()).hexdigest()
    report = {
        "study_name": "immune-associated kidney-damage proxy repeated evaluation",
        "target_warning": bundle["clinical_status"],
        "cohort": {
            "source": "NHANES 1999-2004 surplus-sera ANA subsample",
            "eligibility": "age>=20, eGFR<60 or ACR>=30, ANA actually measured",
            "n": int(len(y)), "positive": int(y.sum()), "negative": int((y == 0).sum()),
            "patient_id_separation": "SEQN; every outer fold is patient-disjoint",
        },
        "protocol": {
            "models": list(evaluations), "outer_folds": outer_folds, "repeats": repeats,
            "total_outer_tests_per_model": outer_folds * repeats,
            "threshold_selection": "inner stratified CV only; maximizes balanced accuracy",
            "selection": "highest patient-pooled repeated-CV AUROC; all candidates reported",
            "untouched_holdout": "none for this new binary proxy; results are repeated nested CV, not external validation",
            "legacy_three_class_holdout_reused": False,
        },
        "evaluations": evaluations,
        "selected_model": best,
        "final_threshold": threshold,
        "final_threshold_oof_metrics": _metrics(y, threshold_oof, threshold),
        "artifact": {"path": "models/immune_kidney_proxy.joblib", "sha256": model_sha},
        "created_utc": created,
    }
    report_path = os.path.join(RESULTS, "nephritis_proxy_repeated_eval.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(os.path.join(MODELS, "immune_kidney_proxy.metadata.json"), "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in bundle.items() if k != "estimator"} | {"sha256": model_sha},
                  f, ensure_ascii=False, indent=2)
    print(json.dumps({
        "selected_model": best,
        "n": len(y), "positive": int(y.sum()),
        "patient_pooled": evaluations[best]["patient_pooled"],
        "artifact_sha256": model_sha,
    }, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--outer-folds", type=int, default=5)
    args = parser.parse_args()
    run(args.seed, args.repeats, args.outer_folds)
