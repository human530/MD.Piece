# -*- coding: utf-8 -*-
"""M2：靜態基線風險分數與分層（只用 t = t_onset 時可得的基線臨床特徵；禁止用未來資訊）。"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

FEATURES = ("upcr_g_per_g", "egfr", "c3_mg_dl", "anti_dsdna_iu_ml", "disease_duration_yr", "prior_flares")
LOG_FEATURES = ("upcr_g_per_g", "anti_dsdna_iu_ml", "disease_duration_yr")


def baseline_matrix(cohort):
    cols = []
    for k in FEATURES:
        v = cohort["features"][k].astype(float)
        cols.append(np.log(v) if k in LOG_FEATURES else v)
    return np.column_stack(cols)


def fit_static(cohort, y=None):
    """在（pilot）世代配適一次；回傳 dict(intercept, coef, features)。"""
    y = cohort["event_24m"] if y is None else y
    m = LogisticRegression(max_iter=2000).fit(baseline_matrix(cohort), y)
    return dict(intercept=float(m.intercept_[0]), coef=[float(c) for c in m.coef_[0]], features=list(FEATURES), log_features=list(LOG_FEATURES))


def score(cohort, coefs):
    z = coefs["intercept"] + baseline_matrix(cohort) @ np.asarray(coefs["coef"])
    return 1.0 / (1.0 + np.exp(-z))


def static_cv_pred(cohort, y, folds, rng_seed):
    y = np.asarray(y).astype(int)
    if y.min() == y.max() or min(np.bincount(y)) < folds:
        return np.full(len(y), y.mean())
    cv = StratifiedKFold(folds, shuffle=True, random_state=int(rng_seed))
    return cross_val_predict(LogisticRegression(max_iter=2000), baseline_matrix(cohort), y, cv=cv, method="predict_proba")[:, 1]


def static_auc(cohort, y=None):
    y = cohort["event_24m"] if y is None else y
    if y.all() or not y.any():
        return np.nan
    m = LogisticRegression(max_iter=2000).fit(baseline_matrix(cohort), y)
    return float(roc_auc_score(y, m.predict_proba(baseline_matrix(cohort))[:, 1]))


def stratify(sc, q):
    r = np.argsort(np.argsort(sc, kind="stable"), kind="stable")
    return (r * q // len(sc)).astype(int)
