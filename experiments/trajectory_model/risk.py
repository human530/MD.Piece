# -*- coding: utf-8 -*-
"""模組二：基線風險分數與分層。

【禁止】只用基線特徵（年齡、性別、x0）。x0 是 X[:,0] 之前的起始值，不是 x(t>0)。
係數若參數檔給 null，就在 pilot 世代（seed+1000，與分析世代不同）配適一次後固定，
之後對所有分析世代只做「套公式」——這模仿真實世界「拿一個已發表的風險方程來分層」。
"""
import numpy as np
from sklearn.linear_model import LogisticRegression

FEATURES = ("age", "male", "x0")


def baseline_matrix(C):
    return np.c_[C["age"], C["male"], C["x0"]].astype(float)


def fit_risk_coefficients(C_pilot):
    """回傳 dict(intercept, coef[3])。"""
    m = LogisticRegression(max_iter=1000).fit(baseline_matrix(C_pilot), C_pilot["event"])
    return dict(intercept=float(m.intercept_[0]), coef=[float(c) for c in m.coef_[0]], features=list(FEATURES))


def risk_score(C, coefs):
    """logistic 十年風險分數（名目上『十年』，模擬世界的結局窗為 T）。"""
    z = coefs["intercept"] + baseline_matrix(C) @ np.asarray(coefs["coef"])
    return 1.0 / (1.0 + np.exp(-z))


def stratify(score, q):
    """分位分層：回傳 0..q-1。用 rank 而非數值切點，避免大量相同分數造成層數不齊。"""
    r = np.argsort(np.argsort(score, kind="stable"), kind="stable")
    return (r * q // len(score)).astype(int)
