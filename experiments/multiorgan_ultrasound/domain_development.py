# -*- coding: utf-8 -*-
"""以 BUS-UCLM 患者分组交叉验证开发跨设备稳健遮罩特征。"""
import hashlib
import io
import json
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from run import ROOT, SEED, mask_features_arrays

PARQUET = ROOT / "data" / "raw" / "BUS_UCLM" / "hf_snapshot" / "data" / "train-00000-of-00001.parquet"
CACHE = ROOT / "data" / "cache" / "bus_uclm_mask_features.npz"
RESULT = ROOT / "results" / "bus_uclm_development.json"
HF_REVISION = "5874ae42ce98f0e403a916981773db8e5fea4c32"
DOMAIN_INVARIANT_INDICES = np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 14, 15, 17])
EXCLUDED_MISMATCHED_MASK_IDS = {"HESN_002", "HESN_003", "HESN_004", "HESN_005"}


def load_uclm():
    columns = ["image", "mask", "image_id", "patient_id", "class_label", "has_doppler", "has_marks", "has_combined"]
    df = pq.read_table(PARQUET, columns=columns).to_pandas()
    df = df[df.class_label.isin(["benign", "malignant"])].reset_index(drop=True)
    df = df[~df.image_id.isin(EXCLUDED_MISMATCHED_MASK_IDS)].reset_index(drop=True)
    df["target"] = (df.class_label == "malignant").astype(int)
    if len(df) != 260 or df.patient_id.nunique() != 35:
        raise ValueError("BUS-UCLM 病灶队列与资料卡不符")
    return df


def extract_features(df):
    if CACHE.exists():
        cached = np.load(CACHE)
        if cached["ids"].tolist() == df.image_id.tolist():
            return cached["features"]
    rows = []
    for row in df.itertuples():
        image = np.asarray(Image.open(io.BytesIO(row.image["bytes"])).convert("L"))
        mask = np.asarray(Image.open(io.BytesIO(row.mask["bytes"])).convert("L"), dtype=bool)
        rows.append(mask_features_arrays(image, mask))
    features = np.stack(rows)
    np.savez_compressed(CACHE, ids=df.image_id.to_numpy(dtype=str), features=features)
    return features


def clustered_bootstrap_auc(df, probabilities, repeats=2000):
    rng = np.random.default_rng(SEED)
    patient_values = df.patient_id.to_numpy()
    patients = df.patient_id.unique()
    clusters = [np.flatnonzero(patient_values == patient) for patient in patients]
    y = df.target.to_numpy()
    values = []
    for _ in range(repeats):
        sampled = rng.integers(0, len(clusters), len(clusters))
        indices = np.concatenate([clusters[index] for index in sampled])
        if np.unique(y[indices]).size == 2:
            values.append(roc_auc_score(y[indices], probabilities[indices]))
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def evaluate(df, features, name):
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.full(len(df), np.nan)
    folds = []
    patient_weight = 1.0 / df.groupby("patient_id").patient_id.transform("size").to_numpy()
    for fold, (train, test) in enumerate(splitter.split(features, df.target, df.patient_id), 1):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000, random_state=SEED),
        )
        model.fit(features[train], df.target.iloc[train], logisticregression__sample_weight=patient_weight[train])
        oof[test] = model.predict_proba(features[test])[:, 1]
        folds.append({
            "fold": fold,
            "patients": int(df.patient_id.iloc[test].nunique()),
            "auc": float(roc_auc_score(df.target.iloc[test], oof[test])),
        })
    return {
        "variant": name,
        "lesion_images": len(df),
        "patients": int(df.patient_id.nunique()),
        "auc": float(roc_auc_score(df.target, oof)),
        "clustered_auc_ci_95": clustered_bootstrap_auc(df, oof),
        "balanced_accuracy_at_0_5": float(balanced_accuracy_score(df.target, oof >= 0.5)),
        "folds": folds,
    }


def main():
    df = load_uclm()
    features = extract_features(df)
    variants = [
        ("all_18_mask_features", features),
        ("domain_invariant_13_mask_features", features[:, DOMAIN_INVARIANT_INDICES]),
    ]
    runs = [evaluate(df, x, name) for name, x in variants]
    with PARQUET.open("rb") as source:
        parquet_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
    result = {
        "dataset": "BUS-UCLM",
        "original_doi": "10.17632/7fvgj4jsp7.3",
        "mirror": "https://huggingface.co/datasets/MedOtter/BUS-UCLM",
        "mirror_revision": HF_REVISION,
        "parquet_sha256": parquet_sha256,
        "split": "5-fold StratifiedGroupKFold by patient; lesion-level metrics with patient-clustered bootstrap",
        "notes": "5/35 eligible lesion patients have both benign and malignant lesions; no patient crosses folds",
        "quality_exclusion": "HESN_002 to HESN_005 excluded because image and mask dimensions differ; no geometric registration is available",
        "runs": runs,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    RESULT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (ROOT / "results" / "runs_log.jsonl").open("a", encoding="utf-8") as log:
        for run in runs:
            log.write(json.dumps({"dataset": "BUS-UCLM", **run, "at": result["created_at"]}, ensure_ascii=False) + "\n")
    for run in runs:
        print(f"{run['variant']}: AUROC {run['auc']:.3f} [{run['clustered_auc_ci_95'][0]:.3f}, {run['clustered_auc_ci_95'][1]:.3f}]")


if __name__ == "__main__":
    main()
