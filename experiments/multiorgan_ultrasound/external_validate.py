# -*- coding: utf-8 -*-
"""在锁定的 TCIA BREAST-LESIONS-USG 外部队列评估 BUS-BRA 模型。"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from run import ROOT, SEED, bootstrap_auc, extract_mask_features, load_manifest, mask_features

TCIA = ROOT / "data" / "raw" / "TCIA_BREAST_LESIONS_USG"
TCIA_IMAGES = TCIA / "BrEaST-Lesions_USG-images_and_masks"
TCIA_CACHE = ROOT / "data" / "cache" / "tcia_mask_features.npz"
LOCK = ROOT / "params" / "external_test_lock.json"
RESULT = ROOT / "results" / "tcia_external_results.json"
PREDICTIONS = ROOT / "results" / "tcia_external_predictions.csv"


def sha256(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def load_external_manifest():
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    archive = TCIA / "images_and_masks.zip"
    clinical = TCIA / "clinical_data.xlsx"
    if sha256(archive) != lock["archive_sha256"] or sha256(clinical) != lock["clinical_sha256"]:
        raise ValueError("外部测试资料校验和与锁定清单不符")
    df = pd.read_excel(clinical, sheet_name=0)
    df = df[df["Classification"].isin(["benign", "malignant"])].copy()
    df["target"] = (df["Classification"] == "malignant").astype(int)
    df["image_path"] = df["Image_filename"].map(lambda name: TCIA_IMAGES / name)
    df["mask_path"] = df["Mask_tumor_filename"].map(lambda name: TCIA_IMAGES / name)
    if len(df) != lock["eligible_patients"] or df.CaseID.nunique() != len(df):
        raise ValueError("外部测试队列人数或患者唯一性与锁定清单不符")
    if not df["image_path"].map(Path.exists).all() or not df["mask_path"].map(Path.exists).all():
        raise FileNotFoundError("外部测试影像或遮罩不完整")
    return df


def extract_external_features(df):
    if TCIA_CACHE.exists():
        cached = np.load(TCIA_CACHE)
        if cached["ids"].tolist() == df["CaseID"].astype(str).tolist():
            return cached["features"]
    features = np.stack([mask_features(row.image_path, row.mask_path) for row in df.itertuples()])
    np.savez_compressed(TCIA_CACHE, ids=df["CaseID"].astype(str).to_numpy(dtype=str), features=features)
    return features


def main():
    if RESULT.exists():
        raise FileExistsError("锁定外部测试已经评估；拒绝重复开启")
    train = load_manifest()
    external = load_external_manifest()
    x_train = extract_mask_features(train)
    x_external = extract_external_features(external)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000, random_state=SEED),
    )
    model.fit(x_train, train.target)
    probability = model.predict_proba(x_external)[:, 1]
    y = external.target.to_numpy()

    subgroup_auc = {}
    for verification, group in external.assign(probability=probability).groupby("Verification"):
        if group.target.nunique() == 2:
            subgroup_auc[verification] = {
                "patients": len(group),
                "auc": float(roc_auc_score(group.target, group.probability)),
            }
    result = {
        "train_dataset": "BUS-BRA",
        "external_dataset": "TCIA BREAST-LESIONS-USG",
        "external_source": "https://doi.org/10.7937/9WKK-Q141",
        "variant": "expert_mask_interpretable_sensitivity",
        "patients": len(external),
        "counts": external.Classification.value_counts().to_dict(),
        "auc": float(roc_auc_score(y, probability)),
        "auc_ci_95": bootstrap_auc(y, probability),
        "balanced_accuracy_at_0_5": float(balanced_accuracy_score(y, probability >= 0.5)),
        "brier": float(brier_score_loss(y, probability)),
        "verification_subgroups": subgroup_auc,
        "protocol": "model fit once on all BUS-BRA patients; no TCIA tuning or calibration",
        "claim_boundary": "expert tumor masks are required; this validates breast lesion malignancy only, not kidney etiology",
        "evaluated_once_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    RESULT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    external.assign(probability=probability)[["CaseID", "Classification", "Verification", "probability"]].to_csv(
        PREDICTIONS, index=False
    )
    with (ROOT / "results" / "runs_log.jsonl").open("a", encoding="utf-8") as log:
        log.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"External AUROC {result['auc']:.3f} [{result['auc_ci_95'][0]:.3f}, {result['auc_ci_95'][1]:.3f}]")


if __name__ == "__main__":
    main()
