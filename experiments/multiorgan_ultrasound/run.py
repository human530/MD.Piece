# -*- coding: utf-8 -*-
"""BUS-BRA 真實超音波病人層級基線。"""
import ast
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "raw" / "BUSBRA"
CACHE = ROOT / "data" / "cache" / "busbra_resnet18.npz"
MASK_CACHE = ROOT / "data" / "cache" / "busbra_mask_features.npz"
RESULTS = ROOT / "results"
SEED = 20260830


def load_manifest(data_root=DATA):
    meta = pd.read_csv(data_root / "bus_data.csv")
    folds = pd.read_csv(data_root / "5-fold-cv.csv")[["ID", "kFold"]]
    df = meta.merge(folds, on="ID", validate="one_to_one")
    if df.groupby("Case")["Pathology"].nunique().max() != 1:
        raise ValueError("同一病人有互相衝突的病理標籤")
    if df.groupby("Case")["kFold"].nunique().max() != 1:
        raise ValueError("同一病人跨到多個外層折，會造成洩漏")
    df["target"] = (df["Pathology"] == "malignant").astype(int)
    df["image_path"] = df["ID"].map(lambda x: data_root / "Images" / f"{x}.png")
    df["mask_path"] = df["ID"].map(lambda x: data_root / "Masks" / f"mask_{x.removeprefix('bus_')}.png")
    if not df["image_path"].map(Path.exists).all() or not df["mask_path"].map(Path.exists).all():
        raise FileNotFoundError("影像或遮罩檔不完整")
    return df


def aggregate_patients(df, probabilities):
    rows = df[["Case", "target", "kFold"]].copy()
    rows["probability"] = probabilities
    return rows.groupby("Case", as_index=False).agg(
        target=("target", "first"), kFold=("kFold", "first"), probability=("probability", "mean")
    )


class UltrasoundDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image = Image.open(row.image_path).convert("RGB")
        x, y, w, h = ast.literal_eval(row.BBOX)
        pad = round(0.2 * max(w, h))
        roi = image.crop((max(0, x - pad), max(0, y - pad), min(image.width, x + w + pad), min(image.height, y + h + pad)))
        return self.transform(image), self.transform(roi)


def extract_features(df):
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    if CACHE.exists():
        cached = np.load(CACHE, allow_pickle=True)  # 兼容首次实验产生的本机 object-ID 缓存
        if cached["ids"].tolist() == df["ID"].tolist():
            return cached["whole"], cached["roi"]
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()
    loader = DataLoader(UltrasoundDataset(df, weights.transforms()), batch_size=32, shuffle=False, num_workers=0)
    whole, roi = [], []
    with torch.inference_mode():
        for whole_batch, roi_batch in loader:
            whole.append(model(whole_batch).numpy())
            roi.append(model(roi_batch).numpy())
    whole, roi = np.concatenate(whole), np.concatenate(roi)
    np.savez_compressed(CACHE, ids=df["ID"].to_numpy(dtype=str), whole=whole, roi=roi)
    return whole, roi


MASK_FEATURE_NAMES = [
    "area_fraction", "bbox_extent", "bbox_aspect", "centroid_x", "centroid_y",
    "perimeter_normalized", "compactness", "eccentricity", "radial_irregularity",
    "lesion_mean", "lesion_std", "lesion_p10", "lesion_p50", "lesion_p90",
    "lesion_entropy", "lesion_gradient_mean", "ring_mean", "lesion_ring_contrast",
]


def mask_features_arrays(gray, mask):
    gray = np.asarray(gray, dtype=np.float32) / 255.0
    mask = np.asarray(mask, dtype=bool)
    if gray.shape != mask.shape or not mask.any():
        raise ValueError("影像与病灶遮罩无效或尺寸不一致")

    yy, xx = np.nonzero(mask)
    area = float(mask.sum())
    height, width = mask.shape
    box_h, box_w = yy.max() - yy.min() + 1, xx.max() - xx.min() + 1
    padded = np.pad(mask, 1)
    perimeter = float(
        np.count_nonzero(padded[1:, :] != padded[:-1, :])
        + np.count_nonzero(padded[:, 1:] != padded[:, :-1])
    )
    covariance = np.cov(np.stack([yy, xx]))
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 1e-12)
    eccentricity = np.sqrt(1.0 - eigenvalues[0] / eigenvalues[1])
    boundary = mask & ~binary_erosion(mask)
    by, bx = np.nonzero(boundary)
    radii = np.hypot(by - yy.mean(), bx - xx.mean())

    lesion = gray[mask]
    gradient = np.hypot(*np.gradient(gray))[mask]
    histogram = np.histogram(lesion, bins=16, range=(0, 1))[0].astype(float)
    histogram /= histogram.sum()
    entropy = -np.sum(histogram[histogram > 0] * np.log2(histogram[histogram > 0]))
    ring = binary_dilation(mask, iterations=5) & ~mask
    ring_mean = float(gray[ring].mean()) if ring.any() else float(lesion.mean())

    return np.asarray([
        area / (height * width),
        area / (box_h * box_w),
        box_w / box_h,
        xx.mean() / width,
        yy.mean() / height,
        perimeter / np.sqrt(area),
        4 * np.pi * area / (perimeter**2),
        eccentricity,
        radii.std() / max(radii.mean(), 1e-12),
        lesion.mean(),
        lesion.std(),
        *np.percentile(lesion, [10, 50, 90]),
        entropy,
        gradient.mean(),
        ring_mean,
        lesion.mean() - ring_mean,
    ], dtype=np.float32)


def mask_features(image_path, mask_path):
    gray = np.asarray(Image.open(image_path).convert("L"))
    mask = np.asarray(Image.open(mask_path).convert("L"), dtype=bool)
    return mask_features_arrays(gray, mask)


def extract_mask_features(df):
    MASK_CACHE.parent.mkdir(parents=True, exist_ok=True)
    if MASK_CACHE.exists():
        cached = np.load(MASK_CACHE)
        if cached["ids"].tolist() == df["ID"].tolist():
            return cached["features"]
    features = np.stack([mask_features(row.image_path, row.mask_path) for row in df.itertuples()])
    if not np.isfinite(features).all():
        raise ValueError("形态／纹理特征含有非有限值")
    np.savez_compressed(MASK_CACHE, ids=df["ID"].to_numpy(dtype=str), features=features)
    return features


def bootstrap_auc(y, p, repeats=2000):
    rng = np.random.default_rng(SEED)
    values = []
    for _ in range(repeats):
        idx = rng.integers(0, len(y), len(y))
        if np.unique(y[idx]).size == 2:
            values.append(roc_auc_score(y[idx], p[idx]))
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def evaluate_variant(df, features, name):
    oof = np.full(len(df), np.nan)
    fold_rows = []
    for fold in range(1, 6):
        train = df["kFold"].to_numpy() != fold
        test = ~train
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000, random_state=SEED),
        )
        model.fit(features[train], df.loc[train, "target"])
        oof[test] = model.predict_proba(features[test])[:, 1]
        patient = aggregate_patients(df.loc[test], oof[test])
        fold_rows.append({"fold": fold, "patients": len(patient), "auc": float(roc_auc_score(patient.target, patient.probability))})
    patient = aggregate_patients(df, oof)
    y, p = patient.target.to_numpy(), patient.probability.to_numpy()
    return {
        "variant": name,
        "patients": len(patient),
        "auc": float(roc_auc_score(y, p)),
        "auc_ci_95": bootstrap_auc(y, p),
        "balanced_accuracy_at_0_5": float(balanced_accuracy_score(y, p >= 0.5)),
        "brier": float(brier_score_loss(y, p)),
        "folds": fold_rows,
    }


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    df = load_manifest()
    whole, roi = extract_features(df)
    mask = extract_mask_features(df)
    variants = [
        ("whole_image_primary", whole),
        ("expert_bbox_roi_sensitivity", roi),
        ("whole_plus_expert_bbox_sensitivity", np.concatenate([whole, roi], axis=1)),
        ("expert_mask_interpretable_sensitivity", mask),
        ("roi_plus_expert_mask_sensitivity", np.concatenate([roi, mask], axis=1)),
        ("whole_roi_plus_expert_mask_sensitivity", np.concatenate([whole, roi, mask], axis=1)),
    ]
    runs = [evaluate_variant(df, features, name) for name, features in variants]
    RESULTS.mkdir(exist_ok=True)
    result = {
        "dataset": "BUS-BRA",
        "source": "https://doi.org/10.5281/zenodo.8231412",
        "archive_md5": hashlib.md5((ROOT / "data" / "raw" / "BUSBRA.zip").read_bytes()).hexdigest(),
        "images": len(df),
        "patients": int(df.Case.nunique()),
        "patient_counts": df.groupby("Case").target.first().value_counts().sort_index().rename(index={0: "benign", 1: "malignant"}).to_dict(),
        "split": "official patient-consistent 5-fold outer CV; patient probabilities are mean across images",
        "model": "frozen ImageNet ResNet-18 embeddings and/or 18 preregistered mask morphology/texture features + balanced L2 logistic regression",
        "mask_features": MASK_FEATURE_NAMES,
        "runs": runs,
        "claim_boundary": "BUS-BRA breast disease results do not validate kidney etiology; bbox variants require expert localization",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (RESULTS / "busbra_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (RESULTS / "runs_log.jsonl").open("a", encoding="utf-8") as log:
        for run in runs:
            log.write(json.dumps({"dataset": "BUS-BRA", **run, "at": result["created_at"]}, ensure_ascii=False) + "\n")
    for run in runs:
        print(f"{run['variant']}: AUROC {run['auc']:.3f} [{run['auc_ci_95'][0]:.3f}, {run['auc_ci_95'][1]:.3f}]")


if __name__ == "__main__":
    main()
