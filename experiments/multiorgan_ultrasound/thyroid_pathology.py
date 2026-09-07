# -*- coding: utf-8 -*-
"""病理标注甲状腺超音波：Batch 1 患者级开发基线。"""
import json
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18

from run import ROOT, SEED, bootstrap_auc

DATA = ROOT / "data" / "raw" / "THYROID_PATHOLOGY"
RESULTS = ROOT / "results"


def load_batch(batch):
    batch_root = DATA / f"{batch}_image"
    labels = pd.read_csv(batch_root / f"{batch}_image_label.csv")
    if labels.groupby("patient_name").histo_label.nunique().max() != 1:
        raise ValueError(f"{batch} 同一病例有冲突病理标签")
    rows = []
    label_map = labels.set_index("patient_name").histo_label.to_dict()
    for path in sorted((batch_root / "dataset").glob("*.Jpg")):
        patient = int(path.name.split("_", 1)[0])
        if patient in label_map:
            rows.append({"batch": batch, "patient": f"{batch}:{patient}", "image_path": path, "target": int(label_map[patient])})
    df = pd.DataFrame(rows)
    expected = {"batch1": (6005, 601), "batch2": (2495, 241)}[batch]
    if (len(df), df.patient.nunique()) != expected:
        raise ValueError(f"{batch} 合格病例清单与预注册人数不符")
    if df.groupby("patient").target.nunique().max() != 1:
        raise ValueError(f"{batch} 同一病例跨病理标签")
    return df


def aggregate_patients(df, probabilities):
    rows = df[["patient", "target"]].copy()
    rows["probability"] = probabilities
    return rows.groupby("patient", as_index=False).agg(target=("target", "first"), probability=("probability", "mean"))


class ThyroidDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        return self.transform(Image.open(self.df.iloc[index].image_path).convert("RGB"))


def extract_features(df, batch):
    cache = ROOT / "data" / "cache" / f"thyroid_{batch}_resnet18.npz"
    ids = df.image_path.map(lambda path: path.name).tolist()
    if cache.exists():
        cached = np.load(cache)
        if cached["ids"].tolist() == ids:
            return cached["features"]
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()
    loader = DataLoader(ThyroidDataset(df, weights.transforms()), batch_size=32, shuffle=False, num_workers=0)
    features = []
    with torch.inference_mode():
        for images in loader:
            features.append(model(images).numpy())
    features = np.concatenate(features)
    np.savez_compressed(cache, ids=np.asarray(ids, dtype=str), features=features)
    return features


def fit_model(features, target, sample_weight):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000, random_state=SEED),
    )
    model.fit(features, target, logisticregression__sample_weight=sample_weight)
    return model


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    df = load_batch("batch1")
    features = extract_features(df, "batch1")
    patient_target = df.groupby("patient").target.first()
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.full(len(df), np.nan)
    weight = 1.0 / df.groupby("patient").patient.transform("size").to_numpy()
    folds = []
    for fold, (train, test) in enumerate(splitter.split(features, df.target, df.patient), 1):
        model = fit_model(features[train], df.target.iloc[train], weight[train])
        oof[test] = model.predict_proba(features[test])[:, 1]
        patient = aggregate_patients(df.iloc[test], oof[test])
        folds.append({"fold": fold, "patients": len(patient), "auc": float(roc_auc_score(patient.target, patient.probability))})
    patient = aggregate_patients(df, oof)
    y, p = patient.target.to_numpy(), patient.probability.to_numpy()
    result = {
        "dataset": "thyroid pathology ultrasound batch 1",
        "source": "https://doi.org/10.6084/m9.figshare.27021604.v1",
        "images": len(df),
        "patients": len(patient),
        "patient_counts": patient_target.value_counts().sort_index().rename(index={0: "benign", 1: "malignant"}).to_dict(),
        "model": "frozen ImageNet ResNet-18 whole-image embeddings + patient-weighted balanced L2 logistic regression",
        "split": "5-fold StratifiedGroupKFold by case; mean probability across each case's images",
        "auc": float(roc_auc_score(y, p)),
        "auc_ci_95": bootstrap_auc(y, p),
        "balanced_accuracy_at_0_5": float(balanced_accuracy_score(y, p >= 0.5)),
        "brier": float(brier_score_loss(y, p)),
        "folds": folds,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "thyroid_batch1_development.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (RESULTS / "runs_log.jsonl").open("a", encoding="utf-8") as log:
        log.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"Thyroid Batch 1 OOF AUROC {result['auc']:.3f} [{result['auc_ci_95'][0]:.3f}, {result['auc_ci_95'][1]:.3f}]")


if __name__ == "__main__":
    main()
