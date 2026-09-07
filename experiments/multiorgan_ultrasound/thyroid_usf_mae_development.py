# -*- coding: utf-8 -*-
"""Ultrasound-specific frozen USF-MAE development on thyroid Batch 1 only.

Batch 2 remains locked. Image subsampling is deterministic and label-blind; set
--max-images-per-patient 0 only after the fixed pilot warrants full extraction.
"""
import argparse
import hashlib
import json
import sys
import time

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, InterpolationMode, Normalize, Resize, ToTensor

from run import ROOT, SEED, bootstrap_auc
from thyroid_pathology import load_batch

CHECKPOINT = ROOT / "data" / "usf_mae_100ep.pth"
FACEBOOK_MAE = ROOT / "data" / "facebook_mae"
CHECKPOINT_SHA256 = "f815c629878c17136985af9f4fdc81c2cfa02a94e4d992c026699957f75ccb66"
USF_MAE_COMMIT = "e58c29127e1a0e707fbc4e754db4eb67fbb964f6"
FACEBOOK_MAE_COMMIT = "efb2a8062c206524e35e47d04501ed4f544c0ae8"


def deterministic_patient_sample(df, maximum):
    if maximum <= 0:
        return df.sort_values(["patient", "image_path"]).reset_index(drop=True)
    selected = []
    for _, group in df.sort_values(["patient", "image_path"]).groupby("patient", sort=True):
        if len(group) <= maximum:
            selected.extend(group.index.tolist())
        else:
            positions = np.linspace(0, len(group) - 1, maximum).round().astype(int)
            selected.extend(group.iloc[np.unique(positions)].index.tolist())
    return df.loc[selected].sort_values(["patient", "image_path"]).reset_index(drop=True)


class ImageDataset(Dataset):
    def __init__(self, paths):
        self.paths = list(paths)
        self.transform = Compose([
            Resize((224, 224), interpolation=InterpolationMode.NEAREST),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        return self.transform(Image.open(self.paths[index]).convert("RGB"))


def load_encoder():
    if not CHECKPOINT.exists() or hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() != CHECKPOINT_SHA256:
        raise RuntimeError("USF-MAE 100-epoch checkpoint missing or checksum mismatch")
    if not (FACEBOOK_MAE / "models_mae.py").exists():
        raise RuntimeError("Pinned facebookresearch/mae source is missing")
    sys.path.insert(0, str(FACEBOOK_MAE))
    import models_mae  # noqa: E402
    model = models_mae.mae_vit_base_patch16()
    model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu", weights_only=True), strict=True)
    model.eval()
    return model


def extract_features(df, maximum, batch_size=8):
    cache = ROOT / "data" / "cache" / f"thyroid_batch1_usf_mae_max{maximum}.npz"
    ids = df.image_path.map(lambda path: path.name).to_numpy(str)
    if cache.exists():
        saved = np.load(cache)
        if saved["ids"].tolist() == ids.tolist() and saved["checkpoint_sha256"].item() == CHECKPOINT_SHA256:
            return {"patch_mean": saved["patch_mean"], "cls": saved["cls"]}
    model = load_encoder()
    loader = DataLoader(ImageDataset(df.image_path), batch_size=batch_size, shuffle=False, num_workers=0)
    patch_mean, cls = [], []
    with torch.inference_mode():
        for batch_index, images in enumerate(loader, 1):
            tokens, _, _ = model.forward_encoder(images, mask_ratio=0)
            patch_mean.append(tokens[:, 1:].mean(dim=1).cpu().numpy())
            cls.append(tokens[:, 0].cpu().numpy())
            if batch_index % 50 == 0:
                print(f"[USF-MAE] extracted {min(batch_index * batch_size, len(df))}/{len(df)} images", flush=True)
    output = {"patch_mean": np.concatenate(patch_mean), "cls": np.concatenate(cls)}
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, ids=ids, checkpoint_sha256=CHECKPOINT_SHA256, **output)
    return output


def pool_patients(df, image_features):
    features, target, patients = [], [], []
    for patient, indices in df.groupby("patient", sort=True).groups.items():
        index = np.asarray(list(indices))
        features.append(image_features[index].mean(axis=0))
        target.append(int(df.loc[index[0], "target"]))
        patients.append(patient)
    return np.stack(features), np.asarray(target), patients


def model(c):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c, class_weight="balanced", max_iter=5000, random_state=SEED),
    )


def choose_c(X, y, seed, candidates=(0.01, 0.1, 1.0, 10.0)):
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
    scores = {}
    for c in candidates:
        probability = np.full(len(y), np.nan)
        for train, valid in cv.split(X, y):
            fitted = model(c).fit(X[train], y[train])
            probability[valid] = fitted.predict_proba(X[valid])[:, 1]
        scores[str(c)] = float(roc_auc_score(y, probability))
    best = max(candidates, key=lambda c: (scores[str(c)], -abs(np.log10(c))))
    return best, scores


def evaluate(X, y, patients, variant):
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    probability = np.full(len(y), np.nan)
    folds = []
    for fold, (train, test) in enumerate(outer.split(X, y), 1):
        chosen_c, inner_scores = choose_c(X[train], y[train], SEED + fold)
        fitted = model(chosen_c).fit(X[train], y[train])
        probability[test] = fitted.predict_proba(X[test])[:, 1]
        folds.append({
            "fold": fold, "train_patients": len(train), "test_patients": len(test),
            "chosen_c": chosen_c, "inner_auc_by_c": inner_scores,
            "test_auc": float(roc_auc_score(y[test], probability[test])),
        })
    return {
        "variant": variant,
        "patients": len(y),
        "auc": float(roc_auc_score(y, probability)),
        "auc_ci_95": bootstrap_auc(y, probability),
        "balanced_accuracy_at_0_5": float(balanced_accuracy_score(y, probability >= 0.5)),
        "brier": float(brier_score_loss(y, probability)),
        "folds": folds,
        "predictions": [{"patient": patient, "target": int(target), "probability": float(p)}
                        for patient, target, p in zip(patients, y, probability)],
    }


def main(maximum=3, batch_size=8):
    full = load_batch("batch1")
    sampled = deterministic_patient_sample(full, maximum)
    image_features = extract_features(sampled, maximum, batch_size)
    runs = []
    for variant, values in image_features.items():
        X, y, patients = pool_patients(sampled, values)
        runs.append(evaluate(X, y, patients, variant))
    result = {
        "dataset": "pathology-annotated thyroid ultrasound Batch 1",
        "representation": "USF-MAE ViT-B/16, 100-epoch OpenUS-46 checkpoint, frozen",
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "upstream_commits": {"USF-MAE": USF_MAE_COMMIT, "facebookresearch/mae": FACEBOOK_MAE_COMMIT},
        "sampling": "label-blind evenly spaced images within each sorted patient sequence",
        "max_images_per_patient": maximum,
        "images_used": len(sampled),
        "patients": sampled.patient.nunique(),
        "evaluation": "nested 5 outer x 4 inner patient-level stratified CV; inner selects logistic C",
        "claim_boundary": "development pilot only; Batch 2 remains locked and was not loaded",
        "runs": runs,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    path = ROOT / "results" / f"thyroid_usf_mae_max{maximum}_development.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (ROOT / "results" / "runs_log.jsonl").open("a", encoding="utf-8") as log:
        for run in runs:
            log.write(json.dumps({k: v for k, v in result.items() if k != "runs"} | run, ensure_ascii=False) + "\n")
    for run in runs:
        print(f"{run['variant']}: AUROC {run['auc']:.3f} [{run['auc_ci_95'][0]:.3f}, {run['auc_ci_95'][1]:.3f}]")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-images-per-patient", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    main(args.max_images_per_patient, args.batch_size)
