# -*- coding: utf-8 -*-
"""Patient-level AUL liver lesion development with a locked one-time holdout."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.spatial import ConvexHull
from scipy.stats import kurtosis, skew
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18

from run import ROOT, SEED, MASK_FEATURE_NAMES, bootstrap_auc, mask_features_arrays

DATA = ROOT / "data" / "raw" / "AUL" / "extracted"
LOCK = ROOT / "params" / "aul_holdout_lock.json"
MODEL_LOCK = ROOT / "params" / "aul_model_lock.json"
RESULTS = ROOT / "results"
CACHE = ROOT / "data" / "cache"
EXPECTED = {"Benign": 200, "Malignant": 435}
ARCHIVES = {
    "Benign.zip": {"md5": "c37fef0cb2730236a79ef57e5315995e", "sha256": "8f07a4d9f4c90e6cb7e080579822e36d8f60b5bafd3faa19a2cb98a1ef6e0d62"},
    "Malignant.zip": {"md5": "63894a9e5654a69c3b94bda84071dfb0", "sha256": "2c1491d61f3e8f71ec60c4aa406ad5321d1915165a1ad8a64867987751d021f7"},
    "Normal.zip": {"md5": "a7e16299b2cf12ca4a6c3468d2e4978f", "sha256": "2eb16a965a30feed102412e3e73e54002ae22e352f439744108b2c9df94b7b16"},
}


def load_manifest():
    rows = []
    for class_name, expected in EXPECTED.items():
        target = int(class_name == "Malignant")
        image_dir = DATA / class_name / "image"
        mass_dir = DATA / class_name / "segmentation" / "mass"
        images = {path.stem: path for path in image_dir.glob("*.jpg")}
        masks = {path.stem: path for path in mass_dir.glob("*.json")}
        if len(images) != expected or set(images) != set(masks):
            raise ValueError(f"AUL {class_name} image/mass manifest mismatch")
        for image_id in sorted(images, key=int):
            rows.append({
                "patient": f"{class_name}:{image_id}", "class_name": class_name, "target": target,
                "image_path": images[image_id], "polygon_path": masks[image_id],
            })
    frame = pd.DataFrame(rows)
    if frame.patient.duplicated().any() or frame.groupby("patient").target.nunique().max() != 1:
        raise ValueError("AUL patient identity is not unique")
    return frame


def create_or_load_holdout(df, fraction=0.2):
    if LOCK.exists():
        document = json.loads(LOCK.read_text(encoding="utf-8"))
        ids = document["patients"]
        digest = hashlib.sha256(json.dumps(sorted(ids)).encode()).hexdigest()
        if digest != document["patient_list_sha256"]:
            raise ValueError("AUL holdout lock checksum mismatch")
        if not set(ids).issubset(set(df.patient)):
            raise ValueError("AUL holdout contains unknown patients")
        return set(ids)
    rng = np.random.default_rng(SEED + 411)
    ids = []
    for target in [0, 1]:
        candidates = df.loc[df.target == target, "patient"].to_numpy()
        ids.extend(rng.choice(candidates, round(fraction * len(candidates)), replace=False).tolist())
    ids = sorted(ids)
    document = {
        "dataset": "Annotated Ultrasound Liver (AUL)",
        "source": "https://doi.org/10.5281/zenodo.7272660",
        "task": "benign versus malignant liver lesion",
        "patient_identity": "one image per patient as documented by the public UltraBench benchmark",
        "fraction": fraction, "seed": SEED + 411,
        "counts": {"benign": sum(value.startswith("Benign:") for value in ids),
                   "malignant": sum(value.startswith("Malignant:") for value in ids)},
        "patients": ids,
        "patient_list_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        "status": "locked_not_evaluated",
        "locked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    LOCK.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[AUL] locked holdout before feature extraction: {document['counts']}")
    return set(ids)


def polygon_mask(image_path, polygon_path):
    with Image.open(image_path) as image:
        size = image.size
    points = json.loads(Path(polygon_path).read_text(encoding="utf-8"))
    if len(points) < 3:
        raise ValueError(f"Invalid polygon: {polygon_path}")
    mask = Image.new("1", size, 0)
    ImageDraw.Draw(mask).polygon([(float(x), float(y)) for x, y in points], fill=1)
    array = np.asarray(mask, dtype=bool)
    if not array.any():
        raise ValueError(f"Empty polygon: {polygon_path}")
    return array


def mass_roi(image_path, polygon_path, padding=0.2):
    image = Image.open(image_path).convert("RGB")
    mask = polygon_mask(image_path, polygon_path)
    yy, xx = np.nonzero(mask)
    pad = round(padding * max(xx.max() - xx.min() + 1, yy.max() - yy.min() + 1))
    return image.crop((max(0, xx.min() - pad), max(0, yy.min() - pad),
                       min(image.width, xx.max() + pad + 1), min(image.height, yy.max() + pad + 1)))


def hog_features(image, size=64, cell=8, bins=9):
    gray = np.asarray(image.convert("L").resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    gy, gx = np.gradient(gray)
    magnitude = np.hypot(gx, gy)
    position = (np.mod(np.arctan2(gy, gx), np.pi) * bins / np.pi)
    low = np.floor(position).astype(int) % bins
    high = (low + 1) % bins
    high_weight = position - np.floor(position)
    cells = size // cell
    histogram = np.zeros((cells, cells, bins), dtype=np.float32)
    for cy in range(cells):
        for cx in range(cells):
            ys, xs = slice(cy * cell, (cy + 1) * cell), slice(cx * cell, (cx + 1) * cell)
            for bucket in range(bins):
                histogram[cy, cx, bucket] = np.sum(
                    magnitude[ys, xs] * (((low[ys, xs] == bucket) * (1 - high_weight[ys, xs]))
                                         + ((high[ys, xs] == bucket) * high_weight[ys, xs]))
                )
    blocks = []
    for cy in range(cells - 1):
        for cx in range(cells - 1):
            block = histogram[cy:cy + 2, cx:cx + 2].ravel()
            blocks.append(block / np.sqrt(np.sum(block**2) + 1e-6))
    output = np.concatenate(blocks)
    if not np.isfinite(output).all():
        raise ValueError("AUL HOG features are invalid")
    return output


EXPANDED_MASK_FEATURE_NAMES = [
    *MASK_FEATURE_NAMES,
    "solidity", "radial_p10", "radial_p50", "radial_p90", "radial_skew",
    "lesion_iqr", "lesion_mad", "lesion_skew", "lesion_kurtosis",
    "gradient_std", "gradient_p50", "gradient_p90", "ring_std", "absolute_ring_contrast",
    "glcm_contrast", "glcm_dissimilarity", "glcm_homogeneity", "glcm_energy",
    "glcm_correlation", "glcm_entropy",
]


def masked_glcm_features(gray, mask, levels=16):
    quantized = np.clip((np.asarray(gray, dtype=float) / 256 * levels).astype(int), 0, levels - 1)
    matrix = np.zeros((levels, levels), dtype=float)
    for dy, dx in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        y0, y1 = max(0, -dy), min(mask.shape[0], mask.shape[0] - dy)
        x0, x1 = max(0, -dx), min(mask.shape[1], mask.shape[1] - dx)
        valid = mask[y0:y1, x0:x1] & mask[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
        left = quantized[y0:y1, x0:x1][valid]
        right = quantized[y0 + dy:y1 + dy, x0 + dx:x1 + dx][valid]
        np.add.at(matrix, (left, right), 1)
        np.add.at(matrix, (right, left), 1)
    if matrix.sum() == 0:
        raise ValueError("Lesion is too small for masked co-occurrence texture")
    matrix /= matrix.sum()
    i, j = np.indices(matrix.shape)
    contrast = np.sum(matrix * (i - j) ** 2)
    dissimilarity = np.sum(matrix * np.abs(i - j))
    homogeneity = np.sum(matrix / (1 + (i - j) ** 2))
    energy = np.sqrt(np.sum(matrix**2))
    mean_i, mean_j = np.sum(matrix * i), np.sum(matrix * j)
    std_i = np.sqrt(np.sum(matrix * (i - mean_i) ** 2))
    std_j = np.sqrt(np.sum(matrix * (j - mean_j) ** 2))
    correlation = np.sum(matrix * (i - mean_i) * (j - mean_j)) / max(std_i * std_j, 1e-12)
    nonzero = matrix[matrix > 0]
    entropy = -np.sum(nonzero * np.log2(nonzero))
    return [contrast, dissimilarity, homogeneity, energy, correlation, entropy]


def expanded_mask_features_arrays(gray, mask):
    gray = np.asarray(gray, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    base = mask_features_arrays(gray, mask)
    yy, xx = np.nonzero(mask)
    boundary = mask & ~binary_erosion(mask)
    by, bx = np.nonzero(boundary)
    radii = np.hypot(by - yy.mean(), bx - xx.mean())
    lesion = gray[mask] / 255.0
    gradient = np.hypot(*np.gradient(gray / 255.0))[mask]
    ring = binary_dilation(mask, iterations=5) & ~mask
    ring_values = gray[ring] / 255.0 if ring.any() else lesion
    points = np.column_stack([xx, yy])
    hull = ConvexHull(points)
    convex = Image.new("1", (mask.shape[1], mask.shape[0]), 0)
    ImageDraw.Draw(convex).polygon([tuple(point) for point in points[hull.vertices]], fill=1)
    convex_area = max(float(np.asarray(convex, dtype=bool).sum()), float(mask.sum()))
    extra = np.asarray([
        mask.sum() / convex_area,
        *np.percentile(radii, [10, 50, 90]),
        skew(radii, bias=False),
        np.percentile(lesion, 75) - np.percentile(lesion, 25),
        np.median(np.abs(lesion - np.median(lesion))),
        skew(lesion, bias=False), kurtosis(lesion, bias=False),
        gradient.std(), *np.percentile(gradient, [50, 90]),
        ring_values.std(), abs(lesion.mean() - ring_values.mean()),
        *masked_glcm_features(gray, mask),
    ], dtype=np.float32)
    output = np.concatenate([base, extra])
    if len(output) != len(EXPANDED_MASK_FEATURE_NAMES) or not np.isfinite(output).all():
        raise ValueError("Expanded AUL radiomics are invalid")
    return output


class AULImages(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image = Image.open(row.image_path).convert("RGB")
        roi = mass_roi(row.image_path, row.polygon_path)
        return self.transform(image), self.transform(roi)


def cache_key(df):
    return hashlib.sha256("\n".join(df.patient).encode()).hexdigest()[:16]


def extract_features(df, role):
    CACHE.mkdir(parents=True, exist_ok=True)
    cache = CACHE / f"aul_{role}_{cache_key(df)}.npz"
    ids = df.patient.to_numpy(str)
    if cache.exists():
        saved = np.load(cache)
        if saved["ids"].tolist() == ids.tolist() and {"expanded_mask", "hog"}.issubset(saved.files):
            return saved["whole"], saved["roi"], saved["mask"], saved["expanded_mask"], saved["hog"]
    weights = ResNet18_Weights.DEFAULT
    encoder = resnet18(weights=weights)
    encoder.fc = torch.nn.Identity()
    encoder.eval()
    loader = DataLoader(AULImages(df, weights.transforms()), batch_size=24, shuffle=False, num_workers=0)
    whole, roi = [], []
    with torch.inference_mode():
        for whole_batch, roi_batch in loader:
            whole.append(encoder(whole_batch).numpy())
            roi.append(encoder(roi_batch).numpy())
    shape_texture, expanded_shape_texture, hog = [], [], []
    for row in df.itertuples():
        gray = np.asarray(Image.open(row.image_path).convert("L"))
        lesion_mask = polygon_mask(row.image_path, row.polygon_path)
        shape_texture.append(mask_features_arrays(gray, lesion_mask))
        expanded_shape_texture.append(expanded_mask_features_arrays(gray, lesion_mask))
        hog.append(hog_features(mass_roi(row.image_path, row.polygon_path)))
    output = (np.concatenate(whole), np.concatenate(roi), np.stack(shape_texture),
              np.stack(expanded_shape_texture), np.stack(hog))
    if not all(np.isfinite(values).all() for values in output):
        raise ValueError("AUL features contain non-finite values")
    np.savez_compressed(cache, ids=ids, whole=output[0], roi=output[1], mask=output[2],
                        expanded_mask=output[3], hog=output[4])
    return output


def estimator(name):
    if name == "LR":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000, random_state=SEED),
        )
    if name.startswith("RBF_C"):
        c = float(name.removeprefix("RBF_C"))
        return make_pipeline(
            StandardScaler(),
            SVC(C=c, kernel="rbf", class_weight="balanced", probability=True, random_state=SEED),
        )
    if name == "HGB":
        return HistGradientBoostingClassifier(
            class_weight="balanced", max_depth=3, learning_rate=0.05, max_iter=250,
            l2_regularization=1.0, random_state=SEED,
        )
    if name == "ExtraTrees":
        return ExtraTreesClassifier(
            n_estimators=500, min_samples_leaf=3, max_features="sqrt",
            class_weight="balanced", random_state=SEED, n_jobs=1,
        )
    raise ValueError(f"Unknown AUL estimator: {name}")


def evaluate_development(df, features, feature_name, classifier):
    y = df.target.to_numpy()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    probability = np.full(len(df), np.nan)
    folds = []
    for fold, (train, test) in enumerate(cv.split(features, y), 1):
        fitted = estimator(classifier).fit(features[train], y[train])
        probability[test] = fitted.predict_proba(features[test])[:, 1]
        folds.append({"fold": fold, "patients": len(test),
                      "auc": float(roc_auc_score(y[test], probability[test]))})
    return summarize(y, probability, feature_name, classifier) | {"folds": folds}


def summarize(y, probability, feature_name, classifier):
    return {
        "feature_variant": feature_name, "classifier": classifier,
        "config": f"{feature_name}__{classifier}", "patients": len(y),
        "auc": float(roc_auc_score(y, probability)), "auc_ci_95": bootstrap_auc(y, probability),
        "balanced_accuracy_at_0_5": float(balanced_accuracy_score(y, probability >= 0.5)),
        "brier": float(brier_score_loss(y, probability)),
    }


def variants(whole, roi, mask, expanded_mask, hog):
    return {
        "whole_image_primary": whole,
        "expert_mass_roi_sensitivity": roi,
        "expert_mass_features_sensitivity": mask,
        "whole_plus_roi_sensitivity": np.concatenate([whole, roi], axis=1),
        "whole_plus_mass_features_sensitivity": np.concatenate([whole, mask], axis=1),
        "roi_plus_mass_features_sensitivity": np.concatenate([roi, mask], axis=1),
        "whole_roi_mass_features_sensitivity": np.concatenate([whole, roi, mask], axis=1),
        "expert_expanded_radiomics_sensitivity": expanded_mask,
        "roi_plus_expanded_radiomics_sensitivity": np.concatenate([roi, expanded_mask], axis=1),
        "whole_roi_expanded_radiomics_sensitivity": np.concatenate([whole, roi, expanded_mask], axis=1),
        "expert_mass_hog_sensitivity": hog,
        "hog_plus_expanded_radiomics_sensitivity": np.concatenate([hog, expanded_mask], axis=1),
        "whole_roi_hog_sensitivity": np.concatenate([whole, roi, hog], axis=1),
        "whole_roi_hog_radiomics_sensitivity": np.concatenate([whole, roi, hog, expanded_mask], axis=1),
    }


def develop():
    df = load_manifest()
    holdout = create_or_load_holdout(df)
    development = df[~df.patient.isin(holdout)].reset_index(drop=True)
    if set(development.patient) & holdout:
        raise AssertionError("AUL development/holdout patient overlap")
    whole, roi, mask, expanded_mask, hog = extract_features(development, "development")
    feature_sets = variants(whole, roi, mask, expanded_mask, hog)
    runs = [evaluate_development(development, values, name, "LR") for name, values in feature_sets.items()]
    nonlinear_features = ["expert_mass_features_sensitivity", "roi_plus_mass_features_sensitivity",
                          "whole_roi_mass_features_sensitivity", "expert_expanded_radiomics_sensitivity",
                          "roi_plus_expanded_radiomics_sensitivity", "whole_roi_expanded_radiomics_sensitivity",
                          "expert_mass_hog_sensitivity", "hog_plus_expanded_radiomics_sensitivity",
                          "whole_roi_hog_sensitivity", "whole_roi_hog_radiomics_sensitivity"]
    for classifier in ["RBF_C0.1", "RBF_C1.0", "RBF_C10.0", "HGB", "ExtraTrees"]:
        for name in nonlinear_features:
            runs.append(evaluate_development(development, feature_sets[name], name, classifier))
    selected = max(runs, key=lambda row: row["auc"])
    result = {
        "dataset": "Annotated Ultrasound Liver (AUL)", "source": "https://doi.org/10.5281/zenodo.7272660",
        "task": "benign versus malignant liver lesion", "archives": ARCHIVES,
        "development_patients": len(development),
        "development_counts": development.class_name.value_counts().to_dict(),
        "holdout_patients_unopened": len(holdout),
        "model": "frozen ImageNet ResNet-18 whole/20%-padded expert mass ROI and/or 18 mask features; LR plus fixed nonlinear sensitivity models",
        "mask_features": MASK_FEATURE_NAMES, "expanded_radiomics": EXPANDED_MASK_FEATURE_NAMES,
        "runs": runs, "selected_config": selected["config"],
        "claim_boundary": "single-source development only; expert variants require manual mass annotation",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (RESULTS / "aul_development.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    MODEL_LOCK.write_text(json.dumps({
        "dataset": result["dataset"], "task": result["task"],
        "selected_feature_variant": selected["feature_variant"], "selected_classifier": selected["classifier"],
        "selected_development_auc": selected["auc"],
        "selection_rule": "highest development OOF AUROC; all v4 configurations reported",
        "version": 4,
        "status": "frozen_before_holdout" if selected["auc"] >= 0.9 else "development_gate_not_met",
        "holdout_gate": "development OOF AUROC >= 0.9",
        "frozen_at": result["created_at"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    with (RESULTS / "runs_log.jsonl").open("a", encoding="utf-8") as log:
        for run in runs:
            log.write(json.dumps({"dataset": "AUL development", **run, "at": result["created_at"]}, ensure_ascii=False) + "\n")
    for run in runs:
        print(f"{run['config']}: AUROC {run['auc']:.3f} [{run['auc_ci_95'][0]:.3f}, {run['auc_ci_95'][1]:.3f}]")
    print(f"[AUL] frozen before holdout: {selected['config']} (development AUROC {selected['auc']:.3f})")
    return result


def final_evaluate():
    result_path = RESULTS / "aul_locked_holdout.json"
    if result_path.exists():
        raise SystemExit("AUL holdout has already been evaluated once")
    if not MODEL_LOCK.exists():
        raise SystemExit("Run development and freeze a model first")
    model_lock = json.loads(MODEL_LOCK.read_text(encoding="utf-8"))
    if model_lock["status"] != "frozen_before_holdout":
        raise ValueError("AUL model lock is invalid")
    df = load_manifest()
    holdout_ids = create_or_load_holdout(df)
    development = df[~df.patient.isin(holdout_ids)].reset_index(drop=True)
    holdout = df[df.patient.isin(holdout_ids)].reset_index(drop=True)
    feature_name = model_lock["selected_feature_variant"]
    classifier = model_lock["selected_classifier"]
    dev_features = variants(*extract_features(development, "development"))[feature_name]
    test_features = variants(*extract_features(holdout, "locked_holdout"))[feature_name]
    fitted = estimator(classifier).fit(dev_features, development.target)
    probability = fitted.predict_proba(test_features)[:, 1]
    result = summarize(holdout.target.to_numpy(), probability, feature_name, classifier)
    result |= {
        "dataset": "AUL locked same-source holdout", "counts": holdout.class_name.value_counts().to_dict(),
        "evaluated_once_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "claim_boundary": "one-time same-source patient holdout, not external-institution validation",
    }
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    lock["status"] = "evaluated_once"
    lock["evaluated_once_at"] = result["evaluated_once_at"]
    LOCK.write_text(json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8")
    with (RESULTS / "runs_log.jsonl").open("a", encoding="utf-8") as log:
        log.write(json.dumps({"dataset": "AUL LOCKED HOLDOUT", **result}, ensure_ascii=False) + "\n")
    print(f"AUL LOCKED HOLDOUT: AUROC {result['auc']:.3f} [{result['auc_ci_95'][0]:.3f}, {result['auc_ci_95'][1]:.3f}]")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    final_evaluate() if args.final else develop()
