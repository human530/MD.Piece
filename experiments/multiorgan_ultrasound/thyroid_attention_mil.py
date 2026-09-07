# -*- coding: utf-8 -*-
"""冻结影像嵌入上的严格嵌套 gated-attention MIL。"""
import json
import time

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from run import ROOT, SEED, bootstrap_auc
from thyroid_pathology import extract_features, load_batch


class GatedAttention(torch.nn.Module):
    def __init__(self, input_dim=512, hidden_dim=64):
        super().__init__()
        self.attention_a = torch.nn.Sequential(torch.nn.Linear(input_dim, hidden_dim), torch.nn.Tanh())
        self.attention_b = torch.nn.Sequential(torch.nn.Linear(input_dim, hidden_dim), torch.nn.Sigmoid())
        self.attention_c = torch.nn.Linear(hidden_dim, 1)
        self.classifier = torch.nn.Linear(input_dim, 1)

    def forward(self, bag):
        scores = self.attention_c(self.attention_a(bag) * self.attention_b(bag)).squeeze(1)
        pooled = torch.sum(torch.softmax(scores, dim=0).unsqueeze(1) * bag, dim=0)
        return self.classifier(pooled).squeeze(0)


def make_bags(df, image_features):
    bags, target, patients = [], [], []
    for patient, indices in df.groupby("patient", sort=True).groups.items():
        index = np.asarray(list(indices))
        bags.append(image_features[index].astype(np.float32))
        target.append(int(df.loc[index[0], "target"]))
        patients.append(patient)
    return bags, np.asarray(target), patients


def scale_bags(bags, train_indices):
    train_images = np.concatenate([bags[index] for index in train_indices])
    mean = train_images.mean(axis=0)
    std = np.maximum(train_images.std(axis=0), 1e-6)
    return [torch.from_numpy(((bag - mean) / std).astype(np.float32)) for bag in bags]


def predict(model, bags, indices):
    model.eval()
    with torch.inference_mode():
        return np.asarray([torch.sigmoid(model(bags[index])).item() for index in indices])


def train_epochs(bags, target, indices, epochs, seed):
    torch.manual_seed(seed)
    model = GatedAttention()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    positives = target[indices].sum()
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(indices) - positives) / positives))
    rng = np.random.default_rng(seed)
    model.train()
    for _ in range(epochs):
        for index in rng.permutation(indices):
            optimizer.zero_grad()
            loss = criterion(model(bags[index]), torch.tensor(float(target[index])))
            loss.backward()
            optimizer.step()
    return model


def choose_epochs(bags, target, outer_train, fold):
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED + fold)
    inner_train_pos, validation_pos = next(split.split(outer_train, target[outer_train]))
    inner_train, validation = outer_train[inner_train_pos], outer_train[validation_pos]
    scaled = scale_bags(bags, inner_train)
    torch.manual_seed(SEED + fold)
    model = GatedAttention()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    positives = target[inner_train].sum()
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(inner_train) - positives) / positives))
    rng = np.random.default_rng(SEED + fold)
    best_auc, best_epoch, stale = -np.inf, 0, 0
    for epoch in range(100):
        model.train()
        for index in rng.permutation(inner_train):
            optimizer.zero_grad()
            loss = criterion(model(scaled[index]), torch.tensor(float(target[index])))
            loss.backward()
            optimizer.step()
        auc = roc_auc_score(target[validation], predict(model, scaled, validation))
        if auc > best_auc + 1e-4:
            best_auc, best_epoch, stale = auc, epoch, 0
        else:
            stale += 1
        if stale >= 15:
            break
    return best_epoch + 1, float(best_auc)


def main():
    df = load_batch("batch1")
    image_features = extract_features(df, "batch1")
    bags, target, _ = make_bags(df, image_features)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.full(len(target), np.nan)
    folds = []
    for fold, (train, test) in enumerate(splitter.split(np.zeros(len(target)), target), 1):
        epochs, validation_auc = choose_epochs(bags, target, train, fold)
        scaled = scale_bags(bags, train)
        model = train_epochs(scaled, target, train, epochs, SEED + 100 + fold)
        oof[test] = predict(model, scaled, test)
        folds.append({
            "fold": fold,
            "patients": len(test),
            "selected_epochs": epochs,
            "inner_validation_auc": validation_auc,
            "outer_auc": float(roc_auc_score(target[test], oof[test])),
        })
    result = {
        "dataset": "thyroid pathology ultrasound batch 1",
        "variant": "frozen_resnet18_gated_attention_mil",
        "patients": len(target),
        "auc": float(roc_auc_score(target, oof)),
        "auc_ci_95": bootstrap_auc(target, oof),
        "balanced_accuracy_at_0_5": float(balanced_accuracy_score(target, oof >= 0.5)),
        "brier": float(brier_score_loss(target, oof)),
        "selection": "epoch selected in an inner patient split; all metrics from untouched outer patients",
        "folds": folds,
        "claim_boundary": "development only; Batch 2 remains locked",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    results = ROOT / "results"
    (results / "thyroid_attention_development.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (results / "runs_log.jsonl").open("a", encoding="utf-8") as log:
        log.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"Attention MIL OOF AUROC {result['auc']:.3f} [{result['auc_ci_95'][0]:.3f}, {result['auc_ci_95'][1]:.3f}]")


if __name__ == "__main__":
    main()
