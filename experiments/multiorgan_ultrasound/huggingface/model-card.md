---
library_name: scikit-learn
pipeline_tag: image-classification
tags:
- medical
- ultrasound
- breast-ultrasound
- interpretable-ml
license: cc-by-4.0
---

# Interpretable ultrasound evidence model — research draft

This card is a local publication draft. No model has been uploaded to Hugging Face.

## Model description

Balanced L2 logistic regression over 18 expert-mask morphology and texture features. The mask is an external expert input; the model does not perform lesion localization. Pathology, diagnosis, verification, and BI-RADS fields are never input features.

## Intended use

Reproducible research on dataset shift in benign-versus-malignant breast lesion classification. It must not be used for autonomous diagnosis, triage, kidney disease classification, or treatment decisions.

## Evaluation

- BUS-BRA patient-level five-fold OOF: AUROC 0.931 (95% CI 0.914–0.948).
- TCIA BREAST-LESIONS-USG locked external test: AUROC 0.807 (95% CI 0.753–0.859).
- BUS-UCLM patient-grouped five-fold development: AUROC 0.925 (clustered 95% CI 0.835–0.978).

The TCIA result is the best current estimate of cross-site performance and is below the 0.90 design target.

An organ-expansion experiment on 601 pathology-labeled thyroid cases yielded AUROC 0.484 for the frozen whole-image baseline and 0.489 for nested gated-attention MIL. The independent 241-case thyroid Batch 2 remains locked and unevaluated.

A checksum-pinned frozen USF-MAE pilot also failed on thyroid Batch 1: patch-token mean AUROC 0.454 (95% CI 0.408–0.500) and CLS-token AUROC 0.447 (95% CI 0.399–0.494). The pilot used 1,796 label-blind sampled images from all 601 patients with nested patient-level validation. The failure gate stopped full extraction and Batch 2 was not opened.

On the AUL liver benign-versus-malignant task, the best of 64 recorded development configurations achieved OOF
AUROC 0.874 (95% CI 0.837–0.907) on 508 patients. The 127-patient preregistered holdout remains unopened because
the 0.90 development gate was not met. This AUL model is not uploaded: the source record requests author contact,
citation, and credit, and no redistribution or derived-weight publication clearance has been established.

## Limitations

Requires expert segmentation, has not been prospectively validated, and shows material domain shift. BUS-UCLM has only 35 eligible patients. Breast and liver results cannot be transferred to kidney etiology claims. No AUL holdout or thyroid Batch 2 result is available.
