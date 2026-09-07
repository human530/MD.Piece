# Patient-Leakage-Aware, Multi-Site Development of Interpretable Ultrasound Evidence Models

> Status: working draft, not submitted or published. The original kidney-cause blood model remains a separate core model; ultrasound is an independent evidence layer until linked multimodal patient data exist.

## Abstract

We investigated whether a compact, interpretable lesion representation transfers across real-world ultrasound sources without patient leakage. BUS-BRA (1,064 patients; 1,875 images) was used for initial development with its official patient-consistent five-fold split. Frozen ImageNet ResNet-18 embeddings achieved patient-level AUROC 0.750 for whole images, 0.854 for expert bounding-box crops, and 0.869 for their concatenation. Eighteen expert-mask morphology and texture features achieved internal out-of-fold AUROC 0.931 (95% bootstrap CI 0.914–0.948). However, the model transferred only modestly to a prospectively locked TCIA BREAST-LESIONS-USG external cohort (252 patients; AUROC 0.807, 95% CI 0.753–0.859). A second development source, BUS-UCLM, yielded patient-grouped AUROC 0.925 for both the full and domain-invariant feature sets, but uncertainty was wide because only 35 eligible patients remained after mask-quality control. These results show that exceeding AUROC 0.90 within development datasets does not establish cross-site generalization. Multi-source training and a new untouched external cohort are required before clinical claims.

## Research question

Can a label-source-free, interpretable ultrasound lesion representation retain malignancy discrimination across devices and institutions while preserving patient-level separation?

## Core invariants

1. Real clinical data only.
2. No pathology, diagnosis, BI-RADS, or verification field may be used as an input feature.
3. All images from a patient stay in the same fold.
4. Every attempted variant is logged, including failures and lower scores.
5. AUROC 0.90 is a design target, not a guaranteed or selectively reported result.
6. Breast-ultrasound performance is not evidence of kidney-etiology performance.

## Data

| Source | Role | Eligible patients | Images/lesions | Ground truth |
|---|---:|---:|---:|---|
| BUS-BRA | initial development | 1,064 | 1,875 images | biopsy-supported pathology |
| TCIA BREAST-LESIONS-USG | locked external test | 252 | 252 lesions | 197 biopsy, 55 follow-up |
| BUS-UCLM | cross-device development | 35 | 260 lesion images | malignant lesions biopsy-confirmed; expert masks |
| OASBUD | RF development only | patient mapping unavailable | 100 lesions / 2 RF planes | biopsy or two-year follow-up |
| Thyroid pathology US Batch 1 | organ-expansion development | 601 | 6,005 images | direct histopathology |
| Thyroid pathology US Batch 2 | locked external test | 241 eligible | 2,495 images | direct histopathology |
| AUL liver ultrasound | organ-expansion development + sealed holdout | 635 | 635 lesion images | expert benign/malignant labels and masks |

Four TCIA normal cases were excluded before evaluation because the task is lesion malignancy and they have no lesion mask. Four BUS-UCLM images from patient `HESN` were excluded because image and mask dimensions differ and no registration transform is supplied.

## Methods

The primary deployable baseline uses the full ultrasound image. Sensitivity analyses use a 20%-padded expert bounding box, an expert segmentation mask, or their combinations. The mask representation contains normalized area, extent, aspect ratio, centroid, perimeter, compactness, eccentricity, radial irregularity, lesion intensity distribution, entropy, gradient, perilesional intensity, and lesion-to-ring contrast. A balanced L2 logistic regression is fitted after standardization.

BUS-BRA predictions are averaged per patient before computing metrics. BUS-UCLM is split with five-fold stratified group cross-validation because five patients contain both benign and malignant lesions; lesion-level AUROC confidence intervals use patient-clustered bootstrap. The TCIA model was trained once on all BUS-BRA cases and applied without target-site tuning or calibration. Confidence intervals use 2,000 bootstrap replicates.

## Results

| Evaluation | Variant | AUROC | 95% CI | Balanced accuracy |
|---|---|---:|---:|---:|
| BUS-BRA patient-level OOF | whole image | 0.750 | 0.718–0.781 | 0.669 |
| BUS-BRA patient-level OOF | expert ROI | 0.854 | 0.829–0.878 | 0.786 |
| BUS-BRA patient-level OOF | whole + ROI | 0.869 | 0.845–0.892 | 0.797 |
| BUS-BRA patient-level OOF | 18 expert-mask features | 0.931 | 0.914–0.948 | 0.875 |
| TCIA locked external | BUS-BRA 18-feature model | 0.807 | 0.753–0.859 | 0.737 |
| BUS-UCLM patient-grouped OOF | all 18 mask features | 0.925 | 0.835–0.978 | 0.874 |
| BUS-UCLM patient-grouped OOF | 13 domain-invariant features | 0.925 | 0.840–0.974 | 0.858 |

## Interpretation and limitations

The expert mask contains clinically meaningful shape information, but it is not available without a radiologist or segmentation model. The large internal-to-external drop demonstrates dataset shift and likely differences in acquisition, lesion prevalence, annotation style, and intensity distribution. BUS-UCLM contributes only 35 eligible patients, so image count must not be mistaken for independent sample size. No current dataset links the same patient's kidney laboratory trajectory and ultrasound, so multimodal fusion remains out of scope. The work is not a diagnostic device and has not undergone prospective clinical validation.

Expansion to pathology-labeled thyroid ultrasound produced near-chance patient-level performance with frozen ImageNet features: whole-image weak-label aggregation AUROC 0.484 (95% CI 0.435–0.532) and nested gated-attention MIL AUROC 0.489 (95% CI 0.441–0.537). Fixed mean, max, and concatenated pooling ranged from 0.441 to 0.473. These negative results were retained. The independent thyroid Batch 2 remains unopened because the representation is not yet adequate.

An additional checksum-locked ultrasound-specific representation did not solve this problem. We evaluated the official 100-epoch USF-MAE ViT-B/16 checkpoint, pretrained on the 46-dataset OpenUS collection, using a label-blind pilot of at most three evenly spaced images per Batch 1 patient (1,796 images; all 601 patients). Nested patient-level validation selected logistic-regression regularization inside each outer fold. Mean patch-token pooling achieved AUROC 0.454 (95% CI 0.408–0.500), and CLS-token pooling achieved 0.447 (95% CI 0.399–0.494). Because both failed the prespecified gate of improving on the 0.484 frozen ImageNet baseline, full-image extraction was stopped and Batch 2 remained unopened.

For liver expansion, the checksum-verified AUL collection provided 200 benign and 435 malignant one-image-per-patient
lesions. Before feature extraction, 127 patients (40 benign, 87 malignant) were deterministically locked as a same-source
holdout. On the remaining 508 development patients, all 64 v4 configurations were retained. The best model combined
whole-image and ROI embeddings with expanded morphology, intensity, gradient, ring, and GLCM radiomics in an RBF SVM
(C=10), reaching OOF AUROC 0.874 (95% CI 0.837–0.907). Adding a dependency-free HOG representation did not improve the
selection. Since the result failed the prespecified 0.90 development gate, the holdout was not evaluated.

## Next locked analysis

Freeze a multi-source, patient-weighted breast representation using BUS-BRA, TCIA, and BUS-UCLM as development data. Use OASBUD only for RF method development because its public MAT file lacks the reported patient mapping. For thyroid, frozen ImageNet and frozen USF-MAE features have both failed; the next justified method is label-efficient end-to-end adaptation or lesion-localized supervision, validated only on Batch 1. For AUL, continue only with materially different GPU-based representation learning on the 508-patient development partition. Open either locked test set once, and only after its method is frozen and its development gate is met. The target is AUROC 0.90 with a confidence interval reported regardless of outcome.
