# Integrated experiment results

## Outcome

The research now contains two deliberately separate evidence tracks. Neither is a deployable medical device.

1. **Kidney core:** routine-laboratory modeling of immune-associated kidney damage in real NHANES participants.
2. **Ultrasound evidence layer:** patient-separated, multi-source lesion experiments in breast and thyroid ultrasound.

No dataset currently links a patient's kidney laboratory trajectory, renal ultrasound, and biopsy-confirmed etiology,
so the tracks were not fused.

## Kidney / immune-associated proxy

The earlier three-class holdout reported AUROC 0.777 (immune), 0.853 (infection), and 0.814 (metabolic), with
balanced accuracy 0.519. Audit found that the analysis included ANA-untested participants as immune-negative,
contrary to the documented ascertainment rule. That locked result is retained as historical provenance but is
**superseded for immune/nephritis interpretation** and was not reopened or reused.

The corrected binary proxy includes only 488 ANA-tested adults with kidney damage (91 positive, 397 negative).
Five repeated five-fold outer evaluations were performed for LR and HGB, with threshold selection confined to
inner folds. HGB was selected: patient-pooled AUROC 0.584 (95% patient-bootstrap CI 0.517–0.649), AUPRC 0.272,
balanced accuracy 0.552, sensitivity 0.275, specificity 0.829, and Brier score 0.166. The result is above chance
only modestly and does not establish nephritis detection.

## Ultrasound experiments

| Dataset / evaluation | Patients | Result |
|---|---:|---:|
| BUS-BRA whole-image frozen ResNet18, patient OOF | 1,064 | AUROC 0.750 (0.718–0.781) |
| BUS-BRA expert-mask features, patient OOF | 1,064 | AUROC 0.931 (0.914–0.948) |
| TCIA locked external test of BUS-BRA mask model | 252 | AUROC 0.807 (0.753–0.859) |
| BUS-UCLM domain-invariant mask features, grouped OOF | 35 | AUROC 0.925 (0.840–0.974) |
| Thyroid Batch 1 frozen whole-image baseline | 601 | AUROC 0.484 (0.435–0.532) |
| Thyroid Batch 1 nested gated-attention MIL | 601 | AUROC 0.489 (0.441–0.537) |
| Thyroid Batch 1 frozen USF-MAE patch mean pilot | 601 | AUROC 0.454 (0.408–0.500) |
| Thyroid Batch 1 frozen USF-MAE CLS pilot | 601 | AUROC 0.447 (0.399–0.494) |
| AUL liver development, whole + ROI + expanded radiomics RBF | 508 | AUROC 0.874 (0.837–0.907) |

The breast internal score above 0.90 did not transfer at that level to TCIA. Thyroid fixed ImageNet features and a
checksum-pinned ultrasound-specific USF-MAE encoder both failed near chance, so locked Thyroid Batch 2 remains unopened.
OASBUD remains RF-method-development-only because its
public MAT file lacks the patient mapping needed for strict patient-level evaluation.

The AUL liver experiment evaluated all 64 recorded v4 configurations on a 508-patient development partition.
HOG did not improve the selected model. Because the best development AUROC was 0.874 rather than the prespecified
0.90 gate, the 127-patient same-source holdout (40 benign, 87 malignant) remains sealed and unevaluated.

## Interpretation

- AUROC and thresholded accuracy answer different questions; imbalance and threshold choice explain why an AUROC
  around 0.8 can coexist with balanced accuracy near 0.5.
- Correct label ascertainment materially lowers the kidney estimate. This is evidence that the previous apparent
  performance partly reflected cohort/measurement structure rather than nephritis biology.
- The complete kidney artifact is reproducible and schema-locked, but its low discrimination makes it a negative
research result, not a clinical model.

For completeness, the corrected exploratory three-class analysis among the 238 ANA-tested participants assigned
to immune, infection, or metabolic classes produced HGB AUROC 0.748 (immune), 0.694 (infection), and 0.771
(metabolic), with balanced accuracy 0.460. Infection had only seven participants, so its confidence interval
(0.465–0.903) is too wide for a stable claim. LR balanced accuracy was 0.534.
- Achieving externally validated AUROC 0.90 for nephritis requires a new cohort with biopsy- or adjudication-based
  renal diagnoses and an untouched institution-level external test.
