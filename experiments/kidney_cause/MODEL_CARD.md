# Immune-associated kidney-damage proxy

## Status

Complete, loadable research artifact; **not a clinical nephritis diagnostic model**.

NHANES 1999–2004 has no kidney-biopsy or clinician-adjudicated nephritis endpoint. The model target is ANA
or specific-autoantibody positivity among adults with kidney damage who were actually tested in the NHANES
surplus-sera ANA study. Calling this target biopsy-confirmed nephritis would be incorrect.

## Cohort and inputs

- 488 real NHANES participants; 91 target-positive and 397 target-negative.
- Eligibility: age at least 20 and eGFR below 60 or ACR at least 30 mg/g.
- All participants have an observed ANA study result. ANA and every specific autoantibody are excluded from inputs.
- 60 routine laboratory, derived kidney, age, and sex features; exact order is stored in the artifact.

## Repeated evaluation

Five repeats of five-fold stratified outer validation were run for both LR and HGB (25 outer tests per model).
The operating threshold in every outer test was selected using only an inner four-fold split of that outer
training set. Each SEQN patient is isolated between train and test within a fold.

| Model | Mean outer-fold AUROC ± SD | Patient-pooled AUROC (95% CI) | AUPRC | Balanced accuracy |
|---|---:|---:|---:|---:|
| LR | 0.526 ± 0.070 | 0.522 (0.455–0.586) | 0.234 | 0.522 |
| HGB | 0.575 ± 0.081 | 0.584 (0.517–0.649) | 0.272 | 0.552 |

HGB was selected by the prespecified highest patient-pooled repeated-CV AUROC rule. There is no untouched
external test for this new binary target, so these are development estimates, not final clinical performance.

## Artifact and inference

- Model: `models/immune_kidney_proxy.joblib`
- Metadata: `models/immune_kidney_proxy.metadata.json`
- CLI: `python predict_nephritis_proxy.py input.csv output.csv`

The inference path rejects missing and unexpected columns instead of silently changing the feature schema.
The artifact contains preprocessing behavior, estimator, feature order, class order, locked threshold, cohort
hash, library version, and creation time.

## Limitations

Performance is weak and does not support clinical deployment. ANA positivity is neither necessary nor sufficient
for nephritis, the target is a cross-sectional proxy, NHANES has no linked renal pathology, and the dataset predates
current practice. The model must not direct diagnosis, triage, biopsy, or treatment.
