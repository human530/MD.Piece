---
license: cc-by-4.0
task_categories:
- image-classification
- image-segmentation
tags:
- ultrasound
- medical-imaging
---

# Multi-site ultrasound experiment manifest — research draft

This is a metadata card for reproducibility, not a redistribution of the source images. Users must obtain each dataset from its original source and comply with its license.

## Sources

- BUS-BRA: DOI `10.5281/zenodo.8231412`, CC BY 4.0.
- TCIA BREAST-LESIONS-USG: DOI `10.7937/9WKK-Q141`, CC BY 4.0.
- BUS-UCLM: DOI `10.17632/7fvgj4jsp7.3`, CC BY 4.0; experiment mirror pinned to revision `5874ae42ce98f0e403a916981773db8e5fea4c32`.
- OASBUD: DOI `10.5281/zenodo.545928`, CC BY 4.0; download and patient-linkage audit pending.
- Thyroid pathology ultrasound: DOI `10.6084/m9.figshare.27021604.v1`, CC BY 4.0; both batches checksum verified, Batch 2 locked.
- AUL liver ultrasound: DOI `10.5281/zenodo.7272660`; official archives checksum verified. The source requests author contact, citation, and credit, so this manifest does not redistribute images.
- USF-MAE: official repository commit `e58c29127e1a0e707fbc4e754db4eb67fbb964f6`; 100-epoch checkpoint SHA256 `f815c629878c17136985af9f4fdc81c2cfa02a94e4d992c026699957f75ccb66`.

Checksums, exclusions, roles, and claim boundaries are recorded in `params/data_sources.json` and `params/external_test_lock.json`.

## Sensitive and clinical considerations

The sources contain de-identified clinical images. They remain unsuitable for re-identification attempts or direct clinical deployment. Dataset labels and expert annotations can encode institution-specific practice and annotation bias.

The AUL 127-patient holdout and thyroid Batch 2 are protocol-locked and have not been evaluated. Their labels must
not be used for feature selection, hyperparameter selection, early stopping, or threshold selection.
