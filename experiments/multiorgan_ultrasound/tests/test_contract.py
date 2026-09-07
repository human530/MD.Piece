import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run import MASK_FEATURE_NAMES, aggregate_patients, load_manifest, mask_features  # noqa: E402
from external_validate import load_external_manifest  # noqa: E402
from domain_development import load_uclm  # noqa: E402
from thyroid_pathology import load_batch  # noqa: E402
from thyroid_mil_development import pool_cases  # noqa: E402
from thyroid_attention_mil import GatedAttention  # noqa: E402


def test_official_split_is_patient_safe():
    df = load_manifest()
    assert len(df) == 1875
    assert df.Case.nunique() == 1064
    assert df.groupby("Case").kFold.nunique().max() == 1
    assert df.groupby("Case").target.nunique().max() == 1


def test_patient_aggregation_does_not_overweight_extra_images():
    df = pd.DataFrame({"Case": [1, 1, 2], "target": [0, 0, 1], "kFold": [1, 1, 2]})
    got = aggregate_patients(df, np.array([0.2, 0.4, 0.9])).set_index("Case")
    assert got.loc[1, "probability"] == pytest.approx(0.3)
    assert got.loc[2, "probability"] == pytest.approx(0.9)


def test_interpretable_mask_features_are_complete_and_finite():
    row = load_manifest().iloc[0]
    got = mask_features(row.image_path, row.mask_path)
    assert got.shape == (len(MASK_FEATURE_NAMES),)
    assert np.isfinite(got).all()


def test_external_test_is_patient_unique_and_prelocked():
    df = load_external_manifest()
    assert len(df) == df.CaseID.nunique() == 252
    assert df.Classification.value_counts().to_dict() == {"benign": 154, "malignant": 98}
    assert df.Mask_tumor_filename.notna().all()
    assert np.isfinite(mask_features(df.iloc[0].image_path, df.iloc[0].mask_path)).all()


def test_uclm_uses_patient_groups_despite_mixed_lesion_labels():
    df = load_uclm()
    assert len(df) == 260
    assert df.patient_id.nunique() == 35
    assert (df.groupby("patient_id").class_label.nunique() == 2).sum() == 5


def test_thyroid_batches_are_case_safe_and_cohort_prefixed():
    batch1 = load_batch("batch1")
    batch2 = load_batch("batch2")
    assert (len(batch1), batch1.patient.nunique()) == (6005, 601)
    assert (len(batch2), batch2.patient.nunique()) == (2495, 241)
    assert set(batch1.patient).isdisjoint(batch2.patient)
    assert batch1.groupby("patient").target.nunique().max() == 1
    assert batch2.groupby("patient").target.nunique().max() == 1


def test_case_pooling_produces_one_row_per_patient():
    df = pd.DataFrame({"patient": ["a", "a", "b"], "target": [0, 0, 1]})
    features = np.asarray([[1, 2], [3, 1], [8, 9]], dtype=float)
    pooled, target, patients = pool_cases(df, features, "mean_max")
    assert pooled.shape == (2, 4)
    assert target.tolist() == [0, 1]
    assert patients == ["a", "b"]
    assert pooled[0].tolist() == [2, 1.5, 3, 2]


def test_attention_mil_returns_one_case_logit():
    torch = pytest.importorskip("torch")
    logit = GatedAttention()(torch.randn(4, 512))
    assert logit.shape == ()
    assert torch.isfinite(logit)
