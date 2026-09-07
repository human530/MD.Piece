import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from thyroid_usf_mae_development import deterministic_patient_sample


def test_patient_sampling_is_label_blind_deterministic_and_capped():
    rows = []
    for patient, count, target in [("a", 7, 0), ("b", 2, 1)]:
        for index in range(count):
            rows.append({"patient": patient, "image_path": Path(f"{patient}_{index}.Jpg"), "target": target})
    frame = pd.DataFrame(rows)
    first = deterministic_patient_sample(frame, 3)
    second = deterministic_patient_sample(frame.sample(frac=1, random_state=9), 3)
    assert first.image_path.map(str).tolist() == second.image_path.map(str).tolist()
    assert first.groupby("patient").size().to_dict() == {"a": 3, "b": 2}
    assert first.groupby("patient").target.nunique().max() == 1
