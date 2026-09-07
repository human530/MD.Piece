import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aul_liver_development import EXPANDED_MASK_FEATURE_NAMES, expanded_mask_features_arrays, hog_features, polygon_mask


def test_polygon_mask_uses_xy_coordinates_and_image_shape(tmp_path):
    image = tmp_path / "image.jpg"
    polygon = tmp_path / "mass.json"
    Image.new("L", (20, 10), 0).save(image)
    polygon.write_text(json.dumps([[2, 3], [8, 3], [8, 7], [2, 7]]), encoding="utf-8")
    mask = polygon_mask(image, polygon)
    assert mask.shape == (10, 20)
    assert mask[5, 5]
    assert not mask[0, 0]
    assert np.count_nonzero(mask) > 0


def test_expanded_radiomics_are_finite_and_schema_locked():
    yy, xx = np.mgrid[:40, :50]
    mask = ((xx - 25) / 12) ** 2 + ((yy - 20) / 8) ** 2 <= 1
    gray = np.clip(40 + 2 * xx + 3 * yy + 15 * np.sin(xx / 3), 0, 255).astype(np.uint8)
    features = expanded_mask_features_arrays(gray, mask)
    assert len(features) == len(EXPANDED_MASK_FEATURE_NAMES)
    assert np.isfinite(features).all()


def test_hog_features_are_fixed_and_finite():
    image = Image.fromarray(np.tile(np.arange(64, dtype=np.uint8), (64, 1)))
    features = hog_features(image)
    assert features.shape == (7 * 7 * 2 * 2 * 9,)
    assert np.isfinite(features).all()
    assert np.linalg.norm(features) > 0
