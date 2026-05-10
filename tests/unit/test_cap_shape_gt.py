"""Tests for cap_shape ground-truth spec wiring."""

from vision.data.ground_truth import GROUND_TRUTH_SPEC
from vision.data.labeling import JSON_PATHS
from vision.data.manifest import _VISIBILITY_REQUIREMENTS


class TestCapShapeGroundTruthSpec:
    def test_cap_shape_registered(self):
        spec = GROUND_TRUTH_SPEC["cap_shape"]
        assert spec["source"] == "features_json"
        assert spec["path"] == ["cap", "shape"]
        assert "derive" not in spec

    def test_cap_shape_in_auto_labeler_paths(self):
        assert "cap_shape" in JSON_PATHS
        assert JSON_PATHS["cap_shape"] == "cap.shape"

    def test_cap_shape_visibility_gate_uses_cap_visible(self):
        # Cap shape requires cap profile to be visible (side or 3/4 view).
        assert _VISIBILITY_REQUIREMENTS["cap_shape"] == "cap_visible"
