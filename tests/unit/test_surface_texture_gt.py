"""Tests for surface_texture ground-truth spec wiring."""

from vision.data.ground_truth import GROUND_TRUTH_SPEC
from vision.data.labeling import JSON_PATHS


class TestSurfaceTextureGroundTruthSpec:
    def test_surface_texture_registered(self):
        spec = GROUND_TRUTH_SPEC["surface_texture"]
        assert spec["source"] == "features_json"
        assert spec["path"] == ["cap", "surface_texture"]
        # Canonical strings live directly at the path; no derive needed.
        assert "derive" not in spec

    def test_surface_texture_in_auto_labeler_paths(self):
        assert "surface_texture" in JSON_PATHS
        assert JSON_PATHS["surface_texture"] == "cap.surface_texture"
