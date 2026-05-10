"""Tests for stem_shape ground-truth spec wiring."""

from vision.data.ground_truth import GROUND_TRUTH_SPEC
from vision.data.labeling import JSON_PATHS
from vision.data.manifest import _VISIBILITY_REQUIREMENTS


class TestStemShapeGroundTruthSpec:
    def test_stem_shape_registered(self):
        spec = GROUND_TRUTH_SPEC["stem_shape"]
        assert spec["source"] == "features_json"
        assert spec["path"] == ["stem", "shape"]
        assert "derive" not in spec

    def test_stem_shape_in_auto_labeler_paths(self):
        assert "stem_shape" in JSON_PATHS
        assert JSON_PATHS["stem_shape"] == "stem.shape"

    def test_stem_shape_visibility_gate_uses_stem_base(self):
        # Stem shape requires the base to be visible (bulbous/clavate/attenuated/
        # rooting all hinge on the lower portion of the stem).
        assert _VISIBILITY_REQUIREMENTS["stem_shape"] == "stem_base_visible"
