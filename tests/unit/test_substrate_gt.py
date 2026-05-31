"""Tests for substrate ground-truth spec wiring."""

from vision.data.ground_truth import GROUND_TRUTH_SPEC
from vision.data.labeling import JSON_PATHS


class TestSubstrateGroundTruthSpec:
    def test_substrate_registered(self):
        spec = GROUND_TRUTH_SPEC["substrate"]
        assert spec["source"] == "features_json"
        assert spec["path"] == ["ecology", "substrate"]
        # Substrate values in the DB are already canonical strings; no derive needed.
        assert "derive" not in spec

    def test_substrate_in_auto_labeler_paths(self):
        assert "substrate" in JSON_PATHS
        assert JSON_PATHS["substrate"] == "ecology.substrate"
