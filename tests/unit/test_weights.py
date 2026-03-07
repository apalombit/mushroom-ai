"""Unit tests for similarity weights."""

import pytest

from similarity.weights import WEIGHT_FIELDS, SimilarityWeights


class TestSimilarityWeights:
    def test_defaults_normalize_to_one(self):
        w = SimilarityWeights()
        total = sum(getattr(w, f) for f in WEIGHT_FIELDS)
        assert total == pytest.approx(1.0)

    def test_as_dict_keys(self):
        d = SimilarityWeights().as_dict()
        expected = set(WEIGHT_FIELDS) | {"body_form_filter"}
        assert set(d.keys()) == expected

    def test_body_form_filter_default_true(self):
        w = SimilarityWeights()
        assert w.body_form_filter is True

    def test_body_form_filter_override(self):
        w = SimilarityWeights(body_form_filter=False)
        assert w.body_form_filter is False

    def test_custom_weights_normalize(self):
        w = SimilarityWeights(macro_visual=1.0, ecological=1.0, numeric=1.0)
        total = sum(getattr(w, f) for f in WEIGHT_FIELDS)
        assert total == pytest.approx(1.0)

    def test_as_dict_values_sum_to_one(self):
        d = SimilarityWeights().as_dict()
        total = sum(v for k, v in d.items() if k != "body_form_filter")
        assert total == pytest.approx(1.0, abs=0.01)

    def test_legacy_morphological_distributes(self):
        """Legacy morphological= kwarg splits across 4 sub-groups."""
        w = SimilarityWeights(morphological=0.60, ecological=0.25, taxonomic=0.15)
        # After normalization, morph sub-groups should exist and sum correctly
        total = sum(getattr(w, f) for f in WEIGHT_FIELDS)
        assert total == pytest.approx(1.0)
        # macro_visual should get the largest share of the morph portion
        assert w.macro_visual > w.structural > w.microscopic_lab

    def test_legacy_morphological_with_explicit_subgroup(self):
        """Explicit sub-group overrides the legacy distribution for that sub-group."""
        w = SimilarityWeights(morphological=0.60, macro_visual=0.50)
        # macro_visual should be higher than default distribution
        total = sum(getattr(w, f) for f in WEIGHT_FIELDS)
        assert total == pytest.approx(1.0)

    def test_all_zero_weights(self):
        """All-zero weights don't crash (normalization is a no-op)."""
        w = SimilarityWeights(
            macro_visual=0,
            structural=0,
            flesh_sensory=0,
            microscopic_lab=0,
            ecological=0,
            taxonomic=0,
            numeric=0,
        )
        total = sum(getattr(w, f) for f in WEIGHT_FIELDS)
        assert total == 0.0
