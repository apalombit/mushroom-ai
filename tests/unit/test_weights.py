"""Unit tests for similarity weights."""

import pytest

from ingestion.rubric import EMBEDDING_GROUPS
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

    def test_body_form_filter_default_from_settings(self):
        from config import settings

        w = SimilarityWeights()
        assert w.body_form_filter is settings.weight_body_form_filter

    def test_body_form_filter_override(self):
        w = SimilarityWeights(body_form_filter=False)
        assert w.body_form_filter is False

    def test_custom_weights_normalize(self):
        first_group = next(iter(EMBEDDING_GROUPS))
        w = SimilarityWeights(weights={first_group: 1.0}, numeric=1.0)
        total = sum(getattr(w, f) for f in WEIGHT_FIELDS)
        assert total == pytest.approx(1.0)

    def test_as_dict_values_sum_to_one(self):
        d = SimilarityWeights().as_dict()
        total = sum(v for k, v in d.items() if k != "body_form_filter")
        assert total == pytest.approx(1.0, abs=0.01)

    def test_weights_dict_partial_override(self):
        """Specified groups get their weights; unspecified get the default."""
        groups = list(EMBEDDING_GROUPS.keys())
        w = SimilarityWeights(weights={groups[0]: 0.5})
        assert getattr(w, groups[0]) > 0
        total = sum(getattr(w, f) for f in WEIGHT_FIELDS)
        assert total == pytest.approx(1.0)

    def test_all_zero_weights(self):
        """All-zero weights don't crash (normalization is a no-op)."""
        w = SimilarityWeights(weights={g: 0 for g in EMBEDDING_GROUPS}, numeric=0)
        total = sum(getattr(w, f) for f in WEIGHT_FIELDS)
        assert total == 0.0
