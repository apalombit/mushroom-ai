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
        expected = set(WEIGHT_FIELDS) | {
            "body_form_filter",
            "hymenium_filter",
            "size_class_filter",
            "group_filter",
            "alpha",
            "morpho_pool_required",
            "morphotype_prefilter",
            "dangerous_filter",
        }
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
        non_weight_keys = {
            "body_form_filter",
            "hymenium_filter",
            "size_class_filter",
            "group_filter",
            "alpha",
            "morpho_pool_required",
            "morphotype_prefilter",
        }
        total = sum(v for k, v in d.items() if k not in non_weight_keys)
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

    def test_group_filter_default_true(self):
        w = SimilarityWeights()
        assert w.group_filter is True

    def test_group_filter_override_false(self):
        w = SimilarityWeights(group_filter=False)
        assert w.group_filter is False

    def test_group_filter_in_as_dict(self):
        d = SimilarityWeights().as_dict()
        assert "group_filter" in d
        assert d["group_filter"] is True

    def test_alpha_default(self):
        w = SimilarityWeights()
        assert w.alpha == 0.5

    def test_alpha_override(self):
        w = SimilarityWeights(alpha=0.8)
        assert w.alpha == 0.8

    def test_alpha_in_as_dict(self):
        d = SimilarityWeights(alpha=0.3).as_dict()
        assert d["alpha"] == 0.3

    def test_hymenium_filter_default_from_settings(self):
        from config import settings

        w = SimilarityWeights()
        assert w.hymenium_filter is settings.weight_hymenium_filter

    def test_hymenium_filter_override(self):
        w = SimilarityWeights(hymenium_filter=False)
        assert w.hymenium_filter is False

    def test_size_class_filter_default_from_settings(self):
        from config import settings

        w = SimilarityWeights()
        assert w.size_class_filter is settings.weight_size_class_filter

    def test_size_class_filter_override(self):
        w = SimilarityWeights(size_class_filter=False)
        assert w.size_class_filter is False

    def test_morpho_pool_required_default_from_settings(self):
        from config import settings

        w = SimilarityWeights()
        assert w.morpho_pool_required is settings.weight_morpho_pool_required

    def test_morpho_pool_required_override(self):
        w = SimilarityWeights(morpho_pool_required=False)
        assert w.morpho_pool_required is False

    def test_morphotype_prefilter_default_from_settings(self):
        from config import settings

        w = SimilarityWeights()
        assert w.morphotype_prefilter is settings.weight_morphotype_prefilter

    def test_morphotype_prefilter_override(self):
        w = SimilarityWeights(morphotype_prefilter=True)
        assert w.morphotype_prefilter is True
