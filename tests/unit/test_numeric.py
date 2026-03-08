"""Unit tests for numeric similarity. No database or LLM required."""

import pytest

from similarity.numeric import (
    _SIZE_BINS,
    category_similarity,
    compute_numeric_similarity,
    range_to_category,
    single_proximity,
)

_CAP_BINS = _SIZE_BINS["cap.diameter_min_cm"]


# ---------------------------------------------------------------------------
# range_to_category
# ---------------------------------------------------------------------------


class TestRangeToCategory:
    def test_both_none_returns_none(self):
        assert range_to_category(None, None, _CAP_BINS) is None

    def test_only_min(self):
        # midpoint = 3 → small (< 5)
        assert range_to_category(3.0, None, _CAP_BINS) == "small"

    def test_only_max(self):
        # midpoint = 20 → large (> 15)
        assert range_to_category(None, 20.0, _CAP_BINS) == "large"

    def test_small_below_threshold(self):
        # midpoint = 2 → small
        assert range_to_category(1.0, 3.0, _CAP_BINS) == "small"

    def test_small_at_threshold(self):
        # midpoint = 5 → small (≤ 5)
        assert range_to_category(4.0, 6.0, _CAP_BINS) == "small"

    def test_medium_above_lower_threshold(self):
        # midpoint = 8 → medium (5 < 8 ≤ 15)
        assert range_to_category(6.0, 10.0, _CAP_BINS) == "medium"

    def test_medium_at_upper_threshold(self):
        # midpoint = 15 → medium (≤ 15)
        assert range_to_category(10.0, 20.0, _CAP_BINS) == "medium"

    def test_large_above_upper_threshold(self):
        # midpoint = 18 → large (> 15)
        assert range_to_category(16.0, 20.0, _CAP_BINS) == "large"

    def test_large_exceeds_upper_bound(self):
        # midpoint = 30 → exceeds all bin upper bounds → fallback "large"
        assert range_to_category(25.0, 35.0, _CAP_BINS) == "large"


# ---------------------------------------------------------------------------
# category_similarity
# ---------------------------------------------------------------------------


class TestCategorySimilarity:
    def test_same_category(self):
        assert category_similarity("small", "small", _CAP_BINS) == 1.0

    def test_adjacent_categories(self):
        assert category_similarity("small", "medium", _CAP_BINS) == 0.5
        assert category_similarity("medium", "large", _CAP_BINS) == 0.5

    def test_opposite_categories(self):
        assert category_similarity("small", "large", _CAP_BINS) == 0.0

    def test_first_none_returns_none(self):
        assert category_similarity(None, "small", _CAP_BINS) is None

    def test_second_none_returns_none(self):
        assert category_similarity("small", None, _CAP_BINS) is None


# ---------------------------------------------------------------------------
# single_proximity
# ---------------------------------------------------------------------------


class TestSingleProximity:
    def test_identical_values(self):
        assert single_proximity(5.0, 5.0, 3.0) == 1.0

    def test_at_tolerance(self):
        assert single_proximity(5.0, 8.0, 3.0) == 0.0

    def test_beyond_tolerance(self):
        assert single_proximity(0.0, 10.0, 3.0) == 0.0

    def test_halfway(self):
        assert single_proximity(5.0, 6.5, 3.0) == pytest.approx(0.5)

    def test_none_returns_none(self):
        assert single_proximity(None, 5.0, 3.0) is None
        assert single_proximity(5.0, None, 3.0) is None


# ---------------------------------------------------------------------------
# compute_numeric_similarity
# ---------------------------------------------------------------------------


class TestComputeNumericSimilarity:
    def test_self_similarity(self, sample_features_json):
        score = compute_numeric_similarity(sample_features_json, sample_features_json)
        assert score == pytest.approx(1.0)

    def test_empty_features(self):
        assert compute_numeric_similarity({}, {}) == 0.0

    def test_partial_data(self):
        a = {"cap": {"diameter_min_cm": 5.0, "diameter_max_cm": 10.0}}
        b = {"cap": {"diameter_min_cm": 5.0, "diameter_max_cm": 10.0}}
        score = compute_numeric_similarity(a, b)
        assert score == pytest.approx(1.0)

    def test_different_ranges(self):
        # cap 1–3 cm → midpoint 2 → "small"; cap 10–15 cm → midpoint 12.5 → "medium"
        # adjacent → 0.5
        a = {"cap": {"diameter_min_cm": 1.0, "diameter_max_cm": 3.0}}
        b = {"cap": {"diameter_min_cm": 10.0, "diameter_max_cm": 15.0}}
        score = compute_numeric_similarity(a, b)
        assert score == pytest.approx(0.5)

    def test_one_side_none(self):
        a = {"cap": {"diameter_min_cm": 5.0, "diameter_max_cm": 10.0}}
        b = {}
        # Only one side has data → no comparable pair → 0.0
        score = compute_numeric_similarity(a, b)
        assert score == 0.0

    def test_string_numeric_parsing(self):
        """Fields stored as strings (e.g. density_per_mm) are parsed."""
        a = {"pores": {"density_per_mm": "2-3 per mm"}}
        b = {"pores": {"density_per_mm": "2-3 per mm"}}
        score = compute_numeric_similarity(a, b)
        assert score == pytest.approx(1.0)
