"""Unit tests for numeric similarity. No database or LLM required."""

import pytest

from similarity.numeric import (
    compute_numeric_similarity,
    range_overlap,
    single_proximity,
)

# ---------------------------------------------------------------------------
# range_overlap
# ---------------------------------------------------------------------------


class TestRangeOverlap:
    def test_identical_ranges(self):
        assert range_overlap(5.0, 10.0, 5.0, 10.0) == 1.0

    def test_disjoint_ranges(self):
        assert range_overlap(1.0, 3.0, 5.0, 8.0) == 0.0

    def test_partial_overlap(self):
        score = range_overlap(1.0, 6.0, 4.0, 9.0)
        # overlap = 2 (4-6), union = 8 (1-9)
        assert score == pytest.approx(2.0 / 8.0)

    def test_subset_range(self):
        score = range_overlap(2.0, 8.0, 3.0, 5.0)
        # overlap = 2 (3-5), union = 6 (2-8)
        assert score == pytest.approx(2.0 / 6.0)

    def test_single_point_identical(self):
        assert range_overlap(5.0, 5.0, 5.0, 5.0) == 1.0

    def test_none_returns_none(self):
        assert range_overlap(None, 10.0, 5.0, 10.0) is None
        assert range_overlap(5.0, None, 5.0, 10.0) is None
        assert range_overlap(5.0, 10.0, None, 10.0) is None
        assert range_overlap(5.0, 10.0, 5.0, None) is None


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
        a = {"cap": {"diameter_min_cm": 1.0, "diameter_max_cm": 3.0}}
        b = {"cap": {"diameter_min_cm": 10.0, "diameter_max_cm": 15.0}}
        score = compute_numeric_similarity(a, b)
        assert score == 0.0

    def test_one_side_none(self):
        a = {"cap": {"diameter_min_cm": 5.0, "diameter_max_cm": 10.0}}
        b = {}
        # Only one range computable, but both sides need values → no scores → 0.0
        score = compute_numeric_similarity(a, b)
        assert score == 0.0

    def test_string_numeric_parsing(self):
        """Fields stored as strings (e.g. density_per_mm) are parsed."""
        a = {"pores": {"density_per_mm": "2-3 per mm"}}
        b = {"pores": {"density_per_mm": "2-3 per mm"}}
        score = compute_numeric_similarity(a, b)
        assert score == pytest.approx(1.0)
