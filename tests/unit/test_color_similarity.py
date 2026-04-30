"""Tests for the fuzzy color similarity helper used by VLM cap_color eval."""

import pytest

from vision.data.color_similarity import (
    DEFAULT_THRESHOLD,
    color_similarity,
    fuzzy_color_match,
    fuzzy_color_score,
)


# ---------------------------------------------------------------------------
# color_similarity — pairwise lookups
# ---------------------------------------------------------------------------


class TestColorSimilarity:
    def test_identical_returns_one(self):
        assert color_similarity("red", "red") == 1.0
        assert color_similarity("brown", "brown") == 1.0

    def test_known_pair_yellow_orange(self):
        # Matrix entry: "yellow-orange": 0.7
        assert color_similarity("yellow", "orange") == 0.7

    def test_pair_is_symmetric(self):
        # Matrix only stores one ordering; lookup must try both
        assert color_similarity("orange", "yellow") == color_similarity(
            "yellow", "orange"
        )
        assert color_similarity("pink", "red") == color_similarity("red", "pink")

    def test_known_pair_orange_red(self):
        assert color_similarity("orange", "red") == 0.6

    def test_known_pair_olive_green(self):
        assert color_similarity("olive", "green") == 0.6

    def test_unrelated_pair_returns_zero(self):
        # No "red-green" entry in the matrix
        assert color_similarity("red", "green") == 0.0
        assert color_similarity("blue", "yellow") == 0.0

    def test_case_insensitive(self):
        assert color_similarity("RED", "red") == 1.0
        assert color_similarity("Yellow", "ORANGE") == 0.7

    def test_unknown_color_returns_zero(self):
        assert color_similarity("ultraviolet", "red") == 0.0


# ---------------------------------------------------------------------------
# fuzzy_color_score — best match against a GT set
# ---------------------------------------------------------------------------


class TestFuzzyColorScore:
    def test_score_against_single_gt(self):
        assert fuzzy_color_score("orange", {"red"}) == 0.6
        assert fuzzy_color_score("orange", {"yellow"}) == 0.7

    def test_score_takes_max_across_gt_set(self):
        # red→orange=0.6, red→yellow=0.0 → max is 0.6
        # but {orange, yellow} has orange→yellow=0.7 → match against orange takes max
        assert fuzzy_color_score("orange", {"red", "yellow"}) == 0.7

    def test_identical_in_gt_set_dominates(self):
        # orange → {red, orange} → orange-orange = 1.0 wins over red-orange = 0.6
        assert fuzzy_color_score("orange", {"red", "orange"}) == 1.0

    def test_null_pred_returns_zero(self):
        assert fuzzy_color_score(None, {"red", "orange"}) == 0.0

    def test_empty_gt_set_returns_zero(self):
        assert fuzzy_color_score("red", set()) == 0.0

    def test_far_pred_returns_zero(self):
        # red against {green, blue} — no matrix entries
        assert fuzzy_color_score("red", {"green", "blue"}) == 0.0


# ---------------------------------------------------------------------------
# fuzzy_color_match — boolean threshold gate
# ---------------------------------------------------------------------------


class TestFuzzyColorMatch:
    def test_close_pair_passes_default_threshold(self):
        # 0.7 ≥ 0.5
        assert fuzzy_color_match("yellow", {"orange"}) is True
        # 0.6 ≥ 0.5
        assert fuzzy_color_match("orange", {"red"}) is True
        # 0.5 ≥ 0.5 (boundary cases)
        assert fuzzy_color_match("orange", {"brown"}) is True
        assert fuzzy_color_match("grey", {"white"}) is True

    def test_far_pair_fails_default_threshold(self):
        # 0.4 < 0.5
        assert fuzzy_color_match("brown", {"olive"}) is False
        assert fuzzy_color_match("black", {"brown"}) is False
        # No matrix entry → 0.0
        assert fuzzy_color_match("red", {"green"}) is False

    def test_identical_always_matches(self):
        assert fuzzy_color_match("red", {"red"}) is True
        assert fuzzy_color_match("brown", {"brown", "yellow"}) is True

    def test_higher_threshold_excludes_boundary(self):
        # orange-brown is 0.5; bumping threshold to 0.6 should exclude it
        assert fuzzy_color_match("orange", {"brown"}, threshold=0.5) is True
        assert fuzzy_color_match("orange", {"brown"}, threshold=0.6) is False

    def test_lower_threshold_includes_more(self):
        # brown-olive is 0.4; threshold 0.4 lets it through
        assert fuzzy_color_match("brown", {"olive"}, threshold=0.5) is False
        assert fuzzy_color_match("brown", {"olive"}, threshold=0.4) is True

    def test_null_pred_never_matches(self):
        assert fuzzy_color_match(None, {"red", "orange"}) is False

    def test_empty_gt_set_never_matches(self):
        assert fuzzy_color_match("red", set()) is False

    def test_default_threshold_value(self):
        # Guard the documented default value
        assert DEFAULT_THRESHOLD == 0.5


# ---------------------------------------------------------------------------
# Integration with multi-label GT pattern (mirrors how the analysis script uses it)
# ---------------------------------------------------------------------------


class TestRealisticUsage:
    @pytest.mark.parametrize(
        "pred,gt_set,expected",
        [
            # Direct match — clear hit
            ("red", {"red", "brown"}, True),
            # Within-family near-miss — the user's red→orange case
            ("orange", {"red"}, True),
            # Within-family near-miss — the user's orange→yellow case
            ("yellow", {"orange"}, True),
            # Within multi-label set with broader options
            ("pink", {"red", "white"}, True),  # red-pink = 0.6 ≥ 0.5
            # Cross-family genuine miss
            ("green", {"red", "brown"}, False),
            # Edge of fuzziness — brown to olive is 0.4, below threshold
            ("brown", {"olive", "green"}, False),
            # Pred is in gt_set directly
            ("brown", {"brown", "white"}, True),
        ],
    )
    def test_realistic_pred_vs_gt_cases(self, pred, gt_set, expected):
        assert fuzzy_color_match(pred, gt_set) is expected
