"""Unit tests for compound morphotype signatures."""

from similarity.morphotype import compute_morphotype_signature, morphotype_match_score


class TestComputeSignature:
    def test_all_known(self):
        features = {
            "overall_body_form": "Agaricoid",
            "hymenium": {"type": "gills"},
            "overall_size_class": "medium",
            "cap": {"shape": "convex"},
            "gills": {"attachment": "free"},
        }
        sig = compute_morphotype_signature(features)
        assert sig == "agaricoid|gills|medium|convex|free"

    def test_partial_known(self):
        features = {
            "overall_body_form": "Agaricoid",
            "hymenium": {"type": "gills"},
        }
        sig = compute_morphotype_signature(features)
        assert sig == "agaricoid|gills|?|?|?"

    def test_all_unknown_returns_none(self):
        sig = compute_morphotype_signature({})
        assert sig is None

    def test_none_values_become_unknown(self):
        features = {
            "overall_body_form": None,
            "hymenium": {"type": "pores"},
        }
        sig = compute_morphotype_signature(features)
        assert sig == "?|pores|?|?|?"

    def test_strips_and_lowercases(self):
        features = {
            "overall_body_form": " Boletoid ",
            "hymenium": {"type": "Pores"},
        }
        sig = compute_morphotype_signature(features)
        assert sig == "boletoid|pores|?|?|?"


class TestMatchScore:
    def test_identical(self):
        sig = "agaricoid|gills|medium|convex|free"
        assert morphotype_match_score(sig, sig) == 1.0

    def test_no_match(self):
        a = "agaricoid|gills|small|convex|free"
        b = "boletoid|pores|large|flat|adnate"
        assert morphotype_match_score(a, b) == 0.0

    def test_partial_match(self):
        a = "agaricoid|gills|medium|convex|free"
        b = "agaricoid|gills|medium|flat|adnate"
        # 3 out of 5 match
        assert morphotype_match_score(a, b) == 0.6

    def test_unknown_excluded(self):
        a = "agaricoid|gills|?|convex|?"
        b = "agaricoid|gills|medium|convex|free"
        # Only 3 comparable fields (body_form, hymenium, cap_shape), all match
        assert morphotype_match_score(a, b) == 1.0

    def test_unknown_both_sides(self):
        a = "agaricoid|?|?|?|?"
        b = "agaricoid|?|?|?|?"
        # Only 1 comparable field, matches
        assert morphotype_match_score(a, b) == 1.0

    def test_all_unknown(self):
        a = "?|?|?|?|?"
        b = "?|?|?|?|?"
        assert morphotype_match_score(a, b) == 0.0

    def test_none_input_a(self):
        assert morphotype_match_score(None, "agaricoid|gills|?|?|?") == 0.0

    def test_none_input_b(self):
        assert morphotype_match_score("agaricoid|gills|?|?|?", None) == 0.0

    def test_both_none(self):
        assert morphotype_match_score(None, None) == 0.0
