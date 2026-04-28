"""Tests for _resolve_color_to_canonical and GT corrections merging."""

from vision.data.ground_truth import _load_corrections, _resolve_color_to_canonical

CANONICAL = {"white", "cream", "yellow", "orange", "red", "pink", "brown", "tan", "grey",
             "olive", "green", "purple", "blue", "black"}

ALIASES = {
    "ivory": "cream",
    "buff": "cream",
    "golden": "yellow",
    "chestnut": "brown",
    "amber": "orange",
    "scarlet": "red",
    "salmon": "pink",
    "beige": "tan",
    "violet": "purple",
    "indigo": "blue",
    "yellowish brown": "brown",
    "reddish brown": "brown",
}


class TestResolveColorToCanonical:
    def test_exact_canonical(self):
        assert _resolve_color_to_canonical("brown", CANONICAL, ALIASES) == {"brown"}

    def test_exact_canonical_case_insensitive(self):
        assert _resolve_color_to_canonical("Brown", CANONICAL, ALIASES) == {"brown"}

    def test_alias_lookup(self):
        assert _resolve_color_to_canonical("ivory", CANONICAL, ALIASES) == {"cream"}

    def test_alias_compound(self):
        assert _resolve_color_to_canonical("yellowish brown", CANONICAL, ALIASES) == {"brown"}

    def test_substring_single(self):
        assert _resolve_color_to_canonical("pale brown", CANONICAL, ALIASES) == {"brown"}

    def test_substring_compound(self):
        # "orangish yellow" — "orange" is NOT a substring of "orangish" (e vs ish)
        result = _resolve_color_to_canonical("orangish yellow", CANONICAL, ALIASES)
        assert result == {"yellow"}

    def test_substring_compound_with_orange(self):
        # "orange-yellow" — both "orange" and "yellow" are substrings
        result = _resolve_color_to_canonical("orange-yellow", CANONICAL, ALIASES)
        assert "orange" in result
        assert "yellow" in result

    def test_substring_golden_brown(self):
        # "golden brown" — alias lookup won't match (it's not in aliases as compound)
        # substring: "brown" is in "golden brown" → {"brown"}
        # "golden" is NOT a canonical value, so no substring hit for it
        result = _resolve_color_to_canonical("golden brown", CANONICAL, ALIASES)
        assert "brown" in result

    def test_empty_string(self):
        assert _resolve_color_to_canonical("", CANONICAL, ALIASES) == set()

    def test_whitespace_only(self):
        assert _resolve_color_to_canonical("   ", CANONICAL, ALIASES) == set()

    def test_no_match(self):
        assert _resolve_color_to_canonical("fluorescent", CANONICAL, ALIASES) == set()

    def test_alias_takes_precedence_over_substring(self):
        # "scarlet" is aliased to "red"; substring would find nothing
        assert _resolve_color_to_canonical("scarlet", CANONICAL, ALIASES) == {"red"}

    def test_canonical_takes_precedence_over_alias(self):
        # "white" is both canonical AND could be an alias target — canonical wins
        assert _resolve_color_to_canonical("white", CANONICAL, ALIASES) == {"white"}

    def test_nearly_orange(self):
        # "nearly orange" — not in aliases, substring: "orange" is in it
        result = _resolve_color_to_canonical("nearly orange", CANONICAL, ALIASES)
        assert result == {"orange"}

    def test_cinnamon_brown(self):
        result = _resolve_color_to_canonical("cinnamon brown", CANONICAL, ALIASES)
        assert "brown" in result

    def test_reddish_orange(self):
        result = _resolve_color_to_canonical("reddish orange", CANONICAL, ALIASES)
        assert "orange" in result
        assert "red" in result


class TestLoadCorrections:
    def test_loads_cap_color_corrections(self):
        corrections = _load_corrections("cap_color")
        assert len(corrections) > 0
        assert "Russula virescens" in corrections
        assert "olive" in corrections["Russula virescens"]

    def test_unknown_feature_returns_empty(self):
        corrections = _load_corrections("nonexistent_feature")
        assert corrections == {}

    def test_corrections_are_sets(self):
        corrections = _load_corrections("cap_color")
        for species, colors in corrections.items():
            assert isinstance(colors, set), f"{species} colors should be a set"

    def test_multi_color_correction(self):
        corrections = _load_corrections("cap_color")
        # Geastrum fornicatum has [white, grey]
        assert corrections["Geastrum fornicatum"] == {"white", "grey"}

    def test_all_14_species_present(self):
        corrections = _load_corrections("cap_color")
        expected = {
            "Russula virescens", "Hygrocybe conica", "Geastrum fornicatum",
            "Guepiniopsis alpina", "Hygrocybe psittacina", "Psilocybe semilanceata",
            "Geastrum rufescens", "Laccaria bicolor", "Amanita echinocephala",
            "Hericium alpestre", "Hydnum albidum", "Leucoagaricus rubrotinctus",
            "Amanita excelsa", "Russula claroflava",
        }
        assert set(corrections.keys()) == expected
