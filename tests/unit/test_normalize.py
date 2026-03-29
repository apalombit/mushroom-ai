"""Unit tests for ingestion.normalize — deterministic pre-merge logic."""


import ingestion.normalize as _norm
from ingestion.normalize import normalize_features, pre_merge_list_fields, pre_merge_numeric_ranges

# Disable semantic model loading for all unit tests
_norm._SEMANTIC_MATCHING = False


# ---------------------------------------------------------------------------
# normalize_features — alias → canonical
# ---------------------------------------------------------------------------


def test_normalize_alias_to_canonical():
    features = {"ecology": {"trophic_mode": "ectomycorrhizal"}}
    out = normalize_features(features)
    assert out["ecology"]["trophic_mode"] == "mycorrhizal"


def test_normalize_alias_case_insensitive():
    features = {"ecology": {"trophic_mode": "Ectomycorrhizal"}}
    out = normalize_features(features)
    assert out["ecology"]["trophic_mode"] == "mycorrhizal"


def test_normalize_canonical_unchanged():
    features = {"ecology": {"trophic_mode": "mycorrhizal"}}
    out = normalize_features(features)
    assert out["ecology"]["trophic_mode"] == "mycorrhizal"


def test_normalize_null_unchanged():
    features = {"ecology": {"trophic_mode": None}}
    out = normalize_features(features)
    assert out["ecology"]["trophic_mode"] is None


def test_normalize_unknown_value_set_to_none():
    # Non-canonical, non-alias values are now dropped to None (enforcement pass)
    features = {"ecology": {"trophic_mode": "some_free_text"}}
    out = normalize_features(features)
    assert out["ecology"]["trophic_mode"] is None


def test_normalize_nested_field():
    features = {"gills": {"thickness": "broad"}}
    out = normalize_features(features)
    assert out["gills"]["thickness"] == "thick"


def test_normalize_multiple_aliases_same_vocab():
    features = {
        "chemical": {
            "KOH_cap": "no reaction",
            "KOH_flesh": "yellowing",
        }
    }
    out = normalize_features(features)
    assert out["chemical"]["KOH_cap"] == "negative"
    assert out["chemical"]["KOH_flesh"] == "yellow"


def test_normalize_does_not_mutate_input():
    features = {"ecology": {"trophic_mode": "ectomycorrhizal"}}
    _ = normalize_features(features)
    assert features["ecology"]["trophic_mode"] == "ectomycorrhizal"


# ---------------------------------------------------------------------------
# pre_merge_numeric_ranges
# ---------------------------------------------------------------------------


def test_pre_merge_numeric_min_max():
    obs1 = {"cap": {"diameter_min_cm": 5.0, "diameter_max_cm": 12.0}}
    obs2 = {"cap": {"diameter_min_cm": 3.0, "diameter_max_cm": 15.0}}
    result = pre_merge_numeric_ranges([obs1, obs2])
    assert result["cap"]["diameter_min_cm"] == 3.0
    assert result["cap"]["diameter_max_cm"] == 15.0


def test_pre_merge_numeric_all_null():
    obs1 = {"cap": {"diameter_min_cm": None}}
    obs2 = {"cap": {"diameter_min_cm": None}}
    result = pre_merge_numeric_ranges([obs1, obs2])
    # all-null → field absent from result
    assert "diameter_min_cm" not in result.get("cap", {})


def test_pre_merge_numeric_partial_null():
    obs1 = {"cap": {"diameter_min_cm": None, "diameter_max_cm": 10.0}}
    obs2 = {"cap": {"diameter_min_cm": 4.0, "diameter_max_cm": None}}
    result = pre_merge_numeric_ranges([obs1, obs2])
    assert result["cap"]["diameter_min_cm"] == 4.0
    assert result["cap"]["diameter_max_cm"] == 10.0


# ---------------------------------------------------------------------------
# pre_merge_list_fields
# ---------------------------------------------------------------------------


def test_pre_merge_list_union():
    obs1 = {"ecology": {"fruiting_months": ["October", "November"]}}
    obs2 = {"ecology": {"fruiting_months": ["August", "September", "October"]}}
    result = pre_merge_list_fields([obs1, obs2])
    # union, sorted by calendar order
    assert result["ecology"]["fruiting_months"] == [
        "August", "September", "October", "November"
    ]


def test_pre_merge_list_dedup():
    obs1 = {"ecology": {"associated_trees": ["birch", "pine"]}}
    obs2 = {"ecology": {"associated_trees": ["pine", "spruce"]}}
    result = pre_merge_list_fields([obs1, obs2])
    trees = result["ecology"]["associated_trees"]
    assert "pine" in trees
    assert trees.count("pine") == 1


def test_pre_merge_list_top_level():
    obs1 = {"known_toxins": ["amatoxins"]}
    obs2 = {"known_toxins": ["gyromitrin"]}
    result = pre_merge_list_fields([obs1, obs2])
    assert set(result["known_toxins"]) == {"amatoxins", "gyromitrin"}


def test_pre_merge_list_empty_all_null():
    obs1 = {"ecology": {"fruiting_months": None}}
    obs2 = {}
    result = pre_merge_list_fields([obs1, obs2])
    assert "fruiting_months" not in result.get("ecology", {})


# ---------------------------------------------------------------------------
# Enforcement tests (semantic matching disabled via _SEMANTIC_MATCHING = False)
# ---------------------------------------------------------------------------


def test_non_canonical_term_set_to_none():
    features = {"cap": {"shape": "fan-shaped"}}
    out = normalize_features(features)
    assert out["cap"]["shape"] is None


def test_canonical_term_preserved():
    features = {"cap": {"shape": "convex"}}
    out = normalize_features(features)
    assert out["cap"]["shape"] == "convex"


def test_alias_resolves_before_enforcement():
    features = {"cap": {"shape": "funnel-shaped"}}
    out = normalize_features(features)
    assert out["cap"]["shape"] == "infundibuliform"


def test_no_enforce_field_preserved():
    # spore_print_color is in _NO_ENFORCE — non-canonical values must not be dropped
    features = {"spore_print_color": "olive-brown"}
    out = normalize_features(features)
    assert out["spore_print_color"] == "olive-brown"
