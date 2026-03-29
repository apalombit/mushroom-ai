"""Unit tests for feature embedding. No database or LLM required."""

import numpy as np

from ingestion.rubric import (
    _NUMERIC_FIELD_PATHS,
    EMBEDDING_GROUPS,
    GROUP_SLOTS,
    MORPHOLOGICAL_FIELDS,
    NUMERIC_FIELDS,
    _load_profile,
    features_to_text,
)


def test_features_to_text_nonempty(sample_features_json):
    """features_to_text produces non-empty text for a complete features dict."""
    text = features_to_text(sample_features_json, MORPHOLOGICAL_FIELDS)
    assert len(text) > 0


def test_features_to_text_contains_key_values(sample_features_json):
    """features_to_text includes expected values from the features dict."""
    text = features_to_text(sample_features_json, MORPHOLOGICAL_FIELDS)
    # Cap shape "convex to flat" should appear
    assert "convex" in text.lower()


def test_features_to_text_ecological(sample_features_json):
    """features_to_text works for ecological fields."""
    text = features_to_text(sample_features_json, EMBEDDING_GROUPS["ecological"])
    assert len(text) > 0
    assert "substrate" in text.lower() or "soil" in text.lower()


def test_features_to_text_taxonomic(sample_features_json):
    """features_to_text works for taxonomic fields."""
    text = features_to_text(sample_features_json, EMBEDDING_GROUPS["taxonomic"])
    assert "Amanitaceae" in text or "Amanita" in text


def test_features_to_text_skips_none_values():
    """features_to_text skips None values and doesn't produce 'None' in output."""
    features = {
        "cap": {
            "shape": "convex",
            "colors": [],
            "surface_texture": None,
            "scales_or_warts": None,
            "diameter_min_cm": None,
            "diameter_max_cm": None,
        },
        "gills": {},
        "stem": {},
        "veil": {},
        "volva": {},
        "flesh": {},
    }
    text = features_to_text(features, MORPHOLOGICAL_FIELDS)
    assert "None" not in text
    assert "none" not in text.lower()
    assert "convex" in text


def test_features_to_text_empty_input():
    """features_to_text returns empty string for empty features."""
    text = features_to_text({}, MORPHOLOGICAL_FIELDS)
    assert text == ""


def test_similarity_ordering():
    """
    Amanita muscaria (Amanita, red cap, volva) should be more similar on macro visual
    to Amanita caesarea (Amanita, orange cap, volva) than to Boletus edulis (Boletus, pores).
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-mpnet-base-v2")

    muscaria = {
        "cap": {"shape": "convex", "colors": ["red", "orange"]},
        "hymenium": {"type": "gills"},
        "overall_size_class": "large",
        "overall_body_form": "agaricoid",
        "spore_print_color": "white",
    }
    caesarea = {
        "cap": {"shape": "convex", "colors": ["orange", "yellow"]},
        "hymenium": {"type": "gills"},
        "overall_size_class": "large",
        "overall_body_form": "agaricoid",
        "spore_print_color": "white",
    }
    edulis = {
        "cap": {"shape": "convex", "colors": ["brown"]},
        "hymenium": {"type": "pores"},
        "overall_size_class": "large",
        "overall_body_form": "boletoid",
        "spore_print_color": "olive-brown",
    }

    # Use cap_color group (colors) which distinguishes these species
    cap_viz_fields = EMBEDDING_GROUPS["cap_color"]
    text_m = features_to_text(muscaria, cap_viz_fields)
    text_c = features_to_text(caesarea, cap_viz_fields)
    text_e = features_to_text(edulis, cap_viz_fields)

    emb_m = model.encode(text_m)
    emb_c = model.encode(text_c)
    emb_e = model.encode(text_e)

    def cosine_sim(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    sim_mc = cosine_sim(emb_m, emb_c)
    sim_me = cosine_sim(emb_m, emb_e)

    assert sim_mc > sim_me, (
        f"Expected muscaria-caesarea ({sim_mc:.3f}) > muscaria-edulis ({sim_me:.3f})"
    )


# ---------------------------------------------------------------------------
# Sub-group coverage tests
# ---------------------------------------------------------------------------


def test_subgroups_cover_all_morphological_fields():
    """Union of active profile groups + numeric paths == MORPHOLOGICAL_FIELDS."""
    union = set()
    for fields in EMBEDDING_GROUPS.values():
        union.update(fields)
    union.update(_NUMERIC_FIELD_PATHS)
    assert union == set(MORPHOLOGICAL_FIELDS)


def test_subgroups_no_overlap():
    """Profile groups are mutually exclusive (no field in two groups)."""
    profile = _load_profile("default")
    seen: set[str] = set()
    for fields in profile.values():
        overlap = seen & set(fields)
        assert not overlap, f"Overlapping fields: {overlap}"
        seen.update(fields)


def test_no_numeric_fields_in_embedding_groups():
    """Numeric fields must not appear in any embedding group."""
    numeric = set(_NUMERIC_FIELD_PATHS)
    for name, fields in EMBEDDING_GROUPS.items():
        overlap = numeric & set(fields)
        assert not overlap, f"Numeric fields in {name}: {overlap}"


def test_embedding_groups_matches_group_slots():
    """Active EMBEDDING_GROUPS keys match GROUP_SLOTS and has at least one entry."""
    assert len(EMBEDDING_GROUPS) > 0
    assert set(EMBEDDING_GROUPS.keys()) == set(GROUP_SLOTS)


def test_each_subgroup_produces_nonempty_text(sample_features_json):
    """Non-bolete-specific sub-groups produce non-empty text for a gill species."""
    # pores/pores_bruise: intentionally empty for gill-bearing species (Amanita fixture)
    # cap_bruise/stem_bruise: Amanita muscaria does not exhibit bruising reactions
    skip = {"pores", "pores_bruise", "cap_bruise", "stem_bruise"}
    for name, fields in EMBEDDING_GROUPS.items():
        if name in skip or not fields:
            continue
        text = features_to_text(sample_features_json, fields)
        assert len(text) > 0, f"Empty text for group '{name}'"


def test_numeric_fields_not_empty():
    """NUMERIC_FIELDS has both range and single entries."""
    assert len(NUMERIC_FIELDS) > 0
    # At least 5 range pairs + 5 single values
    assert len(NUMERIC_FIELDS) >= 10


def test_default_profile_loads():
    """The default profile YAML loads and has all 26 fine-grained groups with non-empty fields."""
    profile = _load_profile("default")
    assert len(profile) == 26
    for fields in profile.values():
        assert len(fields) > 0, "Default profile should have no empty groups"


def test_visual_merged_profile_loads():
    """The visual_merged profile loads and has empty flesh_sensory/microscopic_lab."""
    profile = _load_profile("visual_merged")
    assert len(profile) == 6
    assert len(profile["flesh_sensory"]) == 0
    assert len(profile["microscopic_lab"]) == 0
    assert len(profile["macro_visual"]) > 18  # merged with structural
