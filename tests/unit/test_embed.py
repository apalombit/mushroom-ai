"""Unit tests for feature embedding. No database or LLM required."""

import numpy as np

from ingestion.rubric import (
    _NUMERIC_FIELD_PATHS,
    ECOLOGICAL_FIELDS,
    EMBEDDING_GROUPS,
    FLESH_SENSORY_FIELDS,
    MACRO_VISUAL_FIELDS,
    MICROSCOPIC_LAB_FIELDS,
    MORPHOLOGICAL_FIELDS,
    NUMERIC_FIELDS,
    STRUCTURAL_FIELDS,
    TAXONOMIC_FIELDS,
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
    text = features_to_text(sample_features_json, ECOLOGICAL_FIELDS)
    assert len(text) > 0
    assert "forest" in text.lower() or "birch" in text.lower()


def test_features_to_text_taxonomic(sample_features_json):
    """features_to_text works for taxonomic fields."""
    text = features_to_text(sample_features_json, TAXONOMIC_FIELDS)
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

    text_m = features_to_text(muscaria, MACRO_VISUAL_FIELDS)
    text_c = features_to_text(caesarea, MACRO_VISUAL_FIELDS)
    text_e = features_to_text(edulis, MACRO_VISUAL_FIELDS)

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
# Sub-group coverage tests (Stage 1)
# ---------------------------------------------------------------------------

_MORPH_SUBGROUPS = [
    MACRO_VISUAL_FIELDS,
    STRUCTURAL_FIELDS,
    FLESH_SENSORY_FIELDS,
    MICROSCOPIC_LAB_FIELDS,
]


def test_subgroups_cover_all_morphological_fields():
    """Union of 4 sub-groups + numeric paths == MORPHOLOGICAL_FIELDS."""
    union = set()
    for group in _MORPH_SUBGROUPS:
        union.update(group)
    union.update(_NUMERIC_FIELD_PATHS)
    assert union == set(MORPHOLOGICAL_FIELDS)


def test_subgroups_no_overlap():
    """Sub-groups are mutually exclusive (no field in two groups)."""
    seen: set[str] = set()
    for group in _MORPH_SUBGROUPS:
        overlap = seen & set(group)
        assert not overlap, f"Overlapping fields: {overlap}"
        seen.update(group)


def test_no_numeric_fields_in_embedding_groups():
    """Numeric fields must not appear in any embedding group."""
    numeric = set(_NUMERIC_FIELD_PATHS)
    for name, fields in EMBEDDING_GROUPS.items():
        overlap = numeric & set(fields)
        assert not overlap, f"Numeric fields in {name}: {overlap}"


def test_embedding_groups_has_six_entries():
    """EMBEDDING_GROUPS has exactly 6 entries (4 morph + eco + taxon)."""
    assert len(EMBEDDING_GROUPS) == 6
    expected_keys = {
        "macro_visual",
        "structural",
        "flesh_sensory",
        "microscopic_lab",
        "ecological",
        "taxonomic",
    }
    assert set(EMBEDDING_GROUPS.keys()) == expected_keys


def test_each_subgroup_produces_nonempty_text(sample_features_json):
    """Each morphological sub-group produces non-empty text for a complete features dict."""
    for name, fields in EMBEDDING_GROUPS.items():
        text = features_to_text(sample_features_json, fields)
        assert len(text) > 0, f"Empty text for group '{name}'"


def test_numeric_fields_not_empty():
    """NUMERIC_FIELDS has both range and single entries."""
    assert len(NUMERIC_FIELDS) > 0
    # At least 5 range pairs + 5 single values
    assert len(NUMERIC_FIELDS) >= 10
