"""Unit tests for feature embedding. No database or LLM required."""

import numpy as np

from ingestion.rubric import (
    ECOLOGICAL_FIELDS,
    MORPHOLOGICAL_FIELDS,
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
    Amanita muscaria (Amanita, red cap, volva) should be more morphologically similar
    to Amanita caesarea (Amanita, orange cap, volva) than to Boletus edulis (Boletus, pores).
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")

    muscaria = {
        "cap": {"shape": "convex", "colors": ["red", "orange"]},
        "gills": {"attachment": "free", "color": "white"},
        "veil": {"present": True, "type": "ring-like"},
        "volva": {"present": True, "type": "sac-like"},
        "stem": {"color": "white"},
        "flesh": {"color": "white"},
        "overall_size_class": "large",
    }
    caesarea = {
        "cap": {"shape": "convex", "colors": ["orange", "yellow"]},
        "gills": {"attachment": "free", "color": "yellow"},
        "veil": {"present": True, "type": "ring-like"},
        "volva": {"present": True, "type": "sac-like"},
        "stem": {"color": "yellow"},
        "flesh": {"color": "yellow"},
        "overall_size_class": "large",
    }
    edulis = {
        "cap": {"shape": "convex", "colors": ["brown"]},
        "gills": {"hymenium_type": "pores", "color": "white"},
        "veil": {"present": False},
        "volva": {"present": False},
        "stem": {"color": "brown", "surface_texture": "reticulate"},
        "flesh": {"color": "white"},
        "overall_size_class": "large",
    }

    text_m = features_to_text(muscaria, MORPHOLOGICAL_FIELDS)
    text_c = features_to_text(caesarea, MORPHOLOGICAL_FIELDS)
    text_e = features_to_text(edulis, MORPHOLOGICAL_FIELDS)

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
