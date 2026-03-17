"""Unit tests for LLM schema coercion helpers."""

from llm.schemas import ExtractedSpeciesFeatures


def _minimal_features(**overrides) -> dict:
    """Build a minimal valid ExtractedSpeciesFeatures dict."""
    base: dict = {
        "scientific_name": "Amanita muscaria",
        "cap": {},
        "hymenium": {},
        "gills": {},
        "pores": {},
        "tubes": {},
        "stem": {},
        "veil": {},
        "volva": {},
        "flesh": {},
        "spore": {},
        "microscopic": {},
        "chemical": {},
        "ecology": {},
    }
    base.update(overrides)
    return base


def test_string_null_coerced_to_none():
    data = _minimal_features(cap={"shape": "null"}, stem={"color": "NULL"})
    obj = ExtractedSpeciesFeatures.model_validate(data)
    assert obj.cap.shape is None
    assert obj.stem.color is None


def test_nested_string_null():
    data = _minimal_features(stem={"color": "null"})
    obj = ExtractedSpeciesFeatures.model_validate(data)
    assert obj.stem.color is None


def test_valid_string_not_coerced():
    data = _minimal_features(cap={"shape": "convex"})
    obj = ExtractedSpeciesFeatures.model_validate(data)
    assert obj.cap.shape == "convex"


def test_string_none_coerced_to_none():
    data = _minimal_features(cap={"shape": "none"})
    obj = ExtractedSpeciesFeatures.model_validate(data)
    assert obj.cap.shape is None


def test_bool_field_string_null_becomes_none_not_false():
    """String 'null' in a bool|None field must become None, not False."""
    data = _minimal_features(veil={"cortina_present": "null"})
    obj = ExtractedSpeciesFeatures.model_validate(data)
    assert obj.veil.cortina_present is None
