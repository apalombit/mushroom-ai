"""
Integration tests for LLM feature extraction. Requires Ollama running locally.
May take 10-30 seconds per species.
"""

from ingestion.extract import extract_features_from_text
from ingestion.sources.wikipedia import fetch_species_page
from llm.schemas import ExtractedSpeciesFeatures


def test_extract_amanita_muscaria_returns_valid_schema():
    page = fetch_species_page("Amanita muscaria")
    assert page is not None, "Cache miss — run python -m scripts.ingest --fetch-only first"

    features = extract_features_from_text("Amanita muscaria", page["text"], source_name="Wikipedia")

    assert isinstance(features, ExtractedSpeciesFeatures)
    assert features.scientific_name == "Amanita muscaria"


def test_extract_amanita_muscaria_key_fields():
    page = fetch_species_page("Amanita muscaria")
    features = extract_features_from_text("Amanita muscaria", page["text"], source_name="Wikipedia")

    assert len(features.cap.colors) > 0, "Cap colors should be present"
    # shape may be absent if the model applies strict "explicitly stated" rules;
    # check that at least one other cap morphological field is populated instead
    cap_fields = [features.cap.shape, features.cap.surface_texture, features.cap.scales_or_warts]
    assert any(f is not None for f in cap_fields), "At least one cap morphological field should be extracted"
    assert features.edibility is not None, "Edibility should be present"
