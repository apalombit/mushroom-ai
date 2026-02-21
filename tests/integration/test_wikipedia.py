"""Integration tests for Wikipedia fetcher. Requires network access."""

from ingestion.sources.wikipedia import fetch_species_page


def test_fetch_known_species_returns_text():
    result = fetch_species_page("Amanita caesarea")
    assert result is not None
    assert len(result["text"]) > 100
    assert result["url"].startswith("https://")


def test_fetch_known_species_contains_expected_content():
    result = fetch_species_page("Amanita caesarea")
    assert result is not None
    assert "Amanita" in result["text"]


def test_fetch_nonexistent_species_returns_none():
    result = fetch_species_page("Nonexistentus fakicus")
    assert result is None
