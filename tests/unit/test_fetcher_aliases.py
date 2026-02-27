"""
Unit tests for alias fallback in slug-based fetchers.

All tests are pure unit tests — no network calls. HTTP is mocked via unittest.mock.
"""

from unittest.mock import MagicMock, call, patch

import pytest

from ingestion.sources import firstnature, mushroomexpert, ultimatemushroom


# ---------------------------------------------------------------------------
# mushroomexpert
# ---------------------------------------------------------------------------


class TestMushroomexpertAliases:
    def _make_page(self, text: str = "Some content", url: str = "http://example.com") -> tuple:
        """Return a (html, url) tuple that _get_page would return."""
        html = f"<html><body><td width='380'><p>{text}</p></td></body></html>"
        return html, url

    def test_primary_hit_aliases_not_tried(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mushroomexpert, "CACHE_DIR", tmp_path)
        page = self._make_page("Primary content", "http://mushroomexpert.com/amanita_muscaria.html")

        with patch.object(mushroomexpert, "_get_page", return_value=page) as mock_get:
            result = mushroomexpert.fetch_species_page("Amanita muscaria", aliases=["Amanita alias"])

        assert result is not None
        assert result["text"] == "Primary content"
        # _get_page called once only (primary)
        assert mock_get.call_count == 1

    def test_primary_miss_alias_hit_cached_under_canonical(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mushroomexpert, "CACHE_DIR", tmp_path)
        page = self._make_page("Alias content", "http://mushroomexpert.com/lentinus_edodes.html")

        def _get_page_side_effect(url: str, depth: int = 3):
            if "lentinula" in url:
                return None
            return page

        with patch.object(mushroomexpert, "_get_page", side_effect=_get_page_side_effect):
            result = mushroomexpert.fetch_species_page(
                "Lentinula edodes", aliases=["Lentinus edodes"]
            )

        assert result is not None
        assert result["text"] == "Alias content"

        # Cache written under canonical name
        cache_file = mushroomexpert._cache_path("Lentinula edodes")
        assert cache_file.exists()
        # Alias cache file must NOT exist
        alias_cache = mushroomexpert._cache_path("Lentinus edodes")
        assert not alias_cache.exists()

    def test_primary_miss_no_aliases_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mushroomexpert, "CACHE_DIR", tmp_path)

        with patch.object(mushroomexpert, "_get_page", return_value=None):
            result = mushroomexpert.fetch_species_page("Unknown species")

        assert result is None

    def test_all_aliases_miss_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mushroomexpert, "CACHE_DIR", tmp_path)

        with patch.object(mushroomexpert, "_get_page", return_value=None):
            result = mushroomexpert.fetch_species_page(
                "Unknown species", aliases=["Alias one", "Alias two"]
            )

        assert result is None

    def test_second_alias_used_when_first_misses(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mushroomexpert, "CACHE_DIR", tmp_path)
        page = self._make_page("Second alias content", "http://mushroomexpert.com/alias2.html")

        primary_url = mushroomexpert._build_url("Primary species")
        alias1_url = mushroomexpert._build_url("Alias one")

        def _get_page_side_effect(url: str, depth: int = 3):
            if url in (primary_url, alias1_url):
                return None
            return page

        with patch.object(mushroomexpert, "_get_page", side_effect=_get_page_side_effect):
            result = mushroomexpert.fetch_species_page(
                "Primary species", aliases=["Alias one", "Alias two"]
            )

        assert result is not None
        assert result["text"] == "Second alias content"


# ---------------------------------------------------------------------------
# firstnature
# ---------------------------------------------------------------------------


class TestFirstnatureAliases:
    def _make_response(self, text: str = "Some content") -> MagicMock:
        resp = MagicMock()
        resp.text = f"<html><body><p>{text}</p></body></html>"
        resp.status_code = 200
        return resp

    def _make_404(self) -> None:
        return None

    def test_primary_hit_aliases_not_tried(self, tmp_path, monkeypatch):
        monkeypatch.setattr(firstnature, "CACHE_DIR", tmp_path)
        resp = self._make_response("Primary content")

        with patch.object(firstnature, "_fetch_url", return_value=resp) as mock_fetch:
            result = firstnature.fetch_species_page("Amanita muscaria", aliases=["Amanita alias"])

        assert result is not None
        assert result["text"] == "Primary content"
        assert mock_fetch.call_count == 1

    def test_primary_miss_alias_hit_cached_under_canonical(self, tmp_path, monkeypatch):
        monkeypatch.setattr(firstnature, "CACHE_DIR", tmp_path)
        alias_resp = self._make_response("Alias content")

        primary_url = firstnature._build_url("Lentinula edodes")
        alias_url = firstnature._build_url("Lentinus edodes")

        def _fetch_side(url: str):
            return None if url == primary_url else alias_resp

        with patch.object(firstnature, "_fetch_url", side_effect=_fetch_side):
            result = firstnature.fetch_species_page(
                "Lentinula edodes", aliases=["Lentinus edodes"]
            )

        assert result is not None
        assert result["text"] == "Alias content"
        assert result["url"] == alias_url

        cache_file = firstnature._cache_path("Lentinula edodes")
        assert cache_file.exists()
        alias_cache = firstnature._cache_path("Lentinus edodes")
        assert not alias_cache.exists()

    def test_primary_miss_no_aliases_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(firstnature, "CACHE_DIR", tmp_path)

        with patch.object(firstnature, "_fetch_url", return_value=None):
            result = firstnature.fetch_species_page("Unknown species")

        assert result is None

    def test_all_aliases_miss_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(firstnature, "CACHE_DIR", tmp_path)

        with patch.object(firstnature, "_fetch_url", return_value=None):
            result = firstnature.fetch_species_page(
                "Unknown species", aliases=["Alias one", "Alias two"]
            )

        assert result is None


# ---------------------------------------------------------------------------
# ultimatemushroom
# ---------------------------------------------------------------------------


class TestUltimushroomAliases:
    _INDEX = {
        "lentinus edodes": "https://ultimate-mushroom.com/edible/42-lentinus-edodes.html",
        "amanita muscaria": "https://ultimate-mushroom.com/poisonous/1-amanita-muscaria.html",
    }

    def _make_response(self, text: str = "Some content") -> MagicMock:
        resp = MagicMock()
        resp.text = f"<html><body><article><p>{text}</p></article></body></html>"
        resp.status_code = 200
        return resp

    def test_primary_hit_aliases_not_tried(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ultimatemushroom, "CACHE_DIR", tmp_path)
        monkeypatch.setattr(ultimatemushroom, "INDEX_CACHE", tmp_path / "index.json")

        resp = self._make_response("Primary content")

        with patch.object(ultimatemushroom, "build_index", return_value=self._INDEX):
            with patch("requests.get", return_value=resp):
                result = ultimatemushroom.fetch_species_page(
                    "Amanita muscaria", aliases=["Amanita alias"]
                )

        assert result is not None
        assert result["text"] == "Primary content"

    def test_primary_not_in_index_alias_hit_cached_under_canonical(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ultimatemushroom, "CACHE_DIR", tmp_path)
        monkeypatch.setattr(ultimatemushroom, "INDEX_CACHE", tmp_path / "index.json")

        resp = self._make_response("Alias content")

        with patch.object(ultimatemushroom, "build_index", return_value=self._INDEX):
            with patch("requests.get", return_value=resp):
                result = ultimatemushroom.fetch_species_page(
                    "Lentinula edodes", aliases=["Lentinus edodes"]
                )

        assert result is not None
        assert result["text"] == "Alias content"

        cache_file = ultimatemushroom._cache_path("Lentinula edodes")
        assert cache_file.exists()
        alias_cache = ultimatemushroom._cache_path("Lentinus edodes")
        assert not alias_cache.exists()

    def test_primary_not_in_index_no_aliases_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ultimatemushroom, "CACHE_DIR", tmp_path)
        monkeypatch.setattr(ultimatemushroom, "INDEX_CACHE", tmp_path / "index.json")

        with patch.object(ultimatemushroom, "build_index", return_value=self._INDEX):
            result = ultimatemushroom.fetch_species_page("Unknown species")

        assert result is None

    def test_all_aliases_not_in_index_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ultimatemushroom, "CACHE_DIR", tmp_path)
        monkeypatch.setattr(ultimatemushroom, "INDEX_CACHE", tmp_path / "index.json")

        with patch.object(ultimatemushroom, "build_index", return_value=self._INDEX):
            result = ultimatemushroom.fetch_species_page(
                "Unknown species", aliases=["Also unknown", "Still unknown"]
            )

        assert result is None
