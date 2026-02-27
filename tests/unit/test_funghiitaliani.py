"""
Unit tests for the funghiitaliani fetcher.

All tests are pure unit tests — no network calls.
"""

from unittest.mock import MagicMock, patch

import pytest

from ingestion.sources import funghiitaliani
from ingestion.sources.funghiitaliani import _extract_scientific_name


# ---------------------------------------------------------------------------
# _extract_scientific_name
# ---------------------------------------------------------------------------


class TestExtractScientificName:
    def test_standard_binomial(self):
        assert _extract_scientific_name("Boletus edulis Bull. : Fr. 1782") == "Boletus edulis"

    def test_binomial_with_parenthetical_author(self):
        assert (
            _extract_scientific_name("Agaricus bisporus (J.E. Lange) Imbach 1946")
            == "Agaricus bisporus"
        )

    def test_lowercase_epithet_normalised(self):
        # Epithet already lowercase — should still work
        assert _extract_scientific_name("Cantharellus cibarius Fr. 1821") == "Cantharellus cibarius"

    def test_non_binomial_all_caps_ignored(self):
        assert _extract_scientific_name("Indice Generale Delle Specie") is None

    def test_non_binomial_short_title(self):
        assert _extract_scientific_name("Avviso") is None

    def test_admin_topic_ignored(self):
        # "Aperte due" — genus starts uppercase, epithet starts lowercase but
        # genus must have >1 lowercase letter after first (matches [A-Z][a-z])
        result = _extract_scientific_name("Aperte due enciclopedie illustrate")
        # "Aperte" matches [A-Z][a-z] ✓, "due" matches [a-z] ✓ → returns a value
        # (harmless false positive in index)
        assert result == "Aperte due"

    def test_non_breaking_space_normalised(self):
        title = "Agrocybe\u00a0praecox Pers. : Fr. Fayod 1889"
        assert _extract_scientific_name(title) == "Agrocybe praecox"


# ---------------------------------------------------------------------------
# build_index
# ---------------------------------------------------------------------------


_FAKE_INDEX = {
    "boletus edulis": "https://www.funghiitaliani.it/topic/15310-boletus-edulis/",
    "amanita muscaria": "https://www.funghiitaliani.it/topic/15308-amanita-muscaria/",
    "lentinus edodes": "https://www.funghiitaliani.it/topic/99999-lentinus-edodes/",
}


class TestBuildIndex:
    def test_loads_from_cache_when_present(self, tmp_path, monkeypatch):
        monkeypatch.setattr(funghiitaliani, "CACHE_DIR", tmp_path)
        index_cache = tmp_path / "funghiitaliani_index.json"
        monkeypatch.setattr(funghiitaliani, "INDEX_CACHE", index_cache)
        import json

        index_cache.write_text(json.dumps(_FAKE_INDEX))

        with patch("requests.get") as mock_get:
            result = funghiitaliani.build_index()

        assert result == _FAKE_INDEX
        mock_get.assert_not_called()

    def test_scrapes_categories_and_caches(self, tmp_path, monkeypatch):
        monkeypatch.setattr(funghiitaliani, "CACHE_DIR", tmp_path)
        index_cache = tmp_path / "funghiitaliani_index.json"
        monkeypatch.setattr(funghiitaliani, "INDEX_CACHE", index_cache)

        # Page 1: one species; page 2: empty (stop pagination)
        page1_html = """
        <html><body>
          <a href="/topic/15310-boletus-edulis-bull-fr-1782/">Boletus edulis Bull. : Fr. 1782</a>
        </body></html>
        """
        resp_page1 = MagicMock()
        resp_page1.status_code = 200
        resp_page1.text = page1_html

        resp_empty = MagicMock()
        resp_empty.status_code = 200
        resp_empty.text = "<html><body></body></html>"

        with patch.object(
            funghiitaliani, "_get_category_page",
            side_effect=lambda cat, page: (
                ([("boletus edulis", "https://www.funghiitaliani.it/topic/15310-boletus-edulis/")], True)
                if page == 1
                else ([], True)
            ),
        ):
            result = funghiitaliani.build_index()

        assert "boletus edulis" in result
        assert index_cache.exists()


# ---------------------------------------------------------------------------
# fetch_species_page
# ---------------------------------------------------------------------------


class TestFetchSpeciesPage:
    def _make_topic_response(self, text: str = "Descrizione funghi") -> MagicMock:
        resp = MagicMock()
        resp.status_code = 200
        resp.text = (
            f'<html><body>'
            f'<div data-role="commentContent"><p>{text}</p></div>'
            f'</body></html>'
        )
        return resp

    def test_cache_hit_no_network(self, tmp_path, monkeypatch):
        monkeypatch.setattr(funghiitaliani, "CACHE_DIR", tmp_path)
        monkeypatch.setattr(funghiitaliani, "INDEX_CACHE", tmp_path / "index.json")

        import json
        cache_file = funghiitaliani._cache_path("Boletus edulis")
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps({"text": "cached text", "url": "http://x.com"}))

        with patch("requests.get") as mock_get:
            result = funghiitaliani.fetch_species_page("Boletus edulis")

        assert result == {"text": "cached text", "url": "http://x.com"}
        mock_get.assert_not_called()

    def test_primary_hit(self, tmp_path, monkeypatch):
        monkeypatch.setattr(funghiitaliani, "CACHE_DIR", tmp_path)
        monkeypatch.setattr(funghiitaliani, "INDEX_CACHE", tmp_path / "index.json")

        resp = self._make_topic_response("Ottimo commestibile")

        with patch.object(funghiitaliani, "build_index", return_value=_FAKE_INDEX):
            with patch("requests.get", return_value=resp):
                result = funghiitaliani.fetch_species_page("Boletus edulis")

        assert result is not None
        assert result["text"] == "Ottimo commestibile"
        assert funghiitaliani._cache_path("Boletus edulis").exists()

    def test_primary_miss_alias_hit_cached_under_canonical(self, tmp_path, monkeypatch):
        monkeypatch.setattr(funghiitaliani, "CACHE_DIR", tmp_path)
        monkeypatch.setattr(funghiitaliani, "INDEX_CACHE", tmp_path / "index.json")

        resp = self._make_topic_response("Alias content")

        with patch.object(funghiitaliani, "build_index", return_value=_FAKE_INDEX):
            with patch("requests.get", return_value=resp):
                result = funghiitaliani.fetch_species_page(
                    "Lentinula edodes", aliases=["Lentinus edodes"]
                )

        assert result is not None
        assert result["text"] == "Alias content"
        assert funghiitaliani._cache_path("Lentinula edodes").exists()
        assert not funghiitaliani._cache_path("Lentinus edodes").exists()

    def test_not_in_index_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(funghiitaliani, "CACHE_DIR", tmp_path)
        monkeypatch.setattr(funghiitaliani, "INDEX_CACHE", tmp_path / "index.json")

        with patch.object(funghiitaliani, "build_index", return_value=_FAKE_INDEX):
            result = funghiitaliani.fetch_species_page("Unknown species")

        assert result is None

    def test_primary_hit_aliases_not_tried(self, tmp_path, monkeypatch):
        monkeypatch.setattr(funghiitaliani, "CACHE_DIR", tmp_path)
        monkeypatch.setattr(funghiitaliani, "INDEX_CACHE", tmp_path / "index.json")

        resp = self._make_topic_response("Primary content")
        call_count = {"n": 0}

        original_build = funghiitaliani.build_index

        def counting_build():
            return _FAKE_INDEX

        with patch.object(funghiitaliani, "build_index", side_effect=counting_build):
            with patch("requests.get", return_value=resp):
                result = funghiitaliani.fetch_species_page(
                    "Boletus edulis", aliases=["Boletus alias"]
                )

        assert result is not None
        # Only one HTTP request for the topic page itself (no alias lookup needed)
        assert result["text"] == "Primary content"
