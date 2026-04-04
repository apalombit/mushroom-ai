"""Tests for ui.i18n translation module."""

from unittest.mock import patch

import yaml

from ui.i18n import _DIR, _load, common_names_it, t, tv


def _with_lang(lang: str):
    """Return a mock for st.session_state with the given lang."""
    return patch("ui.i18n.st.session_state", {"lang": lang})


class TestT:
    def test_english_key(self):
        with _with_lang("en"):
            assert t("page_title") == "Mushroom Lookalikes Finder"

    def test_italian_key(self):
        with _with_lang("it"):
            assert t("page_title") == "Trova Sosia Fungini"

    def test_fallback_to_english(self):
        """If a key exists in en.yaml but not it.yaml, t() falls back to English."""
        en_keys = set(_load("en").keys())
        it_keys = set(_load("it").keys())
        # All en keys should exist in it (full coverage)
        assert en_keys == it_keys, f"Missing in it.yaml: {en_keys - it_keys}"

    def test_fallback_to_raw_key(self):
        with _with_lang("en"):
            assert t("nonexistent_key_xyz") == "nonexistent_key_xyz"

    def test_italian_fallback_to_raw_key(self):
        with _with_lang("it"):
            assert t("nonexistent_key_xyz") == "nonexistent_key_xyz"


class TestTv:
    def test_english_passthrough(self):
        with _with_lang("en"):
            assert tv("convex") == "convex"

    def test_italian_shared_color(self):
        with _with_lang("it"):
            assert tv("white") == "bianco"

    def test_italian_feature_specific(self):
        with _with_lang("it"):
            assert tv("convex", "cap_shape") == "convesso"

    def test_unknown_value_passthrough(self):
        with _with_lang("it"):
            assert tv("some_unknown_xyz") == "some_unknown_xyz"

    def test_empty_value(self):
        with _with_lang("it"):
            assert tv("") == ""


class TestCommonNamesIt:
    def test_known_species(self):
        names = common_names_it("Boletus edulis")
        assert len(names) > 0
        assert "Porcino" in names

    def test_unknown_species(self):
        assert common_names_it("Nonexistentus fakeus") == []


class TestCoverage:
    def test_all_en_keys_exist_in_it(self):
        en = _load("en")
        it = _load("it")
        missing = set(en.keys()) - set(it.keys())
        assert not missing, f"Keys in en.yaml missing from it.yaml: {missing}"

    def test_all_it_keys_exist_in_en(self):
        en = _load("en")
        it = _load("it")
        extra = set(it.keys()) - set(en.keys())
        assert not extra, f"Keys in it.yaml not in en.yaml: {extra}"

    def test_vocab_it_has_shared_section(self):
        path = _DIR / "vocab_it.yaml"
        with open(path) as f:
            vocab = yaml.safe_load(f)
        assert "_shared" in vocab
        assert "white" in vocab["_shared"]

    def test_common_names_file_loads(self):
        path = _DIR / "common_names_it.yaml"
        with open(path) as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict)
        assert len(data) > 50
