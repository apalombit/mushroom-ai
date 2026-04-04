"""Internationalization support for the Streamlit UI."""

from pathlib import Path

import streamlit as st
import yaml

_DIR = Path(__file__).parent
_CACHE: dict[str, dict] = {}

LANGUAGES = {"en": "English", "it": "Italiano"}


def _load(lang: str) -> dict:
    if lang not in _CACHE:
        path = _DIR / f"{lang}.yaml"
        with open(path) as f:
            _CACHE[lang] = yaml.safe_load(f) or {}
    return _CACHE[lang]


def t(key: str) -> str:
    """Translate a UI string key. Falls back: selected lang → English → raw key."""
    lang = st.session_state.get("lang", "en")
    val = _load(lang).get(key)
    if val is not None:
        return val
    val = _load("en").get(key)
    if val is not None:
        return val
    return key


# -- Vocabulary value translation ---------------------------------------------

_VOCAB_IT: dict[str, dict[str, str]] | None = None


def _load_vocab_it() -> dict[str, dict[str, str]]:
    global _VOCAB_IT
    if _VOCAB_IT is not None:
        return _VOCAB_IT
    path = _DIR / "vocab_it.yaml"
    with open(path) as f:
        _VOCAB_IT = yaml.safe_load(f) or {}
    return _VOCAB_IT


def tv(value: str, feature_key: str | None = None) -> str:
    """Translate a feature value (e.g. 'convex' → 'convesso').

    Returns original value when lang is English or no translation exists.
    """
    lang = st.session_state.get("lang", "en")
    if lang == "en" or not value:
        return value
    vocab = _load_vocab_it()
    if feature_key and feature_key in vocab:
        translated = vocab[feature_key].get(value.lower())
        if translated:
            return translated
    translated = vocab.get("_shared", {}).get(value.lower())
    if translated:
        return translated
    return value


# -- Italian common names -----------------------------------------------------

_COMMON_NAMES_IT: dict[str, list[str]] | None = None


def common_names_it(scientific_name: str) -> list[str]:
    """Return Italian common names for a species, or empty list."""
    global _COMMON_NAMES_IT
    if _COMMON_NAMES_IT is None:
        path = _DIR / "common_names_it.yaml"
        with open(path) as f:
            _COMMON_NAMES_IT = yaml.safe_load(f) or {}
    return _COMMON_NAMES_IT.get(scientific_name, [])
