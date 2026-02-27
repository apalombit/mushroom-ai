"""
first-nature.com source fetcher.

Fetches species pages from first-nature.com (English mycology reference).
URL pattern: https://www.first-nature.com/fungi/{genus}-{species}.php

Cache: data/cache/first-nature_{safe_name}.json
"""

import json
import logging
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

SOURCE_NAME = "first-nature"
CACHE_DIR = Path("data/cache")
RATE_LIMIT_SECONDS = 1.0

logger = logging.getLogger(__name__)


def _safe_filename(name: str) -> str:
    return re.sub(r"[^\w\-]", "_", name)


def _cache_path(scientific_name: str) -> Path:
    return CACHE_DIR / f"first-nature_{_safe_filename(scientific_name)}.json"


def _load_cache(scientific_name: str) -> dict | None:
    path = _cache_path(scientific_name)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def _save_cache(scientific_name: str, data: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(_cache_path(scientific_name), "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _build_url(scientific_name: str) -> str:
    """Convert 'Amanita muscaria' → 'https://www.first-nature.com/fungi/amanita-muscaria.php'."""
    slug = scientific_name.lower().replace(" ", "-")
    # Remove any special chars except hyphens
    slug = re.sub(r"[^\w\-]", "", slug)
    return f"https://www.first-nature.com/fungi/{slug}.php"


def _fetch_url(url: str) -> requests.Response | None:
    """Fetch a URL, returning the Response or None on 404/error."""
    try:
        time.sleep(RATE_LIMIT_SECONDS)
        response = requests.get(
            url,
            headers={"User-Agent": "mushroom-ai/1.0 (educational project)"},
            timeout=15,
        )
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response
    except requests.RequestException as e:
        logger.warning("HTTP error fetching %s (first-nature): %s", url, e)
        return None


def fetch_species_page(scientific_name: str, aliases: list[str] | None = None) -> dict | None:
    """
    Fetch the first-nature.com page for a species.

    Returns {"text": <page text>, "url": <page url>} or None if not found.
    On primary miss, tries aliases in order and caches under the canonical name.
    Results are cached — delete the cache file to force re-fetch.
    """
    cached = _load_cache(scientific_name)
    if cached is not None:
        logger.debug("Cache hit for %s (first-nature)", scientific_name)
        return cached

    response = _fetch_url(_build_url(scientific_name))
    used_url = _build_url(scientific_name)

    if response is None and aliases:
        for alias in aliases:
            alias_url = _build_url(alias)
            response = _fetch_url(alias_url)
            if response is not None:
                used_url = alias_url
                logger.debug("Alias hit for %s via '%s' (first-nature)", scientific_name, alias)
                break

    if response is None:
        logger.debug("Not found on first-nature: %s", scientific_name)
        return None

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove nav/header/footer noise
    for tag in soup.find_all(["nav", "header", "footer", "script", "style"]):
        tag.decompose()

    # Extract paragraph text from the page body
    paragraphs = []
    for p in soup.find_all("p"):
        text = p.get_text(separator=" ", strip=True)
        if text:
            paragraphs.append(text)

    text = "\n\n".join(paragraphs).strip()
    if not text:
        logger.debug("Empty content for %s (first-nature)", scientific_name)
        return None

    result = {"text": text, "url": used_url}
    _save_cache(scientific_name, result)
    return result
