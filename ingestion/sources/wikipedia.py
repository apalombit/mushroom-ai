"""
Wikipedia source fetcher.

Fetches species pages via the Wikipedia API (CC-BY-SA licensed).
Returns raw text content for each species to be parsed by the extraction LLM.

Cache: each fetched page is saved to data/cache/<safe_name>.json so re-runs
are free. Delete a cache file to force re-fetch for that species.
"""

import json
import logging
import re
import time
from pathlib import Path

import requests

SOURCE_NAME = "Wikipedia"
BASE_URL = "https://en.wikipedia.org/w/api.php"
CACHE_DIR = Path("data/cache")
RATE_LIMIT_SECONDS = 0.5

logger = logging.getLogger(__name__)


def _safe_filename(name: str) -> str:
    """Convert a species name to a safe filesystem name."""
    return re.sub(r"[^\w\-]", "_", name)


def _cache_path(scientific_name: str) -> Path:
    return CACHE_DIR / f"{_safe_filename(scientific_name)}.json"


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


def fetch_species_page(scientific_name: str, aliases: list[str] | None = None) -> dict | None:
    """
    Fetch the Wikipedia page for a species by scientific name.

    Returns {"text": <full page text>, "url": <page url>} or None if not found.
    aliases is accepted for interface uniformity but ignored — the Wikipedia API
    handles synonyms via title normalization and redirects.
    Results are cached to data/cache/ — delete the file to force re-fetch.
    """
    cached = _load_cache(scientific_name)
    if cached is not None:
        logger.debug("Cache hit for %s", scientific_name)
        return cached

    # Normalize binomial: genus capitalized, epithet(s) lowercase ("Russula Vesca" → "Russula vesca")
    parts = scientific_name.split()
    query_title = parts[0] + (" " + " ".join(p.lower() for p in parts[1:]) if len(parts) > 1 else "")

    params = {
        "action": "query",
        "format": "json",
        "titles": query_title,
        "prop": "extracts|info",
        "explaintext": True,
        "redirects": 1,
        "inprop": "url",
    }

    try:
        time.sleep(RATE_LIMIT_SECONDS)
        response = requests.get(
            BASE_URL,
            params=params,
            headers={"User-Agent": "mushroom-ai/1.0 (educational project)"},
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        logger.warning("HTTP error fetching %s: %s", scientific_name, e)
        return None

    pages = data.get("query", {}).get("pages", {})
    if not pages:
        return None

    page = next(iter(pages.values()))

    # -1 means the page does not exist
    if page.get("pageid") == -1 or "missing" in page:
        logger.debug("No Wikipedia page found for %s", scientific_name)
        return None

    text = page.get("extract", "").strip()
    if not text:
        logger.debug("Empty extract for %s", scientific_name)
        return None

    url = page.get("fullurl", f"https://en.wikipedia.org/wiki/{scientific_name.replace(' ', '_')}")
    result = {"text": text, "url": url}
    _save_cache(scientific_name, result)
    return result
