"""
ultimate-mushroom.com source fetcher.

Fetches species pages from ultimate-mushroom.com.
URL scheme uses a category/id-name pattern; requires an index pre-fetch
to resolve scientific name → URL.

Index cache: data/cache/ultimatemushroom_index.json
Page cache:  data/cache/ultimatemushroom_{safe_name}.json
"""

import json
import logging
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

SOURCE_NAME = "ultimate-mushroom"
CACHE_DIR = Path("data/cache")
INDEX_CACHE = CACHE_DIR / "ultimatemushroom_index.json"
INDEX_URL = "https://ultimate-mushroom.com/mushroom-alphabet.html"
RATE_LIMIT_SECONDS = 1.0

logger = logging.getLogger(__name__)


def _safe_filename(name: str) -> str:
    return re.sub(r"[^\w\-]", "_", name)


def _cache_path(scientific_name: str) -> Path:
    return CACHE_DIR / f"ultimatemushroom_{_safe_filename(scientific_name)}.json"


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


def build_index() -> dict[str, str]:
    """
    Fetch the alphabet index and build a {scientific_name_lower: url} mapping.

    Result is cached to data/cache/ultimatemushroom_index.json.
    Returns an empty dict on failure.
    """
    if INDEX_CACHE.exists():
        with open(INDEX_CACHE) as f:
            return json.load(f)

    try:
        time.sleep(RATE_LIMIT_SECONDS)
        response = requests.get(
            INDEX_URL,
            headers={"User-Agent": "mushroom-ai/1.0 (educational project)"},
            timeout=20,
        )
        response.raise_for_status()
    except requests.RequestException as e:
        logger.warning("Failed to fetch ultimate-mushroom index: %s", e)
        return {}

    soup = BeautifulSoup(response.text, "html.parser")
    index: dict[str, str] = {}

    for a in soup.find_all("a", href=True):
        href = a["href"]
        # Species links follow pattern like:
        #   https://ultimate-mushroom.com/poisonous/123-amanita-phalloides.html  (absolute)
        #   /poisonous/123-amanita-phalloides.html  (relative)
        full_url: str | None = None
        if re.search(r"/[a-z\-]+/\d+-[a-z\-]+\.html", href):
            if href.startswith("http"):
                full_url = href
            elif href.startswith("/"):
                full_url = "https://ultimate-mushroom.com" + href
        if full_url:
            link_text = a.get_text(strip=True)
            if link_text:
                index[link_text.lower()] = full_url

    if index:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(INDEX_CACHE, "w") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)
        logger.info("Built ultimate-mushroom index: %d entries", len(index))
    else:
        logger.warning("ultimate-mushroom index is empty — site structure may have changed")

    return index


def fetch_species_page(scientific_name: str, aliases: list[str] | None = None) -> dict | None:
    """
    Fetch the ultimate-mushroom.com page for a species.

    Returns {"text": <page text>, "url": <page url>} or None if not in index.
    On primary miss, tries aliases in order against the index and caches under the canonical name.
    Results are cached — delete the cache file to force re-fetch.
    """
    cached = _load_cache(scientific_name)
    if cached is not None:
        logger.debug("Cache hit for %s (ultimate-mushroom)", scientific_name)
        return cached

    index = build_index()
    if not index:
        return None

    url = index.get(scientific_name.lower())
    if url is None and aliases:
        for alias in aliases:
            url = index.get(alias.lower())
            if url is not None:
                logger.debug("Alias hit for %s via '%s' (ultimate-mushroom)", scientific_name, alias)
                break

    if url is None:
        logger.debug("Not in ultimate-mushroom index: %s", scientific_name)
        return None

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
    except requests.RequestException as e:
        logger.warning("HTTP error fetching %s (ultimate-mushroom): %s", scientific_name, e)
        return None

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove boilerplate
    for tag in soup.find_all(["nav", "header", "footer", "script", "style", "aside"]):
        tag.decompose()

    # Try to find main article content
    article = soup.find("article") or soup.find("main") or soup.find("div", class_=re.compile(r"content|article|post", re.I))
    source = article if article else soup.find("body") or soup

    paragraphs = []
    for p in source.find_all("p"):
        text = p.get_text(separator=" ", strip=True)
        if text:
            paragraphs.append(text)

    text = "\n\n".join(paragraphs).strip()
    if not text:
        logger.debug("Empty content for %s (ultimate-mushroom)", scientific_name)
        return None

    result = {"text": text, "url": url}
    _save_cache(scientific_name, result)
    return result
