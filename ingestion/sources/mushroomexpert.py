"""
mushroomexpert.com source fetcher.

Fetches species pages from mushroomexpert.com (Michael Kuo's mushroom reference).
URL pattern: https://www.mushroomexpert.com/{genus}_{species}.html

Follows redirects — site may redirect varieties to subspecies pages.
Cache: data/cache/mushroomexpert_{safe_name}.json
"""

import json
import logging
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from ingestion.sources._image_utils import filter_content_images

SOURCE_NAME = "mushroomexpert"
CACHE_DIR = Path("data/cache")
RATE_LIMIT_SECONDS = 1.0

logger = logging.getLogger(__name__)


def _safe_filename(name: str) -> str:
    return re.sub(r"[^\w\-]", "_", name)


def _cache_path(scientific_name: str) -> Path:
    return CACHE_DIR / f"mushroomexpert_{_safe_filename(scientific_name)}.json"


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
    """Convert 'Amanita muscaria' → 'https://www.mushroomexpert.com/amanita_muscaria.html'."""
    slug = scientific_name.lower().replace(" ", "_")
    slug = re.sub(r"[^\w]", "", slug)
    return f"https://www.mushroomexpert.com/{slug}.html"


def _get_page(url: str, depth: int = 3) -> tuple[str, str] | None:
    """
    Fetch a URL, following both HTTP redirects and meta-refresh redirects.

    Returns (html_content, final_url) or None on error/404.
    depth limits meta-refresh follow hops.
    """
    if depth == 0:
        return None
    try:
        time.sleep(RATE_LIMIT_SECONDS)
        response = requests.get(
            url,
            allow_redirects=True,
            headers={"User-Agent": "mushroom-ai/1.0 (educational project)"},
            timeout=15,
        )
        if response.status_code == 404:
            return None
        response.raise_for_status()
    except requests.RequestException as e:
        logger.warning("HTTP error fetching %s (mushroomexpert): %s", url, e)
        return None

    html = response.text
    final_url = response.url

    # Detect meta-refresh redirect: <meta HTTP-EQUIV="Refresh" CONTENT="0;URL=...">
    meta_match = re.search(
        r'<meta[^>]+HTTP-EQUIV=["\']?Refresh["\']?[^>]+CONTENT=["\']?\d+;URL=([^"\'>\s]+)',
        html, re.IGNORECASE
    )
    if meta_match:
        redirect_path = meta_match.group(1)
        # Resolve relative path against the current URL's directory
        if not redirect_path.startswith("http"):
            base = final_url.rsplit("/", 1)[0]
            redirect_path = f"{base}/{redirect_path}"
        logger.debug("Meta-refresh redirect: %s → %s", final_url, redirect_path)
        return _get_page(redirect_path, depth - 1)

    return html, final_url


def fetch_species_page(scientific_name: str, aliases: list[str] | None = None) -> dict | None:
    """
    Fetch the mushroomexpert.com page for a species.

    Returns {"text": <page text>, "url": <final url after redirects>} or None if not found.
    Follows both HTTP and meta-refresh redirects.
    On primary miss, tries aliases in order and caches under the canonical name.
    Results are cached — delete the cache file to force re-fetch.
    """
    cached = _load_cache(scientific_name)
    if cached is not None:
        logger.debug("Cache hit for %s (mushroomexpert)", scientific_name)
        return cached

    url = _build_url(scientific_name)
    page = _get_page(url)

    if page is None and aliases:
        for alias in aliases:
            page = _get_page(_build_url(alias))
            if page is not None:
                logger.debug("Alias hit for %s via '%s' (mushroomexpert)", scientific_name, alias)
                break

    if page is None:
        logger.debug("Not found on mushroomexpert: %s", scientific_name)
        return None

    html, final_url = page
    soup = BeautifulSoup(html, "lxml")

    # mushroomexpert uses <td width="380"> as the left content column
    content_td = soup.find("td", {"width": "380"})
    if content_td:
        source = content_td
    else:
        # Fallback: strip nav/header/footer and use full body
        for tag in soup.find_all(["nav", "header", "footer", "script", "style"]):
            tag.decompose()
        source = soup.find("body") or soup

    paragraphs = []
    for p in source.find_all("p"):
        text = p.get_text(separator=" ", strip=True)
        if text:
            paragraphs.append(text)

    text = "\n\n".join(paragraphs).strip()
    if not text:
        logger.debug("Empty content for %s (mushroomexpert)", scientific_name)
        return None

    image_urls = filter_content_images(source.find_all("img"), final_url)
    result = {"text": text, "url": final_url, "image_urls": image_urls}
    _save_cache(scientific_name, result)
    return result
