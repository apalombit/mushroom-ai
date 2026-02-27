"""
funghiitaliani.it source fetcher (AMINT archive).

Fetches species sheets from www.funghiitaliani.it (Associazione Micologica
Italiana Naturalistica Telematica). Species sheets are published as forum
topics in three categories:

  /38-funghi-commestibili/              edible   (~23 pages)
  /39-funghi-velenosi/                  poisonous (~20 pages)
  /40-funghi-non-commestibili-o-sospetti/  non-edible/suspicious (~96 pages)

Topic titles follow "Genus species Author Year" format; we extract
"genus species" (first two words, lowercased) as the index key.

Index cache: data/cache/funghiitaliani_index.json
Page cache:  data/cache/funghiitaliani_{safe_name}.json
"""

import json
import logging
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

SOURCE_NAME = "funghiitaliani"
BASE_URL = "https://www.funghiitaliani.it"
CACHE_DIR = Path("data/cache")
INDEX_CACHE = CACHE_DIR / "funghiitaliani_index.json"
RATE_LIMIT_SECONDS = 1.0

CATEGORIES = [
    "/38-funghi-commestibili/",
    "/39-funghi-velenosi/",
    "/40-funghi-non-commestibili-o-sospetti/",
]

logger = logging.getLogger(__name__)


def _safe_filename(name: str) -> str:
    return re.sub(r"[^\w\-]", "_", name)


def _cache_path(scientific_name: str) -> Path:
    return CACHE_DIR / f"funghiitaliani_{_safe_filename(scientific_name)}.json"


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


def _extract_scientific_name(title: str) -> str | None:
    """
    Extract 'Genus species' from a topic title.

    'Boletus edulis Bull. : Fr. 1782' → 'Boletus edulis'
    Returns None if the title doesn't look like a binomial.
    """
    # Normalize whitespace (titles sometimes use non-breaking spaces)
    title = re.sub(r"\s+", " ", title).strip()
    parts = title.split(" ")
    if len(parts) < 2:
        return None
    genus, epithet = parts[0], parts[1]
    # Genus must start uppercase; epithet must start lowercase (binomial convention)
    if not re.match(r"^[A-Z][a-z]", genus) or not re.match(r"^[a-z]", epithet):
        return None
    return f"{genus} {epithet.lower()}"


def _get_category_page(
    category_path: str, page: int
) -> tuple[list[tuple[str, str]], bool]:
    """
    Fetch one page of a category listing.

    Returns (entries, is_last_page) where entries is a list of
    (scientific_name_lower, url) tuples. is_last_page is True when the HTML
    pagination shows this is the final page (or on any error/404).
    """
    url = f"{BASE_URL}{category_path}?page={page}"
    try:
        time.sleep(RATE_LIMIT_SECONDS)
        response = requests.get(
            url,
            headers={"User-Agent": "mushroom-ai/1.0 (educational project)"},
            timeout=20,
        )
        if response.status_code == 404:
            return [], True
        response.raise_for_status()
    except requests.RequestException as e:
        logger.warning("HTTP error fetching category page %s: %s", url, e)
        return [], True

    soup = BeautifulSoup(response.text, "html.parser")

    # Detect last page by parsing "Pagina X di Y" in the pagination widget.
    # The site returns HTTP 200 for out-of-range pages (mirrors the last page),
    # so we must check the page counter to stop correctly.
    is_last_page = True
    pagination_text = soup.get_text(" ")
    m = re.search(r"Pagina\s+(\d+)\s+di\s+(\d+)", pagination_text, re.IGNORECASE)
    if m:
        current, total = int(m.group(1)), int(m.group(2))
        is_last_page = current >= total

    results = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if not re.search(r"/topic/\d+-[a-z0-9]", href):
            continue
        full_url = href if href.startswith("http") else BASE_URL + href
        title = a.get_text(strip=True)
        name = _extract_scientific_name(title)
        if name:
            results.append((name.lower(), full_url))
    return results, is_last_page


def build_index() -> dict[str, str]:
    """
    Build a {scientific_name_lower: url} index from all three categories.

    Paginates each category until a page returns no species topics.
    Result is cached to data/cache/funghiitaliani_index.json.
    Returns an empty dict on failure.
    """
    if INDEX_CACHE.exists():
        with open(INDEX_CACHE) as f:
            return json.load(f)

    index: dict[str, str] = {}
    for category in CATEGORIES:
        page = 1
        while True:
            entries, is_last = _get_category_page(category, page)
            for name, url in entries:
                if name not in index:  # first category wins on duplicates
                    index[name] = url
            logger.debug(
                "Indexed %s page %d (%d total entries)", category.strip("/"), page, len(index)
            )
            if is_last:
                break
            page += 1

    if index:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(INDEX_CACHE, "w") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)
        logger.info("Built funghiitaliani index: %d entries", len(index))
    else:
        logger.warning("funghiitaliani index is empty — site structure may have changed")

    return index


def fetch_species_page(scientific_name: str, aliases: list[str] | None = None) -> dict | None:
    """
    Fetch the funghiitaliani.it species sheet for a species.

    Returns {"text": <page text>, "url": <page url>} or None if not found.
    On primary miss, tries aliases in order against the index and caches
    under the canonical name.
    Results are cached — delete the cache file to force re-fetch.
    """
    cached = _load_cache(scientific_name)
    if cached is not None:
        logger.debug("Cache hit for %s (funghiitaliani)", scientific_name)
        return cached

    index = build_index()
    if not index:
        return None

    url = index.get(scientific_name.lower())
    if url is None and aliases:
        for alias in aliases:
            url = index.get(alias.lower())
            if url is not None:
                logger.debug(
                    "Alias hit for %s via '%s' (funghiitaliani)", scientific_name, alias
                )
                break

    if url is None:
        logger.debug("Not in funghiitaliani index: %s", scientific_name)
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
        logger.warning("HTTP error fetching %s (funghiitaliani): %s", scientific_name, e)
        return None

    soup = BeautifulSoup(response.text, "html.parser")

    # IPS Community: first post body is in [data-role="commentContent"].
    # We take only the first match (the species sheet, not any replies).
    first_post = soup.find("div", {"data-role": "commentContent"}) or soup.find(
        "div", class_=re.compile(r"cPost_contentWrap|ipsType_richText", re.I)
    )
    source = first_post or soup.find("article") or soup.find("body") or soup

    for tag in source.find_all(["nav", "header", "footer", "script", "style"]):
        tag.decompose()

    paragraphs = []
    for p in source.find_all("p"):
        text = p.get_text(separator=" ", strip=True)
        if text:
            paragraphs.append(text)

    text = "\n\n".join(paragraphs).strip()
    if not text:
        logger.debug("Empty content for %s (funghiitaliani)", scientific_name)
        return None

    result = {"text": text, "url": url}
    _save_cache(scientific_name, result)
    return result
