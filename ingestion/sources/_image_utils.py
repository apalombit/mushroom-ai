"""Shared image filtering for source fetchers."""

import re
from urllib.parse import urljoin

from bs4 import Tag

# Patterns that indicate non-content images (icons, logos, nav elements)
_SKIP_PATTERNS = re.compile(
    r"(logo|icon|banner|button|arrow|nav|sprite|avatar|badge|flag|pixel|spacer"
    r"|tracker|ad[-_]|advert|widget)",
    re.IGNORECASE,
)


def filter_content_images(
    img_tags: list[Tag],
    base_url: str,
    max_images: int = 3,
) -> list[str]:
    """Filter img tags to plausible content photos and return absolute URLs.

    Filters out SVGs, data URIs, tiny images, and non-content images (icons,
    logos, nav elements). Returns up to *max_images* unique absolute URLs.
    """
    seen: set[str] = set()
    result: list[str] = []

    for tag in img_tags:
        src = tag.get("src") or tag.get("data-src") or ""
        if not src or src.startswith("data:"):
            continue

        # Skip SVGs
        if src.lower().endswith(".svg"):
            continue

        url = urljoin(base_url, src)

        # Skip non-content images by filename/path pattern
        if _SKIP_PATTERNS.search(url):
            continue

        # Skip tiny images (explicit width/height attrs)
        for attr in ("width", "height"):
            val = tag.get(attr, "")
            try:
                if val and int(val) < 80:
                    break
            except (ValueError, TypeError):
                continue
        else:
            # No break → passed size check
            if url not in seen:
                seen.add(url)
                result.append(url)
                if len(result) >= max_images:
                    break

    return result
