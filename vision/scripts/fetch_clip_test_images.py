"""Fetch test images for CLIP zero-shot baseline from DB image_urls.

Queries reconciled_species for species grouped by hymenium_type,
downloads one image per species from their scraped image_urls,
and saves to data/images/test_clip/hymenium_type/{class}/{species_slug}.jpg

Usage:
    python -m vision.scripts.fetch_clip_test_images [--max-per-class 10]
"""

import argparse
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from sqlalchemy import text

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from db.connection import get_session
from db.models import ReconciledSpecies

OUTPUT_DIR = PROJECT_ROOT / "data" / "images" / "test_clip" / "hymenium_type"
TIMEOUT = 15  # seconds per download
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) mushroom-ai-research/1.0"
}


def slugify(name: str) -> str:
    """Convert species name to filesystem-safe slug."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def download_image(url: str, dest: Path) -> bool:
    """Download a single image. Returns True on success."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT, stream=True)
        resp.raise_for_status()

        content_type = resp.headers.get("content-type", "")
        if "image" not in content_type and not url.lower().endswith(
            (".jpg", ".jpeg", ".png", ".webp")
        ):
            print(f"    SKIP (not image): {content_type}")
            return False

        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"    FAIL: {e}")
        return False


def get_extension(url: str) -> str:
    """Extract file extension from URL, default to .jpg."""
    path = urlparse(url).path.lower()
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        if path.endswith(ext):
            return ext
    return ".jpg"


def main():
    parser = argparse.ArgumentParser(description="Fetch CLIP test images from DB")
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=10,
        help="Max species to sample per class (default: 10, use 0 for all)",
    )
    args = parser.parse_args()
    max_per_class = args.max_per_class or 999999

    session = get_session()

    # Get all species with hymenium_type and images
    species_rows = (
        session.query(
            ReconciledSpecies.scientific_name,
            ReconciledSpecies.hymenium_type,
            ReconciledSpecies.image_urls,
        )
        .filter(
            ReconciledSpecies.hymenium_type.isnot(None),
            text("jsonb_array_length(COALESCE(image_urls, '[]'::jsonb)) > 0"),
        )
        .order_by(ReconciledSpecies.scientific_name)
        .all()
    )
    session.close()

    # Group by class
    by_class: dict[str, list] = {}
    for name, ht, urls in species_rows:
        by_class.setdefault(ht, []).append((name, urls))

    print(f"Output: {OUTPUT_DIR}")
    print(f"Max per class: {max_per_class}\n")

    total_downloaded = 0
    total_failed = 0

    for cls, species_list in sorted(by_class.items()):
        # For rare classes, take all; for common classes, sample evenly
        selected = species_list[:max_per_class]
        class_dir = OUTPUT_DIR / cls
        class_dir.mkdir(parents=True, exist_ok=True)

        print(f"{cls} ({len(selected)}/{len(species_list)} species):")

        for species_name, image_urls in selected:
            # Pick first image URL
            url_entry = image_urls[0]
            url = url_entry["image_url"] if isinstance(url_entry, dict) else url_entry

            slug = slugify(species_name)
            ext = get_extension(url)
            dest = class_dir / f"{slug}{ext}"

            if dest.exists():
                print(f"  {slug}: already exists, skipping")
                total_downloaded += 1
                continue

            print(f"  {slug}: downloading...", end=" ")
            ok = download_image(url, dest)
            if ok:
                size_kb = dest.stat().st_size / 1024
                print(f"OK ({size_kb:.0f} KB)")
                total_downloaded += 1
            else:
                total_failed += 1

            time.sleep(0.3)  # polite rate limiting

        print()

    print(f"Done: {total_downloaded} downloaded, {total_failed} failed")
    print(f"Images at: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
