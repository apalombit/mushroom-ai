"""Fetch underside/detail images from iNaturalist for CLIP baseline.

Downloads research-grade observation photos for taxa known to display
each hymenium_type clearly, targeting views where the feature is visible.

Usage:
    python -m vision.scripts.fetch_inat_test_images [--per-class 10]
"""

import argparse
import random
import time
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "images" / "test_clip_underside" / "hymenium_type"

INAT_API = "https://api.inaturalist.org/v1/observations"
HEADERS = {"User-Agent": "mushroom-ai-research/1.0 (academic)"}
TIMEOUT = 15

# Taxa + optional search terms to maximize hymenium visibility.
# For gills/pores/teeth: underside or detail shots help.
# For smooth/ridges/alveolate: the feature is visible from any angle.
SEARCH_CONFIG = {
    "gills": [
        # Underside detail shots — search for "gills" in description
        {"taxon_name": "Agaricus", "q": "gills"},
        {"taxon_name": "Russula", "q": "gills"},
        {"taxon_name": "Amanita", "q": "gills"},
        {"taxon_name": "Lactarius", "q": "underside"},
        {"taxon_name": "Mycena", "q": "gills"},
        # Fallback: just good photos of gilled fungi
        {"taxon_name": "Pluteus"},
        {"taxon_name": "Marasmius"},
    ],
    "ridges": [
        # Chanterelles — ridges visible from side/underside naturally
        {"taxon_name": "Cantharellus cibarius"},
        {"taxon_name": "Cantharellus"},
        {"taxon_name": "Craterellus tubaeformis"},
        {"taxon_name": "Craterellus cornucopioides"},
    ],
    "pores": [
        # Bracket fungi show pores naturally; boletes need underside
        {"taxon_name": "Trametes versicolor"},
        {"taxon_name": "Fomes fomentarius"},
        {"taxon_name": "Ganoderma"},
        {"taxon_name": "Boletus", "q": "pores"},
        {"taxon_name": "Suillus", "q": "underside"},
        {"taxon_name": "Fomitopsis"},
    ],
    "teeth": [
        # Hericium/Hydnum — teeth are the defining visual feature
        {"taxon_name": "Hericium erinaceus"},
        {"taxon_name": "Hericium coralloides"},
        {"taxon_name": "Hydnum repandum"},
        {"taxon_name": "Hydnum"},
        {"taxon_name": "Sarcodon"},
    ],
    "smooth": [
        # Jelly fungi, cup fungi — smooth surface is always visible
        {"taxon_name": "Tremella mesenterica"},
        {"taxon_name": "Auricularia auricula-judae"},
        {"taxon_name": "Bulgaria inquinans"},
        {"taxon_name": "Aleuria aurantia"},
        {"taxon_name": "Calocera viscosa"},
        {"taxon_name": "Exidia"},
    ],
    "gleba": [
        # Puffballs — search for cut/cross-section to show internal spore mass
        {"taxon_name": "Lycoperdon", "q": "cross section"},
        {"taxon_name": "Calvatia", "q": "cut"},
        {"taxon_name": "Lycoperdon perlatum"},
        {"taxon_name": "Bovista"},
        {"taxon_name": "Scleroderma citrinum"},
        {"taxon_name": "Calvatia gigantea"},
    ],
    "alveolate": [
        # Morels — pitted surface IS the top surface, always visible
        {"taxon_name": "Morchella esculenta"},
        {"taxon_name": "Morchella elata"},
        {"taxon_name": "Morchella"},
        {"taxon_name": "Gyromitra esculenta"},
        {"taxon_name": "Verpa"},
    ],
}


def search_observations(taxon_name: str, q: str | None = None, per_page: int = 10) -> list:
    """Search iNaturalist for research-grade observations with photos."""
    params = {
        "taxon_name": taxon_name,
        "quality_grade": "research",
        "photos": "true",
        "per_page": per_page,
        "order": "desc",
        "order_by": "votes",  # popular = usually better photos
    }
    if q:
        params["q"] = q

    try:
        resp = requests.get(INAT_API, params=params, headers=HEADERS, timeout=TIMEOUT)
        resp.raise_for_status()
        return resp.json().get("results", [])
    except Exception as e:
        print(f"    API error for {taxon_name}: {e}")
        return []


def get_best_photo_url(observation: dict, size: str = "medium") -> str | None:
    """Extract best photo URL from observation, preferring later photos (often detail shots)."""
    photos = observation.get("photos", [])
    if not photos:
        return None
    # If multiple photos, pick one beyond the first (often habitus) — try index 1 or 2
    if len(photos) > 2:
        photo = photos[min(2, len(photos) - 1)]
    elif len(photos) > 1:
        photo = photos[1]
    else:
        photo = photos[0]
    url = photo.get("url", "")
    return url.replace("square", size) if url else None


def download_image(url: str, dest: Path) -> bool:
    """Download image to dest. Returns True on success."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT, stream=True)
        resp.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"    FAIL: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Fetch iNaturalist test images")
    parser.add_argument("--per-class", type=int, default=10, help="Target images per class")
    args = parser.parse_args()
    target = args.per_class

    print(f"Output: {OUTPUT_DIR}")
    print(f"Target: {target} images per class\n")

    total_downloaded = 0

    for cls, searches in SEARCH_CONFIG.items():
        class_dir = OUTPUT_DIR / cls
        class_dir.mkdir(parents=True, exist_ok=True)

        existing = len(list(class_dir.glob("*.jpg")))
        if existing >= target:
            print(f"{cls}: already have {existing} images, skipping")
            total_downloaded += existing
            continue

        print(f"{cls}:")
        downloaded = existing
        seen_obs_ids = set()

        for search in searches:
            if downloaded >= target:
                break

            taxon = search["taxon_name"]
            q = search.get("q")
            q_str = f" (q={q})" if q else ""
            print(f"  searching {taxon}{q_str}...")

            observations = search_observations(taxon, q=q, per_page=20)
            random.shuffle(observations)

            for obs in observations:
                if downloaded >= target:
                    break

                obs_id = obs["id"]
                if obs_id in seen_obs_ids:
                    continue
                seen_obs_ids.add(obs_id)

                url = get_best_photo_url(obs, size="medium")
                if not url:
                    continue

                species = obs.get("species_guess", "") or obs.get("taxon", {}).get("name", "")
                slug = species.lower().replace(" ", "_").replace(".", "")[:40]
                dest = class_dir / f"{slug}_{obs_id}.jpg"

                if dest.exists():
                    downloaded += 1
                    continue

                ok = download_image(url, dest)
                if ok:
                    size_kb = dest.stat().st_size / 1024
                    print(f"    {slug}_{obs_id}: OK ({size_kb:.0f} KB)")
                    downloaded += 1

                time.sleep(0.5)  # rate limit

            time.sleep(1.0)  # between searches

        print(f"  → {downloaded} images total\n")
        total_downloaded += downloaded

    print(f"Done: {total_downloaded} images across {len(SEARCH_CONFIG)} classes")
    print(f"Images at: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
