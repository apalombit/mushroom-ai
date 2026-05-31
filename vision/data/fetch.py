"""Image fetching from iNaturalist API.

Species-driven fetching with content-hash dedup and DB registration.
"""

import hashlib
import io
import time
from dataclasses import dataclass
from pathlib import Path

import requests
from PIL import Image
from sqlalchemy import text

INAT_API_BASE = "https://api.inaturalist.org/v1"
HEADERS = {"User-Agent": "mushroom-ai-research/1.0 (academic)"}
TIMEOUT = 15
REQUEST_DELAY = 1.0

# Only fetch CC-licensed images
ALLOWED_LICENSES = {"cc-by", "cc-by-nc", "cc-by-sa", "cc-by-nc-sa", "cc0", "cc-by-nd"}


@dataclass
class FetchResult:
    image_id: str
    species: str
    source_id: str
    source_url: str
    license: str
    file_path: str
    already_existed: bool


def content_hash(data: bytes) -> str:
    """SHA256[:12] of raw image bytes."""
    return hashlib.sha256(data).hexdigest()[:12]


def search_taxon(species_name: str) -> int | None:
    """Look up iNaturalist taxon_id for a species name."""
    try:
        resp = requests.get(
            f"{INAT_API_BASE}/taxa",
            params={"q": species_name, "rank": "species", "per_page": 5},
            headers=HEADERS,
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        # Prefer exact name match
        for r in results:
            if r.get("name", "").lower() == species_name.lower():
                return r["id"]
        return results[0]["id"] if results else None
    except Exception as e:
        print(f"  taxon lookup failed for {species_name}: {e}")
        return None


def fetch_observation_photos(
    taxon_id: int,
    max_photos: int = 20,
) -> list[dict]:
    """Fetch research-grade observations with CC-licensed photos.

    Returns list of {obs_id, photo_url, license, obs_url}.
    """
    try:
        resp = requests.get(
            f"{INAT_API_BASE}/observations",
            params={
                "taxon_id": taxon_id,
                "quality_grade": "research",
                "photos": "true",
                "per_page": min(max_photos * 2, 50),
                "order": "desc",
                "order_by": "votes",
            },
            headers=HEADERS,
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
    except Exception as e:
        print(f"  observation fetch failed for taxon {taxon_id}: {e}")
        return []

    photos = []
    for obs in resp.json().get("results", []):
        obs_id = str(obs["id"])
        obs_url = obs.get("uri", f"https://www.inaturalist.org/observations/{obs_id}")
        for photo in obs.get("photos", []):
            license_code = (photo.get("license_code") or "").lower()
            if license_code not in ALLOWED_LICENSES:
                continue
            url = photo.get("url", "")
            if not url:
                continue
            # Use "large" size (~1024px)
            url = url.replace("square", "large")
            photos.append({
                "obs_id": obs_id,
                "photo_url": url,
                "license": license_code,
                "obs_url": obs_url,
            })
            if len(photos) >= max_photos:
                return photos
    return photos


def download_and_register(
    photo_url: str,
    species: str,
    source_id: str,
    source_url: str,
    license_code: str,
    raw_dir: Path,
    session,
) -> FetchResult | None:
    """Download image, compute content hash, save to raw_dir, register in DB.

    Returns None on download failure. Idempotent via content hash dedup.
    """
    try:
        resp = requests.get(photo_url, headers=HEADERS, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.content
    except Exception as e:
        print(f"    download failed: {e}")
        return None

    image_id = content_hash(data)
    file_path = f"{image_id}.jpg"
    dest = raw_dir / file_path

    # Check if already exists
    if dest.exists():
        return FetchResult(
            image_id=image_id,
            species=species,
            source_id=source_id,
            source_url=source_url,
            license=license_code,
            file_path=file_path,
            already_existed=True,
        )

    # Get resolution
    try:
        img = Image.open(io.BytesIO(data))
        w, h = img.size
    except Exception:
        print(f"    corrupt image from {photo_url}")
        return None

    # Write file
    raw_dir.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        f.write(data)

    # Register in DB
    session.execute(
        text(
            "INSERT INTO image_registry "
            "(image_id, file_path, species, source, source_id, source_url, "
            "license, resolution_w, resolution_h) "
            "VALUES (:id, :fp, :sp, :src, :sid, :surl, :lic, :w, :h) "
            "ON CONFLICT (image_id) DO NOTHING"
        ),
        {
            "id": image_id,
            "fp": file_path,
            "sp": species,
            "src": "inaturalist",
            "sid": source_id,
            "surl": source_url,
            "lic": license_code,
            "w": w,
            "h": h,
        },
    )
    session.commit()

    return FetchResult(
        image_id=image_id,
        species=species,
        source_id=source_id,
        source_url=source_url,
        license=license_code,
        file_path=file_path,
        already_existed=False,
    )


def fetch_species(
    species_name: str,
    max_images: int,
    raw_dir: Path,
    session,
) -> list[FetchResult]:
    """Full pipeline: taxon lookup → fetch observations → download + register."""
    taxon_id = search_taxon(species_name)
    if taxon_id is None:
        print(f"  {species_name}: not found on iNaturalist")
        return []

    time.sleep(REQUEST_DELAY)

    photos = fetch_observation_photos(taxon_id, max_photos=max_images)
    if not photos:
        print(f"  {species_name}: no CC-licensed photos found")
        return []

    time.sleep(REQUEST_DELAY)

    results = []
    for photo in photos:
        if len([r for r in results if not r.already_existed]) >= max_images:
            break
        result = download_and_register(
            photo_url=photo["photo_url"],
            species=species_name,
            source_id=photo["obs_id"],
            source_url=photo["obs_url"],
            license_code=photo["license"],
            raw_dir=raw_dir,
            session=session,
        )
        if result:
            results.append(result)
        time.sleep(0.5)

    new = sum(1 for r in results if not r.already_existed)
    existing = sum(1 for r in results if r.already_existed)
    print(f"  {species_name}: {new} new, {existing} existing")
    return results
