"""CLI: fetch images for species from iNaturalist.

Usage:
    python -m vision.scripts.fetch_images --from-db --feature hymenium_type --max-per-species 20
    python -m vision.scripts.fetch_images --species "Boletus edulis" "Amanita muscaria"
    python -m vision.scripts.fetch_images --from-db --max-per-species 15
"""

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from db.connection import get_session
from vision.data.fetch import fetch_species


def get_species_from_db(session, feature: str | None = None) -> list[str]:
    """Get species names from reconciled_species that have Tier 1 features."""
    from sqlalchemy import text

    if feature:
        # Species where the specific feature column is non-null
        # hymenium_type and overall_body_form are direct columns
        direct_cols = {"hymenium_type", "overall_body_form", "overall_size_class"}
        if feature in direct_cols:
            rows = session.execute(
                text(
                    "SELECT scientific_name FROM reconciled_species "
                    f"WHERE {feature} IS NOT NULL "
                    "ORDER BY scientific_name"
                )
            ).fetchall()
        else:
            # features_json dot-path lookup
            json_paths = {
                "cap_shape": "cap.shape",
                "cap_color": "cap.colors",
                "surface_texture": "cap.surface_texture",
                "gill_attachment": "gills.attachment",
                "stem_shape": "stem.shape",
                "cap_surface_moisture": "cap.surface_moisture",
            }
            path = json_paths.get(feature)
            if not path:
                print(f"Unknown feature: {feature}")
                return []
            parts = path.split(".")
            if len(parts) == 2:
                jsonb_path = f"features_json->'{parts[0]}'->'{parts[1]}'"
            else:
                jsonb_path = f"features_json->'{parts[0]}'"
            rows = session.execute(
                text(
                    "SELECT scientific_name FROM reconciled_species "
                    f"WHERE {jsonb_path} IS NOT NULL "
                    "ORDER BY scientific_name"
                )
            ).fetchall()
    else:
        # All species with any Tier 1 feature set
        rows = session.execute(
            text(
                "SELECT scientific_name FROM reconciled_species "
                "WHERE hymenium_type IS NOT NULL "
                "   OR overall_body_form IS NOT NULL "
                "   OR features_json->'cap'->'shape' IS NOT NULL "
                "ORDER BY scientific_name"
            )
        ).fetchall()

    return [r[0] for r in rows]


def main():
    parser = argparse.ArgumentParser(description="Fetch images from iNaturalist")
    parser.add_argument("--species", nargs="+", help="Explicit species list")
    parser.add_argument("--from-db", action="store_true", help="Get species from DB")
    parser.add_argument("--feature", type=str, default=None,
                        help="Filter DB species to those with this feature")
    parser.add_argument("--max-per-species", type=int, default=20,
                        help="Max images per species (default: 20)")
    args = parser.parse_args()

    if not args.species and not args.from_db:
        parser.error("Provide --species or --from-db")

    config = yaml.safe_load(open(PROJECT_ROOT / "vision/config/training.yaml"))
    raw_dir = PROJECT_ROOT / config["paths"]["raw_images"]

    session = get_session()

    if args.species:
        species_list = args.species
    else:
        species_list = get_species_from_db(session, args.feature)

    print(f"Species to fetch: {len(species_list)}")
    print(f"Max per species: {args.max_per_species}")
    print(f"Output: {raw_dir}\n")

    total_new = 0
    total_existing = 0
    total_failed = 0

    for i, species in enumerate(species_list, 1):
        print(f"[{i}/{len(species_list)}] {species}")
        results = fetch_species(species, args.max_per_species, raw_dir, session)
        new = sum(1 for r in results if not r.already_existed)
        existing = sum(1 for r in results if r.already_existed)
        total_new += new
        total_existing += existing
        if not results:
            total_failed += 1

    session.close()

    print(f"\nDone: {total_new} new images, {total_existing} existing, "
          f"{total_failed} species with no images")
    print(f"Images at: {raw_dir}")


if __name__ == "__main__":
    main()
