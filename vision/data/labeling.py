"""Auto-labeling: propagate species-level features to image annotations.

Joins image_registry (species) with reconciled_species (features_json + direct cols)
to populate image_annotations with canonical feature values.
"""

from pathlib import Path

import yaml
from sqlalchemy import text

# Direct columns on reconciled_species (not in features_json)
DIRECT_COLUMNS = {"hymenium_type", "overall_body_form"}

# Mapping from feature name to dot-path in features_json
JSON_PATHS = {
    "cap_shape": "cap.shape",
    "cap_color": "cap.colors",  # list → take [0]
    "surface_texture": "cap.surface_texture",
    "gill_attachment": "gills.attachment",
    "stem_shape": "stem.shape",
    "cap_surface_moisture": "cap.surface_moisture",
}


def _resolve_dot_path(data: dict, path: str):
    """Navigate nested dict via dot-separated path. Returns None if missing."""
    parts = path.split(".")
    current = data
    for part in parts:
        if not isinstance(current, dict):
            return None
        current = current.get(part)
        if current is None:
            return None
    return current


def _normalize_to_canonical(value, feature_name: str, canonical_classes: list[str]) -> str | None:
    """Check if value matches a canonical class. Returns None if unmappable."""
    if value is None:
        return None

    # Handle list values (e.g., cap.colors → take first)
    if isinstance(value, list):
        if not value:
            return None
        value = value[0]

    value_lower = str(value).lower().strip()
    canonical_lower = {c.lower(): c for c in canonical_classes}

    if value_lower in canonical_lower:
        return canonical_lower[value_lower]

    return None


def auto_label_from_species(session, features_config_path: str | Path) -> dict[str, int]:
    """Propagate species features to image_annotations for all registered images.

    Returns dict of {feature_name: count_labeled}.
    """
    with open(features_config_path) as f:
        features_config = yaml.safe_load(f)

    # Get all images with their species
    images = session.execute(
        text("SELECT image_id, species FROM image_registry WHERE species IS NOT NULL")
    ).fetchall()

    if not images:
        print("No images with species in image_registry")
        return {}

    # Get all reconciled species with features
    species_rows = session.execute(
        text(
            "SELECT scientific_name, features_json, hymenium_type, overall_body_form "
            "FROM reconciled_species"
        )
    ).fetchall()

    species_features = {}
    for row in species_rows:
        name = row[0]
        features_json = row[1] or {}
        species_features[name] = {
            "features_json": features_json,
            "hymenium_type": row[2],
            "overall_body_form": row[3],
        }

    # Determine which features to label (Tier 1 + Tier 2)
    target_features = {
        name: cfg["classes"]
        for name, cfg in features_config.items()
        if cfg.get("tier") in (1, 2)
    }

    counts: dict[str, int] = {f: 0 for f in target_features}
    total_inserted = 0

    for image_id, species in images:
        sp_data = species_features.get(species)
        if sp_data is None:
            continue

        for feature_name, canonical_classes in target_features.items():
            # Resolve value from DB
            if feature_name in DIRECT_COLUMNS:
                raw_value = sp_data.get(feature_name)
            elif feature_name in JSON_PATHS:
                raw_value = _resolve_dot_path(
                    sp_data["features_json"], JSON_PATHS[feature_name]
                )
            else:
                continue

            canonical = _normalize_to_canonical(raw_value, feature_name, canonical_classes)
            if canonical is None:
                continue

            session.execute(
                text(
                    "INSERT INTO image_annotations "
                    "(image_id, annotation_type, feature_name, feature_value, "
                    "confidence, annotator) "
                    "VALUES (:iid, 'species_propagated', :fn, :fv, 1.0, 'auto_label') "
                    "ON CONFLICT (image_id, feature_name, annotation_type) DO NOTHING"
                ),
                {"iid": image_id, "fn": feature_name, "fv": canonical},
            )
            counts[feature_name] += 1
            total_inserted += 1

    session.commit()

    print(f"Labeled {total_inserted} annotations across {len(images)} images")
    for feat, count in sorted(counts.items()):
        if count > 0:
            print(f"  {feat}: {count}")

    return counts
