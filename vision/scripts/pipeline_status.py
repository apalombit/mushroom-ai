"""CLI: show image counts at each pipeline stage and annotation stats.

Usage:
    python -m vision.scripts.pipeline_status
"""

import sys
from pathlib import Path

import yaml
from sqlalchemy import text

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from db.connection import get_session


def main():
    config = yaml.safe_load(open(PROJECT_ROOT / "vision/config/training.yaml"))

    raw_dir = PROJECT_ROOT / config["paths"]["raw_images"]
    processed_base = PROJECT_ROOT / config["paths"]["processed_images"]
    embeddings_base = PROJECT_ROOT / config["paths"]["embeddings"]

    print("=" * 60)
    print("  Vision Pipeline Status")
    print("=" * 60)

    # Filesystem counts
    raw_count = len(list(raw_dir.glob("*.jpg"))) if raw_dir.exists() else 0
    print(f"\nRaw images:        {raw_count:>6}  ({raw_dir})")

    if processed_base.exists():
        for version_dir in sorted(processed_base.iterdir()):
            if version_dir.is_dir():
                count = len(list(version_dir.glob("*.jpg")))
                print(f"Processed ({version_dir.name}):  {count:>6}  ({version_dir})")
    else:
        print(f"Processed:         {'0':>6}  (no processed directory)")

    if embeddings_base.exists():
        for backbone_dir in sorted(embeddings_base.iterdir()):
            if backbone_dir.is_dir():
                for version_dir in sorted(backbone_dir.iterdir()):
                    if version_dir.is_dir():
                        count = len(list(version_dir.glob("*.pt")))
                        label = f"{backbone_dir.name}/{version_dir.name}"
                        print(f"Embedded ({label}): {count:>6}")
    else:
        print(f"Embedded:          {'0':>6}  (no embeddings directory)")

    # DB counts
    session = get_session()

    registry_count = session.execute(
        text("SELECT COUNT(*) FROM image_registry")
    ).scalar()
    print(f"\nDB image_registry: {registry_count:>6}")

    species_count = session.execute(
        text("SELECT COUNT(DISTINCT species) FROM image_registry")
    ).scalar()
    print(f"  Distinct species:  {species_count:>4}")

    # Annotations by feature
    annotations = session.execute(
        text(
            "SELECT feature_name, COUNT(*) FROM image_annotations "
            "GROUP BY feature_name ORDER BY feature_name"
        )
    ).fetchall()

    if annotations:
        print(f"\nAnnotations:")
        for feat, count in annotations:
            print(f"  {feat:<25} {count:>6}")
    else:
        print(f"\nAnnotations:         0")

    # Quality / rejections
    rejected = session.execute(
        text("SELECT COUNT(*) FROM image_quality WHERE exclude_reason IS NOT NULL")
    ).scalar()
    verified = session.execute(
        text("SELECT COUNT(*) FROM image_quality WHERE is_verified = TRUE")
    ).scalar()
    print(f"\nQuality:")
    print(f"  Rejected:          {rejected:>4}")
    print(f"  Verified:          {verified:>4}")

    # Orphan detection
    if processed_base.exists():
        processed_ids = set()
        for version_dir in processed_base.iterdir():
            if version_dir.is_dir():
                processed_ids |= {p.stem for p in version_dir.glob("*.jpg")}
        raw_ids = {p.stem for p in raw_dir.glob("*.jpg")} if raw_dir.exists() else set()
        orphaned_processed = processed_ids - raw_ids
        if orphaned_processed:
            print(f"\n  WARNING: {len(orphaned_processed)} processed images without raw source")

    if embeddings_base.exists():
        embedded_ids = set()
        for backbone_dir in embeddings_base.iterdir():
            if backbone_dir.is_dir():
                for version_dir in backbone_dir.iterdir():
                    if version_dir.is_dir():
                        embedded_ids |= {p.stem for p in version_dir.glob("*.pt")}
        if processed_base.exists():
            processed_ids = set()
            for version_dir in processed_base.iterdir():
                if version_dir.is_dir():
                    processed_ids |= {p.stem for p in version_dir.glob("*.jpg")}
            orphaned_embedded = embedded_ids - processed_ids
            if orphaned_embedded:
                print(f"  WARNING: {len(orphaned_embedded)} embeddings without processed image")

    session.close()
    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    main()
