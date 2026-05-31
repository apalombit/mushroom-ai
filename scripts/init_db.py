"""
Initialize database: create tables, enable pgvector extension, load seed data.
Includes migration from old 3-embedding schema to new N-embedding schema.

    python -m scripts.init_db
"""

import logging

import yaml
from sqlalchemy import inspect, text

from db.connection import engine, get_session
from db.models import EMBEDDING_DIM, Base, GroundTruthPair
from ingestion.rubric import EMBEDDING_GROUPS

logger = logging.getLogger(__name__)

GROUND_TRUTH_PATH = "data/seed/known_lookalikes.yaml"

# Old columns/indexes to drop during migration (all 3 old 384-dim columns)
_OLD_COLUMNS = ["embedding_morphological", "embedding_ecological", "embedding_taxonomic"]
_OLD_INDEXES = ["ix_morph_embedding", "ix_eco_embedding", "ix_taxon_embedding"]

# Active profile embedding columns
_EMBEDDING_COLS = [f"embedding_{g}" for g in EMBEDDING_GROUPS]


def _migrate_embeddings() -> None:
    """Detect old 3-column embedding schema and migrate to N-column schema."""
    inspector = inspect(engine)

    if "reconciled_species" not in inspector.get_table_names():
        return  # Fresh install, nothing to migrate

    columns = {col["name"] for col in inspector.get_columns("reconciled_species")}

    if "embedding_morphological" not in columns:
        return  # Already migrated or fresh schema

    logger.info("Detected old embedding schema — migrating to profile-driven embeddings...")

    with engine.connect() as conn:
        # Drop old indexes (ignore if already gone)
        for idx in _OLD_INDEXES:
            conn.execute(text(f"DROP INDEX IF EXISTS {idx}"))

        # Drop old columns
        for col in _OLD_COLUMNS:
            conn.execute(text(f"ALTER TABLE reconciled_species DROP COLUMN IF EXISTS {col}"))

        # Reset embedded_at to force re-embedding with new model
        conn.execute(text("UPDATE reconciled_species SET embedded_at = NULL"))

        conn.commit()

    logger.info("Migration complete — old embedding columns dropped, embedded_at reset.")


def _reconcile_embedding_columns() -> None:
    """
    Add missing embedding columns for the active profile.
    Warn about orphaned columns (columns not in the active profile).
    Resets embedded_at when new columns are added (forces re-embed).
    """
    inspector = inspect(engine)

    if "reconciled_species" not in inspector.get_table_names():
        return  # Will be created by create_all

    columns = {col["name"] for col in inspector.get_columns("reconciled_species")}
    expected = {f"embedding_{g}" for g in EMBEDDING_GROUPS}
    existing_embed = {c for c in columns if c.startswith("embedding_")}

    to_add = expected - existing_embed
    to_drop = existing_embed - expected

    if to_drop:
        logger.warning(
            "Orphaned embedding columns (not in active profile): %s. "
            "Re-run init_db after switching profiles to see this warning.",
            sorted(to_drop),
        )

    extra_varchar = ["overall_body_form", "hymenium_type", "overall_size_class"]
    extra_varchar_long = ["morphotype_signature"]
    to_add_extras = [c for c in extra_varchar if c not in columns]
    to_add_extras_long = [c for c in extra_varchar_long if c not in columns]

    if to_add or to_add_extras or to_add_extras_long:
        with engine.connect() as conn:
            for col in sorted(to_add):
                conn.execute(
                    text(
                        f"ALTER TABLE reconciled_species "
                        f"ADD COLUMN IF NOT EXISTS {col} vector({EMBEDDING_DIM})"
                    )
                )
            for col in to_add_extras:
                conn.execute(
                    text(
                        f"ALTER TABLE reconciled_species "
                        f"ADD COLUMN IF NOT EXISTS {col} VARCHAR(64)"
                    )
                )
            for col in to_add_extras_long:
                conn.execute(
                    text(
                        f"ALTER TABLE reconciled_species "
                        f"ADD COLUMN IF NOT EXISTS {col} VARCHAR(256)"
                    )
                )
            if to_add:
                # New embedding columns added — force re-embed
                conn.execute(text("UPDATE reconciled_species SET embedded_at = NULL"))
            conn.commit()
        if to_add:
            logger.info("Added embedding columns: %s (embedded_at reset)", sorted(to_add))
        if to_add_extras or to_add_extras_long:
            logger.info("Added columns: %s", to_add_extras + to_add_extras_long)


def _add_image_columns() -> None:
    """Add image_urls JSONB columns to both tables if missing."""
    inspector = inspect(engine)

    for table in ("source_observations", "reconciled_species"):
        if table not in inspector.get_table_names():
            continue
        columns = {col["name"] for col in inspector.get_columns(table)}
        if "image_urls" not in columns:
            with engine.connect() as conn:
                conn.execute(
                    text(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS image_urls JSONB")
                )
                conn.commit()
            logger.info("Added image_urls column to %s", table)


def _ensure_vision_tables() -> None:
    """Create vision pipeline tables (image_registry, image_annotations, image_quality)."""
    with engine.connect() as conn:
        conn.execute(
            text("""
            CREATE TABLE IF NOT EXISTS image_registry (
                image_id        TEXT PRIMARY KEY,
                file_path       TEXT NOT NULL,
                species         TEXT,
                source          TEXT NOT NULL,
                source_id       TEXT,
                source_url      TEXT,
                license         TEXT,
                resolution_w    INTEGER,
                resolution_h    INTEGER,
                fetched_at      TIMESTAMP DEFAULT NOW()
            )
        """)
        )
        conn.execute(
            text("""
            CREATE TABLE IF NOT EXISTS image_annotations (
                image_id        TEXT REFERENCES image_registry(image_id),
                annotation_type TEXT NOT NULL,
                feature_name    TEXT NOT NULL,
                feature_value   TEXT NOT NULL,
                confidence      REAL,
                annotator       TEXT,
                created_at      TIMESTAMP DEFAULT NOW(),
                PRIMARY KEY (image_id, feature_name, annotation_type)
            )
        """)
        )
        conn.execute(
            text("""
            CREATE TABLE IF NOT EXISTS image_quality (
                image_id        TEXT REFERENCES image_registry(image_id),
                is_verified     BOOLEAN DEFAULT FALSE,
                quality_score   REAL,
                exclude_reason  TEXT,
                PRIMARY KEY (image_id)
            )
        """)
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_annotations_feature "
                "ON image_annotations(feature_name, feature_value)"
            )
        )
        conn.execute(
            text("CREATE INDEX IF NOT EXISTS idx_annotations_image ON image_annotations(image_id)")
        )
        conn.execute(
            text("CREATE INDEX IF NOT EXISTS idx_registry_species ON image_registry(species)")
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_annotations_vlm_graded "
                "ON image_annotations(image_id, feature_name) "
                "WHERE annotation_type = 'vlm_graded'"
            )
        )
        conn.commit()
    logger.info("Vision tables ensured (image_registry, image_annotations, image_quality).")


def _ensure_embedding_indexes() -> None:
    """Create IVFFlat indexes for each active embedding group."""
    with engine.connect() as conn:
        for group in EMBEDDING_GROUPS:
            col = f"embedding_{group}"
            idx = f"ix_{group}_embedding"
            conn.execute(
                text(
                    f"CREATE INDEX IF NOT EXISTS {idx} ON reconciled_species "
                    f"USING ivfflat ({col} vector_cosine_ops) WITH (lists = 10)"
                )
            )
        conn.commit()
    logger.info("IVFFlat indexes ensured for groups: %s", list(EMBEDDING_GROUPS.keys()))


def load_ground_truth_pairs(path: str = GROUND_TRUTH_PATH) -> None:
    """Load ground truth pairs from known_lookalikes.yaml into the database (upsert).

    Generates one GroundTruthPair per (species, indexed_lookalike) edge.
    Attaches danger_note from the danger_notes field when available.
    """
    with open(path) as f:
        data = yaml.safe_load(f)

    entries = data.get("lookalikes", [])
    session = get_session()
    try:
        loaded = 0
        total = 0
        for entry in entries:
            species = entry["species"]
            # Build danger note lookup for this species
            danger_map: dict[str, str] = {}
            for dn in entry.get("danger_notes", []):
                danger_map[dn["lookalike"]] = dn["note"]

            for lookalike in entry.get("indexed", []):
                if lookalike == species:
                    continue
                total += 1
                # Canonical ordering: alphabetical
                a, b = sorted([species, lookalike])
                existing = (
                    session.query(GroundTruthPair).filter_by(species_a=a, species_b=b).first()
                )
                if existing is None:
                    session.add(
                        GroundTruthPair(
                            species_a=a,
                            species_b=b,
                            danger_note=danger_map.get(lookalike),
                            source="known_lookalikes",
                        )
                    )
                    loaded += 1
        session.commit()
        logger.info("Loaded %d new ground truth pairs (%d edges in file).", loaded, total)
    finally:
        session.close()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Drop all tables before recreating")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.reset:
        logger.warning("--reset: dropping all tables...")
        Base.metadata.drop_all(engine)
        logger.info("Tables dropped.")

    logger.info("Creating database tables...")

    # Enable pgvector
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()

    # Migrate old schema if needed
    _migrate_embeddings()

    # Reconcile embedding columns to match active profile (add missing, warn orphaned)
    _reconcile_embedding_columns()

    # Add image_urls columns if missing
    _add_image_columns()

    # Create vision pipeline tables
    _ensure_vision_tables()

    # Create all tables (only creates tables that don't exist yet)
    Base.metadata.create_all(engine)
    logger.info("Tables created successfully.")

    # Create IVFFlat indexes for active profile groups
    _ensure_embedding_indexes()

    # Load seed data
    load_ground_truth_pairs()
    logger.info("Database initialization complete.")


if __name__ == "__main__":
    main()
