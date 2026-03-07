"""
Initialize database: create tables, enable pgvector extension, load seed data.
Includes migration from old 3-embedding schema to new 6-embedding schema.

    python -m scripts.init_db
"""

import logging

import yaml
from sqlalchemy import inspect, text

from db.connection import engine, get_session
from db.models import EMBEDDING_DIM, Base, GroundTruthPair

logger = logging.getLogger(__name__)

GROUND_TRUTH_PATH = "data/seed/ground_truth_pairs.yaml"

# Old columns/indexes to drop during migration (all 3 old 384-dim columns)
_OLD_COLUMNS = ["embedding_morphological", "embedding_ecological", "embedding_taxonomic"]
_OLD_INDEXES = ["ix_morph_embedding", "ix_eco_embedding", "ix_taxon_embedding"]


def _migrate_embeddings() -> None:
    """Detect old 3-column embedding schema and migrate to 6-column schema."""
    inspector = inspect(engine)

    if "reconciled_species" not in inspector.get_table_names():
        return  # Fresh install, nothing to migrate

    columns = {col["name"] for col in inspector.get_columns("reconciled_species")}

    if "embedding_morphological" not in columns:
        return  # Already migrated or fresh schema

    logger.info("Detected old embedding schema — migrating to 6-group embeddings...")

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


# New columns that need to exist on reconciled_species
_NEW_EMBEDDING_COLS = [
    "embedding_macro_visual",
    "embedding_structural",
    "embedding_flesh_sensory",
    "embedding_microscopic_lab",
    "embedding_ecological",
    "embedding_taxonomic",
]


def _ensure_new_columns() -> None:
    """Add new embedding columns + overall_body_form if they don't exist yet."""
    inspector = inspect(engine)

    if "reconciled_species" not in inspector.get_table_names():
        return  # Will be created by create_all

    columns = {col["name"] for col in inspector.get_columns("reconciled_species")}
    added = []

    with engine.connect() as conn:
        for col in _NEW_EMBEDDING_COLS:
            if col not in columns:
                conn.execute(
                    text(
                        f"ALTER TABLE reconciled_species "
                        f"ADD COLUMN {col} vector({EMBEDDING_DIM})"
                    )
                )
                added.append(col)

        if "overall_body_form" not in columns:
            conn.execute(
                text(
                    "ALTER TABLE reconciled_species "
                    "ADD COLUMN overall_body_form VARCHAR(64)"
                )
            )
            added.append("overall_body_form")

        if added:
            conn.commit()
            logger.info("Added columns: %s", ", ".join(added))


def load_ground_truth_pairs(path: str = GROUND_TRUTH_PATH) -> None:
    """Load ground truth pairs from YAML into the database (upsert — skip duplicates)."""
    with open(path) as f:
        data = yaml.safe_load(f)

    pairs = data.get("pairs", [])
    session = get_session()
    try:
        loaded = 0
        for pair in pairs:
            existing = (
                session.query(GroundTruthPair)
                .filter_by(species_a=pair["species_a"], species_b=pair["species_b"])
                .first()
            )
            if existing is None:
                session.add(
                    GroundTruthPair(
                        species_a=pair["species_a"],
                        species_b=pair["species_b"],
                        danger_note=pair.get("danger_note"),
                        source=pair.get("source"),
                        notes=pair.get("notes"),
                    )
                )
                loaded += 1
        session.commit()
        logger.info("Loaded %d new ground truth pairs (%d total in file).", loaded, len(pairs))
    finally:
        session.close()


def main():
    logging.basicConfig(level=logging.INFO)
    logger.info("Creating database tables...")

    # Enable pgvector
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()

    # Migrate old schema if needed
    _migrate_embeddings()

    # Add new columns to existing tables (create_all won't alter existing tables)
    _ensure_new_columns()

    # Create all tables (only creates tables that don't exist yet)
    Base.metadata.create_all(engine)
    logger.info("Tables created successfully.")

    # Load seed data
    load_ground_truth_pairs()
    logger.info("Database initialization complete.")


if __name__ == "__main__":
    main()
