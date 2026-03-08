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

GROUND_TRUTH_PATH = "data/seed/ground_truth_pairs.yaml"

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

    if "overall_body_form" not in columns:
        to_add_extras = ["overall_body_form"]
    else:
        to_add_extras = []

    if to_add or to_add_extras:
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
                        "ALTER TABLE reconciled_species "
                        "ADD COLUMN IF NOT EXISTS overall_body_form VARCHAR(64)"
                    )
                )
            if to_add:
                # New embedding columns added — force re-embed
                conn.execute(text("UPDATE reconciled_species SET embedded_at = NULL"))
            conn.commit()
        if to_add:
            logger.info("Added embedding columns: %s (embedded_at reset)", sorted(to_add))
        if to_add_extras:
            logger.info("Added columns: %s", to_add_extras)


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

    # Reconcile embedding columns to match active profile (add missing, warn orphaned)
    _reconcile_embedding_columns()

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
