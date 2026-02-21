"""
Initialize database: create tables, enable pgvector extension, load seed data.

    python -m scripts.init_db
"""

import logging

import yaml
from sqlalchemy import text

from db.connection import engine, get_session
from db.models import Base, GroundTruthPair

logger = logging.getLogger(__name__)

GROUND_TRUTH_PATH = "data/seed/ground_truth_pairs.yaml"


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

    # Create all tables
    Base.metadata.create_all(engine)
    logger.info("Tables created successfully.")

    # Load seed data
    load_ground_truth_pairs()
    logger.info("Database initialization complete.")


if __name__ == "__main__":
    main()
