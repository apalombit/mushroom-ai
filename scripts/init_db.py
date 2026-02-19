"""
Initialize database: create tables, enable pgvector extension.

    python -m scripts.init_db

TODO:
    - [ ] Implement table creation from db/models.py
    - [ ] Load ground truth pairs from data/seed/ground_truth_pairs.yaml
"""

import logging

from db.connection import engine
from db.models import Base

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO)
    logger.info("Creating database tables...")

    # Enable pgvector
    with engine.connect() as conn:
        from sqlalchemy import text
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()

    # Create all tables
    Base.metadata.create_all(engine)
    logger.info("Tables created successfully.")

    # TODO: Load ground truth pairs from seed data
    logger.info("Database initialization complete.")


if __name__ == "__main__":
    main()
