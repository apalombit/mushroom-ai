"""
Embed reconciled species features into per-group vectors for pgvector.

For each ReconciledSpecies row:
    1. Convert morphological features → text description → embed → embedding_morphological
    2. Convert ecological features → text description → embed → embedding_ecological
    3. Convert taxonomic features → text description → embed → embedding_taxonomic

Uses sentence-transformers (all-MiniLM-L6-v2, 384 dimensions) for embedding.
The text conversion follows the rubric in ingestion/rubric.py.
"""

import logging
from datetime import datetime
from functools import lru_cache

from sentence_transformers import SentenceTransformer
from sqlalchemy import or_
from sqlalchemy.orm import Session

from config import settings
from db.models import ReconciledSpecies
from ingestion.rubric import (
    ECOLOGICAL_FIELDS,
    MORPHOLOGICAL_FIELDS,
    TAXONOMIC_FIELDS,
    features_to_text,
)

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """Load sentence-transformers model (cached — loads once per process)."""
    logger.info("Loading embedding model: %s", settings.embedding_model)
    return SentenceTransformer(settings.embedding_model)


def embed_species(session: Session, species: ReconciledSpecies) -> None:
    """
    Generate and store three embeddings for a single species.
    Updates embedding_morphological, embedding_ecological, embedding_taxonomic,
    and embedded_at on the species row and commits.
    """
    model = _get_model()
    features = species.features_json or {}

    morph_text = features_to_text(features, MORPHOLOGICAL_FIELDS)
    eco_text = features_to_text(features, ECOLOGICAL_FIELDS)
    taxon_text = features_to_text(features, TAXONOMIC_FIELDS)

    def _embed(text: str) -> list[float] | None:
        """Embed text; return None for empty text (no embedding stored)."""
        if not text.strip():
            return None
        return model.encode(text).tolist()

    species.embedding_morphological = _embed(morph_text)
    species.embedding_ecological = _embed(eco_text)
    species.embedding_taxonomic = _embed(taxon_text)
    species.embedded_at = datetime.utcnow()

    session.commit()
    logger.debug("Embedded: %s", species.scientific_name)


def embed_all(session: Session) -> tuple[int, int]:
    """
    Embed all species where embedded_at is None or older than reconciled_at.

    Returns (embedded_count, skipped_count).
    """
    pending = (
        session.query(ReconciledSpecies)
        .filter(
            or_(
                ReconciledSpecies.embedded_at.is_(None),
                ReconciledSpecies.embedded_at < ReconciledSpecies.reconciled_at,
            )
        )
        .all()
    )

    embedded, skipped = 0, 0
    for species in pending:
        try:
            embed_species(session, species)
            embedded += 1
        except Exception as e:
            logger.error("Failed to embed %s: %s", species.scientific_name, e)
            session.rollback()
            skipped += 1

    return embedded, skipped
