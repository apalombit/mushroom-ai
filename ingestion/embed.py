"""
Embed reconciled species features into per-group vectors for pgvector.

For each ReconciledSpecies row, generates 6 embeddings (one per EMBEDDING_GROUPS entry):
    macro_visual, structural, flesh_sensory, microscopic_lab, ecological, taxonomic

Also populates overall_body_form from features_json for body-form gating.

Uses sentence-transformers (all-mpnet-base-v2, 768 dimensions) for embedding.
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
from ingestion.rubric import EMBEDDING_GROUPS, _get_nested, features_to_text

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """Load sentence-transformers model (cached — loads once per process)."""
    logger.info("Loading embedding model: %s", settings.embedding_model)
    return SentenceTransformer(settings.embedding_model)


def embed_species(session: Session, species: ReconciledSpecies) -> None:
    """
    Generate and store six embeddings for a single species.
    Updates embedding_<group> columns, overall_body_form, and embedded_at.
    """
    model = _get_model()
    features = species.features_json or {}

    def _embed(text: str) -> list[float] | None:
        if not text.strip():
            return None
        return model.encode(text).tolist()

    for group_name, field_list in EMBEDDING_GROUPS.items():
        text = features_to_text(features, field_list)
        col = f"embedding_{group_name}"
        setattr(species, col, _embed(text))

    # Populate body form for gating
    species.overall_body_form = _get_nested(features, "overall_body_form")

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
