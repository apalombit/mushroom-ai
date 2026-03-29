"""
SQLAlchemy models for the three-layer mushroom species data model.

Layer 1 — SourceObservation: One row per species per source. Raw LLM extraction.
Layer 2 — ReconciledSpecies: One canonical row per species. Merged from Layer 1.
Layer 3 — Embeddings: pgvector columns on ReconciledSpecies for similarity search.
           Columns are derived dynamically from the active GROUPING_PROFILE.

The pgvector dimension (768) matches sentence-transformers all-mpnet-base-v2.
"""

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, relationship

from ingestion.rubric import EMBEDDING_GROUPS

EMBEDDING_DIM = 768  # all-mpnet-base-v2 output dimension


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Layer 1 — Raw source observations
# ---------------------------------------------------------------------------


class SourceObservation(Base):
    """
    One row per species per source. Stores the raw LLM-extracted features
    exactly as parsed, plus provenance metadata.
    """

    __tablename__ = "source_observations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    scientific_name = Column(String(256), nullable=False, index=True)

    # Source provenance
    source_name = Column(String(256), nullable=False)  # e.g. "Wikipedia", "MushroomExpert"
    source_url = Column(Text, nullable=True)
    source_text_hash = Column(String(64), nullable=True)  # SHA256 of source text (dedup)

    # Raw extracted features as JSON (the full ExtractedSpeciesFeatures schema)
    features_json = Column(JSONB, nullable=False)

    # Extraction metadata
    extraction_model = Column(String(128), nullable=True)  # e.g. "ollama/llama3.1:8b"
    extraction_timestamp = Column(DateTime, default=datetime.utcnow)
    extraction_notes = Column(Text, nullable=True)

    # Reference images from source page
    image_urls = Column(JSONB, default=list)

    # Link to reconciled species
    species_id = Column(Integer, ForeignKey("reconciled_species.id"), nullable=True)
    species = relationship("ReconciledSpecies", back_populates="source_observations")

    __table_args__ = (
        Index("ix_source_species_source", "scientific_name", "source_name", unique=True),
    )


# ---------------------------------------------------------------------------
# Layer 2 — Reconciled species profiles
# ---------------------------------------------------------------------------


def _build_reconciled_species(groups: dict) -> type:
    """
    Build the ReconciledSpecies model class dynamically, adding one
    embedding_<group> column per entry in the active GROUPING_PROFILE.
    IVFFlat indexes are created separately in init_db.py as raw SQL.
    """
    attrs: dict = {
        "__tablename__": "reconciled_species",
        "__table_args__": (),
        "id": Column(Integer, primary_key=True, autoincrement=True),
        # Identity
        "scientific_name": Column(String(256), nullable=False, unique=True, index=True),
        "common_names": Column(JSONB, default=list),  # list[str]
        "family": Column(String(128), nullable=True, index=True),
        "genus": Column(String(128), nullable=True, index=True),
        "group": Column(String(128), nullable=True, index=True),
        # Full reconciled features as JSON
        "features_json": Column(JSONB, nullable=False),
        # Safety — explicit columns for fast filtering
        "edibility": Column(String(64), nullable=True, index=True),
        "known_toxins": Column(JSONB, default=list),
        "known_lookalikes": Column(JSONB, default=list),
        # Reconciliation metadata
        "reconciliation_confidence": Column(Float, nullable=True),
        "needs_review": Column(Boolean, default=False, index=True),
        "review_notes": Column(Text, nullable=True),
        "human_overrides": Column(JSONB, default=dict),
        "reconciled_at": Column(DateTime, default=datetime.utcnow),
        "source_count": Column(Integer, default=0),
        # Body form — extracted from features_json for fast filtering
        "overall_body_form": Column(String(64), nullable=True, index=True),
        "hymenium_type": Column(String(64), nullable=True, index=True),
        "overall_size_class": Column(String(64), nullable=True, index=True),
        "morphotype_signature": Column(String(256), nullable=True, index=True),
        # Reference images (deduplicated from sources, max 3)
        "image_urls": Column(JSONB, default=list),
        # Layer 3 embedding timestamp
        "embedded_at": Column(DateTime, nullable=True),
        # Relationships
        "source_observations": relationship("SourceObservation", back_populates="species"),
    }
    # Add one vector column per embedding group (profile-driven)
    for group_name in groups:
        attrs[f"embedding_{group_name}"] = Column(Vector(EMBEDDING_DIM), nullable=True)

    return type("ReconciledSpecies", (Base,), attrs)


ReconciledSpecies = _build_reconciled_species(EMBEDDING_GROUPS)


# ---------------------------------------------------------------------------
# Ground truth — known lookalike pairs for validation
# ---------------------------------------------------------------------------


class GroundTruthPair(Base):
    """
    Known lookalike pairs from mycological literature.
    Used to validate the similarity engine produces correct results.
    """

    __tablename__ = "ground_truth_pairs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    species_a = Column(String(256), nullable=False, index=True)
    species_b = Column(String(256), nullable=False, index=True)
    danger_note = Column(Text, nullable=True)  # e.g. "A is edible, B is deadly"
    source = Column(String(256), nullable=True)
    notes = Column(Text, nullable=True)

    __table_args__ = (Index("ix_gt_pair", "species_a", "species_b", unique=True),)
