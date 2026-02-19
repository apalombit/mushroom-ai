"""
SQLAlchemy models for the three-layer mushroom species data model.

Layer 1 — SourceObservation: One row per species per source. Raw LLM extraction.
Layer 2 — ReconciledSpecies: One canonical row per species. Merged from Layer 1.
Layer 3 — Embeddings: pgvector columns on ReconciledSpecies for similarity search.

The pgvector dimension (384) matches sentence-transformers all-MiniLM-L6-v2.
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


EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 output dimension


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

    # Link to reconciled species
    species_id = Column(Integer, ForeignKey("reconciled_species.id"), nullable=True)
    species = relationship("ReconciledSpecies", back_populates="source_observations")

    __table_args__ = (
        Index("ix_source_species_source", "scientific_name", "source_name", unique=True),
    )


# ---------------------------------------------------------------------------
# Layer 2 — Reconciled species profiles
# ---------------------------------------------------------------------------


class ReconciledSpecies(Base):
    """
    One canonical row per species. The reconciled feature profile used for
    similarity computation. Built from Layer 1 by the reconciliation pipeline.

    Structured columns for filterable fields + JSONB for the full feature set
    + pgvector columns for per-group embeddings (Layer 3).
    """

    __tablename__ = "reconciled_species"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Identity
    scientific_name = Column(String(256), nullable=False, unique=True, index=True)
    common_names = Column(JSONB, default=list)  # list[str]
    family = Column(String(128), nullable=True, index=True)
    genus = Column(String(128), nullable=True, index=True)

    # Full reconciled features as JSON (ExtractedSpeciesFeatures schema)
    features_json = Column(JSONB, nullable=False)

    # Safety — explicit columns for fast filtering
    edibility = Column(String(64), nullable=True, index=True)
    known_toxins = Column(JSONB, default=list)
    known_lookalikes = Column(JSONB, default=list)  # From literature (ground truth)

    # Reconciliation metadata
    reconciliation_confidence = Column(Float, nullable=True)
    needs_review = Column(Boolean, default=False, index=True)
    review_notes = Column(Text, nullable=True)
    human_overrides = Column(JSONB, default=dict)  # Fields manually overridden
    reconciled_at = Column(DateTime, default=datetime.utcnow)
    source_count = Column(Integer, default=0)  # How many sources contributed

    # Layer 3 — pgvector embeddings for similarity search
    embedding_morphological = Column(Vector(EMBEDDING_DIM), nullable=True)
    embedding_ecological = Column(Vector(EMBEDDING_DIM), nullable=True)
    embedding_taxonomic = Column(Vector(EMBEDDING_DIM), nullable=True)
    embedded_at = Column(DateTime, nullable=True)

    # Relationships
    source_observations = relationship("SourceObservation", back_populates="species")

    __table_args__ = (
        Index(
            "ix_morph_embedding",
            "embedding_morphological",
            postgresql_using="ivfflat",
            postgresql_with={"lists": 10},
            postgresql_ops={"embedding_morphological": "vector_cosine_ops"},
        ),
        Index(
            "ix_eco_embedding",
            "embedding_ecological",
            postgresql_using="ivfflat",
            postgresql_with={"lists": 10},
            postgresql_ops={"embedding_ecological": "vector_cosine_ops"},
        ),
        Index(
            "ix_taxon_embedding",
            "embedding_taxonomic",
            postgresql_using="ivfflat",
            postgresql_with={"lists": 10},
            postgresql_ops={"embedding_taxonomic": "vector_cosine_ops"},
        ),
    )


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

    __table_args__ = (
        Index("ix_gt_pair", "species_a", "species_b", unique=True),
    )
