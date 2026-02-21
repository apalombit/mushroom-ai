"""
LLM-based feature extraction from source text.

Takes raw text about a mushroom species and extracts structured features
using Instructor-validated LLM calls.

Pipeline:
    Source text → LLM (with ExtractedSpeciesFeatures schema) → validated features
    → store as Layer 1 (SourceObservation) in Postgres
"""

import hashlib
import logging

from config import settings
from db.connection import get_session
from db.models import SourceObservation
from ingestion.sources.wikipedia import SOURCE_NAME
from llm.client import structured_completion
from llm.schemas import ExtractedSpeciesFeatures

logger = logging.getLogger(__name__)

# Truncate source text to keep prompts within a safe token budget.
# gemma3:27b has 8192 max output tokens; 8000 chars ≈ ~2000 tokens of input, leaving
# plenty of room for the structured output.
_MAX_TEXT_CHARS = 8_000

SYSTEM_PROMPT = """\
You are a mycologist extracting structured features from species descriptions.

Rules:
- Extract ONLY what is explicitly stated in the source text.
- Use null for any feature not mentioned in the text — do NOT guess or infer.
- For colors: use descriptive natural language (e.g. "orange-red to yellow").
- For measurements: extract numeric ranges where given (diameter_min_cm / diameter_max_cm).
- For list fields: include every value mentioned (habitat_types, associated_trees, etc.).
- edibility must be exactly one of: edible, conditionally edible, inedible, toxic, deadly.
- Record ambiguity or uncertainty in extraction_notes.
- scientific_name must match the queried species name exactly.
"""


def extract_features_from_text(
    scientific_name: str,
    text: str,
) -> ExtractedSpeciesFeatures:
    """Extract structured features from source text using the LLM."""
    prompt = (
        f"Extract mushroom features for '{scientific_name}' from the following text:\n\n"
        f"{text[:_MAX_TEXT_CHARS]}"
    )
    return structured_completion(
        prompt=prompt,
        response_model=ExtractedSpeciesFeatures,
        system=SYSTEM_PROMPT,
        temperature=0.1,
        max_tokens=4096,
    )


def save_extraction(
    features: ExtractedSpeciesFeatures,
    source_url: str,
    source_text: str,
) -> SourceObservation:
    """
    Upsert extraction result into source_observations (Layer 1).

    If a row for (scientific_name, source_name=Wikipedia) already exists,
    its features_json and metadata are updated (upsert policy).
    """
    text_hash = hashlib.sha256(source_text.encode()).hexdigest()
    features_dict = features.model_dump()
    model_string = f"{settings.llm_provider}/{settings.llm_model}"

    session = get_session()
    try:
        existing = (
            session.query(SourceObservation)
            .filter_by(scientific_name=features.scientific_name, source_name=SOURCE_NAME)
            .first()
        )
        if existing:
            existing.features_json = features_dict
            existing.source_url = source_url
            existing.source_text_hash = text_hash
            existing.extraction_model = model_string
            existing.extraction_notes = features.extraction_notes
            obs = existing
        else:
            obs = SourceObservation(
                scientific_name=features.scientific_name,
                source_name=SOURCE_NAME,
                source_url=source_url,
                source_text_hash=text_hash,
                features_json=features_dict,
                extraction_model=model_string,
                extraction_notes=features.extraction_notes,
            )
            session.add(obs)
        session.commit()
        session.refresh(obs)
        return obs
    finally:
        session.close()
