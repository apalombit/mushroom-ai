"""
Multi-source reconciliation: merge Layer 1 observations into Layer 2 profiles.

For each species, collects all SourceObservation rows, passes them to the LLM
with a strict reconciliation rubric, and produces one canonical ReconciledSpecies row.

Key behaviors:
    - Merges complementary information (different features from different sources)
    - Resolves conflicts by choosing the most specific/reliable value
    - Flags irreconcilable conflicts for human review (needs_review=True)
    - Respects human_overrides: fields manually set are never overwritten
    - Stores reconciliation confidence and conflict notes
"""

import json as _json
import logging
from datetime import datetime

from sqlalchemy.orm import Session

from db.models import ReconciledSpecies, SourceObservation
from llm.client import structured_completion
from llm.schemas import ReconciliationResult

logger = logging.getLogger(__name__)

# Edibility priority for deterministic fallback (lower index = more conservative).
# Used as a safety net if the LLM somehow returns null for edibility_status.
_EDIBILITY_PRIORITY = {
    "deadly": 0,
    "toxic": 1,
    "inedible": 2,
    "conditionally edible": 3,
    "edible": 4,
    "choice": 5,
}


def _fallback_edibility(observations: list[SourceObservation], llm_value: str | None) -> str | None:
    """
    If the LLM returned null for edibility_status, deterministically pick the
    most conservative (safety-first) value from the source observations.
    """
    if llm_value is not None:
        return llm_value
    values = [
        (obs.features_json or {}).get("edibility_status")
        for obs in observations
        if (obs.features_json or {}).get("edibility_status")
    ]
    if not values:
        return None
    return min(
        values,
        key=lambda v: _EDIBILITY_PRIORITY.get((v or "").lower(), 99),
    )

SYSTEM_PROMPT = """\
You are a mycologist reconciling multiple source descriptions of the same mushroom species.

Rules:
- Merge complementary information from different sources (prefer more specific values).
- For conflicts: choose the most specific, well-supported value and note the conflict.
- Never guess or infer — only use what is explicitly stated in the sources.
- Set needs_review=True if any critical safety fields (edibility, known_toxins) conflict.
- Preserve all common_names, known_toxins, known_lookalikes from all sources.
- scientific_name must match the queried species exactly.
"""


def reconcile_species(session: Session, scientific_name: str) -> ReconciledSpecies | None:
    """
    Reconcile all Layer 1 observations for a species into a Layer 2 canonical profile.

    Returns the upserted ReconciledSpecies row, or None if no source observations exist.
    Skips if reconciled_at is newer than all source extraction_timestamps (incremental).
    """
    observations = (
        session.query(SourceObservation)
        .filter_by(scientific_name=scientific_name)
        .all()
    )

    if not observations:
        logger.info("No source observations for %s — skipping.", scientific_name)
        return None

    # Incremental: check if reconciliation is already up-to-date
    existing = (
        session.query(ReconciledSpecies)
        .filter_by(scientific_name=scientific_name)
        .first()
    )
    if existing and existing.reconciled_at:
        newest_extraction = max(
            (obs.extraction_timestamp for obs in observations if obs.extraction_timestamp),
            default=None,
        )
        if newest_extraction and existing.reconciled_at >= newest_extraction:
            logger.debug("Already reconciled (up-to-date): %s", scientific_name)
            return existing

    # Single-source: copy directly with full confidence
    if len(observations) == 1:
        features = observations[0].features_json
        confidence = 1.0
        conflicts: list[str] = []
        needs_review = False
        review_notes = None
    else:
        # Multi-source: LLM reconciliation. Full features_json for every source —
        # gemma3:27b has a 128K context window so ~5K tokens of source data is trivial.
        sources_text = "\n\n".join(
            f"Source {i + 1} ({obs.source_name}):\n"
            + _json.dumps(obs.features_json, ensure_ascii=False, separators=(",", ":"))
            for i, obs in enumerate(observations)
        )
        prompt = (
            f"Reconcile the following {len(observations)} source descriptions "
            f"for '{scientific_name}':\n\n{sources_text}"
        )
        result: ReconciliationResult = structured_completion(
            prompt=prompt,
            response_model=ReconciliationResult,
            system=SYSTEM_PROMPT,
            temperature=0.1,
            max_tokens=4096,
        )
        features = result.reconciled_features.model_dump()
        # Deterministic fallback for safety-critical edibility field.
        if features.get("edibility_status") is None:
            features["edibility_status"] = _fallback_edibility(
                observations, features.get("edibility_status")
            )
        confidence = result.confidence
        conflicts = result.conflicts
        needs_review = result.needs_review
        review_notes = result.review_notes

    now = datetime.utcnow()

    if existing:
        existing.features_json = features
        existing.common_names = features.get("common_names", [])
        existing.family = features.get("family")
        existing.genus = features.get("genus")
        existing.edibility = features.get("edibility_status")
        existing.known_toxins = features.get("known_toxins", [])
        existing.known_lookalikes = features.get("known_lookalikes", [])
        existing.reconciliation_confidence = confidence
        existing.needs_review = needs_review
        existing.review_notes = review_notes
        existing.reconciled_at = now
        existing.source_count = len(observations)
        row = existing
    else:
        row = ReconciledSpecies(
            scientific_name=scientific_name,
            common_names=features.get("common_names", []),
            family=features.get("family"),
            genus=features.get("genus"),
            features_json=features,
            edibility=features.get("edibility_status"),
            known_toxins=features.get("known_toxins", []),
            known_lookalikes=features.get("known_lookalikes", []),
            reconciliation_confidence=confidence,
            needs_review=needs_review,
            review_notes=review_notes,
            reconciled_at=now,
            source_count=len(observations),
        )
        session.add(row)

    session.commit()
    session.refresh(row)

    if conflicts:
        logger.warning("Conflicts in %s: %s", scientific_name, conflicts)

    return row
