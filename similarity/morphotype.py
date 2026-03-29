"""
Compound morphotype signatures for coarse pre-filtering.

A morphotype signature is a pipe-delimited string of 5 diagnostic features:
body_form | hymenium_type | size_class | cap_shape | gill_attachment

Used to filter candidates after embedding retrieval but before scoring.
"""

from __future__ import annotations

from ingestion.rubric import _get_nested

MORPHOTYPE_FIELDS = [
    "overall_body_form",
    "hymenium.type",
    "overall_size_class",
    "cap.shape",
    "gills.attachment",
]
_UNKNOWN = "?"


def compute_morphotype_signature(features: dict) -> str | None:
    """Build a pipe-separated morphotype key from diagnostic fields.

    Returns None if all fields are unknown.
    """
    parts = []
    for field_path in MORPHOTYPE_FIELDS:
        val = _get_nested(features, field_path)
        parts.append((val or _UNKNOWN).lower().strip())
    return "|".join(parts) if any(p != _UNKNOWN for p in parts) else None


def morphotype_match_score(sig_a: str | None, sig_b: str | None) -> float:
    """Fraction of known fields that match between two signatures.

    Unknown fields (?) on either side are excluded from the comparison.
    Returns 0.0 if no fields can be compared, 1.0 if all known fields match.
    """
    if not sig_a or not sig_b:
        return 0.0
    parts_a = sig_a.split("|")
    parts_b = sig_b.split("|")
    comparable = 0
    matching = 0
    for a, b in zip(parts_a, parts_b):
        if a == _UNKNOWN or b == _UNKNOWN:
            continue
        comparable += 1
        if a == b:
            matching += 1
    return matching / comparable if comparable > 0 else 0.0
