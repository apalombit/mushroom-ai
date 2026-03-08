"""
Feature rubric: defines the controlled vocabularies and feature groups
used throughout the system.

This rubric governs:
    - What the ingestion LLM extracts (via schemas in llm/schemas.py)
    - How features are grouped for embedding (6 embedding groups + numeric)
    - What the similarity engine compares

Field-to-group assignments are loaded from a YAML profile under
ingestion/profiles/<name>.yaml. The active profile is set via
GROUPING_PROFILE in .env (default: "default").
"""

from pathlib import Path

import yaml

from config import settings

# ---------------------------------------------------------------------------
# Numeric fields — for direct comparison, not embedded
# ---------------------------------------------------------------------------

# Range pairs: (min_field, max_field)
NUMERIC_RANGE_FIELDS: list[tuple[str, str]] = [
    ("cap.diameter_min_cm", "cap.diameter_max_cm"),
    ("stem.height_min_cm", "stem.height_max_cm"),
    ("stem.diameter_min_cm", "stem.diameter_max_cm"),
    ("spore.length_min_um", "spore.length_max_um"),
    ("spore.width_min_um", "spore.width_max_um"),
]

# Single values: (field, tolerance) — tolerance is max distance for score = 0
NUMERIC_SINGLE_FIELDS: list[tuple[str, float]] = [
    ("pores.density_per_mm", 3.0),
    ("tubes.depth_mm", 15.0),
    ("spore.spine_length_um", 2.0),
    ("spore.spine_base_width_um", 1.0),
    ("microscopic.pileipellis_element_width_um", 5.0),
]

# Combined list for iteration
NUMERIC_FIELDS: list[tuple] = NUMERIC_RANGE_FIELDS + NUMERIC_SINGLE_FIELDS

# All numeric field paths (flattened from range pairs + single values)
_NUMERIC_FIELD_PATHS = [f for pair in NUMERIC_RANGE_FIELDS for f in pair] + [
    f for f, _ in NUMERIC_SINGLE_FIELDS
]

# ---------------------------------------------------------------------------
# YAML profile loading
# ---------------------------------------------------------------------------

_PROFILES_DIR = Path(__file__).parent / "profiles"


def _load_profile(name: str) -> dict[str, list[str]]:
    """Load a grouping profile YAML and validate it."""
    path = _PROFILES_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Grouping profile {name!r} not found at {path}")
    with open(path) as f:
        profile = yaml.safe_load(f)

    if not profile:
        raise ValueError(f"Profile {name!r} is empty")
    # Normalize empty groups to []
    for key in list(profile):
        if profile[key] is None:
            profile[key] = []
    # Validate no numeric fields in embedding groups
    numeric = set(_NUMERIC_FIELD_PATHS)
    for group, fields in profile.items():
        overlap = numeric & set(fields)
        if overlap:
            raise ValueError(
                f"Profile {name!r}, group {group!r}: numeric fields not allowed: {overlap}"
            )
    return profile


def get_active_profile() -> str:
    """Return the name of the active grouping profile."""
    return settings.grouping_profile


# ---------------------------------------------------------------------------
# Embedding groups — each maps to one pgvector column
# ---------------------------------------------------------------------------

EMBEDDING_GROUPS: dict[str, list[str]] = _load_profile(settings.grouping_profile)
GROUP_SLOTS = tuple(EMBEDDING_GROUPS.keys())  # derived from active profile, for backward compat

# ---------------------------------------------------------------------------
# Backward-compatible union — used by build_comparison_table
# ---------------------------------------------------------------------------

MORPHOLOGICAL_FIELDS: list[str] = []
for _fields in EMBEDDING_GROUPS.values():
    MORPHOLOGICAL_FIELDS.extend(_fields)
MORPHOLOGICAL_FIELDS.extend(_NUMERIC_FIELD_PATHS)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_nested(d: dict, path: str):
    """Walk a dot-separated path into a nested dict. Returns None if any key is missing."""
    keys = path.split(".")
    current = d
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
        if current is None:
            return None
    return current


def features_to_text(features: dict, field_list: list[str]) -> str:
    """
    Convert a subset of species features into a natural language description
    suitable for embedding.

    Walks each dot-path field into the nested features dict, skips None/empty
    values, and joins non-null values into concise field-guide prose.

    Args:
        features: The full features_json dict from a ReconciledSpecies row.
        field_list: Which fields to include (e.g., MORPHOLOGICAL_FIELDS).

    Returns:
        A natural language string describing those features.
    """
    parts = []
    for field_path in field_list:
        value = _get_nested(features, field_path)
        if value is None or value == "" or value == []:
            continue

        label = field_path.split(".")[-1].replace("_", " ")

        if isinstance(value, list):
            text = ", ".join(str(v) for v in value if v is not None)
            if text:
                parts.append(f"{label}: {text}")
        elif isinstance(value, bool):
            parts.append(f"{label}: {'yes' if value else 'no'}")
        else:
            parts.append(f"{label}: {value}")

    return ". ".join(parts) + ("." if parts else "")
