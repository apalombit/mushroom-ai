"""Load species-level ground truth from reconciled_species."""

from __future__ import annotations

from pathlib import Path

import yaml
from sqlalchemy import text
from sqlalchemy.orm import Session

_FEATURES_YAML = Path(__file__).resolve().parent.parent / "config" / "features.yaml"
_CORRECTIONS_YAML = Path(__file__).resolve().parent.parent / "config" / "gt_color_corrections.yaml"

def _derive_ring_presence(raw: object) -> str | None:
    """Map veil dict → binary ring presence label, conservative version.

    Only count rings that are reliably *visible* in typical field photos:
    - type ∈ {partial, both} AND ring_persistence != fugacious → "present"
    - type == "absent" → "absent"
    - cortina (rarely visible in mature specimens) → exclude
    - universal-only (volva at base, not a ring on the upper stem) → exclude
    - ring_persistence == "fugacious" (ring drops off early) → exclude
    - missing/unknown → exclude

    Excluded species return None so the eval skips them.

    Why this is conservative: an earlier eval against the broader rule
    (any non-absent type → present) gave 12% recall on "present" because
    cortinate species (Cortinarius), fugacious species (Armillaria,
    Strobilomyces, Cystoderma, Agaricus bisporus) almost never show a
    visible ring in mature photos.  The model was correctly seeing "no
    ring" but the GT was over-permissive at the species level.
    """
    if not isinstance(raw, dict):
        return None
    type_val = (raw.get("type") or "").strip().lower()
    persistence = (raw.get("ring_persistence") or "").strip().lower()
    if type_val == "absent":
        return "absent"
    if type_val in {"partial", "both"} and persistence != "fugacious":
        return "present"
    return None


def _derive_volva_presence(raw: object) -> str | None:
    """Map volva.type → binary volva presence label.

    Any non-absent canonical value (saccate, flaring, circumsessile, zoned,
    friable, napiform) → "present"
    absent → "absent"
    null / unknown / missing → None (excluded from eval)
    """
    if not isinstance(raw, str):
        return None
    val = raw.strip().lower()
    if val == "absent":
        return "absent"
    if val in {"saccate", "flaring", "circumsessile", "zoned", "friable", "napiform"}:
        return "present"
    return None


# Map feature → how to extract ground truth from reconciled_species.
# "column" = direct column, "features_json" = nested JSONB path.
# Optional "derive" = callable applied after raw fetch; returns None to exclude row.
GROUND_TRUTH_SPEC: dict[str, dict] = {
    "hymenium_type": {"source": "column", "column": "hymenium_type"},
    "cap_color": {"source": "features_json", "path": ["cap", "colors", 0]},
    "ring_presence": {
        "source": "features_json",
        "path": ["veil"],
        "derive": _derive_ring_presence,
    },
    "volva_presence": {
        "source": "features_json",
        "path": ["volva", "type"],
        "derive": _derive_volva_presence,
    },
    "substrate": {"source": "features_json", "path": ["ecology", "substrate"]},
    "surface_texture": {"source": "features_json", "path": ["cap", "surface_texture"]},
}

# Multi-label spec: feature → how to extract ALL values (list) from features_json.
_MULTILABEL_SPEC: dict[str, dict] = {
    "cap_color": {"source": "features_json", "path": ["cap", "colors"]},
}


def load_ground_truth(
    session: Session, feature_name: str
) -> dict[str, str | None]:
    """Load species-level ground truth for a feature.

    Returns:
        Dict mapping ``{species_name: feature_value}``.

    Raises:
        KeyError: If *feature_name* is not in GROUND_TRUTH_SPEC.
    """
    spec = GROUND_TRUTH_SPEC[feature_name]

    if spec["source"] == "column":
        rows = session.execute(
            text(
                f"SELECT scientific_name, {spec['column']}"
                " FROM reconciled_species"
            )
        ).fetchall()
        return {r[0]: r[1] for r in rows}

    # features_json path
    rows = session.execute(
        text("SELECT scientific_name, features_json FROM reconciled_species")
    ).fetchall()
    gt: dict[str, str | None] = {}
    path = spec["path"]
    derive = spec.get("derive")
    for name, fj in rows:
        val = fj
        try:
            for key in path:
                val = val[key]
        except (KeyError, IndexError, TypeError):
            val = None
        gt[name] = derive(val) if derive else val
    return gt


def _resolve_color_to_canonical(
    raw: str,
    canonical: set[str],
    aliases: dict[str, str],
) -> set[str]:
    """Resolve a single free-text color string to canonical palette values.

    Strategy (in order):
    1. Exact canonical match
    2. Alias dict lookup
    3. Substring: if any canonical color appears in the free-text, include it
    """
    low = raw.strip().lower()
    if not low:
        return set()

    # 1. Exact canonical
    if low in canonical:
        return {low}

    # 2. Alias lookup
    if low in aliases:
        mapped = aliases[low]
        if mapped in canonical:
            return {mapped}

    # 3. Substring match — check if any canonical color appears in the text
    hits: set[str] = set()
    for c in canonical:
        if c in low:
            hits.add(c)
    return hits


def load_ground_truth_multilabel(
    session: Session,
    feature_name: str,
) -> dict[str, set[str]]:
    """Load ALL canonical values per species, resolved to canonical palette.

    For cap_color: reads ``cap.colors[]``, resolves each free-text color
    to canonical palette via aliases + substring matching.

    Returns:
        ``{species_name: {color1, color2, ...}}``.

    Raises:
        KeyError: If *feature_name* has no multi-label spec.
    """
    from ingestion.normalize import load_vocabulary

    spec = _MULTILABEL_SPEC[feature_name]

    # Load canonical palette from features.yaml
    with open(_FEATURES_YAML) as f:
        all_features = yaml.safe_load(f)
    canonical: set[str] = set(all_features[feature_name]["classes"])

    # Load alias dict
    vocab = load_vocabulary()
    aliases: dict[str, str] = vocab.get("color_palette", {}).get("aliases", {})

    # Fetch features_json from DB
    rows = session.execute(
        text("SELECT scientific_name, features_json FROM reconciled_species")
    ).fetchall()

    gt: dict[str, set[str]] = {}
    path = spec["path"]

    for name, fj in rows:
        val = fj
        try:
            for key in path:
                val = val[key]
        except (KeyError, IndexError, TypeError):
            gt[name] = set()
            continue

        if not isinstance(val, list):
            gt[name] = set()
            continue

        resolved: set[str] = set()
        for raw_color in val:
            if isinstance(raw_color, str):
                resolved |= _resolve_color_to_canonical(raw_color, canonical, aliases)
        gt[name] = resolved

    # Merge manual corrections (audit-driven additions)
    corrections = _load_corrections(feature_name)
    for species, extra_colors in corrections.items():
        if species in gt:
            gt[species] |= extra_colors
        else:
            gt[species] = extra_colors

    return gt


def _load_corrections(feature_name: str) -> dict[str, set[str]]:
    """Load manual GT corrections from gt_color_corrections.yaml."""
    if not _CORRECTIONS_YAML.exists():
        return {}
    with open(_CORRECTIONS_YAML) as f:
        data = yaml.safe_load(f) or {}
    raw = data.get(feature_name, {})
    return {
        species: set(colors) for species, colors in raw.items() if colors
    }
