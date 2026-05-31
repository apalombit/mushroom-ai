"""Pydantic schema for per-image VLM annotations.

Canonical class lists for morphological features are loaded at import time from
`vision/config/features.yaml` so the schema stays in lockstep with the training
vocabulary. The view-angle / image-quality fields are pilot-specific and live
only here.

Cap color is intentionally free text in this pilot — the goal is to see what
colors the VLM proposes before constraining to the 14 canonical values.
"""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

_FEATURES_YAML = Path(__file__).resolve().parent.parent / "config" / "features.yaml"


def _load_classes() -> dict[str, list[str]]:
    with open(_FEATURES_YAML) as f:
        cfg = yaml.safe_load(f)
    return {name: spec["classes"] for name, spec in cfg.items()}


_CLASSES = _load_classes()

# Build Literal types from the canonical class lists. Using tuple() so Literal
# accepts the unpacked values.
CapShape = Literal[tuple(_CLASSES["cap_shape"])]  # type: ignore[valid-type]
SurfaceTexture = Literal[tuple(_CLASSES["surface_texture"])]  # type: ignore[valid-type]
HymeniumType = Literal[tuple(_CLASSES["hymenium_type"])]  # type: ignore[valid-type]
GillAttachment = Literal[tuple(_CLASSES["gill_attachment"])]  # type: ignore[valid-type]
BodyForm = Literal[tuple(_CLASSES["overall_body_form"])]  # type: ignore[valid-type]


class VLMImageAnnotation(BaseModel):
    """Per-image VLM observation. Combines view metadata and morphology in one shot."""

    # ── Image quality / view metadata ──────────────────────────────────────
    subject_present: bool = Field(..., description="True if any mushroom is visible in the image.")
    subject_dominance: Literal["dominant", "partial", "small", "absent"] = Field(
        ...,
        description=(
            "How much of the frame the mushroom occupies. "
            "'dominant' = main subject; 'partial' = visible but not centered; "
            "'small' = tiny/distant; 'absent' = no mushroom."
        ),
    )
    image_quality: Literal["good", "acceptable", "blurry", "occluded", "unusable"] = Field(
        ..., description="Overall usability of the photo for morphological labeling."
    )
    view_angle: Literal[
        "cap_top",
        "side_profile",
        "underside",
        "full_body",
        "cross_section",
        "detail_macro",
        "habitat",
        "microscopy",
        "other",
    ] = Field(..., description="Primary perspective from which the mushroom is photographed.")

    # ── Morphological features (only what is visibly determinable) ─────────
    cap_visible: bool = Field(..., description="True if the cap surface is visible.")
    cap_color_primary: str | None = Field(
        None,
        description=(
            "Dominant color of the cap, in plain English (e.g. 'red', 'reddish-brown'). "
            "Free text — do not constrain to a fixed palette. Null if not visible."
        ),
    )
    cap_color_secondary: str | None = Field(
        None,
        description=(
            "Secondary cap color if clearly present (e.g. white warts on red). Null otherwise."
        ),
    )
    cap_shape: CapShape | None = Field(
        None, description="Cap shape from the canonical vocabulary, or null if not visible."
    )
    cap_surface_texture: SurfaceTexture | None = Field(
        None, description="Cap surface texture from the canonical vocabulary, or null."
    )

    hymenium_visible: bool = Field(
        ..., description="True if the underside of the cap (gills/pores/etc.) is visible."
    )
    hymenium_type: HymeniumType | None = Field(
        None, description="Type of hymenium structure, or null if not visible."
    )
    gill_attachment: GillAttachment | None = Field(
        None, description="Gill-to-stem attachment style, or null if N/A or not visible."
    )

    body_form: BodyForm | None = Field(
        None, description="Overall morphological body form, or null if not determinable."
    )

    # ── Free-text notes & confidence ───────────────────────────────────────
    notes: str | None = Field(
        None,
        description="One-sentence free-text observation the human reviewer can read for context.",
    )
    confidence: Literal["low", "medium", "high"] = Field(
        ..., description="VLM's self-rated confidence in the overall annotation."
    )
