"""Describe-then-classify Pydantic schemas for per-feature VLM extraction.

Each schema follows the VDGD pattern: visible → visual_description → reasoning →
classification → confidence.  Field ordering is critical — left-to-right generation
forces the VLM to ground itself in visual evidence before committing to a class.

Consistency guardrails (model_validator) enforce:
  - not visible  → null classification + cannot_tell
  - cannot_tell  → null classification
  - classification present → visible must be True
"""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# Load canonical class lists (same pattern as vlm_schema.py)
# ---------------------------------------------------------------------------

_FEATURES_YAML = Path(__file__).resolve().parent.parent / "config" / "features.yaml"


def _load_classes() -> dict[str, list[str]]:
    with open(_FEATURES_YAML) as f:
        cfg = yaml.safe_load(f)
    return {name: spec["classes"] for name, spec in cfg.items()}


_CLASSES = _load_classes()

HymeniumType = Literal[tuple(_CLASSES["hymenium_type"])]  # type: ignore[valid-type]
CapColor = Literal[tuple(_CLASSES["cap_color"])]  # type: ignore[valid-type]
RingPresence = Literal[tuple(_CLASSES["ring_presence"])]  # type: ignore[valid-type]
VolvaPresence = Literal[tuple(_CLASSES["volva_presence"])]  # type: ignore[valid-type]
Substrate = Literal[tuple(_CLASSES["substrate"])]  # type: ignore[valid-type]
SurfaceTexture = Literal[tuple(_CLASSES["surface_texture"])]  # type: ignore[valid-type]
StemShape = Literal[tuple(_CLASSES["stem_shape"])]  # type: ignore[valid-type]
CapShape = Literal[tuple(_CLASSES["cap_shape"])]  # type: ignore[valid-type]

Confidence = Literal["high", "low", "cannot_tell"]

# ---------------------------------------------------------------------------
# Per-feature result schemas
# ---------------------------------------------------------------------------


class HymeniumTypeResult(BaseModel):
    """Per-image hymenium type extraction with describe-then-classify ordering."""

    visible: bool = Field(..., description="Is any spore-bearing surface visible in this image?")
    visual_description: str | None = Field(
        None,
        description=(
            "Describe what you see on the underside/fertile surface: "
            "shape, spacing, texture of the structures. 1-2 sentences."
        ),
    )
    reasoning: str | None = Field(
        None,
        description="Which hymenium type does this description match, and why?",
    )
    hymenium_type: HymeniumType | None = Field(
        None,
        description="Classified hymenium type, or null if cannot tell.",
    )
    confidence: Confidence = Field(
        ...,
        description=(
            "high=unambiguous, low=uncertain/borderline, "
            "cannot_tell=not classifiable from this image"
        ),
    )

    @model_validator(mode="after")
    def enforce_consistency(self) -> "HymeniumTypeResult":
        if not self.visible:
            self.hymenium_type = None
            self.visual_description = None
            self.reasoning = None
            self.confidence = "cannot_tell"
        if self.confidence == "cannot_tell":
            self.hymenium_type = None
        if self.hymenium_type is not None:
            self.visible = True
        return self


class CapColorResult(BaseModel):
    """Per-image cap color extraction with describe-then-classify ordering."""

    visible: bool = Field(..., description="Is the cap (pileus) surface visible in this image?")
    visual_description: str | None = Field(
        None,
        description=(
            "Describe the dominant color(s) you see on the cap surface, "
            "noting any gradients, zones, or lighting effects. 1-2 sentences."
        ),
    )
    reasoning: str | None = Field(
        None,
        description="Which canonical color is the closest match, and why?",
    )
    cap_color: CapColor | None = Field(
        None,
        description="Dominant cap color from the canonical palette, or null.",
    )
    confidence: Confidence = Field(
        ...,
        description=(
            "high=unambiguous, low=uncertain/borderline, "
            "cannot_tell=not classifiable from this image"
        ),
    )

    @model_validator(mode="after")
    def enforce_consistency(self) -> "CapColorResult":
        if not self.visible:
            self.cap_color = None
            self.visual_description = None
            self.reasoning = None
            self.confidence = "cannot_tell"
        if self.confidence == "cannot_tell":
            self.cap_color = None
        if self.cap_color is not None:
            self.visible = True
        return self


class RingPresenceResult(BaseModel):
    """Per-image ring (annulus) presence with describe-then-classify ordering."""

    visible: bool = Field(
        ..., description="Is the upper portion of the stem visible in this image?"
    )
    visual_description: str | None = Field(
        None,
        description=(
            "Describe the upper stem region: any membranous skirt, collar, "
            "fibrous zone, or cobweb-like remnants you can see. 1-2 sentences."
        ),
    )
    reasoning: str | None = Field(
        None,
        description=(
            "Does what you see qualify as a ring (annulus, ring-zone, or cortina), "
            "or is the upper stem clearly bare?"
        ),
    )
    ring_presence: RingPresence | None = Field(
        None,
        description="present | absent | null if cannot tell.",
    )
    confidence: Confidence = Field(
        ...,
        description=(
            "high=unambiguous, low=uncertain/borderline, "
            "cannot_tell=upper stem not visible enough to judge"
        ),
    )

    @model_validator(mode="after")
    def enforce_consistency(self) -> "RingPresenceResult":
        if not self.visible:
            self.ring_presence = None
            self.visual_description = None
            self.reasoning = None
            self.confidence = "cannot_tell"
        if self.confidence == "cannot_tell":
            self.ring_presence = None
        if self.ring_presence is not None:
            self.visible = True
        return self


class VolvaPresenceResult(BaseModel):
    """Per-image volva (basal sac) presence with describe-then-classify ordering."""

    visible: bool = Field(
        ..., description="Is the base of the stem visible in this image?"
    )
    visual_description: str | None = Field(
        None,
        description=(
            "Describe the stem base region: any sac-like cup, free margin, "
            "concentric bands or scales, adhering veil patches. 1-2 sentences."
        ),
    )
    reasoning: str | None = Field(
        None,
        description=(
            "Does what you see qualify as a volva (saccate cup or universal-veil "
            "remnants), or is the base just a plain stem or simple swollen bulb?"
        ),
    )
    volva_presence: VolvaPresence | None = Field(
        None,
        description="present | absent | null if cannot tell.",
    )
    confidence: Confidence = Field(
        ...,
        description=(
            "high=unambiguous, low=uncertain/borderline, "
            "cannot_tell=stem base not visible enough to judge"
        ),
    )

    @model_validator(mode="after")
    def enforce_consistency(self) -> "VolvaPresenceResult":
        if not self.visible:
            self.volva_presence = None
            self.visual_description = None
            self.reasoning = None
            self.confidence = "cannot_tell"
        if self.confidence == "cannot_tell":
            self.volva_presence = None
        if self.volva_presence is not None:
            self.visible = True
        return self


class SubstrateResult(BaseModel):
    """Per-image substrate extraction with describe-then-classify ordering."""

    visible: bool = Field(
        ...,
        description=(
            "Is enough of the surroundings/attachment point visible to judge the "
            "substrate the mushroom is growing from?"
        ),
    )
    visual_description: str | None = Field(
        None,
        description=(
            "Describe what the mushroom is growing on or out of: bare soil, leaf "
            "litter, fallen log, living trunk, twigs/bark chips, dung. Note any "
            "visible wood structure under leaves. 1-2 sentences."
        ),
    )
    reasoning: str | None = Field(
        None,
        description="Which substrate class does the visible evidence match, and why?",
    )
    substrate: Substrate | None = Field(
        None,
        description="Classified substrate from the canonical list, or null if cannot tell.",
    )
    confidence: Confidence = Field(
        ...,
        description=(
            "high=unambiguous, low=uncertain/borderline, "
            "cannot_tell=substrate not visible enough to classify"
        ),
    )

    @model_validator(mode="after")
    def enforce_consistency(self) -> "SubstrateResult":
        if not self.visible:
            self.substrate = None
            self.visual_description = None
            self.reasoning = None
            self.confidence = "cannot_tell"
        if self.confidence == "cannot_tell":
            self.substrate = None
        if self.substrate is not None:
            self.visible = True
        return self


class SurfaceTextureResult(BaseModel):
    """Per-image cap surface texture extraction with describe-then-classify ordering."""

    visible: bool = Field(
        ...,
        description="Is the cap (pileus) surface visible at sufficient detail to judge texture?",
    )
    visual_description: str | None = Field(
        None,
        description=(
            "Describe the cap surface in detail: smooth/glossy, fibers, scales, "
            "dust-like coating, wrinkles, pits, or other surface structures. "
            "Note whether the cap looks wet or dry. 1-2 sentences."
        ),
    )
    reasoning: str | None = Field(
        None,
        description="Which canonical texture does the description match, and why?",
    )
    surface_texture: SurfaceTexture | None = Field(
        None,
        description="Classified cap surface texture, or null if cannot tell.",
    )
    confidence: Confidence = Field(
        ...,
        description=(
            "high=unambiguous, low=uncertain/borderline, "
            "cannot_tell=cap surface not visible enough or wet/reflective hides texture"
        ),
    )

    @model_validator(mode="after")
    def enforce_consistency(self) -> "SurfaceTextureResult":
        if not self.visible:
            self.surface_texture = None
            self.visual_description = None
            self.reasoning = None
            self.confidence = "cannot_tell"
        if self.confidence == "cannot_tell":
            self.surface_texture = None
        if self.surface_texture is not None:
            self.visible = True
        return self


class StemShapeResult(BaseModel):
    """Per-image stem (stipe) shape extraction with describe-then-classify ordering."""

    visible: bool = Field(
        ...,
        description=(
            "Is enough of the stem visible — including the base — to judge its shape "
            "in profile?"
        ),
    )
    visual_description: str | None = Field(
        None,
        description=(
            "Describe the stem from apex to base: uniform width, swelling at the "
            "base, tapering toward the base, swelling in the middle, root-like "
            "extension into the substrate, lateral flattening. 1-2 sentences."
        ),
    )
    reasoning: str | None = Field(
        None,
        description="Which canonical stem shape does this match, and why?",
    )
    stem_shape: StemShape | None = Field(
        None,
        description="Classified stem shape, or null if cannot tell.",
    )
    confidence: Confidence = Field(
        ...,
        description=(
            "high=unambiguous, low=uncertain/borderline, "
            "cannot_tell=stem (especially the base) not visible enough to judge"
        ),
    )

    @model_validator(mode="after")
    def enforce_consistency(self) -> "StemShapeResult":
        if not self.visible:
            self.stem_shape = None
            self.visual_description = None
            self.reasoning = None
            self.confidence = "cannot_tell"
        if self.confidence == "cannot_tell":
            self.stem_shape = None
        if self.stem_shape is not None:
            self.visible = True
        return self


class CapShapeResult(BaseModel):
    """Per-image cap (pileus) shape extraction with describe-then-classify ordering."""

    visible: bool = Field(
        ...,
        description=(
            "Is the cap visible from a side or 3/4 angle so its profile shape "
            "can be judged?"
        ),
    )
    visual_description: str | None = Field(
        None,
        description=(
            "Describe the cap profile: rounded dome, flat plate, cone, bell, "
            "central bump, central depression, funnel, ball-shaped, egg-shaped, "
            "irregular. 1-2 sentences."
        ),
    )
    reasoning: str | None = Field(
        None,
        description="Which canonical cap shape does the description match, and why?",
    )
    cap_shape: CapShape | None = Field(
        None,
        description="Classified cap shape, or null if cannot tell.",
    )
    confidence: Confidence = Field(
        ...,
        description=(
            "high=unambiguous, low=uncertain/borderline, "
            "cannot_tell=cap profile not visible enough (e.g., top-down only)"
        ),
    )

    @model_validator(mode="after")
    def enforce_consistency(self) -> "CapShapeResult":
        if not self.visible:
            self.cap_shape = None
            self.visual_description = None
            self.reasoning = None
            self.confidence = "cannot_tell"
        if self.confidence == "cannot_tell":
            self.cap_shape = None
        if self.cap_shape is not None:
            self.visible = True
        return self


# ---------------------------------------------------------------------------
# Feature registry — maps feature name → (schema class, classification field)
# Adding a new feature: add schema above, register here.
# ---------------------------------------------------------------------------

FEATURE_REGISTRY: dict[str, dict] = {
    "hymenium_type": {
        "schema": HymeniumTypeResult,
        "field": "hymenium_type",
    },
    "cap_color": {
        "schema": CapColorResult,
        "field": "cap_color",
    },
    "ring_presence": {
        "schema": RingPresenceResult,
        "field": "ring_presence",
    },
    "volva_presence": {
        "schema": VolvaPresenceResult,
        "field": "volva_presence",
    },
    "substrate": {
        "schema": SubstrateResult,
        "field": "substrate",
    },
    "surface_texture": {
        "schema": SurfaceTextureResult,
        "field": "surface_texture",
    },
    "stem_shape": {
        "schema": StemShapeResult,
        "field": "stem_shape",
    },
    "cap_shape": {
        "schema": CapShapeResult,
        "field": "cap_shape",
    },
}
