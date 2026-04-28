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
}
