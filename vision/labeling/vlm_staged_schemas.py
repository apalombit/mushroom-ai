"""Two-stage chain-of-inquiry schemas + registry (Path C).

Stage 1 — coarse family classification (e.g. linear_radial / punctate / featureless).
Stage 2 — within-family disambiguation, dispatched on stage 1's family value.

Final result is the same Pydantic schema as single-shot extraction, so existing
audit + analysis code works unchanged. Stage 1 output is stashed in the sidecar
JSON under a top-level ``_stage1`` key (added by the batch runner, not the schema).

Adding a new feature is a config addition only — see STAGED_REGISTRY at the bottom.
"""

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from vision.labeling.vlm_feature_schemas import (
    Confidence,
    HymeniumTypeResult,
)

# ---------------------------------------------------------------------------
# Hymenium — stage 1: coarse family
# ---------------------------------------------------------------------------

HymeniumFamily = Literal["linear_radial", "punctate", "featureless_or_internal"]

# Members of each family — used by the analysis script to compute stage-1
# routing accuracy (rows = GT class, cols = stage-1 family).
HYMENIUM_FAMILY_MEMBERS: dict[str, set[str]] = {
    "linear_radial": {"gills", "ridges"},
    "punctate": {"pores", "teeth", "alveolate"},
    "featureless_or_internal": {"smooth", "gleba"},
}


class HymeniumFamilyResult(BaseModel):
    """Stage 1 — coarse hymenium family classification."""

    visible: bool = Field(..., description="Is any spore-bearing surface visible in this image?")
    visual_description: str | None = Field(
        None,
        description=(
            "Describe what you see on the underside/fertile surface — "
            "is the texture line-like, dot-like, or absent? 1-2 sentences."
        ),
    )
    reasoning: str | None = Field(
        None,
        description="Which family does this description match, and why?",
    )
    family: HymeniumFamily | None = Field(
        None,
        description="Coarse family of the fertile surface, or null if cannot tell.",
    )
    confidence: Confidence = Field(
        ...,
        description=(
            "high=unambiguous, low=uncertain/borderline, "
            "cannot_tell=not classifiable from this image"
        ),
    )

    @model_validator(mode="after")
    def enforce_consistency(self) -> "HymeniumFamilyResult":
        if not self.visible:
            self.family = None
            self.visual_description = None
            self.reasoning = None
            self.confidence = "cannot_tell"
        if self.confidence == "cannot_tell":
            self.family = None
        if self.family is not None:
            self.visible = True
        return self


# ---------------------------------------------------------------------------
# Hymenium — stage 2: narrowed within-family schemas
# ---------------------------------------------------------------------------

HymeniumLinear = Literal["gills", "ridges"]
HymeniumPunctate = Literal["pores", "teeth", "alveolate"]
HymeniumFeatureless = Literal["smooth", "gleba"]


class _StagedLeafBase(BaseModel):
    """Shared base for stage-2 leaf results (visible/desc/reasoning/confidence)."""

    visible: bool = Field(default=True, description="Was the relevant surface visible enough to score?")
    visual_description: str | None = Field(
        None,
        description="Brief description of the cues you used to decide. 1-2 sentences.",
    )
    reasoning: str | None = Field(
        None,
        description="Which class does this description match, and why?",
    )
    confidence: Confidence = Field(..., description="high / low / cannot_tell")


class HymeniumLinearResult(_StagedLeafBase):
    """Stage 2 — gills vs ridges within the linear/radial family."""

    hymenium_type: HymeniumLinear | None = Field(
        None, description="gills | ridges | null"
    )

    @model_validator(mode="after")
    def enforce_consistency(self) -> "HymeniumLinearResult":
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


class HymeniumPunctateResult(_StagedLeafBase):
    """Stage 2 — pores vs teeth vs alveolate within the punctate family."""

    hymenium_type: HymeniumPunctate | None = Field(
        None, description="pores | teeth | alveolate | null"
    )

    @model_validator(mode="after")
    def enforce_consistency(self) -> "HymeniumPunctateResult":
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


class HymeniumFeaturelessResult(_StagedLeafBase):
    """Stage 2 — smooth (external featureless) vs gleba (internal mass)."""

    hymenium_type: HymeniumFeatureless | None = Field(
        None, description="smooth | gleba | null"
    )

    @model_validator(mode="after")
    def enforce_consistency(self) -> "HymeniumFeaturelessResult":
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


# ---------------------------------------------------------------------------
# Generic staged-config types (reusable across features)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FamilyConfig:
    """Stage 2 config for a single family branch."""

    schema: type[BaseModel]
    system: str
    user: str


@dataclass(frozen=True)
class StagedConfig:
    """Full two-stage config for a feature."""

    stage1_schema: type[BaseModel]
    stage1_system: str
    stage1_user: str
    families: dict[str, FamilyConfig]
    final_schema: type[BaseModel]
    family_field: str  # name of the family field on stage1 schema
    leaf_field: str  # name of the classification field on stage2 + final schemas


# ---------------------------------------------------------------------------
# Hymenium staged config — wired in vlm_staged_prompts at import time below
# ---------------------------------------------------------------------------

# Late import to avoid circular dependency (prompts file imports nothing from
# this module, but keeping registry assembly here keeps everything in one place).
from vision.labeling.vlm_staged_prompts import (  # noqa: E402
    HYMENIUM_STAGE1_SYSTEM,
    HYMENIUM_STAGE1_USER,
    HYMENIUM_STAGE2_FEATURELESS_SYSTEM,
    HYMENIUM_STAGE2_FEATURELESS_USER,
    HYMENIUM_STAGE2_LINEAR_SYSTEM,
    HYMENIUM_STAGE2_LINEAR_USER,
    HYMENIUM_STAGE2_PUNCTATE_SYSTEM,
    HYMENIUM_STAGE2_PUNCTATE_USER,
)

HYMENIUM_STAGED_CONFIG = StagedConfig(
    stage1_schema=HymeniumFamilyResult,
    stage1_system=HYMENIUM_STAGE1_SYSTEM,
    stage1_user=HYMENIUM_STAGE1_USER,
    families={
        "linear_radial": FamilyConfig(
            schema=HymeniumLinearResult,
            system=HYMENIUM_STAGE2_LINEAR_SYSTEM,
            user=HYMENIUM_STAGE2_LINEAR_USER,
        ),
        "punctate": FamilyConfig(
            schema=HymeniumPunctateResult,
            system=HYMENIUM_STAGE2_PUNCTATE_SYSTEM,
            user=HYMENIUM_STAGE2_PUNCTATE_USER,
        ),
        "featureless_or_internal": FamilyConfig(
            schema=HymeniumFeaturelessResult,
            system=HYMENIUM_STAGE2_FEATURELESS_SYSTEM,
            user=HYMENIUM_STAGE2_FEATURELESS_USER,
        ),
    },
    final_schema=HymeniumTypeResult,
    family_field="family",
    leaf_field="hymenium_type",
)


STAGED_REGISTRY: dict[str, StagedConfig] = {
    "hymenium_type": HYMENIUM_STAGED_CONFIG,
    # Future features (cap_color, gill_attachment, ring_presence, ...) plug in here.
}
