"""Tests for stem_shape Path C (staged) extraction.

Three-family design:
- uniform_equal: {equal}
- widening: {bulbous, clavate}
- other_distinctive: {attenuated, rooting, ventricose, obclavate, compressed}
"""

import re

import pytest

from vision.labeling.vlm_feature_schemas import StemShapeResult
from vision.labeling.vlm_staged_prompts import (
    STEM_SHAPE_STAGE1_SYSTEM,
    STEM_SHAPE_STAGE1_USER,
    STEM_SHAPE_STAGE2_OTHER_SYSTEM,
    STEM_SHAPE_STAGE2_OTHER_USER,
    STEM_SHAPE_STAGE2_UNIFORM_SYSTEM,
    STEM_SHAPE_STAGE2_UNIFORM_USER,
    STEM_SHAPE_STAGE2_WIDENING_SYSTEM,
    STEM_SHAPE_STAGE2_WIDENING_USER,
    _CONFIDENCE_BLOCK,
)
from vision.labeling.vlm_staged_schemas import (
    STAGED_REGISTRY,
    STEM_SHAPE_FAMILY_MEMBERS,
    STEM_SHAPE_STAGED_CONFIG,
    StemShapeFamilyResult,
    StemShapeOtherResult,
    StemShapeUniformResult,
    StemShapeWideningResult,
)


# ---------------------------------------------------------------------------
# Family-members map sanity
# ---------------------------------------------------------------------------


class TestStemShapeFamilyMembersMap:
    def test_three_families(self):
        assert set(STEM_SHAPE_FAMILY_MEMBERS) == {
            "uniform_equal", "widening", "other_distinctive",
        }

    def test_uniform_has_only_equal(self):
        assert STEM_SHAPE_FAMILY_MEMBERS["uniform_equal"] == {"equal"}

    def test_widening_has_bulbous_and_clavate(self):
        assert STEM_SHAPE_FAMILY_MEMBERS["widening"] == {"bulbous", "clavate"}

    def test_other_has_five_classes(self):
        assert STEM_SHAPE_FAMILY_MEMBERS["other_distinctive"] == {
            "attenuated", "rooting", "ventricose", "obclavate", "compressed",
        }

    def test_partition_covers_all_canonical_classes(self):
        """Every canonical stem_shape class must belong to exactly one family."""
        canonical = {
            "equal", "clavate", "obclavate", "bulbous", "attenuated",
            "ventricose", "compressed", "rooting",
        }
        union: set[str] = set().union(*STEM_SHAPE_FAMILY_MEMBERS.values())
        assert union == canonical
        # Disjointness
        seen: set[str] = set()
        for fam in STEM_SHAPE_FAMILY_MEMBERS.values():
            assert seen.isdisjoint(fam), "families overlap"
            seen |= fam


# ---------------------------------------------------------------------------
# Stage 1 schema guardrails
# ---------------------------------------------------------------------------


class TestStemShapeFamilyResult:
    def test_not_visible_clears(self):
        r = StemShapeFamilyResult(
            visible=False,
            visual_description="bulb",
            reasoning="basal swelling",
            family="widening",
            confidence="high",
        )
        assert r.family is None
        assert r.confidence == "cannot_tell"

    def test_cannot_tell_nulls_family(self):
        r = StemShapeFamilyResult(
            visible=True,
            family="uniform_equal",
            confidence="cannot_tell",
        )
        assert r.family is None

    def test_valid_widening(self):
        r = StemShapeFamilyResult(
            visible=True,
            visual_description="base wider than apex",
            reasoning="widening",
            family="widening",
            confidence="high",
        )
        assert r.family == "widening"

    def test_all_three_families_accepted(self):
        for fam in ["uniform_equal", "widening", "other_distinctive"]:
            r = StemShapeFamilyResult(
                visible=True, family=fam, confidence="high"
            )
            assert r.family == fam

    def test_invalid_family_rejected(self):
        with pytest.raises(Exception):
            StemShapeFamilyResult(
                visible=True, family="bulbous", confidence="high"
            )


# ---------------------------------------------------------------------------
# Stage 2 schema guardrails
# ---------------------------------------------------------------------------


class TestStemShapeUniformResult:
    def test_only_equal_accepted(self):
        r = StemShapeUniformResult(
            visible=True, stem_shape="equal", confidence="high"
        )
        assert r.stem_shape == "equal"

    def test_other_class_rejected(self):
        with pytest.raises(Exception):
            StemShapeUniformResult(
                visible=True, stem_shape="bulbous", confidence="high"
            )

    def test_not_visible_clears(self):
        r = StemShapeUniformResult(
            visible=False, stem_shape="equal", confidence="high"
        )
        assert r.stem_shape is None


class TestStemShapeWideningResult:
    def test_bulbous_accepted(self):
        r = StemShapeWideningResult(
            visible=True, stem_shape="bulbous", confidence="high"
        )
        assert r.stem_shape == "bulbous"

    def test_clavate_accepted(self):
        r = StemShapeWideningResult(
            visible=True, stem_shape="clavate", confidence="high"
        )
        assert r.stem_shape == "clavate"

    def test_equal_rejected(self):
        with pytest.raises(Exception):
            StemShapeWideningResult(
                visible=True, stem_shape="equal", confidence="high"
            )

    def test_attenuated_rejected(self):
        with pytest.raises(Exception):
            StemShapeWideningResult(
                visible=True, stem_shape="attenuated", confidence="high"
            )


class TestStemShapeOtherResult:
    @pytest.mark.parametrize(
        "cls",
        ["attenuated", "rooting", "ventricose", "obclavate", "compressed"],
    )
    def test_other_classes_accepted(self, cls):
        r = StemShapeOtherResult(
            visible=True, stem_shape=cls, confidence="high"
        )
        assert r.stem_shape == cls

    @pytest.mark.parametrize("cls", ["equal", "bulbous", "clavate"])
    def test_widening_or_uniform_rejected(self, cls):
        with pytest.raises(Exception):
            StemShapeOtherResult(
                visible=True, stem_shape=cls, confidence="high"
            )


# ---------------------------------------------------------------------------
# Staged config registration
# ---------------------------------------------------------------------------


class TestStemShapeStagedConfig:
    def test_registered_in_staged_registry(self):
        assert "stem_shape" in STAGED_REGISTRY
        assert STAGED_REGISTRY["stem_shape"] is STEM_SHAPE_STAGED_CONFIG

    def test_config_fields(self):
        cfg = STEM_SHAPE_STAGED_CONFIG
        assert cfg.stage1_schema is StemShapeFamilyResult
        assert cfg.final_schema is StemShapeResult
        assert cfg.family_field == "family"
        assert cfg.leaf_field == "stem_shape"
        assert set(cfg.families) == {"uniform_equal", "widening", "other_distinctive"}

    def test_family_schemas_match_members(self):
        cfg = STEM_SHAPE_STAGED_CONFIG
        assert cfg.families["uniform_equal"].schema is StemShapeUniformResult
        assert cfg.families["widening"].schema is StemShapeWideningResult
        assert cfg.families["other_distinctive"].schema is StemShapeOtherResult


# ---------------------------------------------------------------------------
# Prompt contracts
# ---------------------------------------------------------------------------


class TestStemShapeStage1Prompt:
    def test_system_includes_confidence_block(self):
        assert _CONFIDENCE_BLOCK in STEM_SHAPE_STAGE1_SYSTEM

    def test_user_lists_three_families(self):
        for fam in ["uniform_equal", "widening", "other_distinctive"]:
            assert f"- {fam}:" in STEM_SHAPE_STAGE1_USER

    def test_user_has_family_alternation(self):
        assert "uniform_equal | widening | other_distinctive | null" in (
            STEM_SHAPE_STAGE1_USER
        )

    def test_user_handles_invisibility(self):
        assert "cannot_tell" in STEM_SHAPE_STAGE1_USER
        assert "family=null" in STEM_SHAPE_STAGE1_USER

    def test_user_substantive(self):
        assert len(STEM_SHAPE_STAGE1_USER) > 800

    def test_no_stray_format_tokens(self):
        assert not re.search(r"\{\w+\}", STEM_SHAPE_STAGE1_USER)


class TestStemShapeStage2UniformPrompt:
    def test_system_includes_confidence_block(self):
        assert _CONFIDENCE_BLOCK in STEM_SHAPE_STAGE2_UNIFORM_SYSTEM

    def test_user_has_equal_only_alternation(self):
        assert "equal | null" in STEM_SHAPE_STAGE2_UNIFORM_USER

    def test_user_describes_equal(self):
        assert "- equal:" in STEM_SHAPE_STAGE2_UNIFORM_USER

    def test_user_allows_abstention(self):
        assert "cannot_tell" in STEM_SHAPE_STAGE2_UNIFORM_USER
        assert "stem_shape=null" in STEM_SHAPE_STAGE2_UNIFORM_USER


class TestStemShapeStage2WideningPrompt:
    def test_system_includes_confidence_block(self):
        assert _CONFIDENCE_BLOCK in STEM_SHAPE_STAGE2_WIDENING_SYSTEM

    def test_user_has_two_class_alternation(self):
        assert "bulbous | clavate | null" in STEM_SHAPE_STAGE2_WIDENING_USER

    def test_user_describes_both_classes(self):
        assert "- bulbous:" in STEM_SHAPE_STAGE2_WIDENING_USER
        assert "- clavate:" in STEM_SHAPE_STAGE2_WIDENING_USER

    def test_user_has_abrupt_vs_smooth_distinction(self):
        # The whole point of this stage 2 is the abrupt-vs-smooth threshold
        assert "abrupt" in STEM_SHAPE_STAGE2_WIDENING_USER.lower()
        assert "gradual" in STEM_SHAPE_STAGE2_WIDENING_USER.lower() or (
            "smooth" in STEM_SHAPE_STAGE2_WIDENING_USER.lower()
        )


class TestStemShapeStage2OtherPrompt:
    def test_system_includes_confidence_block(self):
        assert _CONFIDENCE_BLOCK in STEM_SHAPE_STAGE2_OTHER_SYSTEM

    def test_user_has_five_class_alternation(self):
        assert (
            "attenuated | rooting | ventricose | obclavate | compressed | null"
            in STEM_SHAPE_STAGE2_OTHER_USER
        )

    def test_user_describes_all_five_classes(self):
        for cls in ["attenuated", "rooting", "ventricose", "obclavate", "compressed"]:
            assert f"- {cls}:" in STEM_SHAPE_STAGE2_OTHER_USER

    def test_user_has_attenuated_vs_rooting_distinction(self):
        # The most likely confusion within this family
        assert "ATTENUATED vs ROOTING" in STEM_SHAPE_STAGE2_OTHER_USER

    def test_user_rare_first_order(self):
        order = [
            "- compressed:",
            "- rooting:",
            "- ventricose:",
            "- obclavate:",
            "- attenuated:",
        ]
        positions = [STEM_SHAPE_STAGE2_OTHER_USER.find(c) for c in order]
        assert all(p > 0 for p in positions)
        assert positions == sorted(positions), "stage-2 other prompt not rare-first"
