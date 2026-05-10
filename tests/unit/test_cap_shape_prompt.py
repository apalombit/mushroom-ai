"""Pin the contract of the cap_shape prompt.

cap_shape is heavily convex-skewed (~67%). The prompt must include rare-first
ordering, MCQA tail, distinction blocks for the dominant confusions
(convex/flat, convex/umbonate, depressed/infundibuliform, conical/campanulate,
conical/parabolic, globose/ovoid), and a top-down/maturity caveat.
"""

import re

from vision.labeling.vlm_feature_prompts import (
    CAP_SHAPE_SYSTEM,
    CAP_SHAPE_USER,
    PROMPT_REGISTRY,
    _CONFIDENCE_BLOCK,
)


class TestCapShapePrompt:
    def test_registered(self):
        assert "cap_shape" in PROMPT_REGISTRY
        sys_p, user_p = PROMPT_REGISTRY["cap_shape"]
        assert sys_p is CAP_SHAPE_SYSTEM
        assert user_p is CAP_SHAPE_USER

    def test_system_includes_confidence_block(self):
        assert _CONFIDENCE_BLOCK in CAP_SHAPE_SYSTEM

    def test_user_has_mcqa_tail(self):
        assert "Final answer rule" in CAP_SHAPE_USER
        assert "multiple-choice" in CAP_SHAPE_USER.lower()
        assert (
            "ovoid | globose | irregular | infundibuliform | depressed | "
            "umbonate | parabolic | campanulate | conical | flat | convex | null"
            in CAP_SHAPE_USER
        )

    def test_user_rare_first_class_order(self):
        order = [
            "- ovoid:",
            "- globose:",
            "- irregular:",
            "- infundibuliform:",
            "- depressed:",
            "- umbonate:",
            "- parabolic:",
            "- campanulate:",
            "- conical:",
            "- flat:",
            "- convex:",
        ]
        positions = [CAP_SHAPE_USER.find(c) for c in order]
        assert all(p > 0 for p in positions), f"missing class lines: {positions}"
        assert positions == sorted(positions), "classes not in rare-first order"

    def test_user_includes_distinction_blocks(self):
        for cue in [
            "CONVEX vs FLAT",
            "CONVEX vs UMBONATE",
            "CONVEX vs PARABOLIC",
            "DEPRESSED vs INFUNDIBULIFORM",
            "CONICAL vs CAMPANULATE",
            "CONICAL vs PARABOLIC",
            "GLOBOSE vs OVOID",
        ]:
            assert cue in CAP_SHAPE_USER, f"missing distinction block: {cue!r}"

    def test_user_has_top_down_caveat(self):
        # Top-down photos cannot judge profile — must trigger cannot_tell, not a convex guess
        assert "top-down" in CAP_SHAPE_USER.lower() or "Top-down" in CAP_SHAPE_USER
        assert "side or 3/4" in CAP_SHAPE_USER

    def test_user_has_maturity_caveat(self):
        # Same species can change shape with maturity; classify as photographed
        assert "maturity" in CAP_SHAPE_USER.lower() or "Maturity" in CAP_SHAPE_USER
        assert "PHOTOGRAPHED" in CAP_SHAPE_USER or "as photographed" in CAP_SHAPE_USER.lower()

    def test_user_handles_invisibility(self):
        assert "cannot_tell" in CAP_SHAPE_USER
        assert "cap_shape=null" in CAP_SHAPE_USER or "confidence=cannot_tell" in CAP_SHAPE_USER

    def test_user_warns_against_convex_default(self):
        assert "most common" in CAP_SHAPE_USER.lower()


class TestPromptHygiene:
    def test_substantive(self):
        assert len(CAP_SHAPE_USER) > 1500
        assert CAP_SHAPE_USER.count("\n\n") >= 4

    def test_no_stray_format_tokens(self):
        assert not re.search(r"\{\w+\}", CAP_SHAPE_USER)
