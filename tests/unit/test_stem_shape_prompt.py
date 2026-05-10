"""Pin the contract of the stem_shape prompt.

stem_shape is heavily equal-skewed (~83%). The prompt must include rare-first
ordering, MCQA tail, named distinction blocks (equal-vs-clavate, clavate-vs-bulbous,
attenuated-vs-rooting, attenuated-vs-equal, ventricose-vs-clavate-or-equal,
bulbous-without-volva), and a buried-base caveat.
"""

import re

from vision.labeling.vlm_feature_prompts import (
    PROMPT_REGISTRY,
    STEM_SHAPE_SYSTEM,
    STEM_SHAPE_USER,
    _CONFIDENCE_BLOCK,
)


class TestStemShapePrompt:
    def test_registered(self):
        assert "stem_shape" in PROMPT_REGISTRY
        sys_p, user_p = PROMPT_REGISTRY["stem_shape"]
        assert sys_p is STEM_SHAPE_SYSTEM
        assert user_p is STEM_SHAPE_USER

    def test_system_includes_confidence_block(self):
        assert _CONFIDENCE_BLOCK in STEM_SHAPE_SYSTEM

    def test_user_has_mcqa_tail(self):
        assert "Final answer rule" in STEM_SHAPE_USER
        assert "multiple-choice" in STEM_SHAPE_USER.lower()
        assert (
            "rooting | compressed | ventricose | obclavate | attenuated | clavate | bulbous | equal | null"
            in STEM_SHAPE_USER
        )

    def test_user_rare_first_class_order(self):
        order = [
            "- rooting:",
            "- compressed:",
            "- ventricose:",
            "- obclavate:",
            "- attenuated:",
            "- clavate:",
            "- bulbous:",
            "- equal:",
        ]
        positions = [STEM_SHAPE_USER.find(c) for c in order]
        assert all(p > 0 for p in positions), f"missing class lines: {positions}"
        assert positions == sorted(positions), "classes not in rare-first order"

    def test_user_includes_distinction_blocks(self):
        for cue in [
            "EQUAL vs CLAVATE",
            "CLAVATE vs BULBOUS",
            "ATTENUATED vs ROOTING",
            "ATTENUATED vs EQUAL",
            "VENTRICOSE vs CLAVATE",
            "BULBOUS without volva",
        ]:
            assert cue in STEM_SHAPE_USER, f"missing distinction block: {cue!r}"

    def test_user_has_buried_base_caveat(self):
        # Buried base must trigger cannot_tell, not an "equal" guess
        assert "buried" in STEM_SHAPE_USER.lower() or "Buried-base" in STEM_SHAPE_USER

    def test_user_handles_invisibility(self):
        assert "cannot_tell" in STEM_SHAPE_USER
        assert "stem_shape=null" in STEM_SHAPE_USER

    def test_user_warns_against_equal_default(self):
        assert "most common" in STEM_SHAPE_USER.lower()


class TestPromptHygiene:
    def test_substantive(self):
        assert len(STEM_SHAPE_USER) > 1500
        assert STEM_SHAPE_USER.count("\n\n") >= 4

    def test_no_stray_format_tokens(self):
        assert not re.search(r"\{\w+\}", STEM_SHAPE_USER)
