"""Pin the contract of the surface_texture prompt.

surface_texture is heavily smooth-skewed (~63% of species). The prompt must
include rare-first ordering, MCQA tail, lighting/wet caveat, and the named
distinction blocks (smooth-vs-silky, silky-vs-fibrillose, fibrillose-vs-velvety,
scaly-vs-squarrose, warty-vs-scaly, wrinkled-vs-reticulate, pruinose-vs-smooth).
"""

import re

from vision.labeling.vlm_feature_prompts import (
    PROMPT_REGISTRY,
    SURFACE_TEXTURE_SYSTEM,
    SURFACE_TEXTURE_USER,
    _CONFIDENCE_BLOCK,
)


class TestSurfaceTexturePrompt:
    def test_registered(self):
        assert "surface_texture" in PROMPT_REGISTRY
        sys_p, user_p = PROMPT_REGISTRY["surface_texture"]
        assert sys_p is SURFACE_TEXTURE_SYSTEM
        assert user_p is SURFACE_TEXTURE_USER

    def test_system_includes_confidence_block(self):
        assert _CONFIDENCE_BLOCK in SURFACE_TEXTURE_SYSTEM

    def test_user_has_mcqa_tail(self):
        assert "Final answer rule" in SURFACE_TEXTURE_USER
        assert "multiple-choice" in SURFACE_TEXTURE_USER.lower()
        # The exhaustive null-terminated alternation
        assert "smooth | null" in SURFACE_TEXTURE_USER

    def test_user_rare_first_class_order(self):
        # rarest first: areolate → ... → smooth
        order = [
            "- areolate:",
            "- pitted:",
            "- reticulate:",
            "- pruinose:",
            "- warty:",
            "- squarrose:",
            "- floccose:",
            "- wrinkled:",
            "- tomentose:",
            "- silky:",
            "- fibrillose:",
            "- velvety:",
            "- scaly:",
            "- smooth:",
        ]
        positions = [SURFACE_TEXTURE_USER.find(c) for c in order]
        assert all(p > 0 for p in positions), f"missing class lines: {positions}"
        assert positions == sorted(positions), (
            "classes not in rare-first order"
        )

    def test_user_includes_distinction_blocks(self):
        # The named distinction blocks that disambiguate the easily-confused pairs
        for cue in [
            "SMOOTH vs SILKY",
            "SILKY vs FIBRILLOSE",
            "FIBRILLOSE vs VELVETY",
            "SCALY vs FIBRILLOSE",
            "SCALY vs SQUARROSE",
            "WARTY vs SCALY",
            "WRINKLED vs RETICULATE",
            "PRUINOSE vs SMOOTH",
        ]:
            assert cue in SURFACE_TEXTURE_USER, f"missing distinction block: {cue!r}"

    def test_user_has_wet_cap_caveat(self):
        # Wet/glossy caps mask texture — model must abstain rather than guess smooth
        assert "wet" in SURFACE_TEXTURE_USER.lower()
        assert "reflective" in SURFACE_TEXTURE_USER.lower() or "glossy" in (
            SURFACE_TEXTURE_USER.lower()
        )

    def test_user_handles_invisibility(self):
        assert "cannot_tell" in SURFACE_TEXTURE_USER
        assert "surface_texture=null" in SURFACE_TEXTURE_USER

    def test_user_warns_against_smooth_default(self):
        assert "most common" in SURFACE_TEXTURE_USER.lower()


class TestPromptHygiene:
    def test_substantive(self):
        assert len(SURFACE_TEXTURE_USER) > 1500
        assert SURFACE_TEXTURE_USER.count("\n\n") >= 4

    def test_no_stray_format_tokens(self):
        assert not re.search(r"\{\w+\}", SURFACE_TEXTURE_USER)
