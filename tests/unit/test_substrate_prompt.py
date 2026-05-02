"""Pin the contract of the substrate prompt.

Substrate is the first non-morphological signal feature. The prompt encodes
mycology-specific anti-traps (buried wood looking soil-attached, moss-covered
logs still being dead wood, twigs vs logs as woody debris vs dead wood).
"""

import re

import pytest

from vision.labeling.vlm_feature_prompts import (
    PROMPT_REGISTRY,
    SUBSTRATE_SYSTEM,
    SUBSTRATE_USER,
    _CONFIDENCE_BLOCK,
)


class TestSubstratePrompt:
    def test_registered(self):
        assert "substrate" in PROMPT_REGISTRY
        sys_p, user_p = PROMPT_REGISTRY["substrate"]
        assert sys_p is SUBSTRATE_SYSTEM
        assert user_p is SUBSTRATE_USER

    def test_system_includes_confidence_block(self):
        assert _CONFIDENCE_BLOCK in SUBSTRATE_SYSTEM

    def test_user_has_mcqa_tail(self):
        assert "Final answer rule" in SUBSTRATE_USER
        assert "multiple-choice" in SUBSTRATE_USER.lower()
        assert "dung | woody debris | leaf litter | living tree | dead wood | soil | null" in (
            SUBSTRATE_USER
        )

    def test_user_rare_first_class_order(self):
        # rarest first → 'dung' before 'woody debris' before 'leaf litter' before
        # 'living tree' before 'dead wood' before 'soil'
        order = [
            "- dung:",
            "- woody debris:",
            "- leaf litter:",
            "- living tree:",
            "- dead wood:",
            "- soil:",
        ]
        positions = [SUBSTRATE_USER.find(c) for c in order]
        assert all(p > 0 for p in positions), f"missing class lines: {positions}"
        assert positions == sorted(positions), (
            f"classes not in rare-first order: {dict(zip(order, positions, strict=True))}"
        )

    def test_user_includes_anti_traps(self):
        # Each named anti-trap must be called out
        for cue in [
            "buried wood",  # buried wood often looks soil-attached
            "moss",  # moss-covered logs are still dead wood
            "size",  # woody debris vs dead wood is a size distinction
            "bulb",  # explicitly disambiguate? not for substrate — drop
        ][:-1]:
            assert cue.lower() in SUBSTRATE_USER.lower(), (
                f"missing anti-trap text: {cue!r}"
            )

    def test_user_handles_invisibility(self):
        assert "cannot_tell" in SUBSTRATE_USER
        assert "substrate=null" in SUBSTRATE_USER

    def test_user_warns_against_soil_default(self):
        # The most common class should not be the default — caution required
        assert "most common" in SUBSTRATE_USER.lower()


class TestPromptHygiene:
    def test_substantive(self):
        assert len(SUBSTRATE_USER) > 1000
        assert SUBSTRATE_USER.count("\n\n") >= 4

    def test_no_stray_format_tokens(self):
        assert not re.search(r"\{\w+\}", SUBSTRATE_USER)


@pytest.mark.parametrize(
    "cls",
    ["soil", "dead wood", "living tree", "leaf litter", "woody debris", "dung"],
)
def test_every_class_described(cls):
    """Each canonical class must have a short definition line in the prompt."""
    assert f"- {cls}:" in SUBSTRATE_USER
