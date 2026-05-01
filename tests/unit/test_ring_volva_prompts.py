"""Pin the contract of the ring + volva prompts.

These prompts encode the validated Path A recipe and the mycology-specific
anti-traps.  Drift in the prompts is a real bug — these tests guard the
non-obvious phrasing choices.
"""

import re

import pytest

from vision.labeling.vlm_feature_prompts import (
    PROMPT_REGISTRY,
    RING_PRESENCE_SYSTEM,
    RING_PRESENCE_USER,
    VOLVA_PRESENCE_SYSTEM,
    VOLVA_PRESENCE_USER,
    _CONFIDENCE_BLOCK,
)


# ---------------------------------------------------------------------------
# Ring prompt
# ---------------------------------------------------------------------------


class TestRingPrompt:
    def test_registered(self):
        assert "ring_presence" in PROMPT_REGISTRY
        sys_p, user_p = PROMPT_REGISTRY["ring_presence"]
        assert sys_p is RING_PRESENCE_SYSTEM
        assert user_p is RING_PRESENCE_USER

    def test_system_includes_confidence_block(self):
        assert _CONFIDENCE_BLOCK in RING_PRESENCE_SYSTEM

    def test_user_has_mcqa_tail(self):
        # Final answer rule + explicit allowed values + null option
        assert "Final answer rule" in RING_PRESENCE_USER
        assert "multiple-choice" in RING_PRESENCE_USER.lower()
        assert "present | absent | null" in RING_PRESENCE_USER

    def test_user_rare_first_class_order(self):
        # 'present' must come before 'absent' in the option enumeration
        present_idx = RING_PRESENCE_USER.find("- present:")
        absent_idx = RING_PRESENCE_USER.find("- absent:")
        assert 0 < present_idx < absent_idx

    def test_user_includes_anti_traps(self):
        # Each named anti-trap must be called out
        for cue in [
            "ring zone",  # ring-zone counts as present
            "cortina",    # cortina counts as present
            "Scales",     # scales-along-stem are NOT a ring
            "swollen base",  # bulb is NOT a ring
        ]:
            assert cue.lower() in RING_PRESENCE_USER.lower(), (
                f"missing anti-trap text: {cue!r}"
            )

    def test_user_handles_invisibility(self):
        assert "cannot_tell" in RING_PRESENCE_USER
        assert "ring_presence=null" in RING_PRESENCE_USER


# ---------------------------------------------------------------------------
# Volva prompt
# ---------------------------------------------------------------------------


class TestVolvaPrompt:
    def test_registered(self):
        assert "volva_presence" in PROMPT_REGISTRY
        sys_p, user_p = PROMPT_REGISTRY["volva_presence"]
        assert sys_p is VOLVA_PRESENCE_SYSTEM
        assert user_p is VOLVA_PRESENCE_USER

    def test_system_includes_confidence_block(self):
        assert _CONFIDENCE_BLOCK in VOLVA_PRESENCE_SYSTEM

    def test_user_has_mcqa_tail(self):
        assert "Final answer rule" in VOLVA_PRESENCE_USER
        assert "multiple-choice" in VOLVA_PRESENCE_USER.lower()
        assert "present | absent | null" in VOLVA_PRESENCE_USER

    def test_user_rare_first_class_order(self):
        present_idx = VOLVA_PRESENCE_USER.find("- present:")
        absent_idx = VOLVA_PRESENCE_USER.find("- absent:")
        assert 0 < present_idx < absent_idx

    def test_user_includes_anti_traps(self):
        # Bulb-without-veil is NOT a volva — the most important volva trap
        for cue in [
            "bulb",
            "swollen bulb",
            "soil",  # debris at base is not a volva
            "ring",  # the upper-stem ring is a separate feature
            "earthstar",  # Geastrum split exoperidium is NOT a volva (added v2)
            "cup fungi",  # cup-without-stipe is not a volva (added v2)
        ]:
            assert cue.lower() in VOLVA_PRESENCE_USER.lower(), (
                f"missing anti-trap text: {cue!r}"
            )

    def test_user_mentions_universal_veil_corroboration(self):
        # Patches on cap from universal veil are corroborating evidence
        assert "universal" in VOLVA_PRESENCE_USER.lower()
        assert "patches" in VOLVA_PRESENCE_USER.lower()

    def test_user_handles_invisibility(self):
        assert "cannot_tell" in VOLVA_PRESENCE_USER
        assert "volva_presence=null" in VOLVA_PRESENCE_USER


# ---------------------------------------------------------------------------
# Cross-cutting
# ---------------------------------------------------------------------------


class TestPromptHygiene:
    @pytest.mark.parametrize(
        "user_prompt,feature_label",
        [
            (RING_PRESENCE_USER, "ring"),
            (VOLVA_PRESENCE_USER, "volva"),
        ],
    )
    def test_prompts_are_substantive(self, user_prompt, feature_label):
        # Sanity: not empty, has multiple paragraphs of guidance
        assert len(user_prompt) > 1000, f"{feature_label} prompt too short"
        assert user_prompt.count("\n\n") >= 4, (
            f"{feature_label} prompt should have multiple sections"
        )

    @pytest.mark.parametrize(
        "user_prompt",
        [RING_PRESENCE_USER, VOLVA_PRESENCE_USER],
    )
    def test_no_stray_format_tokens(self, user_prompt):
        # Defensive: no unexpanded {placeholders}
        assert not re.search(r"\{\w+\}", user_prompt)
