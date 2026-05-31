"""Tests for ring_presence + volva_presence ground-truth derivation."""

import pytest

from vision.data.ground_truth import (
    GROUND_TRUTH_SPEC,
    _derive_ring_presence,
    _derive_volva_presence,
)


# ---------------------------------------------------------------------------
# _derive_ring_presence
# ---------------------------------------------------------------------------


class TestDeriveRingPresence:
    """Refined ring derivation: only count visually-reliable rings.

    Inputs are veil dicts (not bare type strings) since persistence matters.
    """

    @pytest.mark.parametrize(
        "veil,expected",
        [
            # type=partial without persistence info → present
            ({"type": "partial"}, "present"),
            ({"type": "partial", "ring_persistence": None}, "present"),
            ({"type": "partial", "ring_persistence": "persistent"}, "present"),
            # type=both treated like partial
            ({"type": "both", "ring_persistence": "persistent"}, "present"),
            ({"type": "both"}, "present"),
            # absent → absent
            ({"type": "absent"}, "absent"),
            ({"type": "absent", "ring_persistence": None}, "absent"),
            # Excluded: cortina (rarely visible)
            ({"type": "cortina"}, None),
            ({"type": "cortina", "ring_persistence": "persistent"}, None),
            # Excluded: universal-only (volva at base, not a ring)
            ({"type": "universal"}, None),
            # Excluded: fugacious rings (drop off early in field photos)
            ({"type": "partial", "ring_persistence": "fugacious"}, None),
            ({"type": "both", "ring_persistence": "fugacious"}, None),
            # Excluded: ring_zone persistence (only zone visible, no actual ring)
            # ring_zone is allowed since type is partial and persistence != fugacious
            ({"type": "partial", "ring_persistence": "ring_zone"}, "present"),
            # Case + whitespace
            ({"type": "PARTIAL"}, "present"),
            ({"type": "  partial  "}, "present"),
            ({"type": "PARTIAL", "ring_persistence": "FUGACIOUS"}, None),
            # Edge cases
            (None, None),
            ({}, None),
            ({"type": None}, None),
            ({"type": ""}, None),
            ({"type": "unknown"}, None),
            ("partial", None),  # bare string no longer accepted
        ],
    )
    def test_mapping(self, veil, expected):
        assert _derive_ring_presence(veil) == expected


# ---------------------------------------------------------------------------
# _derive_volva_presence
# ---------------------------------------------------------------------------


class TestDeriveVolvaPresence:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("saccate", "present"),
            ("flaring", "present"),
            ("circumsessile", "present"),
            ("zoned", "present"),
            ("friable", "present"),
            ("napiform", "present"),
            ("absent", "absent"),
            (None, None),
            ("", None),
            ("unknown_form", None),
            ("SACCATE", "present"),
            ("  saccate  ", "present"),
            (False, None),  # the volva.present bool field; not allowed here
        ],
    )
    def test_mapping(self, raw, expected):
        assert _derive_volva_presence(raw) == expected


# ---------------------------------------------------------------------------
# GROUND_TRUTH_SPEC integration
# ---------------------------------------------------------------------------


class TestGroundTruthSpecIntegration:
    def test_ring_presence_registered(self):
        spec = GROUND_TRUTH_SPEC["ring_presence"]
        assert spec["source"] == "features_json"
        assert spec["path"] == ["veil"]
        assert spec["derive"] is _derive_ring_presence

    def test_volva_presence_registered(self):
        spec = GROUND_TRUTH_SPEC["volva_presence"]
        assert spec["source"] == "features_json"
        assert spec["path"] == ["volva", "type"]
        assert spec["derive"] is _derive_volva_presence
