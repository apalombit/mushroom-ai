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
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("partial", "present"),
            ("cortina", "present"),
            ("universal", "present"),
            ("both", "present"),
            ("absent", "absent"),
            (None, None),
            ("", None),
            ("unknown", None),
            ("PARTIAL", "present"),  # case-insensitive
            ("  partial  ", "present"),  # stripped
            (123, None),  # non-string
        ],
    )
    def test_mapping(self, raw, expected):
        assert _derive_ring_presence(raw) == expected


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
        assert spec["path"] == ["veil", "type"]
        assert spec["derive"] is _derive_ring_presence

    def test_volva_presence_registered(self):
        spec = GROUND_TRUTH_SPEC["volva_presence"]
        assert spec["source"] == "features_json"
        assert spec["path"] == ["volva", "type"]
        assert spec["derive"] is _derive_volva_presence
