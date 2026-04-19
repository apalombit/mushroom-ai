"""Unit tests for vision/labeling/vlm_db.py."""

from unittest.mock import MagicMock

from vision.labeling.vlm_db import (
    _CONFIDENCE_MAP,
    _QUALITY_SCORE_MAP,
    get_graded_image_ids,
    persist_vlm_annotation,
)
from vision.labeling.vlm_schema import VLMImageAnnotation


def _make_annotation(**overrides) -> VLMImageAnnotation:
    """Build a minimal valid VLMImageAnnotation with sensible defaults."""
    defaults = {
        "subject_present": True,
        "subject_dominance": "dominant",
        "image_quality": "good",
        "view_angle": "cap_top",
        "cap_visible": True,
        "cap_color_primary": "red",
        "cap_shape": None,
        "cap_surface_texture": None,
        "cap_color_secondary": None,
        "hymenium_visible": False,
        "hymenium_type": None,
        "gill_attachment": None,
        "body_form": None,
        "notes": None,
        "confidence": "high",
    }
    defaults.update(overrides)
    return VLMImageAnnotation(**defaults)


def test_persist_vlm_annotation_writes_six_annotation_rows():
    """persist_vlm_annotation inserts 6 rows into image_annotations."""
    session = MagicMock()
    ann = _make_annotation(image_quality="acceptable", confidence="medium")

    persist_vlm_annotation(session, "img_001", ann, model_name="test-model")

    # 6 annotation upserts + 1 quality upsert = 7 execute calls
    assert session.execute.call_count == 7

    # Verify each annotation field was written
    execute_calls = session.execute.call_args_list
    written_features = set()
    for c in execute_calls[:6]:
        params = c[1] if len(c[1]) > 0 else c[0][1]
        written_features.add(params["feature_name"])
        assert params["image_id"] == "img_001"
        assert params["annotator"] == "test-model"
        assert params["confidence"] == _CONFIDENCE_MAP["medium"]

    assert written_features == {
        "image_quality",
        "subject_present",
        "subject_dominance",
        "cap_visible",
        "hymenium_visible",
        "view_angle",
    }


def test_persist_vlm_annotation_quality_table_no_exclusion():
    """Good quality image should have no exclude_reason."""
    session = MagicMock()
    ann = _make_annotation(image_quality="good", subject_present=True)

    persist_vlm_annotation(session, "img_002", ann)

    # Last execute call is the quality upsert
    quality_call = session.execute.call_args_list[-1]
    params = quality_call[0][1]
    assert params["quality_score"] == _QUALITY_SCORE_MAP["good"]
    assert params["exclude_reason"] is None


def test_persist_vlm_annotation_unusable_sets_exclude():
    """Unusable quality should set exclude_reason."""
    session = MagicMock()
    ann = _make_annotation(image_quality="unusable")

    persist_vlm_annotation(session, "img_003", ann)

    quality_call = session.execute.call_args_list[-1]
    params = quality_call[0][1]
    assert params["quality_score"] == _QUALITY_SCORE_MAP["unusable"]
    assert params["exclude_reason"] == "unusable_quality"


def test_persist_vlm_annotation_no_subject_sets_exclude():
    """subject_present=False should set exclude_reason."""
    session = MagicMock()
    ann = _make_annotation(subject_present=False, subject_dominance="absent")

    persist_vlm_annotation(session, "img_004", ann)

    quality_call = session.execute.call_args_list[-1]
    params = quality_call[0][1]
    assert params["exclude_reason"] == "no_subject"


def test_persist_vlm_annotation_absent_dominance_sets_exclude():
    """subject_dominance='absent' should set exclude_reason."""
    session = MagicMock()
    ann = _make_annotation(subject_present=True, subject_dominance="absent", image_quality="good")

    persist_vlm_annotation(session, "img_005", ann)

    quality_call = session.execute.call_args_list[-1]
    params = quality_call[0][1]
    assert params["exclude_reason"] == "subject_absent"


def test_persist_vlm_annotation_boolean_fields_lowercase():
    """Boolean fields should be stored as 'true'/'false' strings."""
    session = MagicMock()
    ann = _make_annotation(cap_visible=True, hymenium_visible=False, subject_present=True)

    persist_vlm_annotation(session, "img_006", ann)

    calls = session.execute.call_args_list[:6]
    values = {c[0][1]["feature_name"]: c[0][1]["feature_value"] for c in calls}

    assert values["cap_visible"] == "true"
    assert values["hymenium_visible"] == "false"
    assert values["subject_present"] == "true"


def test_get_graded_image_ids():
    """get_graded_image_ids returns the correct set of ids."""
    session = MagicMock()
    session.execute.return_value.fetchall.return_value = [
        ("img_a",),
        ("img_b",),
        ("img_c",),
    ]

    result = get_graded_image_ids(session)

    assert result == {"img_a", "img_b", "img_c"}


def test_get_graded_image_ids_empty():
    """get_graded_image_ids returns empty set when no rows."""
    session = MagicMock()
    session.execute.return_value.fetchall.return_value = []

    result = get_graded_image_ids(session)

    assert result == set()
