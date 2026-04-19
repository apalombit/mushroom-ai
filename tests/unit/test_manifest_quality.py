"""Unit tests for VLM quality filtering in vision/data/manifest.py."""

from unittest.mock import MagicMock

from vision.data.manifest import _VISIBILITY_REQUIREMENTS, _build_filtered_query, build_manifest


def test_visibility_requirements_mapping():
    """Verify the expected features have visibility requirements."""
    assert _VISIBILITY_REQUIREMENTS["cap_shape"] == "cap_visible"
    assert _VISIBILITY_REQUIREMENTS["cap_color"] == "cap_visible"
    assert _VISIBILITY_REQUIREMENTS["surface_texture"] == "cap_visible"
    assert _VISIBILITY_REQUIREMENTS["hymenium_type"] == "hymenium_visible"
    assert _VISIBILITY_REQUIREMENTS["gill_attachment"] == "hymenium_visible"
    # overall_body_form should NOT be in the mapping
    assert "overall_body_form" not in _VISIBILITY_REQUIREMENTS


def test_build_filtered_query_includes_global_exclusions():
    """Filtered query includes LEFT JOINs for quality, subject_present, dominance."""
    query, params = _build_filtered_query("overall_body_form")

    sql = str(query)
    assert "vlm_q" in sql  # image_quality join
    assert "vlm_sp" in sql  # subject_present join
    assert "vlm_sd" in sql  # subject_dominance join
    assert "vlm_q.feature_value IS NULL" in sql  # graceful degradation
    assert "'unusable'" in sql
    assert "'absent'" in sql
    assert params["feature_name"] == "overall_body_form"


def test_build_filtered_query_no_visibility_for_body_form():
    """overall_body_form has no visibility requirement — no vlm_vis join."""
    query, params = _build_filtered_query("overall_body_form")

    sql = str(query)
    assert "vlm_vis" not in sql
    assert "visibility_feature" not in params


def test_build_filtered_query_adds_visibility_for_hymenium():
    """hymenium_type requires hymenium_visible — adds vlm_vis join."""
    query, params = _build_filtered_query("hymenium_type")

    sql = str(query)
    assert "vlm_vis" in sql
    assert params["visibility_feature"] == "hymenium_visible"
    assert "vlm_vis.feature_value = 'true'" in sql


def test_build_filtered_query_adds_visibility_for_cap_shape():
    """cap_shape requires cap_visible — adds vlm_vis join."""
    query, params = _build_filtered_query("cap_shape")

    sql = str(query)
    assert "vlm_vis" in sql
    assert params["visibility_feature"] == "cap_visible"


def test_build_manifest_quality_filter_false_uses_simple_query(tmp_path):
    """quality_filter=False uses the original simple JOIN (no LEFT JOINs)."""
    session = MagicMock()
    session.execute.return_value.fetchall.return_value = []

    build_manifest(session, "hymenium_type", tmp_path, quality_filter=False)

    # Only one execute call (the main query), no unfiltered count
    assert session.execute.call_count == 1
    sql = str(session.execute.call_args[0][0])
    assert "vlm_q" not in sql
    assert "vlm_vis" not in sql


def test_build_manifest_quality_filter_true_uses_filtered_query(tmp_path):
    """quality_filter=True uses filtered query with LEFT JOINs."""
    session = MagicMock()
    session.execute.return_value.fetchall.return_value = []

    build_manifest(session, "hymenium_type", tmp_path, quality_filter=True)

    # First call is the filtered query
    sql = str(session.execute.call_args_list[0][0][0])
    assert "vlm_q" in sql
    assert "vlm_vis" in sql


def test_build_manifest_default_quality_filter_is_true(tmp_path):
    """Default quality_filter should be True."""
    session = MagicMock()
    session.execute.return_value.fetchall.return_value = []

    build_manifest(session, "cap_shape", tmp_path)

    sql = str(session.execute.call_args_list[0][0][0])
    assert "vlm_q" in sql  # filtered query was used


def test_build_filtered_query_includes_manual_qa_join():
    """Filtered query includes LEFT JOIN for manual_qa annotations."""
    query, params = _build_filtered_query("hymenium_type")

    sql = str(query)
    assert "mqa" in sql
    assert "manual_qa" in sql
    assert "mqa.feature_value IS NULL OR mqa.feature_value != 'bad'" in sql


def test_build_filtered_query_manual_qa_for_all_features():
    """manual_qa join should be present for all features, not just visibility-gated ones."""
    query, params = _build_filtered_query("overall_body_form")

    sql = str(query)
    assert "mqa" in sql
    assert "manual_qa" in sql
