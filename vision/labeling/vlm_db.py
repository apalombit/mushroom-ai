"""Persist VLM annotations to the database.

Writes per-image VLM grading results (quality, visibility, view angle) into
``image_annotations`` (annotation_type='vlm_graded') and the ``image_quality``
table so that downstream manifest building can filter by quality/visibility.
"""

from sqlalchemy import text
from sqlalchemy.orm import Session

from vision.labeling.vlm_schema import VLMImageAnnotation

_CONFIDENCE_MAP = {"low": 0.33, "medium": 0.66, "high": 1.0}

_QUALITY_SCORE_MAP = {
    "good": 1.0,
    "acceptable": 0.75,
    "blurry": 0.5,
    "occluded": 0.25,
    "unusable": 0.0,
}

_UPSERT_ANNOTATION = text("""
    INSERT INTO image_annotations
        (image_id, feature_name, annotation_type, feature_value, confidence, annotator)
    VALUES (:image_id, :feature_name, 'vlm_graded', :feature_value, :confidence, :annotator)
    ON CONFLICT (image_id, feature_name, annotation_type)
    DO UPDATE SET feature_value = EXCLUDED.feature_value,
                  confidence    = EXCLUDED.confidence,
                  annotator     = EXCLUDED.annotator,
                  created_at    = NOW()
""")

_UPSERT_QUALITY = text("""
    INSERT INTO image_quality (image_id, quality_score, exclude_reason)
    VALUES (:image_id, :quality_score, :exclude_reason)
    ON CONFLICT (image_id)
    DO UPDATE SET quality_score  = EXCLUDED.quality_score,
                  exclude_reason = EXCLUDED.exclude_reason
""")


def persist_vlm_annotation(
    session: Session,
    image_id: str,
    annotation: VLMImageAnnotation,
    model_name: str = "gemma4:31b-cloud",
) -> None:
    """Write a VLMImageAnnotation to the DB (idempotent upsert).

    Inserts 6 rows into ``image_annotations`` (annotation_type='vlm_graded')
    and one row into ``image_quality``.
    """
    conf = _CONFIDENCE_MAP.get(annotation.confidence, 0.5)

    fields = {
        "image_quality": annotation.image_quality,
        "subject_present": str(annotation.subject_present).lower(),
        "subject_dominance": annotation.subject_dominance,
        "cap_visible": str(annotation.cap_visible).lower(),
        "hymenium_visible": str(annotation.hymenium_visible).lower(),
        "view_angle": annotation.view_angle,
    }

    for feat_name, feat_value in fields.items():
        session.execute(
            _UPSERT_ANNOTATION,
            {
                "image_id": image_id,
                "feature_name": feat_name,
                "feature_value": feat_value,
                "confidence": conf,
                "annotator": model_name,
            },
        )

    # Determine exclusion reason (if any)
    exclude_reason = None
    if annotation.image_quality == "unusable":
        exclude_reason = "unusable_quality"
    elif not annotation.subject_present:
        exclude_reason = "no_subject"
    elif annotation.subject_dominance == "absent":
        exclude_reason = "subject_absent"

    session.execute(
        _UPSERT_QUALITY,
        {
            "image_id": image_id,
            "quality_score": _QUALITY_SCORE_MAP.get(annotation.image_quality, 0.5),
            "exclude_reason": exclude_reason,
        },
    )


def get_graded_image_ids(session: Session) -> set[str]:
    """Return the set of image_ids that already have vlm_graded annotations."""
    rows = session.execute(
        text(
            "SELECT DISTINCT image_id FROM image_annotations WHERE annotation_type = 'vlm_graded'"
        )
    ).fetchall()
    return {r[0] for r in rows}
