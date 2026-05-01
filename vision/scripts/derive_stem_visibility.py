"""Backfill stem_upper_visible / stem_base_visible from existing view_angle.

Most images already have a `view_angle` vlm_graded annotation (cap_top,
side_profile, etc.).  We synthesize two boolean visibility tags from it so
ring/volva eval can be gated cheaply, without a second VLM grading pass.

Strict mapping — only true when the stem region is definitely visible:

    view_angle             stem_upper_visible    stem_base_visible
    ───────────────────────────────────────────────────────────────
    side_profile                 true                  true
    full_body                    true                  true
    cross_section                true                  true
    underside                    false                 false  # cap from below
    cap_top                      false                 false
    detail_macro                 false                 false  # ambiguous
    habitat                      false                 false  # too distant
    microscopy / other           false                 false

Idempotent: re-runs upsert.  Counts inserted/updated rows.

Usage:
    python -m vision.scripts.derive_stem_visibility
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import text  # noqa: E402

from db.connection import get_session  # noqa: E402

VISIBLE_VIEW_ANGLES = {"side_profile", "full_body", "cross_section"}

_UPSERT = text(
    """
    INSERT INTO image_annotations
        (image_id, feature_name, annotation_type, feature_value, confidence, annotator)
    VALUES (:image_id, :feature_name, 'vlm_graded', :feature_value, 1.0,
            'derived_from_view_angle')
    ON CONFLICT (image_id, feature_name, annotation_type)
    DO UPDATE SET feature_value = EXCLUDED.feature_value,
                  annotator     = EXCLUDED.annotator,
                  created_at    = NOW()
    """
)


def derive(session) -> tuple[int, int]:
    """Read view_angle for every graded image, write the two boolean visibility tags.

    Returns ``(n_images, n_inserts)``.
    """
    rows = session.execute(
        text(
            """
            SELECT image_id, feature_value
            FROM image_annotations
            WHERE annotation_type = 'vlm_graded'
              AND feature_name = 'view_angle'
            """
        )
    ).fetchall()

    n_images = len(rows)
    n_inserts = 0

    for image_id, view_angle in rows:
        is_visible = view_angle in VISIBLE_VIEW_ANGLES
        value = "true" if is_visible else "false"
        for feat_name in ("stem_upper_visible", "stem_base_visible"):
            session.execute(
                _UPSERT,
                {
                    "image_id": image_id,
                    "feature_name": feat_name,
                    "feature_value": value,
                },
            )
            n_inserts += 1

    session.commit()
    return n_images, n_inserts


def main() -> None:
    with get_session() as session:
        n_images, n_inserts = derive(session)
    print(
        f"Derived stem visibility for {n_images} images "
        f"({n_inserts} annotation upserts)."
    )


if __name__ == "__main__":
    main()
