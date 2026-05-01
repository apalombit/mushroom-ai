"""Build training manifests from DB queries, export to parquet."""

from pathlib import Path

import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

# Per-feature visibility requirements.  If a feature is listed here, the
# manifest will exclude images where the VLM graded the corresponding
# visibility field as 'false'.  Features not in this dict (e.g.
# overall_body_form) have no visibility requirement — any angle can train them.
_VISIBILITY_REQUIREMENTS: dict[str, str] = {
    "cap_shape": "cap_visible",
    "cap_color": "cap_visible",
    "surface_texture": "cap_visible",
    "hymenium_type": "hymenium_visible",
    "gill_attachment": "hymenium_visible",
    "ring_presence": "stem_upper_visible",
    "volva_presence": "stem_base_visible",
}


def build_manifest(
    session: Session,
    feature_name: str,
    embedding_dir: str | Path,
    preprocess_version: str = "v1",
    quality_filter: bool = True,
) -> pd.DataFrame:
    """Build a manifest joining image_registry + image_annotations.

    Args:
        session: SQLAlchemy session.
        feature_name: Feature to build manifest for (e.g. "hymenium_type").
        embedding_dir: Directory containing cached .pt embeddings.
        preprocess_version: Preprocessing version tag.
        quality_filter: When True, exclude images the VLM graded as unusable
            or where the target feature is not visible.  Uses LEFT JOINs so
            images without VLM grades pass through (graceful degradation).

    Returns:
        DataFrame with columns: image_id, species, feature_value, embedding_path.
    """
    embedding_dir = Path(embedding_dir)

    if quality_filter:
        query, params = _build_filtered_query(feature_name)
    else:
        query = text("""
            SELECT r.image_id, r.species, a.feature_value
            FROM image_registry r
            JOIN image_annotations a ON r.image_id = a.image_id
            WHERE a.feature_name = :feature_name
              AND a.annotation_type = 'species_propagated'
        """)
        params = {"feature_name": feature_name}

    rows = session.execute(query, params).fetchall()

    if not rows:
        return pd.DataFrame(columns=["image_id", "species", "feature_value", "embedding_path"])

    df = pd.DataFrame(rows, columns=["image_id", "species", "feature_value"])

    # Count before embedding filter (for stats)
    n_from_db = len(df)

    # Add embedding paths and filter to only images with embeddings on disk
    df["embedding_path"] = df["image_id"].apply(lambda iid: str(embedding_dir / f"{iid}.pt"))
    df = df[df["embedding_path"].apply(lambda p: Path(p).exists())].reset_index(drop=True)

    if quality_filter:
        # Also run unfiltered count for stats
        unfiltered_rows = session.execute(
            text("""
                SELECT COUNT(*) FROM image_registry r
                JOIN image_annotations a ON r.image_id = a.image_id
                WHERE a.feature_name = :feature_name
                  AND a.annotation_type = 'species_propagated'
            """),
            {"feature_name": feature_name},
        ).scalar()
        n_excluded = (unfiltered_rows or 0) - n_from_db
        print(
            f"Manifest for {feature_name}: {len(df)} images, "
            f"{df['feature_value'].nunique()} classes, "
            f"{df['species'].nunique()} species "
            f"({n_excluded} excluded by quality filter)"
        )
    else:
        print(
            f"Manifest for {feature_name}: {len(df)} images, "
            f"{df['feature_value'].nunique()} classes, "
            f"{df['species'].nunique()} species"
        )
    return df


def _build_filtered_query(
    feature_name: str,
) -> tuple[text, dict]:
    """Build SQL with LEFT JOINs for VLM quality + visibility filtering.

    Graceful degradation: LEFT JOINs with ``IS NULL OR`` ensure images
    without any VLM grades pass through unchanged.
    """
    params: dict[str, str] = {"feature_name": feature_name}

    # Global exclusions: unusable quality, no subject, absent dominance
    joins = """
        LEFT JOIN image_annotations vlm_q
            ON r.image_id = vlm_q.image_id
            AND vlm_q.annotation_type = 'vlm_graded'
            AND vlm_q.feature_name = 'image_quality'
        LEFT JOIN image_annotations vlm_sp
            ON r.image_id = vlm_sp.image_id
            AND vlm_sp.annotation_type = 'vlm_graded'
            AND vlm_sp.feature_name = 'subject_present'
        LEFT JOIN image_annotations vlm_sd
            ON r.image_id = vlm_sd.image_id
            AND vlm_sd.annotation_type = 'vlm_graded'
            AND vlm_sd.feature_name = 'subject_dominance'
    """

    where_clauses = """
        AND (vlm_q.feature_value IS NULL
             OR vlm_q.feature_value NOT IN ('unusable'))
        AND (vlm_sp.feature_value IS NULL
             OR vlm_sp.feature_value != 'false')
        AND (vlm_sd.feature_value IS NULL
             OR vlm_sd.feature_value != 'absent')
    """

    # Per-feature visibility requirement
    vis_field = _VISIBILITY_REQUIREMENTS.get(feature_name)
    if vis_field:
        params["visibility_feature"] = vis_field
        joins += """
        LEFT JOIN image_annotations vlm_vis
            ON r.image_id = vlm_vis.image_id
            AND vlm_vis.annotation_type = 'vlm_graded'
            AND vlm_vis.feature_name = :visibility_feature
        """
        where_clauses += """
        AND (vlm_vis.feature_value IS NULL
             OR vlm_vis.feature_value = 'true')
        """

    # Manual QA exclusion: reject images manually marked as 'bad'
    joins += """
        LEFT JOIN image_annotations mqa
            ON r.image_id = mqa.image_id
            AND mqa.annotation_type = 'manual_qa'
            AND mqa.feature_name = :feature_name
    """
    where_clauses += """
        AND (mqa.feature_value IS NULL OR mqa.feature_value != 'bad')
    """

    sql = f"""
        SELECT r.image_id, r.species, a.feature_value
        FROM image_registry r
        JOIN image_annotations a ON r.image_id = a.image_id
        {joins}
        WHERE a.feature_name = :feature_name
          AND a.annotation_type = 'species_propagated'
        {where_clauses}
    """

    return text(sql), params
