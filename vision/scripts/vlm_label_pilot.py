"""CLI: run a small VLM labeling pilot on sampled images.

Writes one JSON sidecar per image plus a manifest.parquet that joins each
sampled image to its species-propagated labels (so the inspection gallery can
show side-by-side comparisons).

Usage examples:
    # Smoke test (default 50 images, stratified, gemma4:31b-cloud via Ollama Cloud)
    python -m vision.scripts.vlm_label_pilot --out data/vlm_pilot/run_001

    # Full pilot
    python -m vision.scripts.vlm_label_pilot --n 200 --out data/vlm_pilot/run_001

    # Pure random
    python -m vision.scripts.vlm_label_pilot --n 100 --strategy random --out data/vlm_pilot/run_002

    # Single species deep dive
    python -m vision.scripts.vlm_label_pilot --strategy species \\
        --species "Amanita muscaria" --out data/vlm_pilot/muscaria

    # Try a larger cloud VLM for comparison
    python -m vision.scripts.vlm_label_pilot --model qwen3-vl:235b-cloud \\
        --out data/vlm_pilot/run_qwen
"""

import argparse
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
from sqlalchemy import text  # noqa: E402

from db.connection import get_session  # noqa: E402
from vision.labeling.vlm_labeler import label_images_batch  # noqa: E402

PROCESSED_DIR = PROJECT_ROOT / "data" / "images" / "processed" / "v1"

# Tier 1 features whose propagated labels we want to compare against the VLM
PROPAGATED_FEATURES = ["hymenium_type", "overall_body_form", "cap_shape", "cap_color"]

# Rare body forms most at risk of weak supervision
RARE_BODY_FORMS = {
    "clavarioid",
    "gasteroid",
    "hydnoid",
    "corticioid",
    "secotioid",
    "hypogeous",
    "tremelloid",
}


def _fetch_image_table(session) -> pd.DataFrame:
    """Pull image_registry + propagated Tier 1 labels into a single DataFrame.

    Returns one row per (image_id, species) with columns:
        image_id, species, propagated_hymenium_type, propagated_overall_body_form,
        propagated_cap_shape, propagated_cap_color
    """
    rows = session.execute(
        text("""
            SELECT r.image_id, r.species
            FROM image_registry r
            WHERE r.species IS NOT NULL
        """)
    ).fetchall()

    base = pd.DataFrame(rows, columns=["image_id", "species"])
    if base.empty:
        return base

    # Pull all propagated annotations for the Tier 1 features in one query.
    ann_rows = session.execute(
        text("""
            SELECT image_id, feature_name, feature_value
            FROM image_annotations
            WHERE annotation_type = 'species_propagated'
              AND feature_name = ANY(:feats)
        """),
        {"feats": PROPAGATED_FEATURES},
    ).fetchall()

    ann_df = pd.DataFrame(ann_rows, columns=["image_id", "feature_name", "feature_value"])
    if ann_df.empty:
        for f in PROPAGATED_FEATURES:
            base[f"propagated_{f}"] = None
        return base

    pivoted = ann_df.pivot_table(
        index="image_id",
        columns="feature_name",
        values="feature_value",
        aggfunc="first",
    ).reset_index()
    pivoted.columns = ["image_id"] + [f"propagated_{c}" for c in pivoted.columns[1:]]

    return base.merge(pivoted, on="image_id", how="left")


def _filter_to_existing_files(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows where the processed image exists on disk."""
    df = df.copy()
    df["processed_path"] = df["image_id"].apply(lambda iid: str(PROCESSED_DIR / f"{iid}.jpg"))
    df = df[df["processed_path"].apply(lambda p: Path(p).exists())].reset_index(drop=True)
    return df


def _sample_random(df: pd.DataFrame, n: int, rng: random.Random) -> pd.DataFrame:
    n = min(n, len(df))
    idx = rng.sample(range(len(df)), n)
    return df.iloc[idx].reset_index(drop=True)


def _sample_species(df: pd.DataFrame, species: str) -> pd.DataFrame:
    return df[df["species"] == species].reset_index(drop=True)


def _sample_stratified(df: pd.DataFrame, n: int, rng: random.Random) -> pd.DataFrame:
    """Stratified pilot sampling — diverse failure-mode coverage.

    30% multi-color cap species (hand-picked list with known multi-color caps).
    30% rare body forms (clavarioid, gasteroid, hydnoid, etc.).
    20% heavily-photographed species (>15 images each, picks within-species variance).
    20% pure random across the rest.
    """
    n_multi = int(round(n * 0.30))
    n_rare = int(round(n * 0.30))
    n_heavy = int(round(n * 0.20))
    n_rand = n - n_multi - n_rare - n_heavy

    multi_color_species = {
        "Amanita muscaria",
        "Amanita pantherina",
        "Russula emetica",
        "Russula virescens",
        "Russula cyanoxantha",
        "Hygrocybe coccinea",
        "Boletus edulis",
        "Suillus luteus",
        "Lactarius deliciosus",
    }

    multi_pool = df[df["species"].isin(multi_color_species)]
    rare_pool = df[df["propagated_overall_body_form"].isin(RARE_BODY_FORMS)]

    species_counts = df["species"].value_counts()
    heavy_species = set(species_counts[species_counts > 15].index)
    heavy_pool = df[df["species"].isin(heavy_species)]

    used_ids: set[str] = set()
    parts: list[pd.DataFrame] = []

    def take(pool: pd.DataFrame, k: int) -> pd.DataFrame:
        avail = pool[~pool["image_id"].isin(used_ids)]
        k = min(k, len(avail))
        if k == 0:
            return avail.iloc[:0]
        picked = avail.sample(n=k, random_state=rng.randint(0, 2**31 - 1))
        used_ids.update(picked["image_id"].tolist())
        return picked

    multi_picked = take(multi_pool, n_multi)
    parts.append(multi_picked)
    print(f"  multi-color species:    {len(multi_picked):4d} / target {n_multi}")

    rare_picked = take(rare_pool, n_rare)
    parts.append(rare_picked)
    print(f"  rare body forms:        {len(rare_picked):4d} / target {n_rare}")

    heavy_picked = take(heavy_pool, n_heavy)
    parts.append(heavy_picked)
    print(f"  heavily-photographed:   {len(heavy_picked):4d} / target {n_heavy}")

    rand_picked = take(
        df,
        n_rand
        + (
            (n_multi - len(multi_picked))
            + (n_rare - len(rare_picked))
            + (n_heavy - len(heavy_picked))
        ),
    )
    parts.append(rand_picked)
    print(f"  random fill:            {len(rand_picked):4d} (incl. shortfall makeup)")

    out = pd.concat(parts, ignore_index=True)
    if len(out) > n:
        out = out.iloc[:n].reset_index(drop=True)
    return out


def main():
    parser = argparse.ArgumentParser(description="VLM labeling pilot")
    parser.add_argument("--n", type=int, default=50, help="Number of images to label")
    parser.add_argument(
        "--strategy",
        choices=["stratified", "random", "species"],
        default="stratified",
    )
    parser.add_argument(
        "--species", type=str, default=None, help="Species name (required for --strategy species)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemma4:31b-cloud",
        help="VLM model name (uses configured provider routing)",
    )
    parser.add_argument(
        "--out", type=str, required=True, help="Output directory (e.g. data/vlm_pilot/run_001)"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.strategy == "species" and not args.species:
        parser.error("--species is required when --strategy=species")

    out_dir = Path(args.out)
    annotations_dir = out_dir / "annotations"
    out_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    print("Loading image registry + propagated labels from DB...")
    with get_session() as session:
        full_df = _fetch_image_table(session)

    print(f"  total images with species: {len(full_df)}")
    full_df = _filter_to_existing_files(full_df)
    print(f"  with processed v1 file:    {len(full_df)}")

    if full_df.empty:
        print("ERROR: no images available.")
        sys.exit(1)

    print(f"\nSampling strategy: {args.strategy}")
    if args.strategy == "random":
        sample_df = _sample_random(full_df, args.n, rng)
    elif args.strategy == "species":
        sample_df = _sample_species(full_df, args.species)
        if len(sample_df) > args.n:
            sample_df = _sample_random(sample_df, args.n, rng)
    else:
        sample_df = _sample_stratified(full_df, args.n, rng)

    print(f"\nSampled {len(sample_df)} images")
    print(f"  unique species: {sample_df['species'].nunique()}")

    # Run VLM
    print(f"\nRunning VLM ({args.model})...")
    results = label_images_batch(
        image_paths=sample_df["processed_path"].tolist(),
        out_dir=annotations_dir,
        model_name=args.model,
        image_ids=sample_df["image_id"].tolist(),
    )

    # Build manifest joining sample + sidecar paths
    sample_df["vlm_json_path"] = sample_df["image_id"].apply(
        lambda iid: str(annotations_dir / f"{iid}.json")
    )
    sample_df["vlm_error"] = [r["error"] for r in results]
    manifest_path = out_dir / "manifest.parquet"
    sample_df.to_parquet(manifest_path, index=False)

    n_ok = sum(1 for r in results if r["error"] is None)
    n_err = len(results) - n_ok
    print(f"\nDone. {n_ok} ok, {n_err} errors.")
    print(f"  manifest: {manifest_path}")
    print(f"  sidecars: {annotations_dir}")
    print(
        f"\nNext: python -m vision.scripts.vlm_pilot_review "
        f"--pilot-dir {out_dir} --out {out_dir}/review.html"
    )


if __name__ == "__main__":
    main()
