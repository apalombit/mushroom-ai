"""
Ingestion pipeline: fetch → extract → reconcile → embed.

Usage:
    python -m scripts.ingest --fetch-only     # Stage 2: fetch source text
    python -m scripts.ingest --extract        # Stage 3: extract to Layer 1
    python -m scripts.ingest --reconcile      # Stage 4: reconcile to Layer 2
    python -m scripts.ingest --embed          # Stage 5: embed to Layer 3
    python -m scripts.ingest                  # Run full pipeline (all stages)

Each stage is STRICTLY independent — re-running skips already-completed work.
If prerequisites are missing, a clear message is printed.
"""

import argparse
import logging
import time

import yaml

from config import settings
from db.connection import get_session
from db.models import ReconciledSpecies, SourceObservation
from ingestion.embed import embed_all
from ingestion.extract import extract_features_from_text, save_extraction
from ingestion.reconcile import reconcile_species
from ingestion.sources import firstnature, funghiitaliani, mushroomexpert, ultimatemushroom, wikipedia

FETCH_SOURCES = [wikipedia, firstnature, mushroomexpert, ultimatemushroom, funghiitaliani]

logger = logging.getLogger(__name__)


def load_seed_species(path: str = "data/seed/species_list.yaml") -> list[dict]:
    """Load species list from seed YAML."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return data.get("species", [])


def _log_mlflow(run_name: str, metrics: dict, params: dict | None = None) -> None:
    """Log metrics to MLflow (best-effort — silently skips if MLflow unavailable)."""
    try:
        import mlflow

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        with mlflow.start_run(run_name=run_name):
            for k, v in metrics.items():
                mlflow.log_metric(k, v)
            for k, v in (params or {}).items():
                mlflow.log_param(k, v)
    except Exception:
        pass


def run_fetch(species_list: list[dict]) -> None:
    """Stage 2: Fetch source text for each species from all sources."""
    total_requests = len(species_list) * len(FETCH_SOURCES)
    logger.info(
        "Fetching %d species × %d sources = %d requests...",
        len(species_list), len(FETCH_SOURCES), total_requests,
    )
    start = time.monotonic()
    ok, failed = 0, []

    for entry in species_list:
        name = entry["scientific_name"]
        aliases = entry.get("aliases", [])
        for source in FETCH_SOURCES:
            result = source.fetch_species_page(name, aliases=aliases)
            if result:
                ok += 1
                logger.debug("Fetched [%s]: %s", source.SOURCE_NAME, name)
            else:
                failed.append(f"{source.SOURCE_NAME}/{name}")
                logger.debug("Not found [%s]: %s", source.SOURCE_NAME, name)

    duration = time.monotonic() - start
    print(f"Fetched {ok}/{total_requests} pages in {duration:.1f}s.")
    if failed:
        print(f"Misses ({len(failed)}): {', '.join(failed[:10])}"
              + (f" ... +{len(failed) - 10} more" if len(failed) > 10 else ""))

    _log_mlflow(
        "fetch",
        {"pages_fetched": ok, "pages_missed": len(failed), "duration_s": duration,
         "success_rate": ok / total_requests if total_requests else 0.0},
    )


def run_extract(species_list: list[dict]) -> None:
    """Stage 3: Extract structured features from all sources → Layer 1 (incremental)."""
    logger.info("Extracting features for %d species × %d sources...",
                len(species_list), len(FETCH_SOURCES))
    start = time.monotonic()
    ok, skipped_existing, failed = 0, 0, []

    # Pre-load set of (scientific_name, source_name) already in Layer 1
    session = get_session()
    try:
        already_done = {
            (row[0], row[1])
            for row in session.query(
                SourceObservation.scientific_name, SourceObservation.source_name
            ).all()
        }
    finally:
        session.close()

    for entry in species_list:
        name = entry["scientific_name"]
        aliases = entry.get("aliases", [])
        for source in FETCH_SOURCES:
            if (name, source.SOURCE_NAME) in already_done:
                skipped_existing += 1
                logger.debug("Already extracted (skip): %s [%s]", name, source.SOURCE_NAME)
                continue

            page = source.fetch_species_page(name, aliases=aliases)
            if page is None:
                logger.debug("No cached page [%s]: %s — run --fetch-only first.", source.SOURCE_NAME, name)
                continue
            try:
                features = extract_features_from_text(name, page["text"], source_name=source.SOURCE_NAME)
                save_extraction(features, page["url"], page["text"], source.SOURCE_NAME)
                ok += 1
                logger.info("Extracted [%s]: %s", source.SOURCE_NAME, name)
            except Exception as e:
                logger.error("Failed to extract [%s] %s: %s", source.SOURCE_NAME, name, e)
                failed.append(f"{source.SOURCE_NAME}/{name}")

    duration = time.monotonic() - start
    total = len(species_list) * len(FETCH_SOURCES)
    print(f"Extracted {ok}/{total} (species×source) in {duration:.1f}s "
          f"({skipped_existing} already done, skipped).")
    if failed:
        print(f"Failed: {', '.join(failed[:10])}"
              + (f" ... +{len(failed) - 10} more" if len(failed) > 10 else ""))

    _log_mlflow(
        "extract",
        {"extractions_done": ok, "extractions_failed": len(failed), "duration_s": duration,
         "success_rate": ok / total if total else 0.0},
    )


def run_reconcile() -> None:
    """Stage 4: Reconcile Layer 1 → Layer 2."""
    session = get_session()
    try:
        names = [
            row[0]
            for row in session.query(SourceObservation.scientific_name).distinct().all()
        ]
    finally:
        session.close()

    if not names:
        print("No Layer 1 data found — run --extract first.")
        return

    logger.info("Reconciling %d species...", len(names))
    start = time.monotonic()
    ok, failed = 0, []

    for name in names:
        s = get_session()
        try:
            result = reconcile_species(s, name)
            if result:
                ok += 1
                logger.info("Reconciled: %s", name)
        except Exception as e:
            logger.error("Failed to reconcile %s: %s", name, e)
            failed.append(name)
        finally:
            s.close()

    duration = time.monotonic() - start
    total = len(names)
    print(f"Reconciled {ok}/{total} species in {duration:.1f}s.")
    if failed:
        print(f"Failed: {', '.join(failed)}")

    _log_mlflow(
        "reconcile",
        {"species_reconciled": ok, "species_failed": len(failed), "duration_s": duration,
         "success_rate": ok / total if total else 0.0},
    )


def run_embed() -> None:
    """Stage 5: Embed Layer 2 features → Layer 3 pgvector columns."""
    session = get_session()
    try:
        count = session.query(ReconciledSpecies).count()
    finally:
        session.close()

    if count == 0:
        print("No Layer 2 data found — run --reconcile first.")
        return

    logger.info("Embedding up to %d species (incremental)...", count)
    start = time.monotonic()

    s = get_session()
    try:
        embedded, skipped = embed_all(s)
    finally:
        s.close()

    duration = time.monotonic() - start
    print(f"Embedded {embedded} species ({skipped} errors) in {duration:.1f}s.")

    _log_mlflow(
        "embed",
        {"species_embedded": embedded, "species_skipped": skipped, "duration_s": duration},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Mushroom AI ingestion pipeline")
    parser.add_argument("--fetch-only", action="store_true", help="Only fetch source text")
    parser.add_argument("--extract", action="store_true", help="Extract to Layer 1")
    parser.add_argument("--reconcile", action="store_true", help="Reconcile to Layer 2")
    parser.add_argument("--embed", action="store_true", help="Embed to Layer 3")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, settings.log_level))

    species_list = load_seed_species()
    logger.info("Loaded %d species from seed list", len(species_list))

    if args.fetch_only:
        run_fetch(species_list)
    elif args.extract:
        run_extract(species_list)
    elif args.reconcile:
        run_reconcile()
    elif args.embed:
        run_embed()
    else:
        # No flag: run full pipeline in order
        run_fetch(species_list)
        run_extract(species_list)
        run_reconcile()
        run_embed()
        logger.info("Ingestion pipeline complete.")


if __name__ == "__main__":
    main()
