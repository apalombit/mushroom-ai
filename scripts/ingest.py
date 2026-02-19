"""
Ingestion pipeline: fetch → extract → reconcile → embed.

Usage:
    python -m scripts.ingest                  # Run full pipeline
    python -m scripts.ingest --fetch-only     # Just fetch source text
    python -m scripts.ingest --extract        # Fetch + extract to Layer 1
    python -m scripts.ingest --reconcile      # + reconcile to Layer 2
    python -m scripts.ingest --embed          # + embed to Layer 3

Each stage is incremental — re-running skips already-completed work.

TODO:
    - [ ] Implement each stage (see IMPLEMENTATION_PLAN.md Stages 2-5)
    - [ ] Wire CLI argument parsing
    - [ ] Add progress bars and summary output
    - [ ] Add MLflow run tracking for ingestion quality metrics
"""

import argparse
import logging
import sys

import yaml

from config import settings

logger = logging.getLogger(__name__)


def load_seed_species(path: str = "data/seed/species_list.yaml") -> list[dict]:
    """Load species list from seed YAML."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return data.get("species", [])


def run_fetch(species_list: list[dict]):
    """Stage 2: Fetch source text for each species."""
    # TODO: Implement — call ingestion/sources/wikipedia.py for each species
    logger.info("Fetching %d species from sources...", len(species_list))
    raise NotImplementedError("Implement fetch stage — see Stage 2 in IMPLEMENTATION_PLAN.md")


def run_extract(species_list: list[dict]):
    """Stage 3: Extract structured features from fetched text → Layer 1."""
    # TODO: Implement — call ingestion/extract.py for each species
    logger.info("Extracting features for %d species...", len(species_list))
    raise NotImplementedError("Implement extract stage — see Stage 3 in IMPLEMENTATION_PLAN.md")


def run_reconcile():
    """Stage 4: Reconcile Layer 1 → Layer 2."""
    # TODO: Implement — call ingestion/reconcile.py for each species
    logger.info("Reconciling source observations...")
    raise NotImplementedError("Implement reconcile stage — see Stage 4 in IMPLEMENTATION_PLAN.md")


def run_embed():
    """Stage 5: Embed Layer 2 features → Layer 3 pgvector columns."""
    # TODO: Implement — call ingestion/embed.py for each species
    logger.info("Embedding reconciled features...")
    raise NotImplementedError("Implement embed stage — see Stage 5 in IMPLEMENTATION_PLAN.md")


def main():
    parser = argparse.ArgumentParser(description="Mushroom AI ingestion pipeline")
    parser.add_argument("--fetch-only", action="store_true", help="Only fetch source text")
    parser.add_argument("--extract", action="store_true", help="Fetch + extract to Layer 1")
    parser.add_argument("--reconcile", action="store_true", help="+ reconcile to Layer 2")
    parser.add_argument("--embed", action="store_true", help="+ embed to Layer 3")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, settings.log_level))

    # Determine which stages to run
    # If no flags, run everything. Otherwise run up to the specified stage.
    run_all = not any([args.fetch_only, args.extract, args.reconcile, args.embed])

    species_list = load_seed_species()
    logger.info("Loaded %d species from seed list", len(species_list))

    # Stage 2: Fetch
    run_fetch(species_list)
    if args.fetch_only:
        return

    # Stage 3: Extract
    run_extract(species_list)
    if args.extract:
        return

    # Stage 4: Reconcile
    run_reconcile()
    if args.reconcile:
        return

    # Stage 5: Embed
    run_embed()

    logger.info("Ingestion pipeline complete.")


if __name__ == "__main__":
    main()
