"""
Optimize per-field Jaccard weights against the expanded lookalike graph.

Uses Optuna (gradient-free) to search over per-field weights that maximize
Recall@K when ranking species by soft-Jaccard similarity.

Usage:
    python -m scripts.optimize_jaccard
    python -m scripts.optimize_jaccard --trials 200 --top-k 5
    python -m scripts.optimize_jaccard --per-group
"""

import argparse
import json
import logging
import time
from collections import defaultdict

import optuna

from db.connection import get_session
from db.models import ReconciledSpecies
from evaluation.recall import load_unified_lookalikes
from ingestion.normalize import _FIELD_TO_VOCAB, load_vocabulary
from similarity.jaccard import _BOOLEAN_FIELDS, _COLOR_FIELDS, compute_soft_jaccard

logger = logging.getLogger(__name__)

# All Jaccard field paths that can be weighted
ALL_FIELDS = list(_FIELD_TO_VOCAB.keys()) + list(_COLOR_FIELDS.keys()) + _BOOLEAN_FIELDS


def _load_all_features(session) -> dict[str, dict]:
    """Load features_json for all reconciled species into memory."""
    rows = session.query(
        ReconciledSpecies.scientific_name,
        ReconciledSpecies.features_json,
        ReconciledSpecies.group,
    ).all()
    return {name: {"features": features or {}, "group": group} for name, features, group in rows}


def _compute_recall_at_k(
    species_data: dict[str, dict],
    edges: list[tuple[str, str]],
    vocab: dict,
    weights: dict[str, float],
    k: int = 5,
) -> float:
    """Compute Recall@K for a given Jaccard weight config.

    For each (query, target) edge direction, rank the target among all
    species by Jaccard score and check if it's in top-K.
    """
    all_species = list(species_data.keys())

    # Pre-compute Jaccard for all edge species against all candidates
    # Build a set of species involved in edges for efficient lookup
    edge_species = set()
    for a, b in edges:
        edge_species.add(a)
        edge_species.add(b)

    # Only evaluate edges where both species have data
    valid_edges = [(a, b) for a, b in edges if a in species_data and b in species_data]
    if not valid_edges:
        return 0.0

    hits = 0
    total = 0

    # Cache Jaccard scores per query species
    for query, target in valid_edges:
        for q, t in [(query, target), (target, query)]:
            total += 1
            q_features = species_data[q]["features"]

            # Compute Jaccard against all species and rank
            scores: list[tuple[str, float]] = []
            for cand_name in all_species:
                if cand_name == q:
                    continue
                c_features = species_data[cand_name]["features"]
                score, _ = compute_soft_jaccard(q_features, c_features, vocab, weights)
                scores.append((cand_name, score))

            scores.sort(key=lambda x: x[1], reverse=True)
            top_names = [name for name, _ in scores[:k]]
            if t in top_names:
                hits += 1

    return hits / total if total > 0 else 0.0


def optimize(
    species_data: dict[str, dict],
    edges: list[tuple[str, str]],
    vocab: dict,
    n_trials: int = 100,
    top_k: int = 5,
    fields: list[str] | None = None,
) -> dict[str, float]:
    """Run Optuna optimization over Jaccard field weights."""
    target_fields = fields or ALL_FIELDS

    # Filter to fields that actually have vocab entries with similarity matrices
    active_fields = []
    for f in target_fields:
        if f in _FIELD_TO_VOCAB:
            vk = _FIELD_TO_VOCAB[f]
            entry = vocab.get(vk)
            if entry and entry.get("similarity_matrix"):
                active_fields.append(f)
        elif f in _COLOR_FIELDS:
            vk = _COLOR_FIELDS[f]
            entry = vocab.get(vk)
            if entry and entry.get("similarity_matrix"):
                active_fields.append(f)
        elif f in _BOOLEAN_FIELDS:
            active_fields.append(f)

    logger.info("Optimizing %d field weights over %d edges", len(active_fields), len(edges))

    def objective(trial: optuna.Trial) -> float:
        weights = {}
        for f in active_fields:
            weights[f] = trial.suggest_float(f, 0.0, 5.0)
        return _compute_recall_at_k(species_data, edges, vocab, weights, k=top_k)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    logger.info("Best Recall@%d: %.1%%", top_k, study.best_value * 100)
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize Jaccard field weights")
    parser.add_argument("--trials", type=int, default=100, help="Number of Optuna trials")
    parser.add_argument("--top-k", type=int, default=5, help="Recall@K depth")
    parser.add_argument(
        "--per-group",
        action="store_true",
        help="Optimize per taxonomic group (separate weight profiles)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    session = get_session()
    try:
        species_data = _load_all_features(session)
        print(f"Loaded {len(species_data)} species from DB")

        edges = load_unified_lookalikes(session)
        # Filter to edges where both species are in DB
        edges = [(a, b) for a, b in edges if a in species_data and b in species_data]
        print(f"Loaded {len(edges)} valid lookalike edges")

        if not edges:
            print("No valid edges — ensure species are ingested and lookalikes extracted.")
            return

        vocab = load_vocabulary()
        start = time.monotonic()

        if args.per_group:
            # Partition species by taxonomic group
            group_species: dict[str, dict[str, dict]] = defaultdict(dict)
            for name, data in species_data.items():
                g = data.get("group") or "unknown"
                group_species[g][name] = data

            # Partition edges by group
            group_edges: dict[str, list[tuple[str, str]]] = defaultdict(list)
            for a, b in edges:
                ga = species_data[a].get("group") or "unknown"
                gb = species_data[b].get("group") or "unknown"
                if ga == gb:
                    group_edges[ga].append((a, b))

            all_weights: dict[str, dict[str, float]] = {}
            for group_name in sorted(group_edges.keys()):
                g_edges = group_edges[group_name]
                g_species = group_species[group_name]
                if len(g_edges) < 3:
                    print(f"\n--- {group_name}: skipping ({len(g_edges)} edges) ---")
                    continue
                print(f"\n--- {group_name} ({len(g_species)} species, {len(g_edges)} edges) ---")
                best = optimize(g_species, g_edges, vocab, args.trials, args.top_k)
                all_weights[group_name] = best

                # Quick eval
                recall = _compute_recall_at_k(g_species, g_edges, vocab, best, args.top_k)
                print(f"  Best Recall@{args.top_k}: {recall:.1%}")

            duration = time.monotonic() - start
            print(f"\nCompleted in {duration:.1f}s")
            print("\nOptimized per-group weights:")
            print(json.dumps(all_weights, indent=2))
        else:
            best = optimize(species_data, edges, vocab, args.trials, args.top_k)
            duration = time.monotonic() - start

            # Final eval with best weights
            recall = _compute_recall_at_k(species_data, edges, vocab, best, args.top_k)
            print(f"\nBest Recall@{args.top_k}: {recall:.1%}")
            print(f"Completed in {duration:.1f}s")
            print(f"\nOptimized weights ({len(best)} fields):")
            print(json.dumps(best, indent=2))

    finally:
        session.close()


if __name__ == "__main__":
    main()
