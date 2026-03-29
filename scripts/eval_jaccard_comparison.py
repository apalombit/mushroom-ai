"""
Compare embedding-only vs Jaccard-blended retrieval on GT pairs and lookalike graph.

Sweeps alpha values: 1.0 (pure embedding), 0.7, 0.5, 0.3, 0.0 (pure Jaccard).
Also tests with/without group pre-filter.

Usage:
    python -m scripts.eval_jaccard_comparison
    python -m scripts.eval_jaccard_comparison --top-k 10
"""

import argparse
import logging
import os
import time

# Suppress SQLAlchemy echo before engine creation
os.environ.setdefault("ENVIRONMENT", "production")

from db.connection import get_session  # noqa: E402
from evaluation.recall import (
    dataset_stats,
    evaluate_recall,
    evaluate_recall_on_graph,
    load_ground_truth,
    load_lookalike_graph,
    per_genus_breakdown,
)
from similarity.weights import SimilarityWeights


def _run_config(session, pairs_or_edges, alpha, group_filter, top_k, use_graph=False):
    """Run one config and return EvalResult."""
    weights = SimilarityWeights(
        alpha=alpha,
        group_filter=group_filter,
        body_form_filter=False,
    )
    if use_graph:
        return evaluate_recall_on_graph(session, pairs_or_edges, weights, top_k=top_k)
    else:
        return evaluate_recall(session, pairs_or_edges, weights, top_k=top_k)


def main():
    parser = argparse.ArgumentParser(description="Compare Jaccard blending strategies")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    logging.basicConfig(level=logging.ERROR)
    logging.getLogger("evaluation.recall").setLevel(logging.ERROR)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.ERROR)
    logging.getLogger("sqlalchemy.engine.Engine").setLevel(logging.ERROR)

    session = get_session()
    try:
        stats = dataset_stats(session)
        print(f"Species: {stats['species_count']}  Embedded: {stats['embedded_count']}")

        # Load both datasets
        gt_pairs = load_ground_truth(session)
        graph_edges = load_lookalike_graph()
        # Filter graph to species in DB
        all_species = {
            r[0]
            for r in session.execute(
                __import__("sqlalchemy").text(
                    "SELECT scientific_name FROM reconciled_species"
                )
            ).fetchall()
        }
        graph_edges = [(a, b) for a, b in graph_edges if a in all_species and b in all_species]

        print(f"GT pairs: {len(gt_pairs)}  Graph edges: {len(graph_edges)}")
        print(f"Top-K: {args.top_k}\n")

        alphas = [1.0, 0.5, 0.0]

        header = f"{'Config':<30} {'R@1':>7} {'R@3':>7} {'R@5':>7} {'Err':>5} {'Time':>7}"

        def _run_sweep(label_prefix, data, use_graph, alphas_list):
            print("=" * 78)
            print(label_prefix)
            print("=" * 78)
            print(header)
            print("-" * 78)
            best_result = None
            best_label = ""
            best_r5 = -1.0
            for alpha in alphas_list:
                for gf in [False, True]:
                    lbl = f"alpha={alpha:.1f} GF={'Y' if gf else 'N'}"
                    start = time.monotonic()
                    res = _run_config(session, data, alpha, gf, args.top_k, use_graph=use_graph)
                    dur = time.monotonic() - start
                    r5 = res.recall_at(5)
                    print(
                        f"{lbl:<30} "
                        f"{res.recall_at(1):>6.1%} "
                        f"{res.recall_at(3):>6.1%} "
                        f"{r5:>6.1%} "
                        f"{res.errors:>5} "
                        f"{dur:>5.1f}s"
                    )
                    if r5 > best_r5:
                        best_r5 = r5
                        best_result = res
                        best_label = lbl
            return best_result, best_label

        # ── GT pairs (fast: 31 pairs × 2 directions × 6 configs) ──
        _run_sweep("GROUND TRUTH PAIRS (DB)", gt_pairs, False, alphas)

        # ── Lookalike graph sample (first 100 edges for speed) ──
        sample_size = min(100, len(graph_edges))
        graph_sample = graph_edges[:sample_size]
        print(f"\n(Using {sample_size}/{len(graph_edges)} graph edges for speed)\n")
        best_result, best_label = _run_sweep(
            f"LOOKALIKE GRAPH (sample={sample_size})", graph_sample, True, alphas
        )
        print("=" * 78)

        # Per-genus breakdown for the best config
        if best_result:
            print(f"\nBest config: {best_label}")
            breakdown = per_genus_breakdown(best_result, k=args.top_k)
            top_genera = sorted(breakdown.items(), key=lambda x: -x[1]["total"])[:15]
            print(f"\n{'Genus':<25} {'Total':>6} {'Hits':>6} {'Recall':>8}")
            print("-" * 50)
            for genus, info in top_genera:
                print(
                    f"{genus:<25} {info['total']:>6.0f} {info['hits']:>6.0f} "
                    f"{info['recall']:>7.1%}"
                )

    finally:
        session.close()


if __name__ == "__main__":
    main()
