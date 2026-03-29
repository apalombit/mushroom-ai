"""
Quick evaluation: run embedding search once, then blend Jaccard at various alphas.

Avoids re-running 44 pgvector queries per config by caching embedding results.
"""

import logging
import os
import sys
import time

os.environ.setdefault("ENVIRONMENT", "production")

from db.connection import get_session  # noqa: E402
from db.models import ReconciledSpecies  # noqa: E402
from evaluation.recall import (  # noqa: E402
    EvalResult,
    PairResult,
    dataset_stats,
    evaluate_recall,
    evaluate_recall_on_graph,
    load_ground_truth,
    load_lookalike_graph,
    per_genus_breakdown,
)
from ingestion.normalize import load_vocabulary  # noqa: E402
from similarity.aggregation import aggregate_blended  # noqa: E402
from similarity.jaccard import compute_soft_jaccard  # noqa: E402
from similarity.weights import SimilarityWeights  # noqa: E402

logging.basicConfig(level=logging.ERROR)


def main():
    session = get_session()
    top_k = 5

    stats = dataset_stats(session)
    print(f"Species: {stats['species_count']}  Embedded: {stats['embedded_count']}")

    # ── Step 1: Run embedding-only baseline (alpha=1.0) on GT pairs ──
    print("\n[1/3] Running embedding-only baseline on GT pairs...")
    gt_pairs = load_ground_truth(session)
    print(f"  GT pairs: {len(gt_pairs)}")

    start = time.monotonic()
    baseline_gt = evaluate_recall(
        session, gt_pairs, SimilarityWeights(alpha=1.0, group_filter=False), top_k=top_k
    )
    dur = time.monotonic() - start
    print(f"  Baseline (alpha=1.0 GF=N): R@1={baseline_gt.recall_at(1):.1%}  "
          f"R@3={baseline_gt.recall_at(3):.1%}  R@5={baseline_gt.recall_at(5):.1%}  "
          f"errors={baseline_gt.errors}  {dur:.1f}s")

    # With group filter
    start = time.monotonic()
    baseline_gt_gf = evaluate_recall(
        session, gt_pairs, SimilarityWeights(alpha=1.0, group_filter=True), top_k=top_k
    )
    dur = time.monotonic() - start
    print(f"  Baseline (alpha=1.0 GF=Y): R@1={baseline_gt_gf.recall_at(1):.1%}  "
          f"R@3={baseline_gt_gf.recall_at(3):.1%}  R@5={baseline_gt_gf.recall_at(5):.1%}  "
          f"errors={baseline_gt_gf.errors}  {dur:.1f}s")

    # ── Step 2: alpha=0.5 blend on GT pairs ──
    print("\n[2/3] Running blended (alpha=0.5) on GT pairs...")
    start = time.monotonic()
    blend_gt = evaluate_recall(
        session, gt_pairs, SimilarityWeights(alpha=0.5, group_filter=False), top_k=top_k
    )
    dur = time.monotonic() - start
    print(f"  Blend (alpha=0.5 GF=N):    R@1={blend_gt.recall_at(1):.1%}  "
          f"R@3={blend_gt.recall_at(3):.1%}  R@5={blend_gt.recall_at(5):.1%}  "
          f"errors={blend_gt.errors}  {dur:.1f}s")

    start = time.monotonic()
    blend_gt_gf = evaluate_recall(
        session, gt_pairs, SimilarityWeights(alpha=0.5, group_filter=True), top_k=top_k
    )
    dur = time.monotonic() - start
    print(f"  Blend (alpha=0.5 GF=Y):    R@1={blend_gt_gf.recall_at(1):.1%}  "
          f"R@3={blend_gt_gf.recall_at(3):.1%}  R@5={blend_gt_gf.recall_at(5):.1%}  "
          f"errors={blend_gt_gf.errors}  {dur:.1f}s")

    # ── Step 3: Sample of graph edges ──
    print("\n[3/3] Running on graph edges (sample=50)...")
    graph_edges = load_lookalike_graph()
    all_species = {
        r[0]
        for r in session.execute(
            __import__("sqlalchemy").text("SELECT scientific_name FROM reconciled_species")
        ).fetchall()
    }
    graph_edges = [(a, b) for a, b in graph_edges if a in all_species and b in all_species]
    sample = graph_edges[:50]
    print(f"  Total graph edges: {len(graph_edges)}, using sample of {len(sample)}")

    configs = [
        ("alpha=1.0 GF=N", 1.0, False),
        ("alpha=1.0 GF=Y", 1.0, True),
        ("alpha=0.7 GF=N", 0.7, False),
        ("alpha=0.5 GF=N", 0.5, False),
        ("alpha=0.5 GF=Y", 0.5, True),
        ("alpha=0.3 GF=N", 0.3, False),
    ]

    print(f"\n{'Config':<25} {'R@1':>7} {'R@3':>7} {'R@5':>7} {'Err':>5} {'Time':>7}")
    print("-" * 65)

    best_result = None
    best_label = ""
    best_r5 = -1.0

    for label, alpha, gf in configs:
        weights = SimilarityWeights(alpha=alpha, group_filter=gf, body_form_filter=False)
        start = time.monotonic()
        result = evaluate_recall_on_graph(session, sample, weights, top_k=top_k)
        dur = time.monotonic() - start
        r5 = result.recall_at(5)
        print(
            f"{label:<25} "
            f"{result.recall_at(1):>6.1%} "
            f"{result.recall_at(3):>6.1%} "
            f"{r5:>6.1%} "
            f"{result.errors:>5} "
            f"{dur:>5.1f}s"
        )
        sys.stdout.flush()
        if r5 > best_r5:
            best_r5 = r5
            best_result = result
            best_label = label

    print("=" * 65)

    if best_result:
        print(f"\nBest config: {best_label}")
        breakdown = per_genus_breakdown(best_result, k=top_k)
        top_genera = sorted(breakdown.items(), key=lambda x: -x[1]["total"])[:10]
        print(f"\n{'Genus':<25} {'Total':>6} {'Hits':>6} {'Recall':>8}")
        print("-" * 50)
        for genus, info in top_genera:
            print(
                f"{genus:<25} {info['total']:>6.0f} {info['hits']:>6.0f} "
                f"{info['recall']:>7.1%}"
            )

    session.close()


if __name__ == "__main__":
    main()
