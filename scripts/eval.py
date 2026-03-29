"""
Evaluate retrieval recall against ground truth pairs.

Logs Recall@1/3/5 to stdout and to MLflow experiment "eval_retrieval".

Usage:
    python -m scripts.eval
    python -m scripts.eval --top-k 10
    python -m scripts.eval --benchmark
"""

import argparse
import logging


def _run_single(session, edges, weights, top_k, run_name, stats, strategy="weighted_avg"):
    """Run a single evaluation config and log to MLflow."""
    from evaluation.mlflow_log import log_eval_to_mlflow
    from evaluation.recall import evaluate_recall_on_graph

    print(f"Weights: {weights.as_dict()}")

    result = evaluate_recall_on_graph(session, edges, weights, top_k=top_k, strategy=strategy)

    print(f"  Recall@1: {result.hits_at(1)}/{result.total} = {result.recall_at(1):.1%}")
    print(f"  Recall@3: {result.hits_at(3)}/{result.total} = {result.recall_at(3):.1%}")
    print(f"  Recall@5: {result.hits_at(5)}/{result.total} = {result.recall_at(5):.1%}")

    if result.errors:
        print(f"  Errors: {result.errors}")

    print(f"  Duration: {result.duration_s}s")

    ok = log_eval_to_mlflow(result, run_name=run_name, dataset_params=stats)
    if ok:
        print("  Logged to MLflow (experiment: eval_retrieval).")
    else:
        print("  MLflow logging skipped (service unavailable).")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval recall")
    parser.add_argument("--top-k", type=int, default=5, help="Retrieval depth (default: 5)")
    parser.add_argument("--run-name", type=str, default=None, help="MLflow run name")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Sweep standard weight configs",
    )
    parser.add_argument(
        "--sweep",
        type=int,
        default=None,
        metavar="N",
        help="Sweep N random weight configs (precompute + rescore)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for --sweep (default: 42)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        choices=["weighted_avg", "rrf", "contrastive_gate", "learned_ranker"],
        help="Run a single aggregation strategy (default: weighted_avg)",
    )
    parser.add_argument(
        "--compare-strategies",
        action="store_true",
        help="Run all aggregation strategies and compare Recall@K",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    from types import SimpleNamespace

    from db.connection import get_session
    from evaluation.recall import (
        BENCHMARK_CONFIGS,
        dataset_stats,
        load_unified_lookalikes,
    )
    from similarity.weights import SimilarityWeights

    session = get_session()
    try:
        edges = load_unified_lookalikes(session)
        if not edges:
            print("No lookalike edges found — check known_lookalikes.yaml and DB species.")
            return

        stats = dataset_stats(session)
        from ingestion.rubric import get_active_profile

        print(f"Grouping profile: {get_active_profile()}")
        print(f"Evaluating {len(edges)} edges (x2 directions) with top_k={args.top_k}")

        if args.benchmark:
            # Sweep all benchmark configs
            results: dict[str, object] = {}
            for label, config in BENCHMARK_CONFIGS.items():
                print(f"\n--- {label} ---")
                cfg = dict(config)  # copy to avoid mutating
                bff = cfg.pop("body_form_filter", False)
                gf = cfg.pop("group_filter", None)
                alpha = cfg.pop("alpha", None)
                numeric_w = cfg.pop("numeric", None)
                morpho = cfg.pop("morpho_pool_required", None)
                weights = SimilarityWeights(
                    weights=cfg, numeric=numeric_w, body_form_filter=bff,
                    group_filter=gf, alpha=alpha, morpho_pool_required=morpho,
                )
                result = _run_single(session, edges, weights, args.top_k, label, stats)
                results[label] = result

            # Comparison table
            print("\n" + "=" * 70)
            print("BENCHMARK COMPARISON")
            print("=" * 70)
            header = f"{'Config':<25} {'R@1':>7} {'R@3':>7} {'R@5':>7} {'Time':>7}"
            print(header)
            print("-" * 70)
            for label, res in results.items():
                print(
                    f"{label:<25} "
                    f"{res.recall_at(1):>6.1%} "
                    f"{res.recall_at(3):>6.1%} "
                    f"{res.recall_at(5):>6.1%} "
                    f"{res.duration_s:>5.1f}s"
                )
            print("=" * 70)
        elif args.compare_strategies:
            strategies = ["weighted_avg", "rrf", "contrastive_gate", "learned_ranker"]
            cfg = dict(BENCHMARK_CONFIGS["blended-best"])
            bff = cfg.pop("body_form_filter", False)
            gf = cfg.pop("group_filter", None)
            alpha = cfg.pop("alpha", None)
            numeric_w = cfg.pop("numeric", None)
            morpho = cfg.pop("morpho_pool_required", None)
            weights = SimilarityWeights(
                weights=cfg, numeric=numeric_w, body_form_filter=bff,
                group_filter=gf, alpha=alpha, morpho_pool_required=morpho,
            )
            results: dict[str, object] = {}
            for strategy in strategies:
                print(f"\n--- {strategy} ---")
                result = _run_single(
                    session, edges, weights, args.top_k, strategy, stats,
                    strategy=strategy,
                )
                results[strategy] = result

            print("\n" + "=" * 60)
            print("STRATEGY COMPARISON")
            print("=" * 60)
            print(f"{'Strategy':<22} {'R@1':>7} {'R@3':>7} {'R@5':>7} {'Time':>7}")
            print("-" * 60)
            for name, res in results.items():
                print(
                    f"{name:<22} "
                    f"{res.recall_at(1):>6.1%} "
                    f"{res.recall_at(3):>6.1%} "
                    f"{res.recall_at(5):>6.1%} "
                    f"{res.duration_s:>5.1f}s"
                )
            print("=" * 60)
        elif args.sweep:
            from evaluation.sweep import sweep_weights
            from similarity.weights import WEIGHT_FIELDS as _WF

            sweep_pairs = [
                SimpleNamespace(species_a=a, species_b=b) for a, b in edges
            ]
            print(f"\nSweeping {args.sweep} random configs over {len(edges)} edges "
                  f"(seed={args.seed})...")
            results = sweep_weights(
                session, sweep_pairs, n_configs=args.sweep,
                top_k=args.top_k, seed=args.seed,
            )

            # Print top-10
            n_show = min(10, len(results))
            print(f"\nTOP {n_show} CONFIGS (of {len(results)} total)")
            print("=" * 140)
            w_headers = [f[:8] for f in _WF]
            header = (
                f"{'#':>3}  {'R@1':>6} {'R@3':>6} {'R@5':>6}"
                f"  {'BFF':>4} {'HF':>3} {'SCF':>4} {'α':>4}  "
                + "  ".join(f"{h:>8}" for h in w_headers)
            )
            print(header)
            print("-" * 140)
            for i, r in enumerate(results[:n_show], 1):
                w = r.weights
                bff = "Y" if w.get("body_form_filter") else "N"
                hf = "Y" if w.get("hymenium_filter") else "N"
                scf = "Y" if w.get("size_class_filter") else "N"
                alpha = w.get("alpha", 1.0)
                weight_vals = "  ".join(f"{w.get(f, 0.0):>8.4f}" for f in _WF)
                print(
                    f"{i:>3}  "
                    f"{r.recall_at.get(1, 0.0):>5.1%} "
                    f"{r.recall_at.get(3, 0.0):>5.1%} "
                    f"{r.recall_at.get(5, 0.0):>5.1%}  "
                    f"{bff:>4} {hf:>3} {scf:>4} {alpha:>4.1f}  "
                    + weight_vals
                )
            print("=" * 140)
        else:
            # Single config run with defaults
            strategy = args.strategy or "weighted_avg"
            weights = SimilarityWeights()
            _run_single(
                session, edges, weights, args.top_k, args.run_name, stats,
                strategy=strategy,
            )
    finally:
        session.close()


if __name__ == "__main__":
    main()
