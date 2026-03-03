"""
Evaluate retrieval recall against ground truth pairs.

Logs Recall@1/3/5 to stdout and to MLflow experiment "eval_retrieval".

Usage:
    python -m scripts.eval
    python -m scripts.eval --top-k 10
    python -m scripts.eval --w-morph 0.8 --w-eco 0.1 --w-taxon 0.1
    python -m scripts.eval --benchmark
"""

import argparse
import logging


def _run_single(session, pairs, weights, top_k, run_name, stats):
    """Run a single evaluation config and log to MLflow."""
    from evaluation.mlflow_log import log_eval_to_mlflow
    from evaluation.recall import evaluate_recall

    print(f"Weights: {weights.as_dict()}")

    result = evaluate_recall(session, pairs, weights, top_k=top_k)

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
        help="Sweep standard weight configs (mutually exclusive with --w-* flags)",
    )

    weight_group = parser.add_argument_group("weight overrides (incompatible with --benchmark)")
    weight_group.add_argument("--w-morph", type=float, default=None, help="Morphological weight")
    weight_group.add_argument("--w-eco", type=float, default=None, help="Ecological weight")
    weight_group.add_argument("--w-taxon", type=float, default=None, help="Taxonomic weight")
    args = parser.parse_args()

    # Validate mutual exclusivity
    has_custom_weights = any(w is not None for w in [args.w_morph, args.w_eco, args.w_taxon])
    if args.benchmark and has_custom_weights:
        parser.error("--benchmark cannot be combined with --w-morph/--w-eco/--w-taxon")

    logging.basicConfig(level=logging.INFO)

    from db.connection import get_session
    from evaluation.recall import BENCHMARK_CONFIGS, dataset_stats, load_ground_truth
    from similarity.weights import SimilarityWeights

    session = get_session()
    try:
        pairs = load_ground_truth(session)
        if not pairs:
            print("No ground truth pairs found — run python -m scripts.init_db first.")
            return

        stats = dataset_stats(session)
        print(f"Evaluating {len(pairs)} pairs (x2 directions) with top_k={args.top_k}")

        if args.benchmark:
            # Sweep all benchmark configs
            results: dict[str, object] = {}
            for label, (w_m, w_e, w_t) in BENCHMARK_CONFIGS.items():
                print(f"\n--- {label} ---")
                weights = SimilarityWeights(morphological=w_m, ecological=w_e, taxonomic=w_t)
                result = _run_single(session, pairs, weights, args.top_k, label, stats)
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
        else:
            # Single config run (original behavior)
            weights = SimilarityWeights(
                morphological=args.w_morph,
                ecological=args.w_eco,
                taxonomic=args.w_taxon,
            )
            _run_single(session, pairs, weights, args.top_k, args.run_name, stats)
    finally:
        session.close()


if __name__ == "__main__":
    main()
