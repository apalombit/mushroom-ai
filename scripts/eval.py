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
    args = parser.parse_args()

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
        from ingestion.rubric import get_active_profile

        print(f"Grouping profile: {get_active_profile()}")
        print(f"Evaluating {len(pairs)} pairs (x2 directions) with top_k={args.top_k}")

        if args.benchmark:
            # Sweep all benchmark configs
            results: dict[str, object] = {}
            for label, config in BENCHMARK_CONFIGS.items():
                print(f"\n--- {label} ---")
                cfg = dict(config)  # copy to avoid mutating
                bff = cfg.pop("body_form_filter", False)
                numeric_w = cfg.pop("numeric", None)
                weights = SimilarityWeights(weights=cfg, numeric=numeric_w, body_form_filter=bff)
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
        elif args.sweep:
            from evaluation.sweep import sweep_weights
            from similarity.weights import WEIGHT_FIELDS as _WF

            print(f"\nSweeping {args.sweep} random configs (seed={args.seed})...")
            results = sweep_weights(
                session, pairs, n_configs=args.sweep, top_k=args.top_k, seed=args.seed
            )

            # Print top-10
            n_show = min(10, len(results))
            print(f"\nTOP {n_show} CONFIGS (of {len(results)} total)")
            print("=" * 120)
            w_headers = [f[:8] for f in _WF]
            header = f"{'#':>3}  {'R@1':>6} {'R@3':>6} {'R@5':>6}  {'BFF':>4}  " + "  ".join(
                f"{h:>8}" for h in w_headers
            )
            print(header)
            print("-" * 120)
            for i, r in enumerate(results[:n_show], 1):
                w = r.weights
                bff = "Y" if w.get("body_form_filter") else "N"
                weight_vals = "  ".join(f"{w.get(f, 0.0):>8.4f}" for f in _WF)
                print(
                    f"{i:>3}  "
                    f"{r.recall_at.get(1, 0.0):>5.1%} "
                    f"{r.recall_at.get(3, 0.0):>5.1%} "
                    f"{r.recall_at.get(5, 0.0):>5.1%}  "
                    f"{bff:>4}  " + weight_vals
                )
            print("=" * 120)
        else:
            # Single config run with defaults
            weights = SimilarityWeights()
            _run_single(session, pairs, weights, args.top_k, args.run_name, stats)
    finally:
        session.close()


if __name__ == "__main__":
    main()
