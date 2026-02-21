"""
Validate similarity engine against known lookalike pairs.

For each ground truth pair, queries species_a and checks whether
species_b appears in the top-K results (and vice versa).

Reports pass rate and logs results to MLflow.

Usage:
    python -m scripts.validate
    python -m scripts.validate --top-k 15
"""

import argparse
import logging

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate against ground truth pairs")
    parser.add_argument("--top-k", type=int, default=10, help="Check if match is in top-K")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    from config import settings
    from db.connection import get_session
    from db.models import GroundTruthPair
    from similarity.search import search_lookalikes
    from similarity.weights import SimilarityWeights

    session = get_session()
    try:
        pairs = session.query(GroundTruthPair).all()
    finally:
        session.close()

    if not pairs:
        print("No ground truth pairs found — run python -m scripts.init_db first.")
        return

    weights = SimilarityWeights()
    total, passed = 0, 0
    failed_pairs: list[tuple[str, str]] = []

    for pair in pairs:
        for query, target in [(pair.species_a, pair.species_b), (pair.species_b, pair.species_a)]:
            s = get_session()
            try:
                _, candidates = search_lookalikes(s, query, weights, top_k=args.top_k)
                found = any(c["scientific_name"] == target for c in candidates)
                total += 1
                if found:
                    passed += 1
                    logger.info("PASS: %s → %s in top-%d", query, target, args.top_k)
                else:
                    failed_pairs.append((query, target))
                    logger.info("FAIL: %s → %s not in top-%d", query, target, args.top_k)
            except ValueError as e:
                logger.warning("Skipping %s: %s", query, e)
            finally:
                s.close()

    if total == 0:
        print("No embedded species available. Run the full ingestion pipeline first.")
        return

    pass_rate = passed / total
    print(f"\nValidation results: {passed}/{total} pairs found (pass rate: {pass_rate:.1%})")

    if failed_pairs:
        print("\nFailed pairs:")
        for q, t in failed_pairs:
            print(f"  {q} → {t}")

    # MLflow logging (best-effort)
    try:
        import mlflow

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        experiment = mlflow.set_experiment("validation")
        with mlflow.start_run(experiment_id=experiment.experiment_id, run_name="gt_validation"):
            mlflow.log_metric("pass_rate", pass_rate)
            mlflow.log_metric("pairs_passed", passed)
            mlflow.log_metric("pairs_total", total)
            mlflow.log_param("top_k", args.top_k)
        print("Logged to MLflow.")
    except Exception as e:
        logger.warning("MLflow logging failed: %s", e)


if __name__ == "__main__":
    main()
