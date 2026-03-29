"""
Train the GBDT lookalike ranker.

Usage:
    python -m scripts.train_ranker
    python -m scripts.train_ranker --neg-ratio 5 --n-folds 5
    python -m scripts.train_ranker --feature-importance
"""

import argparse
import logging
import time

from db.connection import get_session
from db.models import ReconciledSpecies
from evaluation.recall import load_unified_lookalikes
from ingestion.normalize import load_vocabulary


def _load_all_features(session) -> dict[str, dict]:
    """Load features_json for all reconciled species into memory."""
    rows = session.query(
        ReconciledSpecies.scientific_name,
        ReconciledSpecies.features_json,
        ReconciledSpecies.group,
    ).all()
    return {
        name: {"features": features or {}, "group": group}
        for name, features, group in rows
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GBDT lookalike ranker")
    parser.add_argument(
        "--neg-ratio", type=int, default=5, help="Negative:positive ratio (default: 5)"
    )
    parser.add_argument(
        "--n-folds", type=int, default=5, help="GroupKFold splits (default: 5)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--feature-importance",
        action="store_true",
        help="Print top features by importance",
    )
    parser.add_argument(
        "--optimize",
        type=int,
        default=None,
        metavar="N",
        help="Run N Optuna trials to optimize hyperparameters",
    )
    parser.add_argument(
        "--symmetric",
        action="store_true",
        help="Double positives by including both (a,b) and (b,a) directions",
    )
    parser.add_argument(
        "--hard-negatives",
        action="store_true",
        help="Two-pass training: random negs first, then hard negs from model",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    from similarity.ranker import build_training_data, optimize_ranker, train_ranker

    session = get_session()
    try:
        species_data = _load_all_features(session)
        print(f"Loaded {len(species_data)} species from DB")

        edges = load_unified_lookalikes(session)
        edges = [(a, b) for a, b in edges if a in species_data and b in species_data]
        print(f"Loaded {len(edges)} valid lookalike edges")

        if not edges:
            print(
                "No valid edges — ensure species are ingested and lookalikes extracted."
            )
            return

        vocab = load_vocabulary()
        start = time.monotonic()
        best_params = None

        sym_label = " +symmetric" if args.symmetric else ""
        print(f"\nBuilding training data (neg_ratio={args.neg_ratio}{sym_label})...")
        X, y, genera = build_training_data(
            species_data, edges, vocab,
            neg_ratio=args.neg_ratio, seed=args.seed,
            symmetric=args.symmetric,
        )
        print(
            f"Training data: {X.shape[0]} samples "
            f"({int(y.sum())} pos, {int(len(y) - y.sum())} neg), "
            f"{X.shape[1]} features"
        )

        if args.optimize:
            print(f"\nOptimizing with {args.optimize} Optuna trials...")
            opt_result = optimize_ranker(
                X, y, genera,
                n_trials=args.optimize,
                n_folds=args.n_folds,
                seed=args.seed,
            )
            duration = time.monotonic() - start
            best_params = opt_result["best_params"]
            metrics = opt_result["final_metrics"]

            print(f"\nOptimization complete ({duration:.1f}s):")
            print(f"  Best CV AUC: {opt_result['best_auc']:.4f}")
            print("  Best params:")
            for k, v in sorted(best_params.items()):
                print(f"    {k}: {v}")
        else:
            print(f"\nTraining with {args.n_folds}-fold GroupKFold CV...")
            metrics = train_ranker(
                X, y, genera, n_folds=args.n_folds, seed=args.seed,
            )
            duration = time.monotonic() - start

        if args.hard_negatives:
            from similarity.ranker import load_ranker

            print("\nPass 2: rebuilding training data with hard negatives...")
            model_v1 = load_ranker()
            X2, y2, genera2 = build_training_data(
                species_data, edges, vocab,
                neg_ratio=args.neg_ratio, seed=args.seed,
                symmetric=args.symmetric,
                hard_neg_model=model_v1,
            )
            print(
                f"Hard-neg data: {X2.shape[0]} samples "
                f"({int(y2.sum())} pos, {int(len(y2) - y2.sum())} neg)"
            )
            # Clear model cache so the new model can be loaded later
            import similarity.ranker as _ranker_mod
            _ranker_mod._model_cache = None

            metrics = train_ranker(
                X2, y2, genera2, n_folds=args.n_folds, seed=args.seed,
                params=best_params,
            )
            duration = time.monotonic() - start

        print(f"\nCV Results ({duration:.1f}s):")
        print(f"  AUC:      {metrics['auc_mean']:.4f} ± {metrics['auc_std']:.4f}")
        print(
            f"  Logloss:  {metrics['logloss_mean']:.4f} ± {metrics['logloss_std']:.4f}"
        )
        print(f"  Best iteration: {metrics['best_iteration']}")

        # Sanity check: verify model predictions aren't inverted
        import similarity.ranker as _ranker_mod
        _ranker_mod._model_cache = None
        from similarity.ranker import load_ranker as _load

        X_check = X2 if args.hard_negatives else X
        y_check = y2 if args.hard_negatives else y
        _final = _load()
        _preds = _final.predict(X_check)
        _pos_mean = float(_preds[y_check == 1].mean())
        _neg_mean = float(_preds[y_check == 0].mean())
        print(f"  Sanity: mean P(lookalike) — pos={_pos_mean:.3f}, neg={_neg_mean:.3f}")
        if _pos_mean < _neg_mean:
            print("  WARNING: model appears inverted (pos < neg)!")

        if args.feature_importance:
            print("\nTop 20 features by importance (gain):")
            imp = sorted(
                metrics["feature_importance"].items(),
                key=lambda x: x[1],
                reverse=True,
            )
            for i, (feat, score) in enumerate(imp[:20], 1):
                print(f"  {i:>2}. {feat:<40s} {score:>10.1f}")
    finally:
        session.close()


if __name__ == "__main__":
    main()
