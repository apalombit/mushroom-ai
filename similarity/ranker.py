"""Feature-Level Pairwise GBDT Ranker using LightGBM.

Trains a binary classifier on per-field similarity features from known
lookalike pairs. Used as a re-ranker on top of pgvector retrieval.

Feature vector (61 dimensions):
- 47 Jaccard per-field similarities (vocab scalars + colors + booleans)
- 11 numeric per-field comparisons (5 range + 5 single + 1 month overlap)
- 3 gate matches (body_form, hymenium, size_class)
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from pathlib import Path

import numpy as np

from ingestion.rubric import (
    MONTH_OVERLAP_FIELD,
    NUMERIC_RANGE_FIELDS,
    NUMERIC_SINGLE_FIELDS,
    _get_nested,
)
from similarity.jaccard import DEFAULT_JACCARD_WEIGHTS, compute_soft_jaccard
from similarity.numeric import (
    _SIZE_BINS,
    _parse_numeric,
    category_similarity,
    month_overlap,
    range_to_category,
    single_proximity,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical ordered feature list for the GBDT model
# ---------------------------------------------------------------------------

_JACCARD_FIELDS = sorted(DEFAULT_JACCARD_WEIGHTS.keys())
_NUMERIC_RANGE_NAMES = [f"num_{min_f}" for min_f, _ in NUMERIC_RANGE_FIELDS]
_NUMERIC_SINGLE_NAMES = [f"num_{f}" for f, _ in NUMERIC_SINGLE_FIELDS]
_NUMERIC_MONTH_NAME = f"num_{MONTH_OVERLAP_FIELD}"
_GATE_NAMES = ["gate_body_form", "gate_hymenium", "gate_size_class"]

RANKER_FEATURES: list[str] = (
    _JACCARD_FIELDS
    + _NUMERIC_RANGE_NAMES
    + _NUMERIC_SINGLE_NAMES
    + [_NUMERIC_MONTH_NAME]
    + _GATE_NAMES
)

DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "models" / "ranker_model.txt"
)


def _feature_fingerprint() -> str:
    """SHA-256 hex digest of the current RANKER_FEATURES list."""
    return hashlib.sha256(json.dumps(RANKER_FEATURES).encode()).hexdigest()


def _meta_path(model_path: Path) -> Path:
    return model_path.with_suffix(".meta.json")


def _save_meta(model_path: Path) -> None:
    """Write feature fingerprint alongside the model file."""
    meta = {
        "feature_fingerprint": _feature_fingerprint(),
        "feature_count": len(RANKER_FEATURES),
        "features": RANKER_FEATURES,
    }
    _meta_path(model_path).write_text(json.dumps(meta, indent=2))


def _validate_meta(model_path: Path) -> None:
    """Raise RuntimeError if the saved model's feature schema doesn't match current code."""
    mp = _meta_path(model_path)
    if not mp.exists():
        logger.warning(
            "No model metadata at %s — cannot verify feature schema. "
            "Consider retraining: python -m scripts.train_ranker",
            mp,
        )
        return
    meta = json.loads(mp.read_text())
    saved = meta.get("feature_fingerprint", "")
    current = _feature_fingerprint()
    if saved != current:
        raise RuntimeError(
            "Ranker model is stale — feature schema changed since training. "
            "Retrain: python -m scripts.train_ranker --optimize 200 --symmetric --hard-negatives"
        )


DEFAULT_RANKER_PARAMS: dict = {
    "objective": "binary",
    "metric": "binary_logloss",
    "max_depth": 6,
    "num_leaves": 31,
    "min_child_samples": 20,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "learning_rate": 0.05,
    "verbose": -1,
}

_model_cache = None

# Uniform weights for Jaccard — we only use the per-field breakdown
_UNIFORM_JACCARD_WEIGHTS = {f: 1.0 for f in _JACCARD_FIELDS}


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------


def compute_pairwise_features(
    features_a: dict,
    features_b: dict,
    vocab: dict | None = None,
) -> dict[str, float | None]:
    """Compute per-field similarity features between two species.

    Returns a dict mapping each RANKER_FEATURES key to a similarity score
    (float in [0,1]) or None if the field is not computable for this pair.
    """
    result: dict[str, float | None] = {}

    # 1. Jaccard per-field breakdown
    _, field_breakdown = compute_soft_jaccard(
        features_a, features_b, vocab, weights=_UNIFORM_JACCARD_WEIGHTS
    )
    for field in _JACCARD_FIELDS:
        result[field] = field_breakdown.get(field)

    # 2. Numeric range fields
    for (min_field, max_field), feat_name in zip(NUMERIC_RANGE_FIELDS, _NUMERIC_RANGE_NAMES):
        a_min = _parse_numeric(_get_nested(features_a, min_field))
        a_max = _parse_numeric(_get_nested(features_a, max_field))
        b_min = _parse_numeric(_get_nested(features_b, min_field))
        b_max = _parse_numeric(_get_nested(features_b, max_field))
        bins = _SIZE_BINS[min_field]
        cat_a = range_to_category(a_min, a_max, bins)
        cat_b = range_to_category(b_min, b_max, bins)
        result[feat_name] = category_similarity(cat_a, cat_b, bins)

    # 3. Numeric single fields
    for (field, tolerance), feat_name in zip(NUMERIC_SINGLE_FIELDS, _NUMERIC_SINGLE_NAMES):
        a_val = _parse_numeric(_get_nested(features_a, field))
        b_val = _parse_numeric(_get_nested(features_b, field))
        result[feat_name] = single_proximity(a_val, b_val, tolerance)

    # 4. Month overlap
    a_months = _get_nested(features_a, MONTH_OVERLAP_FIELD) or []
    b_months = _get_nested(features_b, MONTH_OVERLAP_FIELD) or []
    result[_NUMERIC_MONTH_NAME] = month_overlap(a_months, b_months)

    # 5. Gate matches
    for gate_name, field_path in [
        ("gate_body_form", "overall_body_form"),
        ("gate_hymenium", "hymenium.type"),
        ("gate_size_class", "overall_size_class"),
    ]:
        val_a = _get_nested(features_a, field_path)
        val_b = _get_nested(features_b, field_path)
        if val_a is None or val_b is None:
            result[gate_name] = None
        else:
            a_str = val_a.lower() if isinstance(val_a, str) else str(val_a).lower()
            b_str = val_b.lower() if isinstance(val_b, str) else str(val_b).lower()
            result[gate_name] = 1.0 if a_str == b_str else 0.0

    return result


# ---------------------------------------------------------------------------
# Training data construction
# ---------------------------------------------------------------------------


def build_training_data(
    species_data: dict[str, dict],
    edges: list[tuple[str, str]],
    vocab: dict,
    neg_ratio: int = 5,
    seed: int = 42,
    symmetric: bool = False,
    hard_neg_model=None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build training data from lookalike edges.

    Returns (X, y, pair_genera) where:
    - X: shape (n_samples, n_features) with NaN for missing values
    - y: shape (n_samples,) with 1 for positive, 0 for negative
    - pair_genera: genus of first species in each pair (for GroupKFold)

    When symmetric=True, adds both (a,b) and (b,a) as positives with negatives
    sampled from each anchor direction.

    When hard_neg_model is provided (a LightGBM Booster), negatives are the
    model's top-scoring non-lookalike species (hard negatives) instead of random.
    """
    all_species = list(species_data.keys())
    edge_set = {tuple(sorted(e)) for e in edges}

    # Pre-compute hard negatives per anchor if model provided
    hard_neg_cache: dict[str, list[str]] = {}
    if hard_neg_model is not None:
        hard_neg_cache = _compute_hard_negatives(
            species_data, edge_set, all_species, vocab, hard_neg_model
        )

    rng = random.Random(seed)
    rows: list[list[float]] = []
    labels: list[int] = []
    genera: list[str] = []

    # Build anchor pairs: [(anchor, partner), ...]
    anchor_pairs: list[tuple[str, str]] = []
    for a, b in edges:
        if a not in species_data or b not in species_data:
            continue
        anchor_pairs.append((a, b))
        if symmetric:
            anchor_pairs.append((b, a))

    for anchor, partner in anchor_pairs:
        # Positive pair
        feats = compute_pairwise_features(
            species_data[anchor]["features"],
            species_data[partner]["features"],
            vocab,
        )
        row = [float("nan") if v is None else v for v in (feats[f] for f in RANKER_FEATURES)]
        rows.append(row)
        labels.append(1)
        genera.append(anchor.split()[0])

        # Negative pairs — mix hard + random when hard neg model is provided
        if hard_neg_model is not None and anchor in hard_neg_cache:
            n_hard = neg_ratio // 2
            n_random = neg_ratio - n_hard
            hard_cands = hard_neg_cache[anchor][:n_hard]
            random_cands = _sample_random_negatives(
                anchor, partner, all_species, edge_set, rng, n_random
            )
            neg_candidates = hard_cands + random_cands
        else:
            neg_candidates = _sample_random_negatives(
                anchor, partner, all_species, edge_set, rng, neg_ratio
            )

        for c in neg_candidates:
            feats = compute_pairwise_features(
                species_data[anchor]["features"],
                species_data[c]["features"],
                vocab,
            )
            row = [float("nan") if v is None else v for v in (feats[f] for f in RANKER_FEATURES)]
            rows.append(row)
            labels.append(0)
            genera.append(anchor.split()[0])

    X = np.array(rows, dtype=np.float64)
    y = np.array(labels, dtype=np.float64)

    # Diagnostic: check feature gap between positives and negatives
    if len(y) > 0 and y.sum() > 0 and (len(y) - y.sum()) > 0:
        with np.errstate(all="ignore"):
            pos_mean = np.nanmean(X[y == 1], axis=0)
            neg_mean = np.nanmean(X[y == 0], axis=0)
            feature_gap = float(np.nanmean(pos_mean - neg_mean))
        logger.info(
            "Training data: %d pos, %d neg, avg feature gap=%.4f",
            int(y.sum()),
            int(len(y) - y.sum()),
            feature_gap,
        )

    return X, y, genera


def _sample_random_negatives(
    anchor: str,
    partner: str,
    all_species: list[str],
    edge_set: set[tuple[str, str]],
    rng: random.Random,
    neg_ratio: int,
) -> list[str]:
    """Sample random non-lookalike species as negatives for an anchor."""
    negatives: list[str] = []
    attempts = 0
    while len(negatives) < neg_ratio and attempts < neg_ratio * 10:
        c = rng.choice(all_species)
        attempts += 1
        if c == anchor or c == partner:
            continue
        if tuple(sorted([anchor, c])) in edge_set:
            continue
        negatives.append(c)
    return negatives


def _compute_hard_negatives(
    species_data: dict[str, dict],
    edge_set: set[tuple[str, str]],
    all_species: list[str],
    vocab: dict,
    model,
    max_negatives: int = 20,
) -> dict[str, list[str]]:
    """For each species, score all non-lookalikes and return top-K hardest."""
    cache: dict[str, list[str]] = {}
    for anchor in species_data:
        # Collect non-lookalike candidates
        candidates = [
            s for s in all_species if s != anchor and tuple(sorted([anchor, s])) not in edge_set
        ]
        if not candidates:
            continue

        # Score all candidates
        cand_features = [species_data[s]["features"] for s in candidates]
        scores = score_candidates(
            species_data[anchor]["features"], cand_features, vocab, model=model
        )

        # Sort by score descending (highest = hardest negatives)
        scored = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        cache[anchor] = [name for name, _ in scored[:max_negatives]]

    return cache


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------


def train_ranker(
    X: np.ndarray,
    y: np.ndarray,
    pair_genera: list[str],
    n_folds: int = 5,
    seed: int = 42,
    model_path: Path | None = None,
    params: dict | None = None,
    save_model: bool = True,
    num_boost_round: int = 500,
    early_stopping_rounds: int = 50,
) -> dict:
    """Train LightGBM ranker with GroupKFold cross-validation.

    Returns dict with CV metrics and feature importance.
    When save_model=False, skips final full-data training and model save (for Optuna).
    """
    import lightgbm as lgb
    from sklearn.metrics import log_loss, roc_auc_score
    from sklearn.model_selection import GroupKFold

    if model_path is None:
        model_path = DEFAULT_MODEL_PATH

    lgb_params = {**DEFAULT_RANKER_PARAMS, "seed": seed}
    if params:
        lgb_params.update(params)

    gkf = GroupKFold(n_splits=n_folds)
    groups = np.array(pair_genera)

    fold_aucs: list[float] = []
    fold_logloss: list[float] = []
    best_iterations: list[int] = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_data = lgb.Dataset(X_train, label=y_train, feature_name=RANKER_FEATURES)
        val_data = lgb.Dataset(
            X_val, label=y_val, feature_name=RANKER_FEATURES, reference=train_data
        )

        model = lgb.train(
            lgb_params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=[val_data],
            callbacks=[
                lgb.early_stopping(early_stopping_rounds),
                lgb.log_evaluation(0),
            ],
        )

        best_iterations.append(model.best_iteration)

        y_pred = model.predict(X_val)
        fold_aucs.append(roc_auc_score(y_val, y_pred))
        fold_logloss.append(log_loss(y_val, y_pred))

        logger.info(
            "Fold %d: AUC=%.4f, logloss=%.4f, best_iter=%d",
            fold + 1,
            fold_aucs[-1],
            fold_logloss[-1],
            model.best_iteration,
        )

    avg_best = int(np.mean(best_iterations))

    result = {
        "auc_mean": float(np.mean(fold_aucs)),
        "auc_std": float(np.std(fold_aucs)),
        "logloss_mean": float(np.mean(fold_logloss)),
        "logloss_std": float(np.std(fold_logloss)),
        "best_iteration": avg_best,
        "feature_importance": {},
        "n_samples": len(y),
        "n_positives": int(y.sum()),
        "n_features": X.shape[1],
    }

    if save_model:
        # Train final model on all data and save
        full_data = lgb.Dataset(X, label=y, feature_name=RANKER_FEATURES)
        final_model = lgb.train(lgb_params, full_data, num_boost_round=avg_best)

        model_path.parent.mkdir(parents=True, exist_ok=True)
        final_model.save_model(str(model_path))
        _save_meta(model_path)
        logger.info("Model saved to %s", model_path)

        result["feature_importance"] = dict(
            zip(
                RANKER_FEATURES,
                final_model.feature_importance(importance_type="gain"),
            )
        )

    return result


# ---------------------------------------------------------------------------
# Hyperparameter optimization
# ---------------------------------------------------------------------------


def optimize_ranker(
    X: np.ndarray,
    y: np.ndarray,
    pair_genera: list[str],
    n_trials: int = 100,
    n_folds: int = 5,
    seed: int = 42,
) -> dict:
    """Optimize LightGBM hyperparameters with Optuna TPE sampler.

    Returns {"best_params": dict, "best_auc": float, "study": optuna.Study}.
    The best model is saved to DEFAULT_MODEL_PATH.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        trial_params = {
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "num_leaves": trial.suggest_int("num_leaves", 8, 128),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.3, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 10.0),
        }
        metrics = train_ranker(
            X,
            y,
            pair_genera,
            n_folds=n_folds,
            seed=seed,
            params=trial_params,
            save_model=False,
        )
        return metrics["auc_mean"]

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_auc = study.best_value

    # Retrain final model with best params and save
    logger.info("Retraining final model with best params (AUC=%.4f)", best_auc)
    final_metrics = train_ranker(
        X,
        y,
        pair_genera,
        n_folds=n_folds,
        seed=seed,
        params=best_params,
        save_model=True,
    )

    return {
        "best_params": best_params,
        "best_auc": best_auc,
        "final_metrics": final_metrics,
        "study": study,
    }


# ---------------------------------------------------------------------------
# Model loading and scoring
# ---------------------------------------------------------------------------


def load_ranker(model_path: Path | None = None):
    """Load trained LightGBM model from disk, cached in module variable."""
    global _model_cache
    if _model_cache is not None:
        return _model_cache

    import lightgbm as lgb

    path = model_path or DEFAULT_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Ranker model not found at {path}. Train it first: python -m scripts.train_ranker"
        )
    _validate_meta(path)
    _model_cache = lgb.Booster(model_file=str(path))
    return _model_cache


def score_candidates(
    query_features: dict,
    candidate_features: list[dict],
    vocab: dict,
    model=None,
) -> list[float]:
    """Score candidates using the GBDT model.

    Returns list of P(lookalike) scores, one per candidate.
    """
    if model is None:
        model = load_ranker()

    n = len(candidate_features)
    if n == 0:
        return []

    X = np.full((n, len(RANKER_FEATURES)), np.nan, dtype=np.float64)
    for i, cand_feats in enumerate(candidate_features):
        feats = compute_pairwise_features(query_features, cand_feats, vocab)
        for j, f in enumerate(RANKER_FEATURES):
            val = feats.get(f)
            if val is not None:
                X[i, j] = val

    return model.predict(X).tolist()
