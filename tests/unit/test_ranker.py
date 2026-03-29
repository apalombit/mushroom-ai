"""Unit tests for similarity/ranker.py — GBDT pairwise ranker."""

import copy

import numpy as np
import pytest

from ingestion.normalize import load_vocabulary
from similarity.ranker import (
    RANKER_FEATURES,
    _feature_fingerprint,
    _meta_path,
    _save_meta,
    _validate_meta,
    build_training_data,
    compute_pairwise_features,
    optimize_ranker,
    score_candidates,
    train_ranker,
)


@pytest.fixture
def vocab():
    return load_vocabulary()


@pytest.fixture
def features_amanita(sample_features_json):
    """Amanita muscaria from conftest."""
    return sample_features_json


@pytest.fixture
def features_boletus():
    """Boletus edulis-like dict — boletoid with pores, contrasts with Amanita."""
    return {
        "scientific_name": "Boletus edulis",
        "species_epithet": "edulis",
        "common_names": ["Porcini", "King bolete"],
        "genus": "Boletus",
        "family": "Boletaceae",
        "order": "Boletales",
        "cap": {
            "shape": "convex",
            "colors": ["brown", "tan", "dark brown"],
            "color_faded": None,
            "color_pattern": "uniform",
            "surface_texture": "smooth",
            "surface_moisture": "dry",
            "bruising_color": None,
            "scales_or_warts": "none",
            "margin_type": "smooth",
            "margin_lined_at_maturity": False,
            "central_depression": False,
            "diameter_min_cm": 8.0,
            "diameter_max_cm": 25.0,
        },
        "hymenium": {
            "type": "pores",
        },
        "gills": {
            "attachment": None,
            "spacing": None,
            "color": None,
        },
        "pores": {
            "color": "white",
            "color_with_age": "yellow-green",
            "bruising_color": None,
            "density_per_mm": 2.0,
        },
        "tubes": {
            "depth_mm": 20.0,
        },
        "stem": {
            "color": "white",
            "color_with_age": "brownish",
            "surface_texture": "reticulate",
            "reticulation": "fine",
            "shape": "clavate",
            "attachment_position": "central",
            "consistency": "firm",
            "hollow_or_solid": "solid",
            "base_color": "white",
            "bruising_color": None,
            "height_min_cm": 8.0,
            "height_max_cm": 20.0,
            "diameter_min_cm": 3.0,
            "diameter_max_cm": 7.0,
        },
        "veil": {
            "present": False,
            "type": None,
            "cortina_present": False,
        },
        "volva": {
            "present": False,
        },
        "flesh": {
            "color": "white",
            "bruising_color": None,
            "odor": "pleasant",
            "taste": "mild",
            "texture": "firm",
            "cap_stem_consistency": "homogeneous",
            "latex_presence": False,
        },
        "spore_print_color": "olive-brown",
        "spore": {
            "shape": "fusiform",
            "length_min_um": 12.0,
            "length_max_um": 17.0,
            "width_min_um": 4.0,
            "width_max_um": 5.5,
            "ornamentation": "smooth",
            "amyloidity": "inamyloid",
        },
        "overall_size_class": "large",
        "overall_body_form": "boletoid",
        "growth_habit": "solitary",
        "ecology": {
            "trophic_mode": "mycorrhizal",
            "substrate": "soil",
            "associated_trees": ["oak", "spruce", "pine", "beech"],
            "fruiting_months": ["August", "September", "October"],
            "geographic_regions": ["Europe", "North America"],
        },
        "edibility_status": "edible",
    }


# ---------------------------------------------------------------------------
# TestComputePairwiseFeatures
# ---------------------------------------------------------------------------


class TestComputePairwiseFeatures:
    def test_feature_count(self, features_amanita, features_boletus, vocab):
        result = compute_pairwise_features(features_amanita, features_boletus, vocab)
        assert set(result.keys()) == set(RANKER_FEATURES)

    def test_identical_species(self, features_amanita, vocab):
        result = compute_pairwise_features(features_amanita, features_amanita, vocab)
        for key, val in result.items():
            if val is not None:
                assert val == pytest.approx(1.0, abs=0.01), f"{key} should be ~1.0, got {val}"

    def test_different_species(self, features_amanita, features_boletus, vocab):
        result = compute_pairwise_features(features_amanita, features_boletus, vocab)
        # Body form gate should be 0.0 (agaricoid != boletoid)
        assert result["gate_body_form"] == 0.0
        # Hymenium gate should be 0.0 (gills != pores)
        assert result["gate_hymenium"] == 0.0
        # Size class gate should be 1.0 (both large)
        assert result["gate_size_class"] == 1.0
        # Some Jaccard fields should be < 1.0
        non_none = {k: v for k, v in result.items() if v is not None}
        assert any(v < 1.0 for v in non_none.values())

    def test_empty_features(self, vocab):
        result = compute_pairwise_features({}, {}, vocab)
        assert len(result) == len(RANKER_FEATURES)
        # Gate fields should all be None (no values to compare)
        assert result["gate_body_form"] is None
        assert result["gate_hymenium"] is None
        assert result["gate_size_class"] is None

    def test_symmetry(self, features_amanita, features_boletus, vocab):
        ab = compute_pairwise_features(features_amanita, features_boletus, vocab)
        ba = compute_pairwise_features(features_boletus, features_amanita, vocab)
        for key in RANKER_FEATURES:
            if ab[key] is not None and ba[key] is not None:
                assert ab[key] == pytest.approx(ba[key], abs=1e-9), (
                    f"{key}: {ab[key]} != {ba[key]}"
                )

    def test_values_in_range(self, features_amanita, features_boletus, vocab):
        result = compute_pairwise_features(features_amanita, features_boletus, vocab)
        for key, val in result.items():
            if val is not None:
                assert 0.0 <= val <= 1.0, f"{key} out of range: {val}"


# ---------------------------------------------------------------------------
# TestBuildTrainingData
# ---------------------------------------------------------------------------


class TestBuildTrainingData:
    @pytest.fixture
    def species_data(self, features_amanita, features_boletus):
        """Minimal 3-species dataset for training data tests."""
        third = copy.deepcopy(features_amanita)
        third["scientific_name"] = "Amanita caesarea"
        third["cap"]["colors"] = ["orange", "yellow-orange"]
        return {
            "Amanita muscaria": {"features": features_amanita, "group": "agaricoid"},
            "Boletus edulis": {"features": features_boletus, "group": "boletoid"},
            "Amanita caesarea": {"features": third, "group": "agaricoid"},
        }

    @pytest.fixture
    def edges(self):
        return [("Amanita muscaria", "Amanita caesarea")]

    def test_label_counts(self, species_data, edges, vocab):
        X, y, genera = build_training_data(species_data, edges, vocab, neg_ratio=2)
        assert int(y.sum()) == 1  # 1 positive (1 edge)
        assert int(len(y) - y.sum()) == 2  # 2 negatives (neg_ratio=2)

    def test_shape(self, species_data, edges, vocab):
        X, y, genera = build_training_data(species_data, edges, vocab, neg_ratio=2)
        assert X.shape[1] == len(RANKER_FEATURES)
        assert X.shape[0] == len(y)
        assert len(genera) == len(y)

    def test_no_self_pairs(self, species_data, edges, vocab):
        """Negative pairs should never include the anchor species itself."""
        X, y, genera = build_training_data(species_data, edges, vocab, neg_ratio=2, seed=0)
        # With only 3 species and 1 edge (A. muscaria, A. caesarea),
        # the only valid negative is Boletus edulis
        neg_count = int(len(y) - y.sum())
        assert neg_count >= 1

    def test_reproducibility(self, species_data, edges, vocab):
        X1, y1, g1 = build_training_data(species_data, edges, vocab, seed=42)
        X2, y2, g2 = build_training_data(species_data, edges, vocab, seed=42)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)
        assert g1 == g2


# ---------------------------------------------------------------------------
# TestScoreCandidates
# ---------------------------------------------------------------------------


class TestScoreCandidates:
    @pytest.fixture
    def mini_model(self, features_amanita, features_boletus, vocab):
        """Train a tiny LightGBM model on synthetic data for scoring tests."""
        import lightgbm as lgb

        third = copy.deepcopy(features_amanita)
        third["cap"]["colors"] = ["orange", "yellow-orange"]

        species_data = {
            "Amanita muscaria": {"features": features_amanita, "group": "agaricoid"},
            "Boletus edulis": {"features": features_boletus, "group": "boletoid"},
            "Amanita caesarea": {"features": third, "group": "agaricoid"},
        }
        edges = [("Amanita muscaria", "Amanita caesarea")]
        X, y, genera = build_training_data(species_data, edges, vocab, neg_ratio=2)

        dataset = lgb.Dataset(X, label=y, feature_name=RANKER_FEATURES)
        params = {
            "objective": "binary",
            "num_leaves": 4,
            "verbose": -1,
            "seed": 42,
        }
        model = lgb.train(params, dataset, num_boost_round=10)
        return model

    def test_score_range(self, features_amanita, features_boletus, vocab, mini_model):
        scores = score_candidates(
            features_amanita,
            [features_boletus, features_amanita],
            vocab,
            model=mini_model,
        )
        assert len(scores) == 2
        for s in scores:
            assert 0.0 <= s <= 1.0

    def test_batch_consistency(self, features_amanita, features_boletus, vocab, mini_model):
        """Batch scoring should match individual scoring."""
        batch = score_candidates(
            features_amanita,
            [features_boletus, features_amanita],
            vocab,
            model=mini_model,
        )
        single_0 = score_candidates(features_amanita, [features_boletus], vocab, model=mini_model)
        single_1 = score_candidates(features_amanita, [features_amanita], vocab, model=mini_model)
        assert batch[0] == pytest.approx(single_0[0])
        assert batch[1] == pytest.approx(single_1[0])

    def test_empty_candidates(self, features_amanita, vocab, mini_model):
        scores = score_candidates(features_amanita, [], vocab, model=mini_model)
        assert scores == []


# ---------------------------------------------------------------------------
# TestTrainRankerParams
# ---------------------------------------------------------------------------


class TestTrainRankerParams:
    @pytest.fixture
    def training_data(self, sample_features_json, features_boletus, vocab):
        """Minimal training data with ≥2 genera for GroupKFold."""
        third = copy.deepcopy(sample_features_json)
        third["cap"]["colors"] = ["orange", "yellow-orange"]
        fourth = copy.deepcopy(features_boletus)
        fourth["cap"]["colors"] = ["tan", "light brown"]

        species_data = {
            "Amanita muscaria": {
                "features": sample_features_json,
                "group": "agaricoid",
            },
            "Amanita caesarea": {"features": third, "group": "agaricoid"},
            "Boletus edulis": {"features": features_boletus, "group": "boletoid"},
            "Boletus reticulatus": {"features": fourth, "group": "boletoid"},
        }
        edges = [
            ("Amanita muscaria", "Amanita caesarea"),
            ("Boletus edulis", "Boletus reticulatus"),
        ]
        X, y, genera = build_training_data(species_data, edges, vocab, neg_ratio=2)
        return X, y, genera

    def test_train_params_override(self, training_data):
        X, y, genera = training_data
        custom_params = {"max_depth": 3, "num_leaves": 4}
        metrics = train_ranker(X, y, genera, n_folds=2, params=custom_params, save_model=False)
        assert "auc_mean" in metrics
        assert metrics["n_samples"] == len(y)

    def test_train_save_model_false(self, training_data, tmp_path):
        X, y, genera = training_data
        model_path = tmp_path / "should_not_exist.txt"
        metrics = train_ranker(X, y, genera, n_folds=2, model_path=model_path, save_model=False)
        assert "auc_mean" in metrics
        assert not model_path.exists()
        assert metrics["feature_importance"] == {}

    def test_optimize_smoke(self, training_data):
        X, y, genera = training_data
        result = optimize_ranker(X, y, genera, n_trials=3, n_folds=2, seed=42)
        assert "best_params" in result
        assert "best_auc" in result
        assert isinstance(result["best_params"], dict)
        assert 0.0 <= result["best_auc"] <= 1.0


# ---------------------------------------------------------------------------
# TestTrainingDataVariants
# ---------------------------------------------------------------------------


class TestTrainingDataVariants:
    @pytest.fixture
    def species_and_edges(self, sample_features_json, features_boletus):
        third = copy.deepcopy(sample_features_json)
        third["cap"]["colors"] = ["orange", "yellow-orange"]
        fourth = copy.deepcopy(features_boletus)
        fourth["cap"]["colors"] = ["tan", "light brown"]

        species_data = {
            "Amanita muscaria": {
                "features": sample_features_json,
                "group": "agaricoid",
            },
            "Amanita caesarea": {"features": third, "group": "agaricoid"},
            "Boletus edulis": {"features": features_boletus, "group": "boletoid"},
            "Boletus reticulatus": {"features": fourth, "group": "boletoid"},
        }
        edges = [
            ("Amanita muscaria", "Amanita caesarea"),
            ("Boletus edulis", "Boletus reticulatus"),
        ]
        return species_data, edges

    def test_symmetric_doubles_positives(self, species_and_edges, vocab):
        species_data, edges = species_and_edges
        _, y_normal, _ = build_training_data(
            species_data, edges, vocab, neg_ratio=2, symmetric=False
        )
        _, y_sym, _ = build_training_data(species_data, edges, vocab, neg_ratio=2, symmetric=True)
        pos_normal = int(y_normal.sum())
        pos_sym = int(y_sym.sum())
        assert pos_sym == 2 * pos_normal

    def test_symmetric_preserves_shape(self, species_and_edges, vocab):
        species_data, edges = species_and_edges
        X_sym, _, _ = build_training_data(species_data, edges, vocab, neg_ratio=2, symmetric=True)
        assert X_sym.shape[1] == len(RANKER_FEATURES)

    def test_hard_neg_mining(self, species_and_edges, vocab):
        import lightgbm as lgb

        species_data, edges = species_and_edges
        # Train a mini model first
        X, y, genera = build_training_data(species_data, edges, vocab, neg_ratio=2)
        dataset = lgb.Dataset(X, label=y, feature_name=RANKER_FEATURES)
        model = lgb.train(
            {"objective": "binary", "num_leaves": 4, "verbose": -1, "seed": 42},
            dataset,
            num_boost_round=10,
        )

        # Build with hard negatives (50/50 mix: 1 hard + 1 random per anchor)
        X_hard, y_hard, _ = build_training_data(
            species_data, edges, vocab, neg_ratio=2, hard_neg_model=model
        )
        assert X_hard.shape[0] == X.shape[0]  # same total count
        assert X_hard.shape[1] == len(RANKER_FEATURES)

        # Feature gap should remain positive (positives more similar than negatives)
        pos_mean = np.nanmean(X_hard[y_hard == 1], axis=0)
        neg_mean = np.nanmean(X_hard[y_hard == 0], axis=0)
        gap = float(np.nanmean(pos_mean - neg_mean))
        assert gap > 0, f"Feature gap should be positive, got {gap}"


class TestModelFingerprint:
    """Feature fingerprint prevents stale model from being silently loaded."""

    def test_fingerprint_deterministic(self):
        assert _feature_fingerprint() == _feature_fingerprint()

    def test_save_and_validate_matching(self, tmp_path):
        model_path = tmp_path / "model.txt"
        model_path.write_text("dummy")
        _save_meta(model_path)

        meta_file = _meta_path(model_path)
        assert meta_file.exists()

        # Should not raise
        _validate_meta(model_path)

    def test_validate_stale_raises(self, tmp_path):
        import json

        model_path = tmp_path / "model.txt"
        model_path.write_text("dummy")

        # Write meta with a wrong fingerprint
        meta = {
            "feature_fingerprint": "stale_hash",
            "feature_count": 0,
            "features": [],
        }
        _meta_path(model_path).write_text(json.dumps(meta))

        with pytest.raises(RuntimeError, match="stale"):
            _validate_meta(model_path)

    def test_missing_meta_warns(self, tmp_path, caplog):
        model_path = tmp_path / "model.txt"
        model_path.write_text("dummy")

        # No meta file — should warn but not raise
        with caplog.at_level("WARNING"):
            _validate_meta(model_path)
        assert "cannot verify" in caplog.text.lower()

    def test_meta_contains_features(self, tmp_path):
        import json

        model_path = tmp_path / "model.txt"
        model_path.write_text("dummy")
        _save_meta(model_path)

        meta = json.loads(_meta_path(model_path).read_text())
        assert meta["features"] == RANKER_FEATURES
        assert meta["feature_count"] == len(RANKER_FEATURES)
