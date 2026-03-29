"""Unit tests for similarity/jaccard.py — soft-Jaccard scorer."""

import copy

import pytest

from ingestion.normalize import load_vocabulary
from similarity.jaccard import (
    DEFAULT_JACCARD_WEIGHTS,
    _color_list_similarity,
    _lookup_matrix_score,
    compute_soft_jaccard,
)


@pytest.fixture
def vocab():
    return load_vocabulary()


@pytest.fixture
def features_a(sample_features_json):
    """Amanita muscaria from conftest."""
    return sample_features_json


@pytest.fixture
def features_b(sample_features_json):
    """Near-identical copy (same species) — should score ~1.0."""
    return copy.deepcopy(sample_features_json)


@pytest.fixture
def features_unrelated():
    """Boletus edulis — a bolete, very different from Amanita muscaria."""
    return {
        "hymenium": {"type": "pores"},
        "overall_body_form": "boletoid",
        "overall_size_class": "large",
        "growth_habit": "solitary",
        "cap": {
            "shape": "convex",
            "colors": ["brown", "tan"],
            "color_pattern": "uniform",
            "surface_texture": "smooth",
            "surface_moisture": "subviscid",
            "scales_or_warts": "none",
            "margin_type": "entire",
            "central_depression": False,
            "margin_lined_at_maturity": False,
        },
        "gills": {
            "attachment": "adnate",
            "spacing": "crowded",
            "color": "white",
            "thickness": "thin",
            "texture": "normal",
        },
        "pores": {
            "color": "white",
            "bruising_color": "none",
        },
        "stem": {
            "color": "white",
            "surface_texture": "reticulate",
            "reticulation": "full",
            "shape": "clavate",
            "attachment_position": "central",
            "consistency": "firm",
            "hollow_or_solid": "solid",
        },
        "veil": {
            "present": False,
            "cortina_present": False,
        },
        "volva": {"present": False},
        "flesh": {
            "color": "white",
            "bruising_color": "none",
            "odor": "pleasant",
            "taste": "nutty",
            "texture": "firm",
            "quantity": "thick",
            "hyphal_structure": "homoiomerous",
            "cap_stem_consistency": "homogeneous",
            "latex_presence": False,
        },
        "spore_print_color": "olive",
        "spore": {
            "shape": "fusiform",
            "ornamentation": "smooth",
            "amyloidity": "inamyloid",
        },
        "ecology": {
            "trophic_mode": "mycorrhizal",
            "substrate": "soil",
            "growth_position": "terrestrial",
        },
    }


class TestLookupMatrixScore:
    def test_identical_terms(self):
        assert _lookup_matrix_score({}, "gills", "gills") == 1.0

    def test_found_forward(self):
        matrix = {"gills-ridges": 0.7}
        assert _lookup_matrix_score(matrix, "gills", "ridges") == 0.7

    def test_found_reverse(self):
        matrix = {"gills-ridges": 0.7}
        assert _lookup_matrix_score(matrix, "ridges", "gills") == 0.7

    def test_not_found(self):
        matrix = {"gills-ridges": 0.7}
        assert _lookup_matrix_score(matrix, "gills", "pores") == 0.0


class TestColorListSimilarity:
    def test_identical_colors(self):
        matrix = {"white-cream": 0.8}
        aliases = {}
        score = _color_list_similarity(["white"], ["white"], matrix, aliases)
        assert score == 1.0

    def test_similar_colors(self):
        matrix = {"white-cream": 0.8}
        aliases = {}
        score = _color_list_similarity(["white"], ["cream"], matrix, aliases)
        assert score == 0.8

    def test_different_length_lists(self):
        matrix = {"red-orange": 0.6, "red-brown": 0.3, "white-cream": 0.8}
        aliases = {"scarlet red": "red"}
        a = ["scarlet red", "white"]
        b = ["orange"]
        score = _color_list_similarity(a, b, matrix, aliases)
        # A→B: max(red→orange=0.6, white→orange=0.0) → avg = 0.3
        # B→A: max(orange→red=0.6, orange→white=0.0) → 0.6
        # Mean: (0.3 + 0.6) / 2 = 0.45
        assert abs(score - 0.45) < 0.01

    def test_alias_normalization(self):
        matrix = {"white-cream": 0.8}
        aliases = {"ivory": "cream"}
        score = _color_list_similarity(["white"], ["ivory"], matrix, aliases)
        assert score == 0.8

    def test_empty_list(self):
        score = _color_list_similarity([], ["white"], {}, {})
        assert score == 0.0


class TestComputeSoftJaccard:
    def test_identical_features(self, features_a, features_b, vocab):
        """Identical features → score near 1.0."""
        score, breakdown = compute_soft_jaccard(features_a, features_b, vocab)
        assert score == pytest.approx(1.0)
        assert all(v == pytest.approx(1.0) for v in breakdown.values())

    def test_unrelated_species(self, features_a, features_unrelated, vocab):
        """Very different species → low score."""
        score, breakdown = compute_soft_jaccard(features_a, features_unrelated, vocab)
        assert score < 0.7

    def test_partial_overlap(self, features_a, features_unrelated, vocab):
        """Partially modified features → intermediate score."""
        partial = copy.deepcopy(features_a)
        # Change a few fields to differ
        partial["hymenium"]["type"] = "pores"
        partial["overall_body_form"] = "boletoid"
        partial["gills"]["attachment"] = "adnate"

        score, _ = compute_soft_jaccard(features_a, partial, vocab)
        assert 0.3 < score < 1.0

    def test_missing_fields_skipped(self, vocab):
        """Fields absent from one dict are skipped, not penalized."""
        a = {"hymenium": {"type": "gills"}, "cap": {"shape": "convex"}}
        b = {"hymenium": {"type": "gills"}}  # no cap

        score, breakdown = compute_soft_jaccard(a, b, vocab)
        # Only hymenium.type should be compared
        assert "hymenium.type" in breakdown
        assert "cap.shape" not in breakdown
        assert score == 1.0

    def test_breakdown_dict_populated(self, features_a, features_unrelated, vocab):
        """Per-field breakdown has entries for all computable fields."""
        _, breakdown = compute_soft_jaccard(features_a, features_unrelated, vocab)
        assert len(breakdown) > 0
        assert all(isinstance(v, float) for v in breakdown.values())
        assert all(0.0 <= v <= 1.0 for v in breakdown.values())

    def test_boolean_fields(self, vocab):
        """Boolean fields: same → 1.0, different → 0.0."""
        a = {"veil": {"present": True}, "volva": {"present": True}}
        b = {"veil": {"present": True}, "volva": {"present": False}}

        _, breakdown = compute_soft_jaccard(a, b, vocab)
        assert breakdown["veil.present"] == 1.0
        assert breakdown["volva.present"] == 0.0

    def test_color_field_string(self, vocab):
        """String color fields use color_palette similarity."""
        a = {"gills": {"color": "white"}}
        b = {"gills": {"color": "cream"}}

        _, breakdown = compute_soft_jaccard(a, b, vocab)
        assert breakdown["gills.color"] == pytest.approx(0.8)

    def test_color_field_list(self, vocab):
        """List color fields (cap.colors) use best-match strategy."""
        a = {"cap": {"colors": ["red", "orange"]}}
        b = {"cap": {"colors": ["red"]}}

        _, breakdown = compute_soft_jaccard(a, b, vocab)
        assert "cap.colors" in breakdown
        # red↔red = 1.0, orange↔red = 0.6
        # A→B: avg(max(red→red)=1.0, max(orange→red)=0.6) = 0.8
        # B→A: avg(max(red→red)=1.0) = 1.0
        # Mean: 0.9
        assert breakdown["cap.colors"] == pytest.approx(0.9)

    def test_custom_weights(self, vocab):
        """Custom per-field weights change the overall score."""
        a = {"hymenium": {"type": "gills"}, "cap": {"shape": "convex"}}
        b = {"hymenium": {"type": "ridges"}, "cap": {"shape": "convex"}}

        score_uniform, _ = compute_soft_jaccard(a, b, vocab)
        # Heavily weight hymenium (where they differ)
        score_weighted, _ = compute_soft_jaccard(
            a, b, vocab, weights={"hymenium.type": 10.0, "cap.shape": 1.0}
        )
        assert score_weighted < score_uniform

    def test_empty_features(self, vocab):
        """Empty feature dicts → score 0.0."""
        score, breakdown = compute_soft_jaccard({}, {}, vocab)
        assert score == 0.0
        assert breakdown == {}

    def test_vocab_auto_loaded(self, features_a, features_b):
        """Passing vocab=None auto-loads the vocabulary."""
        score, _ = compute_soft_jaccard(features_a, features_b, vocab=None)
        assert score == pytest.approx(1.0)

    def test_default_weights_type(self):
        """DEFAULT_JACCARD_WEIGHTS is a dict (possibly empty until optimized)."""
        assert isinstance(DEFAULT_JACCARD_WEIGHTS, dict)

    def test_identical_features_with_default_weights(self, features_a, features_b, vocab):
        """Identical features still score 1.0 with default weights."""
        score, _ = compute_soft_jaccard(features_a, features_b, vocab, weights=None)
        assert score == pytest.approx(1.0)
