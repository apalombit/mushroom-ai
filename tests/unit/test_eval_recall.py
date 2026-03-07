"""Unit tests for evaluation/recall.py — mocked DB and search."""

from unittest.mock import MagicMock, patch

from evaluation.recall import BENCHMARK_CONFIGS, EvalResult, PairResult, evaluate_recall


def _make_pair(a: str, b: str) -> MagicMock:
    pair = MagicMock()
    pair.species_a = a
    pair.species_b = b
    return pair


def _make_candidates(*names: str) -> list[dict]:
    """Build a candidate list with dummy similarity scores."""
    return [
        {
            "scientific_name": n,
            "similarity_overall": round(0.9 - i * 0.05, 4),
            "similarity_macro_visual": 0.85,
            "similarity_structural": 0.80,
            "similarity_flesh_sensory": 0.60,
            "similarity_microscopic_lab": 0.50,
            "similarity_ecological": 0.90,
            "similarity_taxonomic": 0.70,
            "similarity_numeric": 0.65,
        }
        for i, n in enumerate(names)
    ]


@patch("evaluation.recall.search_lookalikes")
def test_hit_at_rank_1(mock_search):
    """Target is the top result → rank 1."""
    mock_search.return_value = (MagicMock(), _make_candidates("B", "C", "D"))
    pair = _make_pair("A", "B")

    result = evaluate_recall(MagicMock(), [pair], top_k=5)

    # A→B should be rank 1, B→A depends on mock (B is not in candidates for B query)
    a_to_b = result.pair_results[0]
    assert a_to_b.query == "A"
    assert a_to_b.target == "B"
    assert a_to_b.rank == 1


@patch("evaluation.recall.search_lookalikes")
def test_hit_at_rank_3(mock_search):
    """Target is at position 3."""
    mock_search.return_value = (MagicMock(), _make_candidates("X", "Y", "B"))
    pair = _make_pair("A", "B")

    result = evaluate_recall(MagicMock(), [pair], top_k=5)

    a_to_b = result.pair_results[0]
    assert a_to_b.rank == 3


@patch("evaluation.recall.search_lookalikes")
def test_miss(mock_search):
    """Target not in candidates → rank is None."""
    mock_search.return_value = (MagicMock(), _make_candidates("X", "Y", "Z"))
    pair = _make_pair("A", "B")

    result = evaluate_recall(MagicMock(), [pair], top_k=5)

    a_to_b = result.pair_results[0]
    assert a_to_b.rank is None


@patch("evaluation.recall.search_lookalikes")
def test_error_handling(mock_search):
    """ValueError from search → recorded as error."""
    mock_search.side_effect = ValueError("Species not found: 'A'")
    pair = _make_pair("A", "B")

    result = evaluate_recall(MagicMock(), [pair], top_k=5)

    assert result.errors == 2  # both directions fail
    assert result.pair_results[0].error is not None


@patch("evaluation.recall.search_lookalikes")
def test_bidirectional(mock_search):
    """Each pair produces two results (A→B and B→A)."""
    mock_search.return_value = (MagicMock(), _make_candidates("other"))
    pairs = [_make_pair("A", "B"), _make_pair("C", "D")]

    result = evaluate_recall(MagicMock(), pairs, top_k=5)

    assert result.total == 4
    queries = [(r.query, r.target) for r in result.pair_results]
    assert ("A", "B") in queries
    assert ("B", "A") in queries
    assert ("C", "D") in queries
    assert ("D", "C") in queries


@patch("evaluation.recall.search_lookalikes")
def test_recall_computation(mock_search):
    """Recall@K computed correctly from hits."""
    # First call: target at rank 1, second call: miss
    mock_search.side_effect = [
        (MagicMock(), _make_candidates("B", "X")),
        (MagicMock(), _make_candidates("X", "Y")),
    ]
    pair = _make_pair("A", "B")

    result = evaluate_recall(MagicMock(), [pair], top_k=5)

    assert result.recall_at(1) == 0.5  # 1 hit out of 2
    assert result.recall_at(5) == 0.5
    assert result.hits_at(1) == 1


def test_eval_result_empty():
    """EvalResult with no pairs returns 0 recall."""
    result = EvalResult(pair_results=[], top_k=5)
    assert result.recall_at(1) == 0.0
    assert result.total == 0
    assert result.errors == 0


def test_pair_result_dataclass():
    """PairResult stores all 7 similarity fields."""
    r = PairResult(
        query="A",
        target="B",
        rank=2,
        sim_overall=0.85,
        sim_macro_visual=0.90,
        sim_structural=0.80,
        sim_flesh_sensory=0.60,
        sim_microscopic_lab=0.50,
        sim_ecological=0.75,
        sim_taxonomic=0.70,
        sim_numeric=0.65,
    )
    assert r.rank == 2
    assert r.sim_numeric == 0.65
    assert r.error is None


def test_benchmark_configs():
    """BENCHMARK_CONFIGS has >= 3 entries, each is a dict with weight keys."""
    assert len(BENCHMARK_CONFIGS) >= 3
    expected_keys = {
        "macro_visual",
        "structural",
        "flesh_sensory",
        "microscopic_lab",
        "ecological",
        "taxonomic",
        "numeric",
        "body_form_filter",
    }
    for label, config in BENCHMARK_CONFIGS.items():
        assert isinstance(label, str)
        assert set(config.keys()) == expected_keys, f"{label} missing keys"
        float_weights = {k: v for k, v in config.items() if k != "body_form_filter"}
        total = sum(float_weights.values())
        assert abs(total - 0.85) < 0.20, f"{label} float weights sum to {total}"
