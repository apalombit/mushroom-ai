"""Unit tests for evaluation/recall.py — mocked DB and search."""

from unittest.mock import MagicMock, patch

from evaluation.recall import (
    BENCHMARK_CONFIGS,
    EvalResult,
    PairResult,
    evaluate_recall,
    load_lookalike_graph,
    per_genus_breakdown,
)


def _make_pair(a: str, b: str) -> MagicMock:
    pair = MagicMock()
    pair.species_a = a
    pair.species_b = b
    return pair


def _make_candidates(*names: str) -> list[dict]:
    """Build a candidate list with dummy similarity scores."""
    from ingestion.rubric import EMBEDDING_GROUPS

    return [
        {
            "scientific_name": n,
            "similarity_overall": round(0.9 - i * 0.05, 4),
            **{f"similarity_{g}": 0.75 for g in EMBEDDING_GROUPS},
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
    """PairResult stores group_sims dict and numeric similarity."""
    r = PairResult(
        query="A",
        target="B",
        rank=2,
        sim_overall=0.85,
        sim_numeric=0.65,
        group_sims={"macro_visual": 0.90, "ecological": 0.75},
    )
    assert r.rank == 2
    assert r.sim_numeric == 0.65
    assert r.group_sims["macro_visual"] == 0.90
    assert r.error is None


@patch("evaluation.recall.search_lookalikes")
def test_evaluate_recall_forwards_strategy(mock_search):
    """strategy param is passed through to search_lookalikes."""
    pair = _make_pair("A", "B")
    mock_search.return_value = (MagicMock(), [])
    evaluate_recall(MagicMock(), [pair], strategy="rrf")
    calls = mock_search.call_args_list
    assert all(c.kwargs.get("strategy") == "rrf" for c in calls)


def test_benchmark_configs():
    """BENCHMARK_CONFIGS has >= 2 entries, each has body_form_filter and float weights."""
    non_weight_keys = {"body_form_filter", "group_filter", "alpha", "morpho_pool_required"}
    assert len(BENCHMARK_CONFIGS) >= 2
    for label, config in BENCHMARK_CONFIGS.items():
        assert isinstance(label, str)
        assert "body_form_filter" in config, f"{label} missing body_form_filter"
        float_weights = {k: v for k, v in config.items() if k not in non_weight_keys}
        assert len(float_weights) > 0, f"{label} has no weight keys"
        total = sum(float_weights.values())
        # configs with >1 weight key set explicit group weights that should sum near 1;
        # configs with only "numeric" rely on equal-split defaults for group weights
        if len(float_weights) > 1:
            assert 0.5 < total <= 1.1, f"{label} float weights sum to {total}"
        else:
            assert total > 0, f"{label} has zero weights"


# ---------------------------------------------------------------------------
# Lookalike graph loader
# ---------------------------------------------------------------------------


def test_load_lookalike_graph_from_yaml(tmp_path):
    """Loader extracts correct bidirectional edges from YAML."""
    yaml_content = {
        "lookalikes": [
            {"species": "Amanita muscaria", "indexed": ["Amanita caesarea", "Amanita pantherina"]},
            {"species": "Amanita caesarea", "indexed": ["Amanita muscaria"]},
            {"species": "Boletus edulis", "indexed": ["Tylopilus felleus"]},
        ]
    }
    p = tmp_path / "test_lookalikes.yaml"
    import yaml

    p.write_text(yaml.dump(yaml_content))

    edges = load_lookalike_graph(p)
    # Amanita muscaria ↔ Amanita caesarea appears from both sides → 1 deduped edge
    # Amanita muscaria ↔ Amanita pantherina → 1 edge
    # Boletus edulis ↔ Tylopilus felleus → 1 edge
    assert len(edges) == 3
    assert ("Amanita caesarea", "Amanita muscaria") in edges
    assert ("Amanita muscaria", "Amanita pantherina") in edges
    assert ("Boletus edulis", "Tylopilus felleus") in edges


def test_load_lookalike_graph_skips_self_references(tmp_path):
    """Self-lookalikes (species == lookalike) are skipped."""
    yaml_content = {
        "lookalikes": [
            {
                "species": "Agaricus augustus",
                "indexed": ["Agaricus augustus", "Agaricus campestris"],
            },
        ]
    }
    p = tmp_path / "test.yaml"
    import yaml

    p.write_text(yaml.dump(yaml_content))

    edges = load_lookalike_graph(p)
    assert len(edges) == 1
    assert ("Agaricus augustus", "Agaricus campestris") in edges


def test_load_lookalike_graph_real_file():
    """The actual known_lookalikes.yaml loads without errors and has many edges."""
    edges = load_lookalike_graph()
    assert len(edges) > 100  # ~847 expected


def test_per_genus_breakdown():
    """Per-genus breakdown computes correctly."""
    results = [
        PairResult(query="Amanita muscaria", target="Amanita caesarea", rank=1),
        PairResult(query="Amanita caesarea", target="Amanita muscaria", rank=None),
        PairResult(query="Boletus edulis", target="Tylopilus felleus", rank=2),
    ]
    eval_result = EvalResult(pair_results=results, top_k=5)
    breakdown = per_genus_breakdown(eval_result, k=5)

    assert "Amanita" in breakdown
    assert breakdown["Amanita"]["total"] == 2
    assert breakdown["Amanita"]["hits"] == 1
    assert breakdown["Amanita"]["recall"] == 0.5

    assert "Boletus" in breakdown
    assert breakdown["Boletus"]["recall"] == 1.0
