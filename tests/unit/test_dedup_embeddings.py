"""Unit tests for the embedding-based dedup script."""

from __future__ import annotations

import torch

from vision.scripts.dedup_embeddings import (
    UnionFind,
    cluster_edges,
    file_matches_image_id,
    find_duplicate_edges,
    pick_keeper,
)


def test_union_find_basic():
    uf = UnionFind(5)
    uf.union(0, 1)
    uf.union(1, 2)
    uf.union(3, 4)
    assert uf.find(0) == uf.find(2)
    assert uf.find(3) == uf.find(4)
    assert uf.find(0) != uf.find(3)


def test_cluster_edges_transitivity():
    # A~B and B~C → all three in one cluster even if no direct A~C edge
    edges = [(0, 1, 0.99), (1, 2, 0.99), (3, 4, 0.99)]
    clusters = cluster_edges(5, edges)
    assert len(clusters) == 2
    sizes = sorted(len(m) for m in clusters.values())
    assert sizes == [2, 3]


def test_cluster_edges_singletons_excluded():
    edges = [(0, 1, 0.99)]
    clusters = cluster_edges(10, edges)
    # Only 0 and 1 are in clusters; 2..9 are singletons not returned
    assert len(clusters) == 1
    members = next(iter(clusters.values()))
    assert sorted(members) == [0, 1]


def test_find_duplicate_edges_matches_naive():
    torch.manual_seed(0)
    n, d = 20, 8
    mat = torch.randn(n, d)
    mat = mat / mat.norm(dim=1, keepdim=True)
    # Force two pairs to be duplicates
    mat[5] = mat[3]
    mat[10] = mat[3]  # forms a chain {3, 5, 10}
    mat[15] = mat[7]

    threshold = 0.99
    chunked = sorted((i, j) for i, j, _ in find_duplicate_edges(mat, threshold, chunk=4))

    sim = mat @ mat.T
    naive = []
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j].item() >= threshold:
                naive.append((i, j))
    naive.sort()
    assert chunked == naive
    # Sanity: contains the planted duplicates
    assert (3, 5) in chunked
    assert (3, 10) in chunked
    assert (5, 10) in chunked
    assert (7, 15) in chunked


def test_find_duplicate_edges_chunk_sizes_equivalent():
    """Different chunk sizes should yield identical edges."""
    torch.manual_seed(1)
    n, d = 30, 16
    mat = torch.randn(n, d)
    mat = mat / mat.norm(dim=1, keepdim=True)
    mat[2] = mat[0]
    mat[20] = mat[1]
    e1 = sorted((i, j) for i, j, _ in find_duplicate_edges(mat, 0.999, chunk=5))
    e2 = sorted((i, j) for i, j, _ in find_duplicate_edges(mat, 0.999, chunk=17))
    e3 = sorted((i, j) for i, j, _ in find_duplicate_edges(mat, 0.999, chunk=n))
    assert e1 == e2 == e3


def test_pick_keeper_prefers_vlm_count():
    ranks = {
        "a": (5, 0, 1000, "a"),
        "b": (3, 99, 9999, "b"),
        "c": (5, 1, 500, "c"),
    }
    # a and c tied on vlm=5; a wins on sp (0 vs 1 → c wins on sp because higher).
    # Wait: rank tuple is (n_vlm, n_sp, res, image_id); higher is better for first three.
    # a: (5, 0, 1000), c: (5, 1, 500) → c has higher sp → c wins
    assert pick_keeper(["a", "b", "c"], ranks) == "c"


def test_pick_keeper_resolution_tiebreak():
    ranks = {
        "a": (0, 0, 1000, "a"),
        "b": (0, 0, 9999, "b"),
        "c": (0, 0, 500, "c"),
    }
    assert pick_keeper(["a", "b", "c"], ranks) == "b"


def test_pick_keeper_lexicographic_tiebreak():
    ranks = {
        "z": (0, 0, 0, "z"),
        "a": (0, 0, 0, "a"),
        "m": (0, 0, 0, "m"),
    }
    assert pick_keeper(["z", "a", "m"], ranks) == "a"


def test_file_matches_image_id_exact():
    assert file_matches_image_id("abc123", "abc123") is True
    assert file_matches_image_id("abc123_thumb", "abc123") is True
    assert file_matches_image_id("abc123_896_cc", "abc123") is True


def test_file_matches_image_id_rejects_prefix_only():
    assert file_matches_image_id("abc1234", "abc123") is False
    assert file_matches_image_id("abc123other", "abc123") is False
    assert file_matches_image_id("other", "abc123") is False
