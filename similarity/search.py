"""
Similarity search engine.

Finds lookalike species by running independent pgvector similarity queries
per feature group (morphological, ecological, taxonomic), then merging
and reranking with tunable weights.

Query-time pipeline:
    1. Look up query species embeddings from ReconciledSpecies
    2. Run 3 pgvector cosine similarity queries (one per group)
    3. Merge candidate sets, compute weighted score
    4. Apply user context (region/season) as filter or boost
    5. Return top-K with per-group similarity breakdown

TODO:
    - [ ] Implement search_by_group() — single pgvector query
    - [ ] Implement search_lookalikes() — full pipeline with merge/rerank
    - [ ] Implement context filtering (region, season boost/penalty)
    - [ ] Build comparison table (query species features vs each candidate)
"""
