"""
Embed reconciled species features into per-group vectors for pgvector.

For each ReconciledSpecies row:
    1. Convert morphological features → text description → embed → embedding_morphological
    2. Convert ecological features → text description → embed → embedding_ecological
    3. Convert taxonomic features → text description → embed → embedding_taxonomic

Uses sentence-transformers (all-MiniLM-L6-v2, 384 dimensions) for embedding.
The text conversion follows the rubric in ingestion/rubric.py.

TODO:
    - [ ] Implement embed_species() for a single species
    - [ ] Implement embed_all() batch pipeline
    - [ ] Implement incremental re-embedding (only species with changed profiles)
    - [ ] Handle missing features (skip from text, don't embed "None")
"""
