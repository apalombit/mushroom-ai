"""
LLM-based feature extraction from source text.

Takes raw text about a mushroom species and extracts structured features
using Instructor-validated LLM calls.

Pipeline:
    Source text → LLM (with ExtractedSpeciesFeatures schema) → validated features
    → store as Layer 1 (SourceObservation) in Postgres

TODO:
    - [ ] Implement extract_features_from_text()
    - [ ] Add extraction prompt with rubric instructions
    - [ ] Store results in source_observations table with provenance
    - [ ] Handle extraction failures gracefully (log, skip, retry)
    - [ ] Track extraction quality metrics in MLflow
"""
