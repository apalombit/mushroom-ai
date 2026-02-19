"""
Multi-source reconciliation: merge Layer 1 observations into Layer 2 profiles.

For each species, collects all SourceObservation rows, passes them to the LLM
with a strict reconciliation rubric, and produces one canonical ReconciledSpecies row.

Key behaviors:
    - Merges complementary information (different features from different sources)
    - Resolves conflicts by choosing the most specific/reliable value
    - Flags irreconcilable conflicts for human review (needs_review=True)
    - Respects human_overrides: fields manually set are never overwritten
    - Stores reconciliation confidence and conflict notes

Human-in-the-loop:
    When needs_review=True, the species appears in a review queue.
    Human decisions are stored in a directives file (data/seed/review_directives.yaml)
    that the reconciliation LLM reads on subsequent runs.

TODO:
    - [ ] Implement reconcile_species() using structured_completion + ReconciliationResult
    - [ ] Load and apply human review directives
    - [ ] Handle incremental re-reconciliation (only re-run affected species)
    - [ ] Respect human_overrides column (skip those fields)
"""
