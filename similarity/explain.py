"""
LLM explanation generator.

Takes the structured comparison table (query species vs lookalike candidates
with per-feature similarity breakdown) and generates a human-readable
"at a glance" summary.

This is the only LLM call at query time. The similarity search itself
is deterministic (pgvector + weighted merge).

TODO:
    - [ ] Implement generate_explanation() using structured_completion + LookalikeExplanation
    - [ ] Design the explanation prompt (receives comparison table as structured input)
    - [ ] Handle cases where all candidates are low-similarity (explain that no strong lookalikes exist)
    - [ ] Track explanation quality/latency in MLflow
"""
