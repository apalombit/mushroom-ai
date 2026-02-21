"""
LLM explanation generator.

Takes the structured comparison table (query species vs lookalike candidates
with per-feature similarity breakdown) and generates a human-readable
"at a glance" summary.

This is the only LLM call at query time. The similarity search itself
is deterministic (pgvector + weighted merge).
"""

import logging
from typing import Any

from llm.client import structured_completion
from llm.schemas import LookalikeExplanation

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a mycologist writing safety-focused species comparisons.

Given a query mushroom species and its visual lookalikes, write a clear explanation
to help foragers avoid dangerous confusion.

Rules:
- summary: 2-3 sentences covering the most notable lookalikes and the key reason they
  cause confusion.
- notable_pairs: 2-3 sentences, each explaining what makes a specific pair confusable
  and the single most critical distinguishing feature.
- safety_warning: must be non-empty if any lookalike is toxic, deadly, or inedible.
  Be direct and unambiguous. If all lookalikes are safe, write "No toxic lookalikes
  identified in this result set."
"""


def generate_explanation(
    query_name: str,
    comparison_table: list[dict[str, Any]],
) -> LookalikeExplanation:
    """
    Generate a human-readable explanation of lookalike relationships.

    Args:
        query_name: Scientific name of the queried species.
        comparison_table: Output of build_comparison_table — list of candidate dicts
            with similarity scores and feature_comparisons.

    Returns:
        LookalikeExplanation with summary, notable_pairs, safety_warning.
    """
    # Format top-5 candidates for prompt (limit for token budget)
    lines = []
    for cand in comparison_table[:5]:
        name = cand["scientific_name"]
        edibility = cand.get("edibility") or "unknown"
        overall = cand.get("similarity_overall", 0.0)
        lines.append(f"- {name} (edibility: {edibility}, similarity: {overall:.2f})")

    prompt = (
        f"Query species: {query_name}\n\n"
        f"Top lookalike candidates:\n"
        + "\n".join(lines)
        + "\n\nGenerate a concise explanation of why these species cause confusion "
        "and how to distinguish them safely."
    )

    return structured_completion(
        prompt=prompt,
        response_model=LookalikeExplanation,
        system=SYSTEM_PROMPT,
        temperature=0.3,
    )
