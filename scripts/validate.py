"""
Validate similarity engine against known lookalike pairs.

For each ground truth pair, queries species_a and checks whether
species_b appears in the top-K results (and vice versa).

Reports pass rate and logs results to MLflow.

Usage:
    python -m scripts.validate
    python -m scripts.validate --top-k 15

TODO:
    - [ ] Implement after Stage 6 (similarity search is working)
    - [ ] Wire up to MLflow experiment "validation"
    - [ ] Print detailed report: which pairs pass, which fail, at what rank
"""

import argparse
import logging

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Validate against ground truth pairs")
    parser.add_argument("--top-k", type=int, default=10, help="Check if match is in top-K")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # TODO:
    # 1. Load ground truth pairs from DB (ground_truth_pairs table)
    # 2. For each pair, run search_lookalikes(species_a) and check if species_b is in top-K
    # 3. Also run search_lookalikes(species_b) and check if species_a is in top-K
    # 4. Compute pass rate
    # 5. Log to MLflow
    # 6. Print report

    raise NotImplementedError(
        "Implement after Stage 6 — see Stage 10 in IMPLEMENTATION_PLAN.md"
    )


if __name__ == "__main__":
    main()
