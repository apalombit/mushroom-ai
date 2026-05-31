"""CLI: launch the manual image QA labeling tool.

Usage:
    python -m vision.scripts.run_manual_qa --feature hymenium_type
    python -m vision.scripts.run_manual_qa --feature cap_shape --port 8003
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from vision.labeling.manual_qa_app import run  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Manual image QA labeling tool")
    parser.add_argument("--feature", required=True, help="Feature name (e.g. hymenium_type)")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--preprocess-version", default="v1")
    args = parser.parse_args()

    run(
        feature_name=args.feature,
        port=args.port,
        preprocess_version=args.preprocess_version,
    )


if __name__ == "__main__":
    main()
