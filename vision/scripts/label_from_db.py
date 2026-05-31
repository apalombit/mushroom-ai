"""CLI: auto-label images from species DB (reconciled_species features).

Usage:
    python -m vision.scripts.label_from_db
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from db.connection import get_session
from vision.data.labeling import auto_label_from_species

FEATURES_CONFIG = PROJECT_ROOT / "vision/config/features.yaml"


def main():
    session = get_session()
    counts = auto_label_from_species(session, FEATURES_CONFIG)
    session.close()

    if not counts or all(v == 0 for v in counts.values()):
        print("No labels generated. Check that image_registry has species "
              "matching reconciled_species.")


if __name__ == "__main__":
    main()
