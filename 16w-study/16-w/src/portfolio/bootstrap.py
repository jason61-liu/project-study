from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[4]
WEEK15_SRC = PROJECT_DIR / "15-w" / "src"


def load_week15() -> None:
    path = str(WEEK15_SRC)
    if path not in sys.path:
        sys.path.insert(0, path)

