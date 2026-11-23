"""Point d'entree commun : `python yolov3.py --help`."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from yolov3_project.pipeline import main  # noqa: E402


if __name__ == "__main__":
    main()
