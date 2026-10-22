"""Make the backend source package importable before project tooling is set up."""

import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1] / "backend"
ROOT = BACKEND.parent
for source in (ROOT, BACKEND):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
