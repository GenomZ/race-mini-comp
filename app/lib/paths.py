"""Where things live in the repo (works from any working directory).

    from app.lib.paths import REPO_ROOT, RACES_DIR, DRIVER_FOLDERS
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]          # app/lib/paths.py -> repo root
APP_DIR = REPO_ROOT / "app"
RACES_DIR = REPO_ROOT / "races"
DEMO_DIR = RACES_DIR / "demo"
RESULTS_DIR = REPO_ROOT / "results"
DRIVERS_DIR = REPO_ROOT / "drivers"

# Driver folders, in the order the app lists them (label, folder).
DRIVER_FOLDERS: list[tuple[str, Path]] = [
    ("submissions", DRIVERS_DIR / "submissions"),
    ("examples", DRIVERS_DIR / "examples"),
    ("checkpoints", REPO_ROOT / "lesson" / "checkpoints"),
]
TEMPLATE = DRIVERS_DIR / "template.py"


def ensure_repo_on_path() -> Path:
    """Make `import agentic_gp` and `import app...` work when Streamlit runs a page script."""
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return REPO_ROOT


def rel(path: str | Path) -> str:
    """Repo-relative path with forward slashes (for display), or the path itself if outside the repo."""
    p = Path(path).resolve()
    try:
        return p.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(p)
