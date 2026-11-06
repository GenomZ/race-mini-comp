"""Re-export the fruit_fly_antigravity agentic driver for agentic_gp.agents namespace."""
import sys
from pathlib import Path

# Ensure repo root is in sys.path
repo_root = str(Path(__file__).resolve().parents[2])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from fruit_fly_antigravity import *  # noqa: F401, F403
