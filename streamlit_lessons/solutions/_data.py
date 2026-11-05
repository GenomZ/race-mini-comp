"""The solutions use the lessons' _data.py (one folder up). This file only forwards to it.

`import _data` in a solution finds THIS file first (Streamlit puts the script's folder on
sys.path). It loads ../_data.py under the same name, so a solution file needs no extra lines and
`git diff --no-index` against the lesson shows only the real changes.
"""

import importlib.util
import sys
from pathlib import Path

REAL = Path(__file__).resolve().parents[1] / "_data.py"
_spec = importlib.util.spec_from_file_location("_data", REAL)
_real = importlib.util.module_from_spec(_spec)
sys.modules["_data"] = _real  # `import _data` hands out the real module from now on
_spec.loader.exec_module(_real)
