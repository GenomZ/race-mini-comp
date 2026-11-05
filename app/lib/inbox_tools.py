"""Scrutineering helpers for the Inbox page (instructor): file names, the smoke test, a status summary.

    from app.lib.inbox_tools import sanitize, smoke_check, status_of
    sanitize("Ada Lovelace (v3).py")      # -> "ada_lovelace_v3.py"
    passed, output = smoke_check(path)    # `race.py check` in its own process, mock LLM, time limit

Student code is never imported into the app process: the static check parses it, the smoke test
runs it in a separate Python process that can be killed.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

from .paths import REPO_ROOT
from .sandbox import API_KEY_VARS


def sanitize(filename: str) -> str:
    """'Ada Lovelace (v3).py' -> 'ada_lovelace_v3.py' (lower case, a-z 0-9 _, starts with a letter)."""
    stem = re.sub(r"[^a-z0-9_]+", "_", Path(filename).stem.lower()).strip("_") or "driver"
    if not stem[0].isalpha():
        stem = "d_" + stem
    return stem[:40] + ".py"


SMOKE_TIMEOUT_S = 60.0               # a clean 40-tick check takes 2-10 s; longer = something is stuck


def smoke_check(path: str | Path, timeout_s: float = SMOKE_TIMEOUT_S) -> tuple[bool, str]:
    """`race.py check --isolate <file>` in a separate process (mock LLM, no API key, time limit): (passed, output).

    The output always says something: with no printed output it reports the exit code (a file
    that calls os._exit() dies silently otherwise).
    """
    env = dict(os.environ, LLM_PROVIDER="mock", PYTHONIOENCODING="utf-8", LLM_NO_DOTENV="1")
    for var in API_KEY_VARS:                         # no key anywhere near an uploaded file
        env.pop(var, None)
    try:
        # --isolate: the file runs in a driver process under the race guard, never in the check process
        p = subprocess.run([sys.executable, str(REPO_ROOT / "race.py"), "check", "--isolate", str(path)],
                           cwd=str(REPO_ROOT),
                           env=env, capture_output=True, text=True, encoding="utf-8", errors="replace",
                           timeout=timeout_s)
        out = (p.stdout + p.stderr).strip()
        passed = p.returncode == 0 and "PASSED" in out
        if not out:
            out = (f"(no output) - the check process exited with code {p.returncode}"
                   + (": the driver probably ended the process itself (os._exit / sys.exit)" if p.returncode else ""))
        elif not passed and p.returncode not in (0, 1):
            out += f"\n\n(exit code {p.returncode})"
        return passed, out[-3000:]
    except subprocess.TimeoutExpired:
        return False, (f"timed out after {timeout_s:.0f} s and was stopped - an endless loop at import, in plan() "
                       f"or in drive()? The file cannot race like this.")


def status_of(path: str | Path) -> dict:
    """Static facts about a driver file: check errors/warnings, NAME, AUTHOR, KIND, LLM use, checkpoint."""
    from agentic_gp.race.loader import check_driver_source

    from .checkpoints import where_am_i
    from .drivers import inspect_driver

    issues = check_driver_source(path)
    info = inspect_driver(path)
    try:
        level = where_am_i(path).level
    except Exception:                                    # noqa: BLE001 - the summary must never fail
        level = "?"
    return {"errors": [i.message for i in issues if i.level == "error"],
            "warnings": [i.message for i in issues if i.level != "error"],
            "name": info.name, "author": info.author, "kind": info.kind, "llm": ", ".join(info.llm_in) or "-",
            "style": info.style, "level": level}
