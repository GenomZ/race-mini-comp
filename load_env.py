"""Tiny helper: load KEY=VALUE pairs from a .env file into os.environ.

Why a custom loader instead of `python-dotenv`?
    - Zero new dependencies (the project pins langgraph, langchain, etc., but
      not dotenv, and adding a dep just to read a 6-line config file is silly).
    - Works the same on every Python 3.10+ install.
    - Easy to read in one screen — a student can audit it without reading docs.

Usage:
    from load_env import load_env
    load_env()              # reads ./.env if it exists; no-op if it does not
    load_env("/path/to/.env", override=False)

Format:
    KEY=value
    KEY="value with spaces"
    KEY='single quoted'
    # KEY=ignored       (comment)
    blank lines are ignored

It does NOT expand variables, does NOT support multi-line values, does NOT
do anything clever. If you need any of that, use python-dotenv.
"""

from __future__ import annotations

import os
import shlex
from pathlib import Path


def load_env(path: str | os.PathLike | None = None, override: bool = False) -> list[str]:
    """Load KEY=VALUE pairs from `path` into `os.environ`. Return the keys set.

    Args:
        path:    Path to a .env file. Defaults to <repo root>/.env.
        override: If True, overwrite variables that are already in os.environ.
                  If False (the default), already-set variables win — this lets
                  a user override .env with a shell export.

    Returns:
        A list of the variable names that were actually set (or overwritten)
        from the file. Useful for the caller to print "loaded N vars from .env".
    """
    if path is None:
        path = Path(__file__).resolve().parent / ".env"
    env_path = Path(path)
    if not env_path.is_file():
        return []

    loaded: list[str] = []
    with env_path.open() as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            # Strip an optional `export ` prefix so .env works the same as a
            # shell script.
            if line.startswith("export "):
                line = line[len("export "):].lstrip()
            # Split on the first '=' only — values may contain '='.
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if not key:
                continue
            # Strip matching surrounding quotes if present.
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            else:
                # Use shlex to handle escapes, trailing comments, etc.
                try:
                    value = shlex.split(value)[0]
                except ValueError:
                    pass   # fall back to the raw string

            if not override and key in os.environ:
                continue
            os.environ[key] = value
            loaded.append(key)

    return loaded