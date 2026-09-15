"""Run the teaching agent (my_agent_simple) against the MiniMax M3 API.

This script is the one-line way to run the educational reference driver.
It is intentionally explicit about *which* API key is being used so anyone
watching the terminal output (or this file's source) can see exactly where
the credentials come from.

USAGE:
    python run_simple.py                 # 1 attempt, default track, with plot
    python run_simple.py --track oval    # smaller, faster track
    python run_simple.py --attempts 3    # run more attempts for averaging

================================================================================
  CONFIGURATION
================================================================================
All configuration is read from environment variables. The order of precedence
(highest wins):

    1. Variables already exported in your shell.
    2. Values in a .env file in this directory (if present).

So you can put a one-time setup in .env and override per-run in your shell,
or vice versa.

Variables recognised:

    MINIMAX_API_KEY     your MiniMax API key (sk-cp-...)
    RACE_LLM_PROVIDER   always 'minimax' for this script
    RACE_LLM_MODEL      always 'MiniMax-M3' for this script

OUTPUTS (all written to results/ by default):
    results/simple.json           lap summary (written by evaluate.py)
    results/simple.png            track plot, requires --plot (default-on here)
    results/simple_<ts>.log       per-tick log: state, prompt, decision, events
                                  use --no-log to skip, --log PATH to relocate

If you do not set MINIMAX_API_KEY, the script still runs — the agent's
call_minimax() helper falls back to BALANCED mode when the API is
unreachable, so you get a rule-based lap instead of an LLM-driven one.

If you do not create a .env file, the script simply uses whatever is in the
shell environment, and prints "no .env found" so you know.

================================================================================
  WHAT THE AGENT DOES
================================================================================
`my_agent_simple` is the *teaching* reference. Its control policy is plain
Python (steer toward the next waypoint + push away from walls + brake before
sharp corners). It does NOT use an LLM in its inner loop. The MiniMax M3
API call is wired into the agent's optional `SYSTEM_PROMPT_FOR_CONTROLS`
helper, which a student can plug in if they want to swap the rule-based
executor for a one-shot prompt to the model. See the bottom of
`my_agent_simple.py` for the hooks.
================================================================================
"""

from __future__ import annotations

import os
import sys
import time

# ---------------------------------------------------------------------------
# 1. Load .env FIRST so the banner can show what was loaded.
#    We never override shell-exported values — env vars win.
# ---------------------------------------------------------------------------
from load_env import load_env                                       # noqa: E402

_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
_loaded_keys = load_env(_env_path, override=False)


def _read_key_from(path: str) -> str | None:
    """Read a key from a file. Returns None if the file is missing."""
    try:
        with open(path) as f:
            return f.read().strip()
    except (FileNotFoundError, PermissionError):
        return None


def _resolve_minimax_api_key() -> tuple[str, str]:
    """Return (api_key, source_description). Never prints the key itself.

    Resolution order:
      1. The MINIMAX_API_KEY env var (shell or .env).
      2. The file /home/kbuzar/workspace/ny/.secret_key (the developer's
         local token-plan key on this VM — used for one-off demos).
    """
    env_key = os.getenv("MINIMAX_API_KEY")
    if env_key:
        # Distinguish shell export vs .env-loaded for the banner.
        source = (
            "environment variable MINIMAX_API_KEY (exported in shell)"
            if "MINIMAX_API_KEY" not in _loaded_keys
            else ".env file in this directory"
        )
        return env_key, source

    # Convenience for this VM only — the user said they would reset the
    # secret_key file later. If you fork this script, delete this block.
    file_key = _read_key_from("/home/kbuzar/workspace/ny/.secret_key")
    if file_key:
        return file_key, "file /home/kbuzar/workspace/ny/.secret_key (fallback)"

    return "", "<not found>"


api_key, key_source = _resolve_minimax_api_key()
os.environ["MINIMAX_API_KEY"] = api_key
os.environ.setdefault("RACE_LLM_PROVIDER", "minimax")
os.environ.setdefault("RACE_LLM_MODEL", "MiniMax-M3")


# How many variables in the .env file were either newly set OR would have been
# set if not already in the environment. Useful for the banner.
def _count_env_lines(path: str) -> int:
    if not os.path.isfile(path):
        return 0
    count = 0
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].lstrip()
            key, sep, _ = line.partition("=")
            if sep and key.strip():
                count += 1
    return count


_env_line_count = _count_env_lines(_env_path)


# ---------------------------------------------------------------------------
# 2. Banner: surface everything that matters before the race starts.
# ---------------------------------------------------------------------------

print("=" * 72)
print("  run_simple.py — Agentic Grand Prix, teaching edition")
print("=" * 72)
print(f"  agent:           my_agent_simple")
print(f"  LLM provider:    {os.environ['RACE_LLM_PROVIDER']}")
print(f"  LLM model:       {os.environ['RACE_LLM_MODEL']}")
print(f"  API key source:  {key_source}")
print(f"  API key status:  {'set (' + str(len(api_key)) + ' chars)' if api_key else 'NOT SET'}")
if not os.path.isfile(_env_path):
    print(f"  .env file:       none at {_env_path} (using shell env)")
elif _loaded_keys:
    print(f"  .env file:       loaded {len(_loaded_keys)} new variable(s) from {_env_path} (others already in shell)")
else:
    print(f"  .env file:       found {_env_path} with {_env_line_count} variable(s), but all already set in shell")
print(f"  Working dir:     {os.getcwd()}")
print("=" * 72)
print()


# ---------------------------------------------------------------------------
# 3. Hand off to evaluate.py. Equivalent to:
#       python evaluate.py my_agent_simple --name simple --attempts 1 --plot
# ---------------------------------------------------------------------------

import evaluate  # noqa: E402  (must come AFTER env-var setup above)

if __name__ == "__main__":
    # Per-tick trace default: results/simple_<timestamp>.log, so all artefacts
    # for one run (JSON summary, PNG plot, per-tick log) live together in
    # results/. Pass --no-log to disable the per-tick file, or --log PATH to
    # write it elsewhere.
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--no-log", action="store_true",
                        help="disable per-tick log file")
    parser.add_argument("--log", default=None, metavar="PATH",
                        help="write per-tick log to PATH "
                             "(default: results/simple_<ts>.log)")
    parsed, extra = parser.parse_known_args(sys.argv[1:])
    log_path = None
    if not parsed.no_log:
        if parsed.log:
            log_path = parsed.log
        else:
            ts = time.strftime("%Y%m%d_%H%M%S")
            log_dir = os.path.join(os.getcwd(), "results")
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, f"simple_{ts}.log")

    argv = [
        "my_agent_simple",
        "--name", "simple",
        "--attempts", "1",
        "--plot",
        *extra,
    ]
    if log_path:
        argv += ["--agent-arg", f"log_path={log_path}"]
    evaluate.main(argv)