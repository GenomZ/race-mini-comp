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
  WHERE THE MiniMax API KEY COMES FROM
================================================================================
This script reads the MiniMax API key from the environment variable
`MINIMAX_API_KEY`. There are three ways to set it; pick whichever fits your
workflow:

  1. Export it in your shell before running:

         export MINIMAX_API_KEY=***        # ← paste your key here
         export RACE_LLM_PROVIDER=minimax
         export RACE_LLM_MODEL=MiniMax-M3
         python run_simple.py

  2. Put it in a `.env` file in this directory (and use a tool like
     `python-dotenv` or `direnv` to load it). .env is git-ignored.

  3. Hard-code it for one run inline (do not commit this):

         MINIMAX_API_KEY=*** python run_simple.py

The script will:
  - Look for the key in the environment.
  - If found, print a one-line banner saying so (NEVER printing the key
    itself — only "set" / "NOT SET").
  - Set `RACE_LLM_PROVIDER=minimax` and `RACE_LLM_MODEL=MiniMax-M3` for you,
    so you do not have to remember them.
  - Hand off to `evaluate.main()` which drives the agent through the race.

If the key is missing, the script still runs — the agent's `call_minimax()`
helper falls back to `BALANCED` mode when the API is unreachable, so the
race completes even without an API key. You just get a rule-based lap
instead of an LLM-driven one.

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

For the *LLM-as-planner* teaching agent, see `my_agent_simple_llm_planner.py.bak`.
That one calls MiniMax every ~10 ticks to pick AGGRESSIVE / BALANCED / CAUTIOUS.
================================================================================
"""

from __future__ import annotations

import os
import sys


# ---------------------------------------------------------------------------
# 1. Surface the API key situation BEFORE we import anything that uses it.
#    Printing this here is the whole point of having a dedicated run script:
#    anyone reading the terminal sees exactly which credentials are live.
# ---------------------------------------------------------------------------

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
      1. The MINIMAX_API_KEY env var (set by the user in their shell).
      2. The file /home/kbuzar/workspace/ny/.secret_key (the developer's
         local token-plan key on this VM — used for one-off demos).
    """
    env_key = os.getenv("MINIMAX_API_KEY")
    if env_key:
        return env_key, "environment variable MINIMAX_API_KEY"

    # Convenience for this VM only — the user said they would reset the
    # secret_key file later. If you fork this script, delete this block.
    file_key = _read_key_from("/home/kbuzar/workspace/ny/.secret_key")
    if file_key:
        return file_key, "file /home/kbuzar/workspace/ny/.secret_key"

    return "", "<not found>"


api_key, key_source = _resolve_minimax_api_key()
os.environ["MINIMAX_API_KEY"] = api_key
os.environ["RACE_LLM_PROVIDER"] = "minimax"
os.environ["RACE_LLM_MODEL"] = "MiniMax-M3"

print("=" * 72)
print("  run_simple.py — Agentic Grand Prix, teaching edition")
print("=" * 72)
print(f"  agent:           my_agent_simple")
print(f"  LLM provider:    {os.environ['RACE_LLM_PROVIDER']}")
print(f"  LLM model:       {os.environ['RACE_LLM_MODEL']}")
print(f"  API key source:  {key_source}")
print(f"  API key status:  {'set (' + str(len(api_key)) + ' chars)' if api_key else 'NOT SET'}")
print(f"  Working dir:     {os.getcwd()}")
print("=" * 72)
print()


# ---------------------------------------------------------------------------
# 2. Hand off to evaluate.py. This is the same as:
#       python evaluate.py my_agent_simple --name simple --attempts 1 --plot
#    We pass through any extra CLI args the user supplied.
# ---------------------------------------------------------------------------

import evaluate  # noqa: E402  (must come AFTER env-var setup above)

if __name__ == "__main__":
    argv = [
        "my_agent_simple",
        "--name", "simple",
        "--attempts", "1",
        "--plot",
        *sys.argv[1:],   # forward --track oval, --attempts 3, etc.
    ]
    evaluate.main(argv)