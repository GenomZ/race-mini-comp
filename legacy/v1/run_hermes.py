"""Run the Hermes baseline (zero-shot LLM driver) and record its lap time.

    ollama pull hermes3 && ollama serve          # default provider
    python run_hermes.py                         # one attempt, plots results/hermes.png
    python run_hermes.py --attempts 3 --agent-arg history=2

Configuration is read from environment variables; a .env file in this
directory is loaded first. Shell exports always override .env values.

Switch model/provider with RACE_LLM_PROVIDER / RACE_LLM_MODEL (see .env.example).

OUTPUTS (all written to results/ by default):
    results/hermes.json            lap summary (written by evaluate.py)
    results/hermes.png             track plot (with --plot)
    results/hermes_trace.log       per-tick log: state, prompt, llm reply,
                                   parsed-from, decision, latency_ms, events
                                   (committed to repo as a reference artefact;
                                   use --no-log to skip, --log PATH to relocate)
"""

from __future__ import annotations

import os
import sys

from load_env import load_env

load_env(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"), override=False)

import evaluate  # noqa: E402  (import after env-var setup)

if __name__ == "__main__":
    # Default: log per-tick trace to results/hermes_trace.log so the trace is
    # preserved on disk for after-the-fact debugging. Pass --no-log (or
    # --agent-arg log_path=) to disable or relocate.
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--no-log", action="store_true",
                        help="disable per-tick log file")
    parser.add_argument("--log", default=None, metavar="=",
                        help="write per-tick log to PATH "
                             "(default: results/hermes_trace.log)")
    parsed, extra = parser.parse_known_args(sys.argv[1:])
    log_path = None
    if not parsed.no_log:
        if parsed.log:
            log_path = parsed.log
        else:
            log_dir = os.path.join(os.getcwd(), "results")
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, "hermes_trace.log")

    argv = [
        "agentic_gp.agents.hermes",
        "--name", "hermes",
        "--plot",
        *extra,
    ]
    if log_path:
        argv += ["--agent-arg", f"log_path={log_path}"]
    evaluate.main(argv)