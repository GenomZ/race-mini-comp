"""Run the Hermes baseline (zero-shot LLM driver) and record its lap time.

    ollama pull hermes3 && ollama serve          # default provider
    python run_hermes.py                         # one attempt, plots results/hermes.png
    python run_hermes.py --attempts 3 --agent-arg history=2

Configuration is read from environment variables; a .env file in this
directory is loaded first. Shell exports always override .env values.

Switch model/provider with RACE_LLM_PROVIDER / RACE_LLM_MODEL (see .env.example).
"""

import os
import sys

from load_env import load_env

load_env(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"), override=False)

import evaluate  # noqa: E402  (import after env-var setup)

if __name__ == "__main__":
    evaluate.main(["agentic_gp.agents.hermes", "--name", "hermes", "--plot", *sys.argv[1:]])