"""Run the Hermes baseline (zero-shot LLM driver) and record its lap time.

    ollama pull hermes3 && ollama serve          # default provider
    python run_hermes.py                         # one attempt, plots results/hermes.png
    python run_hermes.py --attempts 3 --agent-arg history=2

Switch model/provider with RACE_LLM_PROVIDER / RACE_LLM_MODEL (see .env.example).
"""

import sys

import evaluate

if __name__ == "__main__":
    evaluate.main(["agentic_gp.agents.hermes", "--name", "hermes", "--plot", *sys.argv[1:]])
