"""Run the reference Agentic Workflow agent (planning, execution, reflection) and plot the result.

    python run_agentic.py
    python run_agentic.py --attempts 3 --plot

This evaluates the full cognitive architecture: global track planning from get_track_map,
tactical execution, episodic memory, and reflection across attempts.

Configuration is read from environment variables; a .env file in this
directory is loaded first. Shell exports always override .env values.
"""

import os
import sys

from load_env import load_env

load_env(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"), override=False)

import evaluate  # noqa: E402  (import after env-var setup)

if __name__ == "__main__":
    evaluate.main(["agentic_gp.agents.agentic", "--name", "agentic", "--attempts", "3", "--plot", *sys.argv[1:]])