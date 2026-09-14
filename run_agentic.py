"""Run the reference Agentic Workflow agent (planning, execution, reflection) and plot the result.

    python run_agentic.py
    python run_agentic.py --attempts 3 --plot

This evaluates the full cognitive architecture: global track planning from get_track_map,
tactical execution, episodic memory, and reflection across attempts.
"""

import sys

import evaluate

if __name__ == "__main__":
    evaluate.main(["agentic_gp.agents.agentic", "--name", "agentic", "--attempts", "3", "--plot", *sys.argv[1:]])
