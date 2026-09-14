"""Run the dummy agent (drives straight, crashes) and plot the result.

    python run_dummy.py
"""

import sys

import evaluate

if __name__ == "__main__":
    evaluate.main(["agentic_gp.agents.dummy", "--name", "dummy", "--attempts", "1", "--plot", *sys.argv[1:]])
