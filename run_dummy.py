"""Run the dummy agent (drives straight, crashes) and plot the result.

Configuration is read from environment variables; a .env file in this
directory is loaded first. Shell exports always override .env values.

    python run_dummy.py
"""

import os
import sys

from load_env import load_env

load_env(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"), override=False)

import evaluate  # noqa: E402  (import after env-var setup)

if __name__ == "__main__":
    evaluate.main(["agentic_gp.agents.dummy", "--name", "dummy", "--attempts", "1", "--plot", *sys.argv[1:]])