"""Run the fruit-fly-connectome racing agent.

This script is the one-line way to run `agentic_gp.agents.fruit_fly_minimax`.
On first run it will CEM-train a readout across `--attempts` racing attempts
(default 30) and save the best weights to `fruit_fly_weights.json`. On
subsequent runs it loads the saved weights and replays them.

    python run_fruit_fly_minimax.py                          # train (~16 s)
    python run_fruit_fly_minimax.py --attempts 1             # use saved weights
    python run_fruit_fly_minimax.py --attempts 60 --no-save  # longer training

Source / credit: https://github.com/cobanov/awesome-fly
"""

from __future__ import annotations

import os
import sys

from load_env import load_env

load_env(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"), override=False)

import evaluate  # noqa: E402  (import after env-var setup)

if __name__ == "__main__":
    argv = [
        "agentic_gp.agents.fruit_fly_minimax",
        "--name", "fruit_fly_minimax",
        "--plot",
        *sys.argv[1:],
    ]
    evaluate.main(argv)