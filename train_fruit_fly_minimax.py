"""Train the fruit-fly-connectome agent for N training laps.

This script wraps `evaluate.py` and the `agentic_gp.agents.fruit_fly_minimax`
agent. Each training lap is one full evaluation attempt through the racing
gym; the agent uses the first ~90% of attempts for CEM search and the
final attempt as an evaluation of the best readout found.

Usage:
    python train_fruit_fly_minimax.py --training-laps 30 --track default --plot
    python train_fruit_fly_minimax.py --training-laps 100 --track default

The weights and the evaluate.py outputs are saved with the number of
training laps in the filename:

    fruit_fly_minimax_<N>lap_weights.json   <- trained readout
    results/fruit_fly_minimax_<N>lap.json    <- lap summary (committed)
    results/fruit_fly_minimax_<N>lap.png     <- track plot (committed)

This way you can compare several training budgets on the leaderboard side
by side without overwriting each other. Each <N> gets its own row.

Notes on cost:
    On the default Grand Prix circuit, a 100-lap training run takes
    ~25 seconds on this VM (one tick is deterministic and cheap). On the
    small 'oval' track it's even faster, ~8 seconds for 100 laps.

Caveats:
    - Training is deterministic given the same random seeds (built into
      the connectome builder + CEM), so two runs at the same N produce
      the same weights and the same best lap.
    - The agent file uses a *fixed* tiny toy connectome (37 neurons) so
      even 100 CEM candidates don't reliably discover great policies;
      this script is for demonstrating the training pipeline, not for
      beating the tuned my_agent.py.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from load_env import load_env


REPO_ROOT = Path(__file__).resolve().parent
MAX_TRAINING_LAPS = 100


def _project_root() -> Path:
    return Path(os.getcwd()).resolve()


def _weights_path(num_laps: int) -> Path:
    """Where the CEM-trained readout lives for this training budget."""
    return _project_root() / f"fruit_fly_minimax_{num_laps}lap_weights.json"


def _result_json_path(num_laps: int) -> Path:
    """Where evaluate.py writes its lap summary."""
    return _project_root() / "results" / f"fruit_fly_minimax_{num_laps}lap.json"


def _result_png_path(num_laps: int) -> Path:
    return _project_root() / "results" / f"fruit_fly_minimax_{num_laps}lap.png"


def train(num_laps: int, track: str = "default", plot: bool = True,
         verbose: bool = True) -> dict:
    """Run a single training run of `num_laps` laps and capture the result.

    Returns a summary dict with the best lap, weights path, and the path to
    the saved results/<name>.json file. Caller decides whether to commit.
    """
    if not 1 <= num_laps <= MAX_TRAINING_LAPS:
        raise ValueError(
            f"--training-laps must be 1..{MAX_TRAINING_LAPS}, got {num_laps}"
        )

    weights_file = _weights_path(num_laps)
    json_file = _result_json_path(num_laps)
    png_file = _result_png_path(num_laps)

    # Wipe stale artefacts so we know the result we read is from THIS run.
    for p in (weights_file, json_file, png_file):
        if p.exists():
            p.unlink()

    if verbose:
        print(f"[train] Training fruit_fly_minimax with {num_laps} training laps "
              f"on track '{track}'.")
        print(f"[train] Weights -> {weights_file}")
        print(f"[train] Results -> {json_file} (+ {png_file})")

    argv = [
        sys.executable,
        "evaluate.py",
        "agentic_gp.agents.fruit_fly_minimax",
        "--name", f"fruit_fly_minimax_{num_laps}lap",
        "--track", track,
        "--attempts", str(num_laps),
        "--agent-arg", f"weights_path={weights_file}",
    ]
    if plot:
        argv.append("--plot")

    env = os.environ.copy()
    env["FRUIT_FLY_WEIGHTS"] = str(weights_file)

    result = subprocess.run(argv, cwd=str(REPO_ROOT), env=env)
    if result.returncode != 0:
        raise RuntimeError(
            f"evaluate.py exited with code {result.returncode}. See output above."
        )

    # Read back the JSON evaluate.py wrote so we can return a summary.
    if not json_file.exists():
        raise RuntimeError(f"Expected {json_file} was not produced.")
    with open(json_file) as f:
        summary = json.load(f)

    # Read the best_lap from the weights file too (the CEM track).
    if weights_file.exists():
        with open(weights_file) as f:
            weights = json.load(f)
        summary["cem_best_lap_during_training"] = weights.get("best_lap")

    summary["weights_path"] = str(weights_file)
    summary["training_laps"] = num_laps
    return summary


def main(argv: list[str]) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--training-laps", type=int, required=True,
                        help=f"how many laps to CEM-train for (1..{MAX_TRAINING_LAPS})")
    parser.add_argument("--track", default="default", choices=["default", "oval"])
    parser.add_argument("--no-plot", action="store_true",
                        help="skip the track-plot PNG")
    args = parser.parse_args(argv)

    summary = train(
        num_laps=args.training_laps,
        track=args.track,
        plot=not args.no_plot,
    )
    print()
    print(f"[train] DONE  best_lap={summary['best_lap']}s  "
          f"crashes={summary['total_crashes']}  "
          f"weight file={summary['weights_path']}")
    print(f"[train] Saved {summary['training_laps']}-lap run to "
          f"results/fruit_fly_minimax_{summary['training_laps']}lap.{{json,png}}")
    return 0


if __name__ == "__main__":
    # Load .env first so MiniMax / RACE_LLM_* vars are available if needed.
    load_env(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"), override=False)
    sys.exit(main(sys.argv[1:]))