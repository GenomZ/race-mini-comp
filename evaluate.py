"""Evaluate an agent on the competition track and record the result.

    python evaluate.py my_agent --attempts 3 --plot
    python evaluate.py agentic_gp.agents.hermes --name hermes --attempts 1
    python evaluate.py path/to/team_agent.py --name team-rocket --attempts 3 --plot

The agent module must expose:

    def run(env: RaceEnv, tools: list[StructuredTool], attempts: int = 1, **kwargs) -> None

Results are written to results/<name>.json (and results/<name>.png with --plot),
then aggregated into LEADERBOARD.md by `python -m agentic_gp.leaderboard`.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from agentic_gp import RaceEnv, PhysicsConfig, make_tools
from agentic_gp.tools import RaceTools
from agentic_gp.track import TRACKS
from agentic_gp.llm import describe_model


def load_agent(spec: str):
    """Import an agent by module name ("my_agent") or file path ("agents/foo.py")."""
    if spec.endswith(".py") or "/" in spec or "\\" in spec:
        path = Path(spec).resolve()
        mod_spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(mod_spec)
        sys.modules[path.stem] = module
        mod_spec.loader.exec_module(module)
        return module
    sys.path.insert(0, str(Path.cwd()))
    return importlib.import_module(spec)


def parse_kv(items: list[str]) -> dict:
    out = {}
    for item in items:
        k, _, v = item.partition("=")
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("agent", help="module name (my_agent, agentic_gp.agents.hermes) or path to a .py file")
    ap.add_argument("--name", help="entry name for the leaderboard (default: agent module name)")
    ap.add_argument("--attempts", type=int, default=1)
    ap.add_argument("--track", choices=sorted(TRACKS), default="default")
    ap.add_argument("--max-ticks", type=int, default=PhysicsConfig.max_ticks)
    ap.add_argument("--plot", action="store_true", help="save results/<name>.png of the best attempt")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--log-dir", default="logs")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--agent-arg", action="append", default=[], metavar="KEY=VALUE",
                    help="extra keyword argument passed to run(), e.g. --agent-arg history=3")
    args = ap.parse_args(argv)

    module = load_agent(args.agent)
    name = args.name or args.agent.rsplit(".", 1)[-1].replace(".py", "")
    cfg = PhysicsConfig(max_ticks=args.max_ticks)
    env = RaceEnv(track=TRACKS[args.track](), config=cfg, log_dir=args.log_dir, run_name=f"{name}_{int(time.time())}")
    raw = RaceTools(env)
    tools = make_tools(env, raw)

    print(f"== Evaluating '{name}' on track '{args.track}' ({env.track.length:.0f} units, "
          f"{env.track.n_gates} waypoints), {args.attempts} attempt(s), max {cfg.max_ticks} ticks each ==")
    t0 = time.perf_counter()
    try:
        module.run(env, tools, attempts=args.attempts, verbose=not args.quiet, **parse_kv(args.agent_arg))
    except KeyboardInterrupt:
        print("\n[interrupted]")
    finally:
        env.close()
    wall = time.perf_counter() - t0

    attempts = [a.to_dict() for a in env.attempts]
    result = {
        "name": name,
        "agent": args.agent,
        "track": args.track,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": describe_model() if "hermes" in args.agent or "llm" in args.agent else "-",
        "attempts": attempts,
        "best_lap": env.best_lap,
        "completed_attempts": sum(1 for a in env.attempts if a.completed),
        "max_waypoints_passed": max([a.waypoints_passed for a in env.attempts], default=0),
        "total_crashes": sum(a.crashes for a in env.attempts),
        "tool_calls": len(raw.calls),
        "wall_clock_seconds": round(wall, 1),
        "log": str(env.log_path) if env.log_path else None,
    }

    print("\n-- Summary --")
    for a in env.attempts:
        status = f"lap {a.lap_time:.2f}s" if a.completed else f"DNF ({a.ended_by})"
        print(f"attempt {a.attempt}: {status:>18}  sim {a.sim_time:6.1f}s  penalties {a.penalty_time:5.1f}s  "
              f"crashes {a.crashes:2d}  waypoints {a.waypoints_passed:2d}/{env.track.n_gates}  ticks {a.ticks}")
    best = "none" if env.best_lap is None else f"{env.best_lap:.2f}s"
    print(f"best lap: {best} | tool calls: {len(raw.calls)} | wall clock: {wall:.1f}s")

    if not args.no_save:
        out_dir = Path(args.results_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"{name}.json"
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"saved {out}")
        if args.plot:
            from agentic_gp.render import plot_env

            png = plot_env(env, out_dir / f"{name}.png")
            if png:
                print(f"saved {png}")
    return result


if __name__ == "__main__":
    main()
