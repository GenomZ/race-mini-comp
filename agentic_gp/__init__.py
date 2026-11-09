"""Agentic Grand Prix - a deterministic 2D racing gym for single-file AI drivers.

    from agentic_gp import run_session, SessionConfig
    result = run_session("drivers/examples/l2_planner.py", SessionConfig(track="sprint", attempts=2))
    print(result.best_lap)
    result.to_replay().to_json("races/sprint.json")       # for the Streamlit / HTML / Unity viewers

    from agentic_gp import run_grand_prix, GrandPrixConfig
    gp = run_grand_prix(["drivers/submissions"], GrandPrixConfig(track="grand_prix", race_laps=3))
    gp.save("races", "final")

Public API:
    RaceEnv, PhysicsConfig, Track, TRACKS, make_track     - the simulation (agentic_gp.sim)
    load_driver, run_session, SessionConfig               - one driver, one track (agentic_gp.race)
    run_grand_prix, GrandPrixConfig, Replay               - many drivers + the replay JSON contract
Driver files only import agentic_gp.driverkit and agentic_gp.llm (never the sim or the runner).
Command line: `python race.py --help` (agentic_gp/cli.py).
"""

from .sim.config import PhysicsConfig
from .sim.track import TRACKS, Track, make_track
from .sim.env import RaceEnv
from .race.loader import load_driver
from .race.session import SessionConfig, run_session
from .race.grand_prix import GrandPrixConfig, run_grand_prix
from .race.replay import Replay

__all__ = ["RaceEnv", "PhysicsConfig", "Track", "TRACKS", "make_track", "load_driver", "run_session",
           "SessionConfig", "run_grand_prix", "GrandPrixConfig", "Replay"]
__version__ = "2.0.0"
