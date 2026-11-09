"""The race runner: load drivers, run sessions and grand prix events, write replays.

    from agentic_gp.race import load_driver, run_session, SessionConfig
    result = run_session("drivers/examples/l2_planner.py", SessionConfig(track="sprint", attempts=2))
    result.to_replay().to_json("races/sprint.json")

    from agentic_gp.race import run_grand_prix, GrandPrixConfig
    gp = run_grand_prix(["drivers/submissions"], GrandPrixConfig(race_laps=3))
    gp.save("races", "final")              # final.json (race replay), final_qualifying.json, final_results.json

Drivers never import this package (the runner owns the simulation).
"""

from .loader import (
    DriverDied, DriverError, DriverLoadError, DriverLoadTimeout, DriverTimeout, Issue, IsolatedDriver, LoadedDriver,
    check_driver_source, close_all_drivers, coerce_params, color_for, load_driver, parse_param_args,
)
from .replay import (
    Replay, ReplayCar, ReplayEvent, ReplayMessage, ReplayTrack, car_state_at, replay_car_from_trace,
    standings_at,
)
from .session import AttemptReport, SessionConfig, SessionEvent, SessionResult, iter_session, run_session
from .grand_prix import DriverEntry, GrandPrixConfig, GrandPrixResult, find_driver_files, run_grand_prix

__all__ = [
    "DriverDied", "DriverError", "DriverLoadError", "DriverLoadTimeout", "DriverTimeout", "Issue", "IsolatedDriver",
    "LoadedDriver",
    "check_driver_source", "close_all_drivers", "coerce_params", "color_for", "load_driver", "parse_param_args",
    "Replay", "ReplayCar", "ReplayEvent", "ReplayMessage", "ReplayTrack", "car_state_at", "replay_car_from_trace",
    "standings_at", "AttemptReport", "SessionConfig", "SessionEvent", "SessionResult", "iter_session",
    "run_session", "DriverEntry", "GrandPrixConfig", "GrandPrixResult", "find_driver_files", "run_grand_prix",
]
