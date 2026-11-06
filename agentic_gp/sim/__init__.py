"""The simulation: physics, tracks and the single-car RaceEnv.

    from agentic_gp.sim import RaceEnv, make_track
    env = RaceEnv(track=make_track("sprint"), laps=2)

Drivers never import this package; the race runner owns the env.
"""

from .config import PhysicsConfig, DEFAULT_CONFIG
from .track import Track, Zone, TRACKS, PUBLIC_TRACKS, make_track, make_random_track, load_track_file
from .env import RaceEnv, AttemptResult, OBS_KEYS

__all__ = [
    "PhysicsConfig", "DEFAULT_CONFIG", "Track", "Zone", "TRACKS", "PUBLIC_TRACKS", "make_track",
    "make_random_track", "load_track_file", "RaceEnv", "AttemptResult", "OBS_KEYS",
]
