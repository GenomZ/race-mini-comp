"""Agentic Grand Prix - a lightweight 2D racing gym for LLM agents.

Public API:
    RaceEnv          - the simulation environment (reset / step / sensors)
    PhysicsConfig    - tunable physics + competition constants
    Track            - track geometry (walls, centerline, waypoint gates)
    make_default_track()
    make_tools(env)  - LangChain StructuredTool bindings for an env
"""

from .config import PhysicsConfig
from .track import Track, make_default_track
from .env import RaceEnv
from .tools import make_tools

__all__ = ["RaceEnv", "PhysicsConfig", "Track", "make_default_track", "make_tools"]
__version__ = "0.1.0"
