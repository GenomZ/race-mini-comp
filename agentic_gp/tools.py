"""LangChain tool bindings for a RaceEnv.

    from agentic_gp import RaceEnv, make_tools
    env = RaceEnv()
    tools = make_tools(env)          # [read_sensors, drive, reset_environment, get_track_map]
    by_name = {t.name: t for t in tools}
    by_name["drive"].invoke({"throttle": 0.8, "steering": 0.0})

Every tool returns a JSON string so that it can be dropped straight into an LLM
conversation. The same functions are also exposed as plain Python callables via
`RaceTools` for agents that mix LLM planning with code-level execution.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from .env import RaceEnv


class DriveInput(BaseModel):
    # Bounds are documented rather than enforced: out-of-range values are clipped by
    # the environment so a slightly wrong LLM output does not abort the race.
    throttle: float = Field(
        ...,
        description="-1.0 = full brake (or slow reverse when stopped), 0 = coast, 1.0 = full acceleration. "
        "Values outside [-1, 1] are clipped.",
    )
    steering: float = Field(
        ...,
        description="-1.0 = full left, 0 = straight, 1.0 = full right. Values outside [-1, 1] are clipped.",
    )


class NoInput(BaseModel):
    pass


SENSOR_DOC = (
    "JSON with: speed; distance_to_wall_front / _left_20 / _right_20 / _left_45 / _right_45 / "
    "_left_90 / _right_90 (LiDAR-like ray distances, max 200); distance_to_next_waypoint; "
    "angle_to_next_waypoint (degrees, negative = waypoint is to the LEFT, positive = RIGHT); "
    "distance/angle_to_waypoint_after_next; next_waypoint_index; waypoints_passed; position [x, y]; "
    "heading_deg; sim_time; penalty_time; crashes; tick; ticks_remaining; lap_complete; attempt_over."
)


class RaceTools:
    """Plain-Python versions of the tools (handy for code-level executors)."""

    def __init__(self, env: RaceEnv, log_calls: bool = True) -> None:
        self.env = env
        self.calls: list[dict[str, Any]] = []
        self._log_calls = log_calls

    def _record(self, name: str, args: dict, result: dict) -> None:
        if self._log_calls:
            self.calls.append({"tool": name, "args": args, "tick": result.get("tick")})

    def read_sensors(self) -> dict:
        obs = self.env.sensors()
        self._record("read_sensors", {}, obs)
        return obs

    def drive(self, throttle: float, steering: float) -> dict:
        obs = self.env.step(throttle, steering)
        self._record("drive", {"throttle": throttle, "steering": steering}, obs)
        return obs

    def reset_environment(self) -> dict:
        obs = self.env.reset()
        self._record("reset_environment", {}, obs)
        return obs

    def get_track_map(self) -> dict:
        m = self.env.track_map()
        self._record("get_track_map", {}, {"tick": self.env.tick})
        return m


def _dumps(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"))


def make_tools(env: RaceEnv, raw: RaceTools | None = None) -> list[StructuredTool]:
    """Build the four LangChain tools bound to `env`."""
    rt = raw or RaceTools(env)

    read_sensors = StructuredTool.from_function(
        func=lambda: _dumps(rt.read_sensors()),
        name="read_sensors",
        description="ReadSensorsTool: returns the current state of the car without advancing time. " + SENSOR_DOC,
        args_schema=NoInput,
    )

    def _drive(throttle: float, steering: float) -> str:
        return _dumps(rt.drive(throttle, steering))

    drive = StructuredTool.from_function(
        func=_drive,
        name="drive",
        description=(
            "DriveTool: apply throttle and steering, advance the simulation by 0.5 virtual seconds "
            "and return the new sensor reading plus an 'events' list (crashes, waypoints passed, lap_complete). "
            "Hitting a wall sets speed to 0 and adds a 5 second penalty. " + SENSOR_DOC
        ),
        args_schema=DriveInput,
    )

    reset_environment = StructuredTool.from_function(
        func=lambda: _dumps(rt.reset_environment()),
        name="reset_environment",
        description=(
            "ResetTool: put the car back on the start line and start a NEW attempt (the previous attempt "
            "is recorded as-is). Only use this when starting a new attempt."
        ),
        args_schema=NoInput,
    )

    get_track_map = StructuredTool.from_function(
        func=lambda: _dumps(rt.get_track_map()),
        name="get_track_map",
        description=(
            "TrackMapTool: static description of the circuit - waypoint centers, track heading at each "
            "waypoint, tightest corner radius between waypoints, track width, lap length and the physics "
            "constants. Useful for planning a lap before driving."
        ),
        args_schema=NoInput,
    )

    return [read_sensors, drive, reset_environment, get_track_map]


def tools_by_name(tools: list[StructuredTool]) -> dict[str, StructuredTool]:
    return {t.name: t for t in tools}
