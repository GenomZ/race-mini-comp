"""The driver protocol: what a driver file looks like and what it receives.

A driver is ONE .py file. Either a function style driver:

    NAME = "Rocket"; AUTHOR = "Ada"; COLOR = "#e63946"; KIND = "code"   # optional metadata

    def drive(obs: dict):              # REQUIRED: called every tick (0.5 s of sim time)
        return 1.0, 0.0                # (throttle, steering), both in [-1, 1]

    def plan(track: dict): ...         # optional: once per session, before the first attempt
    def reflect(report: dict): ...     # optional: after every attempt (your memory / learning hook)

or a class named `Driver` (instantiated once with no arguments; the SAME instance drives every attempt
of a session, so attributes are your memory):

    class Driver:
        NAME = "Max"; KIND = "llm"; MODEL = "MiniMax-M3"
        PARAMS = {"aggression": 1.0}   # optional tunable knobs (CLI --param, Streamlit sliders)
        def plan(self, track): ...
        def drive(self, obs): return {"throttle": 1.0, "steering": 0.0, "say": "flat out!"}
        def reflect(self, report): ...

`drive` may return (throttle, steering), (throttle, steering, say), a dict with "throttle", "steering",
optional "say" (aliases "message", "thought") and optional "debug" (small JSON dict for live views),
or None (= coast). Anything odd becomes 0; values are clipped to [-1, 1].

Driver-frame convention everywhere: "+ = right" (steering, ray and waypoint angles, offset_from_center).

OBS_FIELDS, TRACK_FIELDS and REPORT_FIELDS below document every key a driver receives; they are the
single source of truth for the lesson docs and the Streamlit app.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

KINDS = ("code", "llm", "fly", "human", "other")
ANATOMY_TAGS = ("perceive", "act", "map", "plan", "memory", "reflect", "llm", "tools", "roles", "spiking")
PHASES = ("practice", "qualifying", "race", "session")

SAY_MAX_CHARS = 160
DEBUG_MAX_BYTES = 2048


@dataclass(frozen=True)
class Action:
    """One decision of a driver: controls + an optional radio message + optional debug info."""

    throttle: float = 0.0      # -1 full brake/reverse .. +1 full throttle
    steering: float = 0.0      # -1 full left .. +1 full right
    say: str | None = None     # "radio message" shown in viewers (<= 160 chars)
    debug: dict | None = None  # small JSON dict for live views (session frames only, never replays)

    def as_tuple(self) -> tuple[float, float]:
        return self.throttle, self.steering


COAST = Action()


def _control(value: Any) -> float:
    """A number in [-1, 1]; anything that is not a finite number becomes 0."""
    if isinstance(value, bool):
        return float(value)
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(f) or math.isinf(f):
        return 0.0
    return max(-1.0, min(1.0, f))


def _say(value: Any) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).split())       # one line, no runs of whitespace
    return text[:SAY_MAX_CHARS] or None


def _debug(value: Any) -> dict | None:
    """Keep a debug dict only if it is JSON-serialisable and small (<= 2 KB)."""
    if not isinstance(value, dict) or not value:
        return None
    try:
        text = json.dumps(value, allow_nan=False, default=_json_default)
    except (TypeError, ValueError):
        return {"error": "debug dict is not JSON-serialisable"}
    if len(text) > DEBUG_MAX_BYTES:
        return {"error": f"debug dict too large ({len(text)} bytes > {DEBUG_MAX_BYTES})"}
    return json.loads(text)


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "tolist"):                # numpy arrays and scalars
        return obj.tolist()
    raise TypeError(f"not JSON-serialisable: {type(obj).__name__}")


def normalize_action(raw: Any) -> Action:
    """Turn whatever drive() returned into a clean Action.

        (0.8, -0.2)                         -> Action(0.8, -0.2)
        (1, 0, "flat out")                  -> Action(1.0, 0.0, say="flat out")
        {"throttle": 2, "steering": "x"}    -> Action(1.0, 0.0)          (clipped / not a number -> 0)
        {"throttle": 1, "message": "hi"}    -> Action(1.0, 0.0, say="hi")
        None                                -> Action(0.0, 0.0)          (coast)
    """
    if raw is None:
        return COAST
    if isinstance(raw, Action):
        return Action(_control(raw.throttle), _control(raw.steering), _say(raw.say), _debug(raw.debug))
    if isinstance(raw, dict):
        say = next((raw[k] for k in ("say", "message", "thought") if raw.get(k) is not None), None)
        return Action(_control(raw.get("throttle")), _control(raw.get("steering")), _say(say),
                      _debug(raw.get("debug")))
    if isinstance(raw, (list, tuple)) or hasattr(raw, "tolist"):
        seq = raw.tolist() if hasattr(raw, "tolist") else list(raw)
        if not isinstance(seq, (list, tuple)) or not seq:     # a numpy scalar / empty array
            return COAST
        throttle = _control(seq[0])
        steering = _control(seq[1]) if len(seq) > 1 else 0.0
        say = _say(seq[2]) if len(seq) > 2 else None
        return Action(throttle, steering, say)
    return COAST


ACTION_KEYS = ("throttle", "steering", "say", "message", "thought", "debug")


def _is_number(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _number_problem(name: str, value: Any) -> str | None:
    """Why `value` is not a usable throttle/steering (None = it is; out-of-range values are clipped)."""
    if isinstance(value, bool):
        return None
    if not _is_number(value):
        return f"drive() returned {name} = {value!r} - it must be a number in [-1, 1]"
    if not math.isfinite(float(value)):
        return (f"drive() returned {name} = {float(value)!r} - not a finite number (a division by zero?); "
                "the car coasts this tick")
    return None


def action_problem(raw: Any) -> str | None:
    """Why drive()'s return value is not an action (None = it is one). The runner records the reason
    as a driver error; normalize_action() turns the same value into COAST.

        action_problem((1, 0)) -> None          action_problem("full throttle") -> "drive() returned str ..."
    """
    if raw is None:
        return None
    if isinstance(raw, Action):
        return _number_problem("throttle", raw.throttle) or _number_problem("steering", raw.steering)
    if isinstance(raw, dict):
        if any(k in raw for k in ACTION_KEYS):
            for name in ("throttle", "steering"):
                value = raw.get(name)
                problem = None if value is None else _number_problem(name, value)
                if problem:
                    return problem
            return None
        keys = ", ".join(map(str, list(raw)[:6])) or "none"
        return (f"drive() returned a dict without 'throttle'/'steering' (its keys: {keys}) - "
                'return {"throttle": t, "steering": s} or (t, s)')
    if isinstance(raw, (list, tuple)) or hasattr(raw, "tolist"):
        seq = raw.tolist() if hasattr(raw, "tolist") else list(raw)
        if not isinstance(seq, (list, tuple)):              # a numpy scalar, e.g. np.float64(0.7)
            return f"drive() returned one number ({seq!r}) - return (throttle, steering)"
        if not seq:
            return f"drive() returned an empty {type(raw).__name__} - return (throttle, steering)"
        for name, value in zip(("throttle", "steering"), seq[:2]):
            problem = _number_problem(name, value)
            if problem:
                return problem
        return None
    if isinstance(raw, (int, float)):
        return f"drive() returned one number ({raw!r}) - return (throttle, steering)"
    text = repr(raw)
    return (f"drive() returned {type(raw).__name__} {text[:40]}{'...' if len(text) > 40 else ''} - "
            "return (throttle, steering), a dict or an Action")


class Driver:
    """Optional base class: inherit it to get no-op plan/reflect and default metadata."""

    NAME: str | None = None
    AUTHOR: str = "anonymous"
    COLOR: str | None = None
    KIND: str = "code"
    MODEL: str | None = None
    MOTTO: str = ""
    ANATOMY: list[str] = []
    PARAMS: dict = {}

    def plan(self, track: dict) -> None:
        """Called once per session with the track map (see TRACK_FIELDS)."""

    def drive(self, obs: dict) -> Any:
        """Called every tick with the observation (see OBS_FIELDS). Default: coast."""
        return 0.0, 0.0

    def reflect(self, report: dict) -> None:
        """Called after every attempt with the lap report (see REPORT_FIELDS)."""


# --------------------------------------------------------------------------- documentation
OBS_FIELDS: dict[str, str] = {
    "speed": "current forward speed (units/s, 0..40; negative when reversing)",
    "distance_to_wall_front": "distance to the wall straight ahead (units, capped at 200)",
    "distance_to_wall_left_20": "ray 20 deg to the left of the heading",
    "distance_to_wall_right_20": "ray 20 deg to the right of the heading",
    "distance_to_wall_left_45": "ray 45 deg to the left",
    "distance_to_wall_right_45": "ray 45 deg to the right",
    "distance_to_wall_left_90": "ray 90 deg to the left (distance to the left wall beside you)",
    "distance_to_wall_right_90": "ray 90 deg to the right",
    "distance_to_next_waypoint": "straight-line distance to the centre of the next waypoint gate",
    "angle_to_next_waypoint": "angle from your heading to the next waypoint (deg, + = right, -180..180)",
    "distance_to_waypoint_after_next": "distance to the waypoint after the next one",
    "angle_to_waypoint_after_next": "angle to the waypoint after the next one (deg, + = right)",
    "next_waypoint_index": "index of the next gate to cross (gate 0 = start/finish line)",
    "waypoints_passed": "gates crossed so far in this attempt (all laps)",
    "waypoints_total": "number of gates per lap",
    "position": "[x, y] world position (for maps; the track dict uses the same coordinates)",
    "heading_deg": "world heading (deg, counter-clockwise from +x, like the track map)",
    "offset_from_center": "signed distance from the centerline (units, + = right of it, - = left)",
    "segment": "waypoint segment you are on = the gate you passed last ((next_waypoint_index - 1) mod total)",
    "progress": "fraction of the CURRENT lap done (0 at the start line .. ~1 just before finishing)",
    "lap": "current lap, 1-based (stays at laps_total after finishing)",
    "laps_total": "laps in this attempt",
    "race_progress": "laps completed + progress (used for live standings)",
    "sim_time": "simulated seconds since the start of the attempt",
    "penalty_time": "penalty seconds so far (5 s per crash + 1 s per tick too fast in a zone)",
    "race_time": "the race clock = sim_time + penalty_time (what you are ranked on)",
    "crashes": "wall hits so far in this attempt",
    "tick": "ticks done in this attempt (one tick = dt seconds)",
    "ticks_remaining": "ticks left before the attempt ends as DNF (500 per lap)",
    "finished": "True once all laps are done",
    "attempt_over": "True when the attempt has ended (finished, DNF, stopped)",
    "zone_speed_limit": "speed limit of the zone you are in (e.g. a yellow flag), None outside zones",
    "phase": "'practice' | 'qualifying' | 'race' | 'session' (set by the runner)",
    "dt": "seconds per tick (0.5): how long your answer is applied before you are asked again",
    "events": "what happened in the tick just driven, as text: 'waypoint_passed: 3', 'crash: ...', "
              "'lap_complete: ...', 'zone_violation: ...', 'finished: ...' ([] at the start)",
}

TRACK_FIELDS: dict[str, str] = {
    "key": "registry key of the track (e.g. 'grand_prix', 'random:7')",
    "name": "display name",
    "track_width": "wall-to-wall width (units)",
    "half_width": "centerline-to-wall distance (units)",
    "lap_length": "centerline length of one lap (units)",
    "n_waypoints": "number of waypoint gates per lap",
    "start_position": "[x, y] of the start grid (just behind gate 0)",
    "start_heading_deg": "start heading (deg, counter-clockwise from +x)",
    "waypoints": "list of {index, center, track_heading_deg, min_turn_radius_until_next, centerline_index}",
    "centerline": "dense centerline [[x, y], ...] (~4 units apart; plan your speeds on THIS)",
    "left_wall": "left wall polyline [[x, y], ...] (same indexing as centerline)",
    "right_wall": "right wall polyline [[x, y], ...]",
    "corners": "auto-detected corners [{id 'T1', start_index, apex_index, end_index, min_radius, direction, "
               "v_max, from_wp, to_wp}]",
    "briefing": "race-control text (yellow flags / speed limits are ONLY announced here)",
    "notes": "hints about sensors and rules",
    "physics": "physics constants: tick_seconds, max_speed, accel, brake, drag, max_lateral_accel, ...",
    "laps": "laps per attempt in this session",
}

REPORT_FIELDS: dict[str, str] = {
    "attempt": "attempt number in this session (1-based)",
    "phase": "phase of this attempt: 'practice' | 'qualifying' | 'race' | 'session'",
    "next_phase": "what comes next: 'practice' | 'qualifying' | 'race' | None (session over)",
    "completed": "True if all laps were finished",
    "race_time": "sim time + penalties (None if DNF)",
    "lap_times": "race-clock time of every completed lap (penalties included)",
    "best_lap": "fastest lap of this attempt (None if no lap)",
    "best_lap_so_far": "fastest lap of the whole session so far",
    "sim_time": "simulated seconds driven",
    "penalty_time": "penalty seconds (crashes + zone violations)",
    "crashes": "number of wall hits",
    "zone_violations": "list of {tick, t, lap, segment, speed, limit} ticks that ended too fast in a zone",
    "waypoints_passed": "gates crossed",
    "laps": "laps planned",
    "laps_completed": "laps finished",
    "ticks": "ticks driven",
    "ended_by": "'finished' | 'max_ticks' | 'stopped' | 'stalled' | 'timeout' | 'reset'",
    "crash_log": "list of {tick, t, lap, segment, progress, position, speed_before, throttle, steering, rays, "
                 "corner}",
    "segments": "per waypoint segment: {segment, time, entry_speed, min_speed, max_speed, avg_speed, crashes, ticks}",
    "corners": "per corner: {id, direction, v_max, passes, time, entry_speed, min_speed, exit_speed, crashes}",
    "telemetry": "per tick: {tick, t, lap, segment, progress, x, y, speed, throttle, steering, offset_from_center}",
    "stalls": "ticks where drive() answered too late (the car coasted)",
    "driver_errors": "list of {tick, phase, error} for exceptions in your code",
    "llm_usage": "LLM usage this session so far: {calls, prompt_tokens, completion_tokens, seconds, errors, "
                 "mode, calls_left}",
    "summary_text": "compact human/LLM-readable summary of all of the above (paste it into a prompt)",
}
