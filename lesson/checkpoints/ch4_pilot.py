"""HANDRAIL CP4 (Ch4 Talking to MiniMax) - <your_handle>_pilot.py: the LLM drives, every tick.

This is a LESSON TOOL, not your race car: your main file stays the Ch3 learner. The pilot is your Ch1
reflex file with the LLM put in charge of drive(): the sensors go to MiniMax as JSON every tick, the
answer {"throttle", "steering", "say"} is validated and driven. After MAX_LLM_TICKS LLM decisions
(class budget: 20) - or whenever the answer is unusable, or the LLM is not allowed - your Ch1
reflexes drive. Official events forbid LLM calls in drive(), so there it is just your Ch1 car.

Stuck in Ch4? Copy this file to drivers/submissions/<your_handle>_pilot.py. Compare it with your own:

    git diff --no-index lesson/checkpoints/ch1_reflex.py lesson/checkpoints/ch4_pilot.py

What the Ch4 steps added to the Ch1 file:
    2-3. llm.chat_json(sensors as JSON, system=SYSTEM, effort="none") -> {"throttle", "steering", "say"}
    5.   the prompt knows the sign rule ("+ = RIGHT") and the stopping rule - re-run the exam to compare
    -    ask_llm() validates: numbers, finite, clipped to -1..1; anything else -> None -> reflexes
    -    _memory["latencies"] keeps the seconds of every call (p50 / p95 for the Latency Lab)

Run it (real calls on the shared key: at most 20 per drive, 1 per exam state):
    python race.py exam lesson/checkpoints/ch4_pilot.py --states races/demo/exam_states.json --n 10
    (own states: python race.py drive X.py --dump-states races/exam_states.json, then --states that)
    python race.py drive lesson/checkpoints/ch4_pilot.py                 # turn-based: the sim waits
    python race.py drive lesson/checkpoints/ch4_pilot.py --realtime      # the world does NOT wait
    LLM_PROVIDER=mock python race.py drive lesson/checkpoints/ch4_pilot.py   # (PowerShell: $env:LLM_PROVIDER="mock")
With the mock LLM every answer is unusable ("{}"), so the Ch1 reflexes drive the whole lap - that is
the fallback working. Measured numbers: lesson/instructor/CALIBRATION.md.

Exercise: ticks per call = p50 / 0.5. What is the best lap a realtime LLM driver could hope for?
Gold: a hybrid - every 10 ticks the LLM only picks "push" or "careful" (a cap), the code steers.
"""

from __future__ import annotations

import json
import math
import time

from agentic_gp import driverkit as dk
from agentic_gp import llm

# --------------------------------------------------------------------------- TODO 0: who are you?
NAME = "CP4 Pilot"          # <= 16 characters, shown in the standings and the 3D viewer
AUTHOR = "your_handle"      # your handle (pseudonyms welcome)
COLOR = "#7209b7"           # your car colour (hex)
KIND = "llm"                # "code" | "llm" | "fly"
MODEL = "MiniMax-M3"
MOTTO = "I think, therefore I'm late."
ANATOMY = ["perceive", "act", "llm"]

# Numeric knobs: tune them with --param MAX_LLM_TICKS=5 (and Streamlit sliders) without editing code.
PARAMS = {
    "MAX_LLM_TICKS": 20,    # Ch4: LLM decisions per run (class budget), then the reflexes drive
    # your Ch1 reflexes (the fallback)
    "cap": 34.0, "steer_gain": 1.0, "margin": 4.0, "wall_gain": 1.5,
    "blend_far": 40.0, "blend_max": 0.6, "turn_k": 0.6,
}

# Ch4 step 5: the sign rule and the stopping rule are IN the prompt (re-run the exam to see the effect).
SYSTEM = ("You drive a race car in a 2D sim, one decision every 0.5 s. Sign rule: + = RIGHT, - = LEFT for "
          "every angle and for steering. Steer towards the next waypoint: steering ~ angle_to_next_waypoint / 30. "
          "Stopping rule: brake (throttle < 0) when distance_to_wall_front < speed * speed / 50 + 10. "
          'Reply ONLY with JSON: {"throttle": -1..1, "steering": -1..1, "say": "<= 6 words"}')

# The ten sensor values the model gets (short prompt = fast answer).
KEYS = ("speed", "distance_to_wall_front", "distance_to_wall_left_45", "distance_to_wall_right_45",
        "distance_to_wall_left_90", "distance_to_wall_right_90", "angle_to_next_waypoint",
        "distance_to_next_waypoint", "angle_to_waypoint_after_next", "zone_speed_limit")

# The seven wall rays in the driver frame: (angle, obs key). + = right, - = left.
RAYS = [(-90, "left_90"), (-45, "left_45"), (-20, "left_20"), (0, "front"),
        (20, "right_20"), (45, "right_45"), (90, "right_90")]

_memory: dict = {"llm_ticks": 0, "failures": 0, "latencies": []}     # + recover_if_stuck's state


def drive(obs: dict) -> dict:
    # ----------------------------------------------------------------------- Ch4: the LLM decides
    if _memory["llm_ticks"] < PARAMS["MAX_LLM_TICKS"] and llm.llm_allowed():
        _memory["llm_ticks"] += 1
        action = ask_llm(obs)
        if action is not None:
            return action
        _memory["failures"] += 1
    # ----------------------------------------------------------------------- budget used / bad answer
    throttle, steering = reflexes(obs)
    return {"throttle": throttle, "steering": steering}


def ask_llm(obs: dict) -> dict | None:
    """One LLM decision, validated. None if anything goes wrong (the reflexes take this tick)."""
    sensors = {k: (round(obs[k], 1) if isinstance(obs.get(k), float) else obs.get(k)) for k in KEYS}
    t0 = time.perf_counter()
    try:
        answer = llm.chat_json(json.dumps(sensors), system=SYSTEM, effort="none", max_tokens=200)
    except llm.LLMError:
        return None
    finally:
        _memory["latencies"].append(round(time.perf_counter() - t0, 3))
    try:
        throttle, steering = float(answer["throttle"]), float(answer["steering"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (math.isfinite(throttle) and math.isfinite(steering)):
        return None
    say = str(answer.get("say") or "")[:60] or None
    return {"throttle": dk.clamp(throttle), "steering": dk.clamp(steering), "say": say}


# =========================================================================== your Ch1 reflexes (fallback)
def free_distance(obs: dict, angle: float) -> float:
    """How far we can see in direction `angle` (between the two nearest rays)."""
    angle = max(-90.0, min(90.0, angle))
    for (a0, k0), (a1, k1) in zip(RAYS, RAYS[1:]):
        if a0 <= angle <= a1:
            d0, d1 = obs["distance_to_wall_" + k0], obs["distance_to_wall_" + k1]
            return d0 + (angle - a0) / (a1 - a0) * (d1 - d0)
    return obs["distance_to_wall_front"]


def reflexes(obs: dict) -> tuple[float, float]:
    """Your Ch1 drive(obs), unchanged (only renamed)."""
    P = PARAMS
    recover = dk.recover_if_stuck(obs, _memory)
    if recover is not None:
        return recover
    d_next = obs["distance_to_next_waypoint"]
    w = dk.clamp(1.0 - (d_next - 10.0) / (P["blend_far"] - 10.0), 0.0, P["blend_max"])
    angle = (1.0 - w) * obs["angle_to_next_waypoint"] + w * obs["angle_to_waypoint_after_next"]
    if free_distance(obs, angle) < min(d_next, 40.0):
        open_angle = max((a for a, _ in RAYS if abs(a) <= 45), key=lambda a: free_distance(obs, a))
        angle = 0.5 * angle + 0.5 * open_angle
    steering = P["steer_gain"] * dk.steer_towards(angle, obs["speed"])
    front = obs["distance_to_wall_front"]
    v_stop = math.sqrt(2 * 25.0 * max(front - P["margin"] - 0.5 * obs["speed"], 0.0))
    v_turn = math.sqrt(20.0 * P["turn_k"] * max(front, 0.0))
    turn_ahead = max(abs(angle), 0.7 * abs(obs["angle_to_waypoint_after_next"]) if d_next < P["blend_far"] else 0.0)
    v_corner = P["cap"] * max(0.35, math.cos(math.radians(min(turn_ahead, 80.0))))
    target = max(5.0, min(P["cap"], v_stop, v_turn, v_corner))
    if obs["zone_speed_limit"] is not None:
        target = min(target, obs["zone_speed_limit"] - 1.0)
    dv = target - obs["speed"]
    throttle = dk.clamp((dv + 0.075 * obs["speed"]) / 6.0 if dv >= 0 else dv / 12.5)
    steering += dk.wall_repulsion(obs, P["wall_gain"])
    return throttle, dk.clamp(steering)
