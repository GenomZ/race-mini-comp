"""HANDRAIL CP1 (Ch1 Reflexes) - the template after all five Ch1 steps: a simple reflex agent.

Stuck in Ch1? Copy this file over your drivers/submissions/<your_handle>.py, put your NAME / AUTHOR /
COLOR back, and carry on with Ch2. Compare it with your own file first - the variable names are the
template's, so the diff shows exactly which lines each step changed:

    git diff --no-index drivers/template.py lesson/checkpoints/ch1_reflex.py

What the Ch1 steps changed (TODO numbers as in drivers/template.py):
    TODO 1  AIM: look through the corner - blend in the waypoint after next as the gate gets close
    TODO 1  steer_towards(angle, speed): "the angle I want" -> "the command I need" at this speed
    TODO 2  SPEED: higher cap, smaller margin, plus two limits: "can I still TURN AWAY from the wall
            ahead?" (v = sqrt(grip * k * front)) and "how sharp is the turn after the next gate?"
    TODO 3  WALLS: a gate hidden behind a wall -> aim halfway towards the most open ray (line of sight)
    TODO 3b recover_if_stuck stays: insurance. This finished file never gets stuck, so deleting it
            changes nothing here (66.70 s, 0 crashes even with --crash-rule classic); the Chaser demo
            in Ch1 shows the crash loop it protects against

Still a simple reflex agent: it only sees this tick's obs (no map, no memory), so it cannot brake
for a corner it cannot see yet, and it cannot learn.

Check it (numbers measured, see lesson/instructor/CALIBRATION.md):
    python race.py check lesson/checkpoints/ch1_reflex.py
    python race.py drive lesson/checkpoints/ch1_reflex.py               # grand_prix, clean (CP1 band)
    python race.py validate lesson/checkpoints/ch1_reflex.py            # every public track, clean

Push (Gold): --param cap=40 and friends: faster on grand_prix? Now try sprint and speedway too.
"""

from __future__ import annotations

import math

from agentic_gp import driverkit as dk

# --------------------------------------------------------------------------- TODO 0: who are you?
NAME = "CP1 Reflex"         # <= 16 characters, shown in the standings and the 3D viewer
AUTHOR = "your_handle"      # your handle (pseudonyms welcome)
COLOR = "#0096c7"           # your car colour (hex)
KIND = "code"               # "code" | "llm" | "fly"
MOTTO = "See wall, turn early."
ANATOMY = ["perceive", "act"]          # add "map", "plan", "memory", "reflect", "llm", "tools" as you grow

# Numeric knobs: tune them with --param cap=30 (and Streamlit sliders) without editing code.
PARAMS = {
    "cap": 34.0,            # TODO 2: our top speed (the car can do 40)
    "steer_gain": 1.0,      # TODO 1: how hard we steer towards the waypoint
    "margin": 4.0,          # TODO 2: free space we keep in front of the car
    "wall_gain": 1.5,       # TODO 3: how hard we push away from close walls
    "blend_far": 40.0,      # TODO 1: start looking through to the next-but-one gate at this distance
    "blend_max": 0.6,       # TODO 1: ... and put at most this much weight on it
    "turn_k": 0.6,          # TODO 2: "turn away" limit v = sqrt(20 * turn_k * front)
}

# The seven wall rays in the driver frame: (angle, obs key). + = right, - = left.
RAYS = [(-90, "left_90"), (-45, "left_45"), (-20, "left_20"), (0, "front"),
        (20, "right_20"), (45, "right_45"), (90, "right_90")]

_memory: dict = {}          # recover_if_stuck keeps its little state here


def free_distance(obs: dict, angle: float) -> float:
    """How far we can see in direction `angle` (between the two nearest rays)."""
    angle = max(-90.0, min(90.0, angle))
    for (a0, k0), (a1, k1) in zip(RAYS, RAYS[1:]):
        if a0 <= angle <= a1:
            d0, d1 = obs["distance_to_wall_" + k0], obs["distance_to_wall_" + k1]
            return d0 + (angle - a0) / (a1 - a0) * (d1 - d0)
    return obs["distance_to_wall_front"]


def drive(obs: dict) -> tuple[float, float]:
    P = PARAMS

    # ----------------------------------------------------------------------- TODO 3b: unstick
    # Stopped with a wall in front of you? Reverse towards the open side for a moment.
    recover = dk.recover_if_stuck(obs, _memory)
    if recover is not None:
        return recover

    # ----------------------------------------------------------------------- TODO 1: steer to the waypoint
    # Look through the corner: the closer the gate, the more we aim at the waypoint after it.
    d_next = obs["distance_to_next_waypoint"]
    w = dk.clamp(1.0 - (d_next - 10.0) / (P["blend_far"] - 10.0), 0.0, P["blend_max"])
    angle = (1.0 - w) * obs["angle_to_next_waypoint"] + w * obs["angle_to_waypoint_after_next"]
    # Line of sight: is there a wall in the aim direction, nearer than the gate? Then the gate is around
    # a corner - aim halfway towards the most open ray instead (TODO 3 upgrade).
    if free_distance(obs, angle) < min(d_next, 40.0):
        open_angle = max((a for a, _ in RAYS if abs(a) <= 45), key=lambda a: free_distance(obs, a))
        angle = 0.5 * angle + 0.5 * open_angle
    steering = P["steer_gain"] * dk.steer_towards(angle, obs["speed"])

    # ----------------------------------------------------------------------- TODO 2: control speed
    # Never faster than you can stop before the wall ahead: v = sqrt(2 * brake * distance), brake = 25.
    # Nor faster than you can still turn away from it: v = sqrt(grip * turn_k * distance), grip = 20.
    front = obs["distance_to_wall_front"]
    v_stop = math.sqrt(2 * 25.0 * max(front - P["margin"] - 0.5 * obs["speed"], 0.0))
    v_turn = math.sqrt(20.0 * P["turn_k"] * max(front, 0.0))
    turn_ahead = max(abs(angle), 0.7 * abs(obs["angle_to_waypoint_after_next"]) if d_next < P["blend_far"] else 0.0)
    v_corner = P["cap"] * max(0.35, math.cos(math.radians(min(turn_ahead, 80.0))))
    target = max(5.0, min(P["cap"], v_stop, v_turn, v_corner))
    if obs["zone_speed_limit"] is not None:              # yellow flag! (+1 s per tick too fast)
        target = min(target, obs["zone_speed_limit"] - 1.0)
    dv = target - obs["speed"]                           # accelerate gently (12 u/s^2), brake hard (25)
    throttle = dk.clamp((dv + 0.075 * obs["speed"]) / 6.0 if dv >= 0 else dv / 12.5)

    # ----------------------------------------------------------------------- TODO 3: avoid walls
    # wall_repulsion pushes you away from walls close beside you (+ = away from a close LEFT wall).
    steering += dk.wall_repulsion(obs, P["wall_gain"])

    return throttle, dk.clamp(steering)
