"""HANDRAIL CP2 (Ch2 The Map) - your Ch1 file after the Ch2 steps: a model-based, goal-based agent.

Stuck in Ch2? Copy this file over your drivers/submissions/<your_handle>.py, put your NAME / AUTHOR /
COLOR back, and carry on. Compare it with your own file first:

    git diff --no-index lesson/checkpoints/ch1_reflex.py lesson/checkpoints/ch2_planner.py

What the Ch2 steps changed:
    1. The template's LEVEL UP skeleton: `class Driver` with plan() (once per session) and drive()
       (every tick). The Ch1 reflexes moved into drive() as the fallback (renamed reflexes()).
    2. plan(): radii = dk.curvature_radii(track["centerline"]) -> grip limit v = sqrt(20 R) per point.
    3. plan(): the backward braking pass - ONE line inside the loop (marked "the blank line").
    4. drive(): dk.PathFollower(points, speeds) - pure pursuit along the centerline at the planned
       speeds; its stopping-distance safety net keeps the Ch1 rule "never faster than you can stop".

No per-track numbers anywhere: everything comes from the map it is given, so it is clean on tracks
it has never seen (that is the Validation column).

Check it (numbers measured, see lesson/instructor/CALIBRATION.md):
    python race.py check lesson/checkpoints/ch2_planner.py
    python race.py drive lesson/checkpoints/ch2_planner.py              # grand_prix, clean (CP2 band)
    python race.py validate lesson/checkpoints/ch2_planner.py           # public + random tracks, clean

Push: --param aggression=1.1 (faster or crash?); Gold: points = dk.racing_line(track, 3.0).
"""

from __future__ import annotations

import math

import numpy as np

from agentic_gp import driverkit as dk

# --------------------------------------------------------------------------- TODO 0: who are you?
NAME = "CP2 Planner"        # <= 16 characters, shown in the standings and the 3D viewer
AUTHOR = "your_handle"      # your handle (pseudonyms welcome)
COLOR = "#0077b6"           # your car colour (hex)
KIND = "code"               # "code" | "llm" | "fly"
MOTTO = "Plan the lap, drive the plan."
ANATOMY = ["perceive", "act", "map", "plan"]

# Numeric knobs: tune them with --param aggression=1.1 (and Streamlit sliders) without editing code.
PARAMS = {
    "aggression": 1.0,      # x the physics corner speed (1.0 = exactly the grip limit)
    # the Ch1 reflexes (the fallback in drive())
    "cap": 34.0, "steer_gain": 1.0, "margin": 4.0, "wall_gain": 1.5,
    "blend_far": 40.0, "blend_max": 0.6, "turn_k": 0.6,
}


class Driver:
    def __init__(self) -> None:
        self.follower: dk.PathFollower | None = None
        self.memory: dict = {}                    # recover_if_stuck keeps its little state here

    # ----------------------------------------------------------------------- plan: once per session
    def plan(self, track: dict) -> None:
        phys = dk.physics_of(track)               # grip 20, brake 25, top speed 40 (this sim's numbers)
        points = np.asarray(track["centerline"], dtype=float)

        # Step 2: how tight is the track at every point, and how fast may we go there?  v = sqrt(a_lat * R)
        radii = dk.curvature_radii(points)
        v = np.sqrt(phys["max_lateral_accel"] * np.minimum(radii, 1e9)) * PARAMS["aggression"]
        v = np.clip(v, 5.0, phys["max_speed"])

        # Step 3: backward braking pass. Walking BACKWARDS round the lap, a point may be no faster than
        # the speed from which we can still brake to the next point's speed: v^2 = v_next^2 + 2 * brake * ds.
        ds = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)     # distance to the next point
        n = len(points)
        for _ in range(2):                        # twice: the lap is a loop (the last corner limits the start)
            for i in reversed(range(n)):
                nxt = (i + 1) % n
                v[i] = min(v[i], math.sqrt(v[nxt] ** 2 + 2 * phys["brake"] * ds[i]))     # <- the blank line

        # Step 4: the executor - pure pursuit along the same points at our planned speeds.
        self.follower = dk.PathFollower.from_track(track, speeds=v)
        slowest = int(np.argmin(v))
        self.note = f"Plan: slowest point {v[slowest]:.0f} u/s at index {slowest} of {n}"

    # ----------------------------------------------------------------------- drive: every tick
    def drive(self, obs: dict) -> dict:
        if self.follower is None:                 # plan() failed? The Ch1 reflexes still bring the car home.
            throttle, steering = reflexes(obs, self.memory)
            return {"throttle": throttle, "steering": steering}
        throttle, steering = self.follower.control(obs)
        say = self.note if obs["tick"] == 0 else None
        return {"throttle": throttle, "steering": steering, "say": say, "debug": dict(self.follower.debug)}


# =========================================================================== the Ch1 reflexes (fallback)
RAYS = [(-90, "left_90"), (-45, "left_45"), (-20, "left_20"), (0, "front"),
        (20, "right_20"), (45, "right_45"), (90, "right_90")]


def free_distance(obs: dict, angle: float) -> float:
    """How far we can see in direction `angle` (between the two nearest rays)."""
    angle = max(-90.0, min(90.0, angle))
    for (a0, k0), (a1, k1) in zip(RAYS, RAYS[1:]):
        if a0 <= angle <= a1:
            d0, d1 = obs["distance_to_wall_" + k0], obs["distance_to_wall_" + k1]
            return d0 + (angle - a0) / (a1 - a0) * (d1 - d0)
    return obs["distance_to_wall_front"]


def reflexes(obs: dict, memory: dict) -> tuple[float, float]:
    """Your Ch1 drive(obs), unchanged (only renamed, and `_memory` became an argument)."""
    P = PARAMS
    recover = dk.recover_if_stuck(obs, memory)
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
