"""HANDRAIL CP0 (Ch0 Pit Lane) - exactly what you have after Ch0: the template with YOUR metadata.

Stuck in Ch0? Copy this file to drivers/submissions/<your_handle>.py, put your handle in AUTHOR,
and carry on with Ch1. Compare it with your own file:   git diff --no-index drivers/template.py lesson/checkpoints/ch0_pit_lane.py

What changed against drivers/template.py: only TODO 0 (NAME, AUTHOR, COLOR, KIND, MOTTO). The driving
code is the template's, untouched: slow and careful, clean on every public track.

Check it (numbers measured, see lesson/instructor/CALIBRATION.md):
    python race.py check lesson/checkpoints/ch0_pit_lane.py          # PASSED
    python race.py drive lesson/checkpoints/ch0_pit_lane.py          # grand_prix 85.85 s, 0 crashes

Ch0 warm-up: --param steer_gain=0 --param wall_gain=0 switches the steering off and the marshals carry
the car round (368.55 s, 34 crashes). Then only --param steer_gain=0: 119.35 s and clean - which line
steers now?
"""

from __future__ import annotations

import math

from agentic_gp import driverkit as dk

# --------------------------------------------------------------------------- TODO 0: who are you?
NAME = "Pit Lane"           # <= 16 characters, shown in the standings and the 3D viewer
AUTHOR = "your_handle"      # your handle (pseudonyms welcome)
COLOR = "#00b4d8"           # your car colour (hex)
KIND = "code"               # "code" | "llm" | "fly"
MOTTO = "First lap, first car, my colours."
ANATOMY = ["perceive", "act"]          # add "map", "plan", "memory", "reflect", "llm", "tools" as you grow

# Numeric knobs: tune them with --param cap=30 (and Streamlit sliders) without editing code.
PARAMS = {
    "cap": 22.0,            # TODO 2: our top speed (the car can do 40)
    "steer_gain": 1.0,      # TODO 1: how hard we steer towards the waypoint
    "margin": 6.0,          # TODO 2: free space we keep in front of the car
    "wall_gain": 1.5,       # TODO 3: how hard we push away from close walls
}

_memory: dict = {}          # recover_if_stuck keeps its little state here


def drive(obs: dict) -> tuple[float, float]:
    P = PARAMS

    # ----------------------------------------------------------------------- TODO 3b: unstick
    # Stopped with a wall in front of you? Reverse towards the open side for a moment.
    # Insurance: a good driver rarely gets stuck, but one bad corner without it can end your race.
    recover = dk.recover_if_stuck(obs, _memory)
    if recover is not None:
        return recover

    # ----------------------------------------------------------------------- TODO 1: steer to the waypoint
    # angle_to_next_waypoint: + = the gate is to the RIGHT. Steering is a fraction of grip, so the same
    # command turns much less at high speed: steer_towards converts "the angle I want" into a command.
    # Upgrade: when the gate is close, blend in angle_to_waypoint_after_next (look through the corner).
    angle = obs["angle_to_next_waypoint"]
    # Line of sight: is the wall on the gate's side much closer than the gate? Then the gate is around
    # a corner (a hairpin) - turn in half as hard, or we steer straight into the inside wall.
    side = obs["distance_to_wall_left_45"] if angle < 0 else obs["distance_to_wall_right_45"]
    if side < 0.5 * min(obs["distance_to_next_waypoint"], 40.0):
        angle *= 0.5
    steering = P["steer_gain"] * dk.steer_towards(angle, obs["speed"])

    # ----------------------------------------------------------------------- TODO 2: control speed
    # Never faster than you can stop before the wall ahead: v = sqrt(2 * brake * distance), brake = 25.
    # Upgrade: also slow down when the waypoint is at a big angle (a corner is coming).
    front = obs["distance_to_wall_front"]
    v_stop = math.sqrt(2 * 25.0 * max(front - P["margin"] - 0.5 * obs["speed"], 0.0))
    v_corner = P["cap"] * max(0.4, math.cos(math.radians(min(abs(angle), 80.0))))
    target = max(4.0, min(P["cap"], v_stop, v_corner))
    if obs["zone_speed_limit"] is not None:              # yellow flag! (+1 s per tick too fast)
        target = min(target, obs["zone_speed_limit"] - 1.0)
    throttle = dk.clamp((target - obs["speed"]) / 6.0)

    # ----------------------------------------------------------------------- TODO 3: avoid walls
    # wall_repulsion pushes you away from walls close beside you (+ = away from a close LEFT wall).
    # Upgrade: the line-of-sight rule in TODO 1 only turns in less. Better: when the gate is hidden
    # behind a wall, aim at the most open ray instead (see drivers/examples/l1_reflex.py).
    steering += dk.wall_repulsion(obs, P["wall_gain"])

    return throttle, dk.clamp(steering)


# =========================================================================== LEVEL UP (Ch2 / Ch3)
# When your reflexes are good, turn this file into a class with a plan and a memory. Delete
# drive() above and un-comment this skeleton (look at drivers/examples/l2_planner.py and
# l3_learner.py for complete versions):
#
# class Driver:
#     NAME, AUTHOR, COLOR, KIND = "My Driver", "your_handle", "#3a86ff", "code"
#     PARAMS = {"aggression": 1.0}
#
#     def plan(self, track):                      # once per session: the whole map is yours
#         self.follower = dk.PathFollower.from_track(track, aggression=self.PARAMS["aggression"])
#         self.history = []                       # memory across attempts
#
#     def drive(self, obs):                       # every tick: execute the plan
#         return self.follower.control(obs)
#
#     def reflect(self, report):                  # after every attempt: learn
#         self.history.append(report["race_time"])
#         if report["next_phase"] in ("qualifying", "race"):
#             return                              # lock in: no experiments when it counts
#         ...                                     # e.g. crash -> slow that segment down
