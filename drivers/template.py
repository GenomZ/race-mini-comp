"""MY DRIVER - copy me to drivers/submissions/<your_name>.py and make me yours.

DRIVER CARD (fill in before you submit, one line each):
    Approach:      <e.g. reflexes -> map + speed plan -> learns per segment>
    LLM:           <what the LLM does in your driver, or "none, because ...">
    reflect():     <what your driver learns between attempts>
    Anatomy:       <perceive, act, map, plan, memory, reflect, llm, tools>
    Another day:   <what you would build next>

This file already drives: slowly and carefully. It is clean on every public track (grand_prix ~86 s)
and finished every one of 300 random tracks we tried (with a crash on 7 of them). Your job is to make
it faster, then smarter. Work through the TODOs from top to bottom; run after every change:

    python race.py check  drivers/submissions/<your_name>.py         # contract + 40-tick smoke test
    python race.py drive  drivers/submissions/<your_name>.py --attempts 3
    python race.py drive  drivers/submissions/<your_name>.py --track sprint --replay races/me.json
    python race.py validate drivers/submissions/<your_name>.py       # all public + random tracks

How a driver works: the runner calls drive(obs) every tick (0.5 s of sim time) with what the car
senses (obs, see agentic_gp/driver.py OBS_FIELDS) and you return (throttle, steering), both -1..1.
ONE rule for every angle, offset and steering value: + = RIGHT, - = LEFT.

Ch0 warm-up: --param steer_gain=0 --param wall_gain=0 switches the steering off (~370 s, 34 crashes:
the marshals carry it round). Then only steer_gain=0: it still finishes cleanly - which line steers now?
Ch1: each TODO below has an "Upgrade" - the goal is a clean lap well under 75 s (l1_reflex.py: 67 s).

Submitting: one file named <your_name>.py in drivers/submissions/, NAME and AUTHOR set,
`python race.py check` green, driver card filled in. Rules: drivers/submissions/README.md.
"""

from __future__ import annotations

import math

from agentic_gp import driverkit as dk

# --------------------------------------------------------------------------- TODO 0: who are you?
NAME = "My Driver"          # <= 16 characters, shown in the standings and the 3D viewer
AUTHOR = "your_handle"      # your handle (pseudonyms welcome)
COLOR = "#3a86ff"           # your car colour (hex)
KIND = "code"               # "code" | "llm" | "fly"
MOTTO = "Slow is smooth, smooth is fast."
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
