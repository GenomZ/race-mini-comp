"""HANDRAIL CP3 (Ch3 Practice Makes Perfect) - your Ch2 planner + reflect(): a learning agent.

Stuck in Ch3? Copy this file over your drivers/submissions/<your_handle>.py, put your NAME / AUTHOR /
COLOR back, and carry on. Compare it with your own file first:

    git diff --no-index lesson/checkpoints/ch2_planner.py lesson/checkpoints/ch3_learner.py

What the Ch3 steps changed (Russell & Norvig's learning agent, mapped onto the car):
    performance element = drive()          (the Ch2 planner, unchanged)
    critic              = the lap report   (race_time, crashes, crash_log with the segment of each crash)
    learning element    = reflect()        (keep-best, revert, change the knobs)
    problem generator   = ONE change per attempt
    1. The knobs: one speed factor per waypoint segment (self.seg_factor, x the grip limit) and the
       follower's safety margin (free space it keeps in front of the car). _build() turns knobs -> speeds.
    2. reflect(): keep the best lap; a crash -> that segment and the one before x crash_cut (0.85);
       a clean lap -> try something from the BEST knobs: push every segment x push (1.05), or trust
       the walls (margin - 1). A change that made it slower is reverted and its step halved (the sim
       is deterministic: the same change would give the same bad lap). next_phase "qualifying" or
       "race" -> lock in the best knobs, no more experiments.
    Memory: self.history (episodic, one row per attempt) and self.best (what worked).

Check it (numbers measured, see lesson/instructor/CALIBRATION.md):
    python race.py check lesson/checkpoints/ch3_learner.py
    python race.py drive lesson/checkpoints/ch3_learner.py --track sprint --attempts 6 --save
    # CP3: a learning curve - best of 6 at least 3 % faster than attempt 1, no DNF

Exercise: write a rule that stops "the same crash twice in a row" (self.history[-1]["crash_segments"]).
"""

from __future__ import annotations

import copy
import math

import numpy as np

from agentic_gp import driverkit as dk

# --------------------------------------------------------------------------- TODO 0: who are you?
NAME = "CP3 Learner"        # <= 16 characters, shown in the standings and the 3D viewer
AUTHOR = "your_handle"      # your handle (pseudonyms welcome)
COLOR = "#023e8a"           # your car colour (hex)
KIND = "code"               # "code" | "llm" | "fly"
MOTTO = "Every lap teaches me something."
ANATOMY = ["perceive", "act", "map", "plan", "memory", "reflect"]

# Numeric knobs: tune them with --param push=1.08 (and Streamlit sliders) without editing code.
PARAMS = {
    "aggression": 1.0,      # x the physics corner speed (1.0 = exactly the grip limit)
    "crash_cut": 0.85,      # Ch3: a crash slows that segment and the one before by this factor
    "push": 1.05,           # Ch3: a clean lap -> next attempt tries every segment this much faster
    "start_margin": 3.0,    # Ch3: safety margin of attempt 1 (units kept free in front of the car)
    "min_margin": 1.0,      # Ch3: never brake later than this
    # the Ch1 reflexes (the fallback in drive())
    "cap": 34.0, "steer_gain": 1.0, "margin": 4.0, "wall_gain": 1.5,
    "blend_far": 40.0, "blend_max": 0.6, "turn_k": 0.6,
}
EXPERIMENTS = ("push all", "trust walls")     # the problem generator takes turns


class Driver:
    def __init__(self) -> None:
        self.follower: dk.PathFollower | None = None
        self.memory: dict = {}                    # recover_if_stuck keeps its little state here
        self.history: list[dict] = []             # episodic memory: one row per attempt
        self.say_next: str | None = None

    # ----------------------------------------------------------------------- plan: once per session
    def plan(self, track: dict) -> None:
        self.phys = phys = dk.physics_of(track)
        self.points = points = np.asarray(track["centerline"], dtype=float)

        # Ch2 step 2: the grip limit at every point, v = sqrt(a_lat * R)
        radii = dk.curvature_radii(points)
        self.grip = np.clip(np.sqrt(phys["max_lateral_accel"] * np.minimum(radii, 1e9)), 5.0, phys["max_speed"])
        self.ds = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)

        # Ch3 step 1: the knobs. Segment g runs from gate g to gate g+1 (the same number as obs["segment"]
        # and as "segment" in the report's crash_log).
        self.segment_of = dk.segment_ids(len(points), dk.waypoint_indices(track, points))
        self.seg_factor = [PARAMS["aggression"]] * len(track["waypoints"])
        self.margin = PARAMS["start_margin"]
        self.steps = {"push all": PARAMS["push"] - 1.0, "trust walls": 1.0}
        self.best: dict | None = None             # {"seg_factor", "margin", "time", "attempt"}
        self.experiment = "baseline"
        self.turn = 0                             # whose turn it is in EXPERIMENTS
        self.locked = False
        self._build()
        self.say_next = f"Baseline plan, margin {self.margin:.1f}"

    def _build(self) -> None:
        """Knobs -> planned speeds -> executor (Ch2 steps 2-4 with one factor per segment)."""
        v = self.grip * np.array(self.seg_factor)[self.segment_of]
        v = np.clip(v, 5.0, self.phys["max_speed"])
        n = len(v)
        for _ in range(2):                        # Ch2 step 3: the backward braking pass
            for i in reversed(range(n)):
                nxt = (i + 1) % n
                v[i] = min(v[i], math.sqrt(v[nxt] ** 2 + 2 * self.phys["brake"] * self.ds[i]))
        self.follower = dk.PathFollower(self.points, v, physics=self.phys, segments=self.segment_of,
                                        safety_margin=self.margin, blind_factor=min(1.0, self.margin / 3.0))

    # ----------------------------------------------------------------------- drive: every tick
    def drive(self, obs: dict) -> dict:
        if self.follower is None:                 # plan() failed? The Ch1 reflexes still bring the car home.
            throttle, steering = reflexes(obs, self.memory)
            return {"throttle": throttle, "steering": steering}
        throttle, steering = self.follower.control(obs)
        say, self.say_next = self.say_next, None
        return {"throttle": throttle, "steering": steering, "say": say, "debug": dict(self.follower.debug)}

    # ----------------------------------------------------------------------- reflect: after every attempt
    def reflect(self, report: dict) -> None:
        if self.follower is None:
            return
        time = report["race_time"] if report["completed"] else None          # a DNF has no lap time
        self.history.append({"attempt": report["attempt"], "experiment": self.experiment,
                             "seg_factor": list(self.seg_factor), "margin": self.margin, "time": time,
                             "crashes": report["crashes"],
                             "crash_segments": [c["segment"] for c in report["crash_log"]]})

        # CRITIC: is this the best lap so far?  (keep-best)
        improved = time is not None and (self.best is None or time < self.best["time"])
        if improved:
            self.best = {"seg_factor": list(self.seg_factor), "margin": self.margin, "time": time,
                         "attempt": report["attempt"]}
        elif self.experiment in self.steps:
            self.steps[self.experiment] *= 0.5    # that change hurt: a smaller step next time

        # LOCK-IN: qualifying or the race is next -> no more experiments, drive the best knobs we know.
        if self.locked or report["next_phase"] in ("qualifying", "race"):
            self.locked = True
            if self.best:
                self.seg_factor, self.margin = list(self.best["seg_factor"]), self.best["margin"]
                self._build()
                self.say_next = f"Locked in: knobs of attempt {self.best['attempt']} ({self.best['time']:.2f} s)"
            return

        # LEARN: a crash -> slow down where it happened; else start from the best knobs (revert) + ONE change.
        n = len(self.seg_factor)
        if report["crashes"]:
            for crash in report["crash_log"]:
                for seg in (crash["segment"], (crash["segment"] - 1) % n):
                    self.seg_factor[seg] *= PARAMS["crash_cut"]
            self.margin = PARAMS["start_margin"]
            self.experiment = "crash fix"
            self.say_next = f"{report['crashes']} crash(es): slowing those segments down"
        else:
            if self.best:
                self.seg_factor, self.margin = list(self.best["seg_factor"]), self.best["margin"]
            self.experiment = EXPERIMENTS[self.turn % len(EXPERIMENTS)]
            self.turn += 1
            step = self.steps[self.experiment]
            if self.experiment == "push all":
                self.seg_factor = [min(1.6, f * (1.0 + step)) for f in self.seg_factor]
            else:
                self.margin = max(PARAMS["min_margin"], self.margin - step)
            result = "new best" if improved else "slower - back to the best knobs"
            self.say_next = f"{time:.2f} s ({result}); next: {self.experiment}" if time else \
                f"DNF ({report['ended_by']}); next: {self.experiment}"
        self._build()


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
