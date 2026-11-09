"""HANDRAIL CP5 (Ch5 The Strategist) - your Ch3 learner + an LLM that writes the plan in plan().

Stuck in Ch5? Copy this file over your drivers/submissions/<your_handle>.py, put your NAME / AUTHOR /
COLOR back, and carry on. Compare it with your own file first:

    git diff --no-index lesson/checkpoints/ch3_learner.py lesson/checkpoints/ch5_strategist.py

The LLM lives on the SESSION clock (plan(): seconds are fine there); drive() never calls it.

    plan(track): corner table + physics + BRIEFING --> MiniMax (JSON, may call 2 calculator tools)
                 --> validate() --> problems? ONE retry with the problems --> clamp --> fallback
                 --> corner factors + zones --> your Ch3 executor (and your Ch3 reflect() keeps learning)

What the Ch5 steps added to the Ch3 file:
    1. prompt = dk.track_brief(track) (physics, corner table T1..Tn, the briefing) + the JSON schema
       {"cap", "corners": [{"id", "speed_factor", "note"}], "zones": [{"from_wp", "to_wp", "max_speed"}], "say"}
    2. llm.chat_json(prompt, system=SYSTEM, tools=[corner_speed, braking_distance]): the model asks,
       code does the arithmetic
    3. validate(): every field checked; speed_factor clamped to 0.6 .. 1.15; unknown corner -> dropped;
       a zone whose numbers are not in the briefing -> dropped (grounding: no invented rules).
       No LLM at all (mock, no key, error)? The physics plan (= your Ch3 car) races.
    4. _build(): grip x segment knob (Ch3) x corner factor (LLM), then the zones: the executor brakes
       BEFORE the yellow flag instead of discovering it inside.

Check it (numbers measured, see lesson/instructor/CALIBRATION.md):
    python race.py check lesson/checkpoints/ch5_strategist.py                  # mock LLM
    python race.py drive lesson/checkpoints/ch5_strategist.py --track briefing_demo
    python race.py grand-prix lesson/checkpoints/ch5_strategist.py lesson/checkpoints/ch3_learner.py --track briefing_demo --practice 0 --laps 1
CP5: within 5 % of your Ch3 time, 0 zone violations on briefing_demo (needs the real LLM: the mock
cannot read), at most 3 LLM calls in plan(), and it still finishes with LLM_PROVIDER=mock.
After plan(): print(self.raw_plan) and self.problems - what did the model write, what was fixed?
"""

from __future__ import annotations

import copy
import json
import math
import re

import numpy as np

from agentic_gp import driverkit as dk
from agentic_gp import llm

# --------------------------------------------------------------------------- TODO 0: who are you?
NAME = "CP5 Strategist"     # <= 16 characters, shown in the standings and the 3D viewer
AUTHOR = "your_handle"      # your handle (pseudonyms welcome)
COLOR = "#3a0ca3"           # your car colour (hex)
KIND = "llm"                # "code" | "llm" | "fly"
MODEL = "MiniMax-M3"
MOTTO = "Think slow, drive fast."
ANATOMY = ["perceive", "act", "map", "plan", "memory", "reflect", "llm", "tools"]

# Numeric knobs: tune them with --param max_factor=1.1 (and Streamlit sliders) without editing code.
PARAMS = {
    "aggression": 1.0,      # x the physics corner speed (1.0 = exactly the grip limit)
    "crash_cut": 0.85,      # Ch3: a crash slows that segment and the one before by this factor
    "push": 1.05,           # Ch3: a clean lap -> next attempt tries every segment this much faster
    "start_margin": 3.0,    # Ch3: safety margin of attempt 1 (units kept free in front of the car)
    "min_margin": 1.0,      # Ch3: never brake later than this
    "min_factor": 0.6,      # Ch5: clamp range for the LLM's corner speed factors
    "max_factor": 1.15,
    "max_calls": 3,         # Ch5: LLM calls we allow ourselves in plan() (official events allow 2)
    # the Ch1 reflexes (the fallback in drive())
    "cap": 34.0, "steer_gain": 1.0, "margin": 4.0, "wall_gain": 1.5,
    "blend_far": 40.0, "blend_max": 0.6, "turn_k": 0.6,
}
EXPERIMENTS = ("push all", "trust walls")     # the problem generator takes turns

# Ch5 step 1: the strategist's job description and the JSON it must return.
SYSTEM = """You are the race strategist of a 2D racing team. A code executor drives the car along the
track centerline; you choose how fast it takes each corner. speed_factor 1.0 = the physics grip limit
v_max = sqrt(20 * R) of that corner; allowed 0.6 .. 1.15 (above 1.0 is aggressive). The briefing from
race control is binding: every speed-limit / yellow-flag zone in it MUST appear in "zones" (waypoint
numbers as written there). Reply with ONE JSON object only:
{"cap": <top speed u/s>, "corners": [{"id": "T1", "speed_factor": 1.0, "note": "<= 6 words"}],
 "zones": [{"from_wp": 9, "to_wp": 11, "max_speed": 15}], "say": "<radio message, <= 80 chars>"}"""


# Ch5 step 2: tools - the model asks, code does the arithmetic (docstrings = the tool descriptions).
def corner_speed(radius: float) -> float:
    """Fastest speed (u/s) through a corner of this radius at the grip limit (20 u/s^2, top speed 40).

    Args:
        radius: corner radius in units
    """
    return round(min(40.0, math.sqrt(20.0 * max(float(radius), 0.0))), 2)


def braking_distance(v_from: float, v_to: float) -> float:
    """Distance (units) needed to brake from v_from to v_to (u/s) at 25 u/s^2.

    Args:
        v_from: speed before braking
        v_to: target speed
    """
    return round(max(0.0, (float(v_from) ** 2 - float(v_to) ** 2) / 50.0), 2)


class Driver:
    def __init__(self) -> None:
        self.follower: dk.PathFollower | None = None
        self.memory: dict = {}                    # recover_if_stuck keeps its little state here
        self.history: list[dict] = []             # episodic memory: one row per attempt
        self.say_next: str | None = None
        self.raw_plan: object = None              # what the model wrote
        self.problems: list[str] = []             # what validate() fixed
        self.source = "physics"                   # "llm" | "llm (fixed)" | "physics"

    # ----------------------------------------------------------------------- plan: once per session
    def plan(self, track: dict) -> None:
        self.track = track
        self.phys = phys = dk.physics_of(track)
        self.points = points = np.asarray(track["centerline"], dtype=float)
        radii = dk.curvature_radii(points)
        self.grip = np.clip(np.sqrt(phys["max_lateral_accel"] * np.minimum(radii, 1e9)), 5.0, phys["max_speed"])
        self.ds = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
        self.segment_of = dk.segment_ids(len(points), dk.waypoint_indices(track, points))
        self.seg_factor = [PARAMS["aggression"]] * len(track["waypoints"])
        self.margin = PARAMS["start_margin"]
        self.steps = {"push all": PARAMS["push"] - 1.0, "trust walls": 1.0}
        self.best: dict | None = None
        self.experiment = "baseline"
        self.turn = 0
        self.locked = False

        # Ch5: ask the strategist (never more than max_calls LLM calls; any failure -> physics plan)
        self.corners = dk.track_corners(track)
        self.strategy = self.ask_strategist()
        self._build()
        zones = "; ".join(f"WP{z['from_wp']}-{z['to_wp']} max {z['max_speed']:g}" for z in self.strategy["zones"])
        self.say_next = (self.strategy["say"] or f"Plan ({self.source})") + (f" | zones: {zones}" if zones else "")

    def ask_strategist(self) -> dict:
        prompt = (dk.track_brief(self.track) + "\n\nWrite the corner plan (every corner id) and the zones "
                  "from the briefing (an empty list if there are none).")
        calls_before = llm.llm_stats()["calls"]
        try:
            self.raw_plan = llm.chat_json(prompt, system=SYSTEM, tools=[corner_speed, braking_distance], max_rounds=2)
        except llm.LLMError as exc:                   # no LLM: the physics plan races
            self.problems, self.source = [f"LLM unavailable: {exc}"[:200]], "physics"
            return self.validate({})[0]
        strategy, self.problems = self.validate(self.raw_plan)
        calls_left = PARAMS["max_calls"] - (llm.llm_stats()["calls"] - calls_before)
        if self.problems and calls_left > 0 and llm.llm_allowed():      # ONE retry, WITH the problems
            fix = (prompt + "\n\nYour previous answer:\n" + json.dumps(self.raw_plan)[:1500] +
                   "\n\nProblems found by the validator:\n- " + "\n- ".join(self.problems) +
                   "\nReturn the corrected, complete JSON.")
            try:
                raw2 = llm.chat_json(fix, system=SYSTEM)
                strategy2, problems2 = self.validate(raw2)
                if len(problems2) <= len(self.problems):
                    self.raw_plan, strategy, self.problems = raw2, strategy2, problems2
            except llm.LLMError as exc:               # the retry failed: keep the first (usable) plan
                self.problems.append(f"retry failed, first plan kept: {exc}"[:200])
        self.source = "llm" if not self.problems else "llm (fixed)"
        if not strategy["factors"] and not strategy["zones"]:
            self.source = "physics"
        return strategy

    # Ch5 step 3: the validator. Model output -> (a plan we can drive, the list of problems). Never raises.
    def validate(self, raw: object) -> tuple[dict, list[str]]:
        problems: list[str] = []
        raw = raw if isinstance(raw, dict) else {}
        plan = {"cap": self.phys["max_speed"], "factors": {}, "notes": {}, "zones": [], "say": None}
        cap = number(raw.get("cap"))
        if cap is not None:
            plan["cap"] = min(max(cap, 10.0), self.phys["max_speed"])
        known = {c["id"] for c in self.corners}
        for item in raw.get("corners") if isinstance(raw.get("corners"), list) else []:
            cid = item.get("id") if isinstance(item, dict) else None
            f = number(item.get("speed_factor")) if isinstance(item, dict) else None
            if cid not in known or f is None:
                problems.append(f"bad corner entry {str(item)[:60]} (ids: {', '.join(sorted(known))})")
                continue
            clamped = min(max(f, PARAMS["min_factor"]), PARAMS["max_factor"])
            if clamped != f:
                problems.append(f"{cid}: speed_factor {f} clamped to {clamped}")
            plan["factors"][cid] = clamped
            plan["notes"][cid] = str(item.get("note") or "")[:40]
        said = {float(x) for x in re.findall(r"\d+(?:\.\d+)?", self.track.get("briefing") or "")}
        n_wp = len(self.track["waypoints"])
        for z in raw.get("zones") if isinstance(raw.get("zones"), list) else []:
            a, b, v = (number(z.get(k)) if isinstance(z, dict) else None for k in ("from_wp", "to_wp", "max_speed"))
            if None in (a, b, v) or not (0 <= a < n_wp and 0 <= b < n_wp and 3 <= v <= 40):
                problems.append(f"bad zone {str(z)[:60]} (from_wp/to_wp 0..{n_wp - 1}, max_speed 3..40)")
            elif not {a, b, v} <= said:          # grounding: only zones race control announced
                problems.append(f"zone WP{a:g}-{b:g} max {v:g} is not in the briefing (dropped)")
            else:
                plan["zones"].append({"from_wp": int(a), "to_wp": int(b), "max_speed": float(v)})
        if isinstance(raw.get("say"), str):
            plan["say"] = raw["say"].strip()[:80] or None
        return plan, problems

    def _build(self) -> None:
        """Knobs -> planned speeds -> executor: grip x segment knob (Ch3) x corner factor (Ch5), + zones."""
        corner = np.ones(len(self.points))
        n = len(self.points)
        for c in self.corners:                    # Ch5 step 4: the LLM's factor inside each corner
            idx = np.arange(c["start_index"], c["end_index"] + 1 + (n if c["end_index"] < c["start_index"] else 0)) % n
            corner[idx] = self.strategy["factors"].get(c["id"], 1.0)
        v = self.grip * np.array(self.seg_factor)[self.segment_of] * corner
        v = np.clip(v, 5.0, min(self.strategy["cap"], self.phys["max_speed"]))
        for _ in range(2):                        # Ch2 step 3: the backward braking pass
            for i in reversed(range(n)):
                nxt = (i + 1) % n
                v[i] = min(v[i], math.sqrt(v[nxt] ** 2 + 2 * self.phys["brake"] * self.ds[i]))
        self.follower = dk.PathFollower(self.points, v, physics=self.phys, segments=self.segment_of,
                                        safety_margin=self.margin, blind_factor=min(1.0, self.margin / 3.0))
        for z in self.strategy["zones"]:          # brake BEFORE the yellow flag
            self.follower.set_speed_limit(z["from_wp"], z["to_wp"], z["max_speed"])

    # ----------------------------------------------------------------------- drive: every tick, no LLM
    def drive(self, obs: dict) -> dict:
        if self.follower is None:                 # plan() failed? The Ch1 reflexes still bring the car home.
            throttle, steering = reflexes(obs, self.memory)
            return {"throttle": throttle, "steering": steering}
        throttle, steering = self.follower.control(obs)
        say, self.say_next = self.say_next, None
        return {"throttle": throttle, "steering": steering, "say": say,
                "debug": dict(self.follower.debug, plan=self.source)}

    # ----------------------------------------------------------------------- reflect: your Ch3 learner
    def reflect(self, report: dict) -> None:
        if self.follower is None:
            return
        time = report["race_time"] if report["completed"] else None          # a DNF has no lap time
        self.history.append({"attempt": report["attempt"], "experiment": self.experiment,
                             "seg_factor": list(self.seg_factor), "margin": self.margin, "time": time,
                             "crashes": report["crashes"],
                             "crash_segments": [c["segment"] for c in report["crash_log"]]})
        improved = time is not None and (self.best is None or time < self.best["time"])
        if improved:
            self.best = {"seg_factor": list(self.seg_factor), "margin": self.margin, "time": time,
                         "attempt": report["attempt"]}
        elif self.experiment in self.steps:
            self.steps[self.experiment] *= 0.5
        if self.locked or report["next_phase"] in ("qualifying", "race"):
            self.locked = True
            if self.best:
                self.seg_factor, self.margin = list(self.best["seg_factor"]), self.best["margin"]
                self._build()
                self.say_next = f"Locked in: knobs of attempt {self.best['attempt']} ({self.best['time']:.2f} s)"
            return
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


def number(x: object) -> float | None:
    """A finite float, or None (the model may write "fast", null, NaN, ...)."""
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


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
