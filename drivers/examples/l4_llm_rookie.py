"""L4 Rookie - the LLM IS the driver: sensor JSON in, {throttle, steering} out, every tick.

Agent type (Russell & Norvig): a reflex agent whose rules are written by a language model at run
time. Agentic concept: the LATENCY WALL. A tick is 0.5 s of sim time; an LLM answer takes 0.5-3 s of
wall time. Turn-based (the sim waits) it is merely slow and erratic; in realtime (--realtime) the car
drives blind while the model thinks. This file is a lesson tool, not a racer: after MAX_LLM_TICKS
LLM decisions the code reflexes (an L1-style fallback) take over.

    every tick:
      obs ──> compact sensor JSON ──> MiniMax (thinking off, effort "none") ──> {"throttle","steering","say"}
                                          │ error / bad JSON / budget used up / not allowed
                                          v
                                     L1-style reflex rules (code)  ──> (throttle, steering)
      between decisions (DECIDE_EVERY_N_TICKS > 1): hold the last LLM action

How it works
- The prompt is short on purpose (latency!): the sign rule "+ = right", one physics hint, and the
  10 most useful sensor values as JSON.
- llm.chat_json(..., effort="none"): MiniMax-M3 with thinking disabled answers in ~0.5-1 s.
- Every answer is validated: numbers, clipped to [-1, 1]; anything else -> fallback for this tick.
- PARAMS: DECIDE_EVERY_N_TICKS (ask the LLM every N ticks, hold the action in between) and
  MAX_LLM_TICKS (default 20, the class-wide budget rule for Ch4); then the code drives.
- Official events forbid LLM calls in drive(): llm.llm_allowed() is False there, so the rookie
  races on its fallback reflexes (and says so on the radio).

Run it:
    python race.py drive drivers/examples/l4_llm_rookie.py                      # turn-based, 20 LLM ticks
    python race.py drive drivers/examples/l4_llm_rookie.py --realtime           # feel the latency wall
    python race.py drive drivers/examples/l4_llm_rookie.py --param MAX_LLM_TICKS=0   # fallback only

Expected on grand_prix: with MiniMax ~78 s - the 20 LLM ticks are slow and hesitant (p50 ~1.2 s per
call), then the fallback finishes the lap. With the mock LLM: fallback from tick 0, 71.8 s, clean.

Try this
1. Measure: self.latencies holds the seconds per call. Ticks per call = p50 / 0.5. What lap time
   could a realtime LLM driver reach at best?
2. Add "brake when distance_to_wall_front < 25" to the prompt. Fewer crashes? (race.py exam helps.)
3. --param DECIDE_EVERY_N_TICKS=4: fewer calls, smoother or worse?
4. Hybrid (Ch4 Gold): let the LLM choose only "push" or "careful" every 10 ticks, and let the code
   reflexes do the steering.
"""

from __future__ import annotations

import json
import math
import time

from agentic_gp import driverkit as dk
from agentic_gp import llm

try:
    _MODEL = llm.describe_llm()
except Exception:
    _MODEL = "unknown"

SYSTEM = ("You drive a race car in a 2D sim, one decision every 0.5 s. Sign rule: + = RIGHT, - = LEFT for "
          "every angle and for steering. Steer towards the next waypoint (steering ~ angle/30). Walls closer "
          "than 20 ahead: brake (throttle < 0). Corner speed limit ~ sqrt(20 * radius). Reply ONLY with JSON: "
          '{"throttle": -1..1, "steering": -1..1, "say": "<= 6 words"}')

KEYS = ("speed", "distance_to_wall_front", "distance_to_wall_left_45", "distance_to_wall_right_45",
        "distance_to_wall_left_90", "distance_to_wall_right_90", "angle_to_next_waypoint",
        "distance_to_next_waypoint", "angle_to_waypoint_after_next", "zone_speed_limit")


class Driver:
    NAME = "Rookie"
    AUTHOR = "Agentic GP"
    COLOR = "#ff006e"
    KIND = "llm"
    MODEL = _MODEL
    MOTTO = "I think, therefore I'm late."
    ANATOMY = ["perceive", "act", "llm"]
    PARAMS = {
        "DECIDE_EVERY_N_TICKS": 1,    # ask the LLM every N ticks (hold the action in between)
        "MAX_LLM_TICKS": 20,          # LLM decisions per session, then the code fallback drives
    }

    def __init__(self) -> None:
        self.llm_ticks = 0            # LLM decisions so far (successful or not)
        self.failures = 0
        self.latencies: list[float] = []
        self.held: dict | None = None
        self.recovery: dict = {}
        self.last_say: str | None = None

    def plan(self, track: dict) -> None:
        self.held = None              # a new session: forget the held action (the budget stays spent)

    # ------------------------------------------------------------------ drive: every tick
    def drive(self, obs: dict) -> dict:
        P = self.PARAMS
        if obs["tick"] == 0:
            self.last_say = None                                     # a new attempt: speak again
        n = max(1, int(P["DECIDE_EVERY_N_TICKS"]))
        if obs["tick"] % n != 0 and self.held is not None:          # hold the last LLM action
            return dict(self.held, say=None)
        if self.llm_ticks < int(P["MAX_LLM_TICKS"]):
            if not llm.llm_allowed():                               # official events: drive() is LLM-free
                return self._fallback(obs, "LLM not allowed in drive() - reflexes only")
            self.llm_ticks += 1
            action = self._ask(obs)
            if action is not None:
                self.held = action
                return dict(action, say=self._radio(action.get("say")))
            self.failures += 1
            return self._fallback(obs, "LLM answer unusable - reflexes this tick")
        self.held = None
        return self._fallback(obs, f"LLM budget used ({self.llm_ticks} ticks) - reflexes drive")

    def _ask(self, obs: dict) -> dict | None:
        """One LLM decision, validated. None if anything goes wrong."""
        sensors = {k: (round(obs[k], 1) if isinstance(obs.get(k), float) else obs.get(k)) for k in KEYS}
        t0 = time.perf_counter()
        try:
            raw = llm.chat_json(json.dumps(sensors), system=SYSTEM, effort="none", max_tokens=200)
        except llm.LLMError:
            return None
        finally:
            self.latencies.append(round(time.perf_counter() - t0, 3))
        try:
            throttle, steering = float(raw["throttle"]), float(raw["steering"])
        except (KeyError, TypeError, ValueError):
            return None
        if not (math.isfinite(throttle) and math.isfinite(steering)):
            return None
        return {"throttle": dk.clamp(throttle), "steering": dk.clamp(steering),
                "say": str(raw.get("say") or "")[:60] or None}

    # ------------------------------------------------------------------ the code fallback (L1-style)
    def _fallback(self, obs: dict, why: str) -> dict:
        recover = dk.recover_if_stuck(obs, self.recovery)
        if recover is not None:
            return {"throttle": recover[0], "steering": recover[1], "say": self._radio(why)}
        speed, front = obs["speed"], obs["distance_to_wall_front"]
        d_next = obs["distance_to_next_waypoint"]
        w = dk.clamp(1.0 - (d_next - 10.0) / 30.0, 0.0, 0.6)          # look through to the next-but-one
        aim = (1 - w) * obs["angle_to_next_waypoint"] + w * obs["angle_to_waypoint_after_next"]
        side = obs["distance_to_wall_left_45"] if aim < 0 else obs["distance_to_wall_right_45"]
        if side < min(d_next, 40.0) * 0.5:                              # gate hidden behind a wall
            aim *= 0.5
        steering = dk.steer_towards(aim, speed) + dk.wall_repulsion(obs, 1.5)
        target = min(30.0, math.sqrt(2 * 25 * max(front - 4.0 - 0.5 * speed, 0.0)),
                     math.sqrt(12.0 * max(front, 0.0)),
                     30.0 * max(0.35, math.cos(math.radians(min(abs(aim), 80.0)))))
        if obs.get("zone_speed_limit") is not None:
            target = min(target, obs["zone_speed_limit"] - 0.5)
        target = max(target, 5.0)
        dv = target - speed
        throttle = (dv + 0.075 * speed) / 6.0 if dv >= 0 else dv / 12.5
        return {"throttle": dk.clamp(throttle), "steering": dk.clamp(steering), "say": self._radio(why)}

    def _radio(self, text: str | None) -> str | None:
        if not text or text == self.last_say:
            return None
        self.last_say = text
        return text
