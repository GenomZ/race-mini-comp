"""L5 Strategist - the LLM plans once (slow thinking), code executes every tick (fast acting).

Agent type (Russell & Norvig): goal-based agent whose PLANNER is an LLM. The LLM lives on the
SESSION clock (one conversation in plan(), seconds are fine there); the executor lives on the TICK
clock (0.5 s, no LLM allowed). Agentic concepts: structured output (JSON), validation + clamping,
fallback, tools for arithmetic, and reading natural-language rules (the race briefing).

    plan(track)                                                  drive(obs) (every 0.5 s)
    ┌───────────────────────────────────────────────────────┐    ┌──────────────────────────┐
    │ corner table + physics + BRIEFING ──> MiniMax (JSON)   │    │ PathFollower on the      │
    │   tools: corner_speed(R), braking_distance(v1, v2)     │    │ racing line, speeds =    │
    │ JSON ──> validate ──> clamp [0.6, 1.15] x physics      │──> │ physics x LLM factors,   │
    │   invalid?  one retry WITH the error message           │    │ yellow-flag zones capped │
    │   still bad? per-corner fallback = physics speed       │    │ radio: the LLM's notes   │
    └───────────────────────────────────────────────────────┘    └──────────────────────────┘

How it works
- The prompt = driverkit.track_brief(track) (physics, corner table T1..Tn with radius and v_max,
  the briefing) + the JSON schema: {"cap", "corners": [{"id", "speed_factor", "note"}],
  "zones": [{"from_wp", "to_wp", "max_speed"}], "say"}.
- The model may call two pure-Python tools (corner_speed, braking_distance): it asks, code does the
  arithmetic.
- validate() checks every field. Bad corner -> physics speed for that corner only; factor out of
  range -> clamped to [0.6, 1.15]; bad zone -> dropped; a zone whose numbers are not in the briefing
  text (an invented one) -> dropped (a grounding check). Problems are sent back ONCE so the model can
  fix them (budget: <= 2 calls in plan() at official events, so the retry only happens if allowed; if
  the retry fails, the first plan is kept). Wrong JSON types never crash the validator.
- No LLM (mock, no key, error, budget used up)? The physics plan drives: the car always races.
- The briefing is the LLM's real edge: "Yellow flag between waypoints 9 and 11: max speed 15"
  becomes a zone the executor brakes for BEFORE it gets there.

Run it:
    python race.py drive drivers/examples/l5_llm_strategist.py --track briefing_demo --attempts 2
    LLM_PROVIDER=mock python race.py drive drivers/examples/l5_llm_strategist.py   # fallback plan

Expected: with MiniMax ~49-50 s on grand_prix and 0 zone violations on briefing_demo (~57-58 s; it
depends on the plan the model writes - run it twice!); with the mock LLM the physics plan: 49.95 s,
clean (but 2 zone violations on briefing_demo: no LLM, no briefing reader).

Try this
1. Print self.raw_plan and self.problems after plan(): what did the model write, what was fixed?
2. Remove the briefing from the prompt: how many zone violations on briefing_demo now?
3. --param use_tools=0: compare the factors with and without the calculator tools.
4. Force bad JSON (return "banana" from a mock responder) and watch the fallback work.
"""

from __future__ import annotations

import json
import math
import re

import numpy as np

from agentic_gp import driverkit as dk
from agentic_gp import llm

try:
    _MODEL = llm.describe_llm()
except Exception:                      # never fail at import time because of LLM settings
    _MODEL = "unknown"

SYSTEM = """You are the race strategist of a 2D racing team. A code executor drives the car along a
racing line; you choose how fast it takes each corner. speed_factor 1.0 = the physics grip limit
v_max = sqrt(grip * R) of that corner; allowed range 0.6 .. 1.15 (values above 1.0 rely on the line
being wider than the centerline - aggressive). The briefing from race control is binding: every
speed-limit / yellow-flag zone in it MUST appear in "zones" (waypoint numbers as written there).
Reply with ONE JSON object only:
{"cap": <top speed u/s>, "corners": [{"id": "T1", "speed_factor": 1.05, "note": "<= 6 words"}],
 "zones": [{"from_wp": 9, "to_wp": 11, "max_speed": 15}], "say": "<radio message, <= 80 chars>"}"""


class Driver:
    NAME = "Strategist"
    AUTHOR = "Agentic GP"
    COLOR = "#8338ec"
    KIND = "llm"
    MODEL = _MODEL
    MOTTO = "Think slow, drive fast."
    ANATOMY = ["perceive", "act", "map", "plan", "llm", "tools"]
    PARAMS = {
        "min_factor": 0.6,      # clamp range for the LLM's speed factors
        "max_factor": 1.15,
        "line_margin": 3.0,     # racing line distance to the walls (units)
        "use_tools": 1,         # 1 = offer corner_speed / braking_distance tools to the model
        "max_calls": 3,         # LLM calls we allow ourselves in plan() (official events: 2)
    }

    def __init__(self) -> None:
        self.follower: dk.PathFollower | None = None
        self.raw_plan: dict | None = None       # what the model wrote
        self.plan_used: dict = {}               # what we drive (validated + clamped + fallback)
        self.problems: list[str] = []
        self.source = "physics"                 # "llm" | "llm (fixed)" | "physics"
        self.last_say: str | None = None

    # ================================================================== plan: once per session
    def plan(self, track: dict) -> None:
        self.track = track
        self.phys = dk.physics_of(track)
        self.corners = dk.track_corners(track)
        self.n_wp = len(track["waypoints"])
        prompt = (dk.track_brief(track) + "\n\nWrite the corner plan (every corner id) and the zones from "
                  "the briefing (an empty list if there are none).")

        raw, used, problems = None, None, []
        calls_before = llm.llm_stats()["calls"]
        try:
            raw = self._ask(prompt)
            used, problems = self.validate(raw)
            self.source = "llm" if not problems else "llm (fixed)"
        except llm.LLMError as exc:                                    # no LLM: physics plan
            problems = [f"LLM unavailable: {exc}"[:200]]
            used, self.source = None, "physics"
        if used is not None and problems and self._may_call(calls_before):   # ONE retry, with the errors
            fix = (prompt + "\n\nYour previous answer:\n" + _short_json(raw) +
                   "\n\nProblems found by the validator:\n- " + "\n- ".join(problems) +
                   "\nReturn the corrected, complete JSON.")
            try:                                     # a failed retry keeps the first (usable) plan
                raw2 = llm.chat_json(fix, system=SYSTEM)
                used2, problems2 = self.validate(raw2)
                if len(problems2) <= len(problems):
                    raw, used, problems = raw2, used2, problems2
                    self.source = "llm" if not problems else "llm (fixed)"
            except llm.LLMError as exc:
                problems = problems + [f"retry failed, first plan kept: {exc}"[:200]]
        if used is None or not used["corners"] and not used["zones"]:
            used, self.source = self.validate({})[0], "physics"
        self.raw_plan, self.plan_used, self.problems = raw, used, problems
        self._build(used)

    def _ask(self, prompt: str) -> dict:
        """One LLM conversation -> a dict (LLMError if it fails)."""
        if self.PARAMS["use_tools"]:
            return llm.chat_json(prompt, system=SYSTEM, tools=[corner_speed, braking_distance], max_rounds=2)
        return llm.chat_json(prompt, system=SYSTEM)

    def _may_call(self, calls_before: int) -> bool:
        used = llm.llm_stats()["calls"] - calls_before
        return used < self.PARAMS["max_calls"] and llm.llm_allowed()

    # ------------------------------------------------------------------ the validator
    def validate(self, raw: object) -> tuple[dict, list[str]]:
        """Model output -> (plan we can drive, list of problems). Never raises."""
        try:
            return self._validate(raw)
        except Exception as exc:          # a shape we did not foresee: physics plan, and say why
            plan, _ = self._validate({})
            return plan, [f"unreadable plan ({type(exc).__name__}: {exc})"[:200]]

    def _validate(self, raw: object) -> tuple[dict, list[str]]:
        P = self.PARAMS
        problems: list[str] = []
        raw = raw if isinstance(raw, dict) else {}
        plan = {"cap": self.phys["max_speed"], "factors": {}, "notes": {}, "corners": [], "zones": [],
                "say": None}

        cap = _number(raw.get("cap"))
        if cap is not None:
            plan["cap"] = min(max(cap, 10.0), self.phys["max_speed"])
            if plan["cap"] != cap:
                problems.append(f"cap {cap} clamped to {plan['cap']:g} (allowed 10..{self.phys['max_speed']:g})")

        known = {c["id"] for c in self.corners}
        corners = _as_list(raw.get("corners"), "corners", problems)
        for item in corners:
            cid = item.get("id") if isinstance(item, dict) else None
            if not isinstance(cid, str) or cid not in known:
                problems.append(f"unknown corner entry {str(item)[:60]} (ids: {', '.join(sorted(known))})")
                continue
            f = _number(item.get("speed_factor"))
            if f is None:
                problems.append(f"{cid}: speed_factor must be a number")
                continue
            clamped = min(max(f, P["min_factor"]), P["max_factor"])
            if clamped != f:
                problems.append(f"{cid}: speed_factor {f} clamped to {clamped} "
                                f"(allowed {P['min_factor']}..{P['max_factor']})")
            plan["factors"][cid] = clamped
            plan["notes"][cid] = str(item.get("note") or "")[:40]
        plan["corners"] = sorted(plan["factors"])
        missing = sorted(known - set(plan["factors"]))
        if raw and missing:
            problems.append(f"no speed_factor for {', '.join(missing)} (physics speed used)")

        said = _numbers_in(self.track.get("briefing"))          # grounding: what race control wrote
        for z in _as_list(raw.get("zones"), "zones", problems):
            a, b, v = (_number(z.get(k)) if isinstance(z, dict) else None for k in ("from_wp", "to_wp", "max_speed"))
            if None in (a, b, v) or not (0 <= a < self.n_wp and 0 <= b < self.n_wp) or not 3 <= v <= 40:
                problems.append(f"bad zone {str(z)[:60]} (need from_wp/to_wp 0..{self.n_wp - 1}, max_speed 3..40)")
                continue
            if not {a, b, v} <= said:                            # an invented zone would cost seconds
                problems.append(f"zone WP{a:g}-{b:g} max {v:g} is not in the briefing (dropped): only copy "
                                f"zones race control announced")
                continue
            plan["zones"].append({"from_wp": int(a), "to_wp": int(b), "max_speed": float(v)})
        if raw and _briefing_mentions_zone(self.track.get("briefing")) and not plan["zones"]:
            problems.append("the briefing announces a speed limit but 'zones' is empty")
        say = raw.get("say")
        plan["say"] = (say.strip()[:80] or None) if isinstance(say, str) else None
        return plan, problems

    # ------------------------------------------------------------------ plan -> executor
    def _build(self, plan: dict) -> None:
        p = self.phys
        line = dk.racing_line(self.track, margin=self.PARAMS["line_margin"])
        grip = np.minimum(p["max_speed"], np.sqrt(p["max_lateral_accel"] * np.minimum(dk.curvature_radii(line), 1e9)))
        v = grip.copy()
        for c in self.corners:                                  # the LLM's factor inside each corner
            mask = _corner_mask(len(line), c)
            v[mask] = grip[mask] * plan["factors"].get(c["id"], 1.0)
        v = np.clip(v, 5.0, min(plan["cap"], p["max_speed"]))
        speeds = dk.braking_pass(v, line, p["brake"])
        self.follower = dk.PathFollower.from_track(self.track, line=line, speeds=speeds)
        for z in plan["zones"]:                                  # brake BEFORE the yellow flag
            self.follower.set_speed_limit(z["from_wp"], z["to_wp"], z["max_speed"])
        zones = "; ".join(f"WP{z['from_wp']}-{z['to_wp']} max {z['max_speed']:g}" for z in plan["zones"])
        self.plan_note = (plan["say"] or f"Plan ({self.source})") + (f" | zones: {zones}" if zones else "")
        self.last_say = None

    # ================================================================== drive: every tick, no LLM
    def drive(self, obs: dict) -> dict:
        if self.follower is None:
            return {"throttle": 0.3, "steering": dk.clamp(obs["angle_to_next_waypoint"] / 30.0)}
        throttle, steering = self.follower.control(obs)
        return {"throttle": throttle, "steering": steering, "say": self._radio(obs, throttle),
                "debug": dict(self.follower.debug, plan=self.source)}

    def _radio(self, obs: dict, throttle: float) -> str | None:
        """Plan at the start, then the LLM's note for each corner as we brake for it."""
        if obs["tick"] == 0:
            text = self.plan_note
        elif throttle < -0.2:
            idx = self.follower.idx or 0
            ahead = [c for c in self.corners if 0 <= c["apex_index"] - idx < 60]
            note = self.plan_used.get("notes", {}).get(ahead[0]["id"]) if ahead else None
            text = f"{ahead[0]['id']}: {note}" if note else None
        else:
            text = None
        if text is None or text == self.last_say:
            return None
        self.last_say = text
        return text


# ====================================================================== tools for the model
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


# ====================================================================== small helpers
def _number(x: object) -> float | None:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _numbers_in(text: str | None) -> set[float]:
    return {float(x) for x in re.findall(r"\d+(?:\.\d+)?", text or "")}


def _as_list(x: object, field: str, problems: list[str]) -> list:
    """A JSON list field, or [] (and a problem) if the model wrote something else."""
    if x is None:
        return []
    if isinstance(x, list):
        return x
    problems.append(f"'{field}' must be a list, got {type(x).__name__}")
    return []


def _corner_mask(n: int, corner: dict) -> np.ndarray:
    a, b = int(corner["start_index"]), int(corner["end_index"])
    idx = np.arange(n)
    return (idx >= a) & (idx <= b) if a <= b else (idx >= a) | (idx <= b)


def _briefing_mentions_zone(text: str | None) -> bool:
    return bool(text) and re.search(r"(yellow flag|speed limit|max(imum)? speed)\s.*\d", text, re.I) is not None \
        and "no speed restrictions" not in text.lower()


def _short_json(d: object) -> str:
    return json.dumps(d)[:1500]
