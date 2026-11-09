"""Max - the house racer: a MiniMax crew of Strategist, Race Engineer, Steward and Driver.

Agent type (Russell & Norvig): a LEARNING, GOAL-BASED agent built as a small TEAM with ROLES.
Agentic concepts: tool use (the LLM calls Python functions), reflection (a debrief after every
practice lap), memory (a Reflexion-style lessons notebook + the best plan so far), guardrails
(a code Steward checks every LLM proposal), budgets (<= 2 LLM calls per plan / per reflect), and
lock-in (no experiments in qualifying and the race). "The LLM advises, code keeps the car safe."

            plan(track)                     drive(obs) every 0.5 s               reflect(report)
    ┌──────────────────────────┐   ┌──────────────────────────────┐   ┌──────────────────────────────┐
    │ STRATEGIST (MiniMax)     │   │ DRIVER (code)                │   │ guardrails (code): keep-best,│
    │ reads corner table +     │   │ PathFollower on the racing   │   │ revert, crash -> slow corner │
    │ BRIEFING, calls tools:   │   │ line; logs per corner which  │   │ RACE ENGINEER (MiniMax)      │
    │ set_aggression,          │──>│ limit held it back (plan or  │──>│ reads summary_text + own     │
    │ set_corner_factor,       │   │ safety net) -> telemetry     │   │ telemetry + lessons, tools:  │
    │ set_safety_margin,       │   │ radio: plan, corner calls    │   │ corner_stats, crash_detail,  │
    │ add_speed_limit          │   └──────────────────────────────┘   │ set_*, add_lesson            │
    │      │                   │                 ^                     │      │                       │
    │      v                   │                 │  next attempt       │      v                       │
    │ STEWARD (code): bounds,  │                 └─────────────────────│ STEWARD: <= 10 % per step,   │
    │ zones from the briefing  │                                       │ bounds, zones; lock-in for   │
    └──────────────────────────┘                                       │ qualifying / race            │
                                                                       └──────────────────────────────┘

How it works
- Knobs (the "plan"): one speed factor per corner (x the grip limit sqrt(20 R) on the racing line),
  an overall aggression for everything else, the executor's safety margin (free space it keeps in
  front of the car: less = later braking) and the speed-limit zones from the briefing.
- STRATEGIST (plan, <= 2 LLM calls): gets the corner table and the briefing, sets the knobs with
  tools in its first reply, then answers with a radio message.
- DRIVER (drive, no LLM): follows the plan and writes its own telemetry: per corner, how many ticks
  the plan or the safety net was the limit. That tells the engineer WHICH knob to turn.
- RACE ENGINEER (reflect after a practice lap, <= 2 LLM calls): reads report["summary_text"], the
  telemetry and the lessons notebook, may look things up (corner_stats, crash_detail), proposes
  changes with tools and writes a lesson.
- STEWARD (code): rejects factors outside [0.6, 1.6], trims any change to <= 10 % per step, keeps
  the safety margin in [0.8, 4], makes sure every zone in the briefing is respected (it parses the
  briefing itself as a cross-check) and rejects zones the briefing does not contain (grounding: an
  invented zone would cost the whole weekend). The first plan must not be estimated > 5 % slower than
  the code plan (sanity check). Rejections go back to the model as the tool result.
- Guardrails before the engineer: a crash slows that corner down; a lap clearly slower than the best
  (more than the 0.5 % tick noise) reverts to the best plan. Next phase qualifying/race: lock in the best plan, no more LLM.
- No LLM (mock, no key, refused, timeout): the code plan races safely (racing line, factor 1.0).
- Memory lives only inside this session (no files): plan -> practice -> qualifying -> race.

Run it:
    python race.py drive drivers/examples/l6_llm_crew_chief.py --attempts 3
    python race.py drive drivers/examples/l6_llm_crew_chief.py --track briefing_demo --attempts 3
    LLM_PROVIDER=mock python race.py drive drivers/examples/l6_llm_crew_chief.py   # code fallback

Expected on grand_prix: code fallback 49.95 s (clean, = the racing line); with MiniMax 44.6-45.8 s
after 3 practice laps (e.g. 46.05 -> 46.05 -> 44.65), clean, and 0 zone violations on briefing_demo
(~53.5-54 s). 6 LLM calls per weekend, 4-45 s each. Max is good, but beatable: a hand-tuned FIXED
setting (aggression 1.3, margin 0.8) is faster on most tracks - see drivers/README.md.

Try this
1. Read self.crew_log after a session: every tool call, its arguments and the Steward's verdict.
2. Ablation "no memory": clear self.notebook in reflect(). Does the engineer start to oscillate?
3. Ablation "no steward": --param max_step=1 --param max_factor=3 --param min_margin=0.1. Crashes?
4. Give the engineer a new tool (e.g. set_lookahead) and prove from the trace that it used it.
"""

from __future__ import annotations

import copy
import json
import math
import re
import time

import numpy as np

from agentic_gp import driverkit as dk
from agentic_gp import llm

try:
    _MODEL = llm.describe_llm()
except Exception:                       # never fail at import time because of LLM settings
    _MODEL = "unknown"


STRATEGIST_SYSTEM = """You are the STRATEGIST of Max's race team (2D racing sim). A code executor drives the
car along a racing line. You set its plan with tools; a code Steward checks every value.
Knobs:
- set_aggression(factor): speed factor for every corner without its own factor (1.0 = grip limit).
- set_corner_factor(corner, factor): speed in that corner = factor x grip limit on our line.
  Allowed {min_factor} .. {max_factor}. The line is wider than the centerline, so values above 1.0 can work.
- set_safety_margin(units): free space the executor keeps in front of the car (default 3, allowed
  {min_margin} .. {max_margin}). Lower = brakes later = faster, but less room for mistakes.
- add_speed_limit(from_wp, to_wp, max_speed): REQUIRED for every yellow flag / speed limit in the
  briefing (numbers exactly as written there). Zones that are not in the briefing are rejected.
Team experience on this sim (25 test tracks): aggression 1.0 with margin 3 is always clean but slow.
Plans with aggression 1.15-1.45 and margin 0.8-1.5 were clean and a median 6 % faster (up to 14 %),
but the best combination differs from track to track, and maxing every knob out was slower on most tracks (the car
brakes for the walls) or crashed. Read the corner table (tight corners, corners after long straights)
and choose. Remarks in the briefing without a number (marshals,
weather, ...) are colour, not rules - only numeric limits are.
Keep your reasoning SHORT (a few lines), then act.
You have exactly TWO replies. Reply 1: call ALL the tools you need at once (parallel calls).
Reply 2: one line "RADIO: <message to the driver, max 90 characters>"."""

ENGINEER_SYSTEM = """You are the RACE ENGINEER of Max's team. After each practice lap you read the lap report
and the car's own telemetry, then adjust the plan for the next lap. Goal: the fastest CLEAN lap (a crash
costs 5 s, a tick too fast in a zone 1 s). A code Steward checks every change: factors {min_factor} ..
{max_factor}, safety margin {min_margin} .. {max_margin}, at most {max_step:.0%} change per knob per debrief.
Telemetry: "net%" = share of ticks in a corner where the SAFETY NET (margin) limited the speed,
"plan%" = where our planned corner speed did.
Recipe: (1) many corners with net% > 50 -> lower the safety margin by 0.2-0.5; (2) a corner with
plan% >= 60 and no crash -> raise its factor 3-8 %; (3) a crash -> lower that corner 5-10 %;
(4) add_lesson with what you learned. Faster is NOT always better: more speed or less margin can make
the car brake harder for the walls - if a push made the lap slower, the notebook says so; try another
knob instead. Lap times jitter by a few tenths (0.5 s ticks): judge trends.
The pit wall is waiting: keep your reasoning SHORT (a few lines), then act.
You have exactly TWO replies. Reply 1: call the tools at once (parallel calls allowed). Reply 2: one
line "RADIO: <message to the driver, max 90 characters>"."""


class Driver:
    NAME = "Max"
    AUTHOR = "House"
    COLOR = "#d4a017"                     # #0, black and gold
    KIND = "llm"
    MODEL = _MODEL
    MOTTO = "Plan it. Drive it. Debrief it."
    ANATOMY = ["perceive", "act", "map", "plan", "memory", "reflect", "llm", "tools", "roles"]
    PARAMS = {
        "start_aggression": 1.0,      # code (fallback) plan: grip limit everywhere ...
        "start_margin": 3.0,          # ... with the default safety margin
        "min_factor": 0.6,            # Steward bounds
        "max_factor": 1.6,
        "max_step": 0.10,             # Steward: max relative change per knob per debrief
        "min_margin": 0.8,
        "max_margin": 4.0,
        "max_margin_step": 1.0,       # Steward: max safety-margin change per debrief (units)
        "line_margin": 3.0,           # racing line distance to the walls
        "crash_cut": 0.9,             # guardrail: a crash multiplies that corner's factor by this
        "explore_tolerance": 0.005,   # a clean lap within 0.5 % of the best is "noise": keep exploring from it
        "sanity_slack": 0.05,         # Steward: a first plan estimated > 5 % slower than the code plan is refused
        "llm_calls": 2,               # LLM calls per plan() and per reflect() (official limit: 2)
        "call_timeout": 45.0,         # seconds per LLM request: a stuck request must not eat the whole debrief
    }

    def __init__(self) -> None:
        self.follower: dk.PathFollower | None = None
        self.plan_now: dict = {}              # the plan the Driver executes
        self.best: dict | None = None         # {"plan", "time", "attempt"}
        self.notebook: list[str] = []         # Reflexion-style lessons (this session only)
        self.history: list[dict] = []         # one entry per attempt
        self.crew_log: list[dict] = []        # every LLM call / tool call / Steward verdict
        self.last_report: dict = {}
        self.locked = False
        self.radio_queue: list[str] = []
        self.last_say: str | None = None

    # ===================================================================== plan (STRATEGIST)
    def plan(self, track: dict) -> None:
        self.track = track
        self.phys = dk.physics_of(track)
        self.corners = dk.track_corners(track)
        self.corner_by_id = {c["id"]: c for c in self.corners}
        self.n_wp = len(track["waypoints"])
        self.line = dk.racing_line(track, margin=self.PARAMS["line_margin"])
        radii = np.minimum(dk.curvature_radii(self.line), 1e9)
        self.grip = np.minimum(self.phys["max_speed"], np.sqrt(self.phys["max_lateral_accel"] * radii))
        self.corner_at_point = np.full(len(self.line), -1)          # which corner each line point is in
        for k, c in enumerate(self.corners):
            self.corner_at_point[_corner_mask(len(self.line), c, pad=3)] = k
        self.briefing_zones = parse_briefing(track.get("briefing") or "", self.n_wp)

        P = self.PARAMS
        code_plan = {"version": 1, "source": "code", "aggression": P["start_aggression"], "corners": {},
                     "margin": P["start_margin"], "zones": [dict(z, source="briefing") for z in self.briefing_zones],
                     "radio": None}
        self.plan_now = code_plan
        steward = Steward(self, code_plan, first_plan=True)
        run = self._ask_crew("Strategist", self._system(STRATEGIST_SYSTEM), self._strategist_prompt(),
                             steward.tools(self, strategist=True))
        new_plan = steward.finish(reference=code_plan)
        if steward.accepted and not steward.refused:
            new_plan["source"] = "llm"
        new_plan["radio"] = _radio_line(run.final_text if run else "") or self._plan_summary(new_plan)
        self._apply(new_plan)

    def _system(self, template: str) -> str:
        """The system prompt with the Steward's bounds from PARAMS (so --param changes reach the model)."""
        P = self.PARAMS
        return template.format(min_factor=P["min_factor"], max_factor=P["max_factor"], min_margin=P["min_margin"],
                               max_margin=P["max_margin"], max_step=P["max_step"])

    def _strategist_prompt(self) -> str:
        lines = [dk.track_brief(self.track), "",
                 f"Zones the Steward found in the briefing (you must add them too): "
                 f"{json.dumps(self.briefing_zones) if self.briefing_zones else 'none'}",
                 f"Current (default) plan: aggression {self.plan_now['aggression']}, "
                 f"safety margin {self.plan_now['margin']}.",
                 "Make the plan now: call the tools in your first reply."]
        return "\n".join(lines)

    # ===================================================================== drive (DRIVER, no LLM)
    def drive(self, obs: dict) -> dict:
        if self.follower is None:
            return {"throttle": 0.3, "steering": dk.clamp(obs["angle_to_next_waypoint"] / 30.0)}
        throttle, steering = self.follower.control(obs)
        self._log_tick(obs)
        return {"throttle": throttle, "steering": steering, "say": self._radio(obs, throttle),
                "debug": {**self.follower.debug, "plan": self.plan_now.get("version")}}

    def _log_tick(self, obs: dict) -> None:
        """Own telemetry: in which corner are we, and which limit set the speed this tick?"""
        idx = self.follower.idx
        if obs["tick"] == 0:
            self.ticks = {c["id"]: {"ticks": 0, "net": 0, "plan": 0, "min_speed": 99.0} for c in self.corners}
        if idx is None or self.corner_at_point[idx] < 0:
            return
        row = self.ticks[self.corners[self.corner_at_point[idx]]["id"]]
        row["ticks"] += 1
        row["min_speed"] = min(row["min_speed"], round(float(obs["speed"]), 1))
        d = self.follower.debug
        if "v_plan" in d and "v_target" in d:
            row["net" if d["v_target"] < d["v_plan"] - 0.5 else "plan"] += 1

    def _radio(self, obs: dict, throttle: float) -> str | None:
        text = None
        if obs["tick"] == 0 and self.plan_now.get("radio"):
            text = self.plan_now["radio"]
        elif self.radio_queue:
            text = self.radio_queue.pop(0)
        elif throttle < -0.3 and obs["lap"] == 1:            # corner calls on the first lap only
            idx = self.follower.idx or 0
            ahead = [c for c in self.corners if 0 <= c["apex_index"] - idx < 50]
            if ahead:
                f = self.plan_now["corners"].get(ahead[0]["id"], self.plan_now["aggression"])
                text = f"{ahead[0]['id']} at {f:.2f}"
        if text is None or text == self.last_say:
            return None
        self.last_say = text
        return text[:160]

    # ===================================================================== reflect (ENGINEER)
    def reflect(self, report: dict) -> None:
        if self.follower is None:
            return
        P = self.PARAMS
        self.last_report = report
        time_ = report["race_time"] if report.get("completed") else None
        entry = {"attempt": report.get("attempt"), "phase": report.get("phase"), "plan": copy.deepcopy(self.plan_now),
                 "time": time_, "crashes": report.get("crashes", 0),
                 "zone_violations": len(report.get("zone_violations") or []),
                 "telemetry": copy.deepcopy(getattr(self, "ticks", {}))}
        self.history.append(entry)

        # --- qualifying / race: the plan is locked, nothing to learn from here on
        if self.locked or report.get("phase") in ("qualifying", "race"):
            return

        # --- guardrail 1: keep the best (a DNF is never best)
        improved = time_ is not None and (self.best is None or time_ < self.best["time"])
        if improved:
            self.best = {"plan": copy.deepcopy(self.plan_now), "time": time_, "attempt": entry["attempt"]}
        lap = entry["attempt"]
        if time_ is None:
            self.notebook.append(f"Lap {lap} (v{self.plan_now['version']}): DNF ({report.get('ended_by')}) - too risky.")
        elif not improved and self.best and time_ > self.best["time"] * (1 + P["explore_tolerance"]):
            self.notebook.append(f"Lap {lap} (v{self.plan_now['version']}): {time_:.2f}s, slower than the best "
                                 f"{self.best['time']:.2f}s (v{self.best['plan']['version']}) - reverted to v"
                                 f"{self.best['plan']['version']}.")
        elif not improved and self.best and self.plan_now["version"] != self.best["plan"]["version"]:
            self.notebook.append(f"Lap {lap} (v{self.plan_now['version']}): {time_:.2f}s, within tick noise of the "
                                 f"best {self.best['time']:.2f}s - exploring on from v{self.plan_now['version']}.")

        # --- lock-in: qualifying / race next (or the session is over) -> drive the best plan, no LLM
        if report.get("next_phase") in ("qualifying", "race", None):
            if self.best and self.plan_now["version"] != self.best["plan"]["version"]:
                self._apply(copy.deepcopy(self.best["plan"]))
            if report.get("next_phase") is not None:
                self.locked = True
                best = f" ({self.best['time']:.2f}s)" if self.best else ""
                self.plan_now["radio"] = f"{report['next_phase'].title()}: plan v{self.plan_now['version']} locked in{best}"
            return

        # --- guardrail 2: start from the best plan (or from this one if it was clean and within the
        #     tick noise of the best: sideways moves keep the search going); a crash slows that corner
        near_best = (time_ is not None and self.best is not None and not report.get("crashes")
                     and time_ <= self.best["time"] * (1 + P["explore_tolerance"]))
        base = copy.deepcopy(self.plan_now if near_best else (self.best["plan"] if self.best else self.plan_now))
        base["version"] = max(h["plan"]["version"] for h in self.history) + 1
        crashed = [self._crash_corner(c) for c in report.get("crash_log") or []]
        for cid in [c for c in crashed if c]:
            old = base["corners"].get(cid, base["aggression"])
            base["corners"][cid] = round(max(P["min_factor"], old * P["crash_cut"]), 3)
            base["margin"] = min(P["max_margin"], base["margin"] + 0.5)
        if crashed:
            self.notebook.append(f"Lap {lap}: crash at {', '.join(c or '?' for c in crashed)} - slowed down there.")

        # --- the RACE ENGINEER proposes, the STEWARD disposes
        steward = Steward(self, base, first_plan=False)
        prompt = self._engineer_prompt(report, base)
        run = self._ask_crew("Engineer", self._system(ENGINEER_SYSTEM), prompt, steward.tools(self, strategist=False))
        new_plan = steward.finish()
        if steward.accepted:
            new_plan["source"] = "llm"
        if self.best and _same_plan(new_plan, self.best["plan"]):
            new_plan = copy.deepcopy(self.best["plan"])           # nothing changed: same version, no new name
        radio = _radio_line(run.final_text if run else "")
        new_plan["radio"] = radio or self._plan_summary(new_plan)
        if time_ is not None:
            self.radio_queue.append(f"Debrief: {time_:.2f}s" + (" - new best" if improved else ""))
        self._apply(new_plan)

    def _engineer_prompt(self, report: dict, base: dict) -> str:
        rows = []
        tele = self.history[-1]["telemetry"]
        for c in self.corners:
            t = tele.get(c["id"], {})
            n = max(t.get("ticks", 0), 1)
            rows.append(f"  {c['id']:>4} {c['direction']:>5} R{c['min_radius']:>5.1f} factor "
                        f"{base['corners'].get(c['id'], base['aggression']):.2f}  min speed {t.get('min_speed', 0):>4}"
                        f"  net% {100 * t.get('net', 0) // n:>3}  plan% {100 * t.get('plan', 0) // n:>3}")
        best = (f"Best so far: {self.best['time']:.2f}s with plan v{self.best['plan']['version']}."
                if self.best else "No clean lap yet.")
        return "\n".join([
            "LAP REPORT (from race control):", _without_segments(report.get("summary_text")
                                                                or dk.summarize_report(report, self.track)),
            "", "OUR TELEMETRY (this lap, per corner):", *rows, "",
            f"PLAN FOR THE NEXT LAP (v{base['version']}, your starting point - the code already reverted to the best "
            f"plan if this lap was clearly slower and slowed corners where we crashed; 'factor' in the telemetry "
            f"table is from this starting point): aggression {base['aggression']:.2f}, safety margin {base['margin']:.2f}, "
            f"zones {json.dumps([{k: z[k] for k in ('from_wp', 'to_wp', 'max_speed')} for z in base['zones']])}",
            best, "", "LESSONS NOTEBOOK:", *(f"- {x}" for x in self.notebook[-8:] or ["(empty)"]), "",
            "Debrief now: call the tools in your first reply."])

    def _crash_corner(self, crash: dict) -> str | None:
        cid = crash.get("corner")
        if not cid and crash.get("position"):
            cid = dk.corner_at(self.track, crash["position"])
        if cid and cid.startswith("before "):
            cid = cid[len("before "):]
        return cid if cid in self.corner_by_id else None

    # ===================================================================== the LLM call (budgeted)
    def _ask_crew(self, role: str, system: str, prompt: str, tools: list) -> llm.ToolRun | None:
        """One budgeted conversation (<= PARAMS llm_calls calls). Never raises: None = no LLM."""
        budget = int(self.PARAMS["llm_calls"])
        timeout = float(self.PARAMS["call_timeout"])
        before, t0 = llm.llm_stats(), time.perf_counter()
        run, error = None, None
        try:
            if budget >= 1 and llm.llm_allowed():
                run = llm.ask_with_tools(prompt, tools, system=system, max_rounds=budget, timeout=timeout)
                if not run.calls and run.rounds < budget and llm.llm_allowed():   # answered without tools
                    nudge = run.messages + [{"role": "user", "content": "Use the tools now (call them), then "
                                                                        "answer with the RADIO line."}]
                    run2 = llm.ask_with_tools(nudge, tools, max_rounds=budget - run.rounds, timeout=timeout)
                    run = llm.ToolRun(run2.final_text, run.calls + run2.calls, run.rounds + run2.rounds,
                                      run2.messages, run2.stopped)
        except llm.LLMError as exc:
            error = f"{type(exc).__name__}: {exc}"[:200]
        after = llm.llm_stats()
        self.crew_log.append({
            "role": role, "attempt": len(self.history), "calls": after["calls"] - before["calls"],
            "tokens": after["prompt_tokens"] + after["completion_tokens"] - before["prompt_tokens"] - before["completion_tokens"],
            "seconds": round(time.perf_counter() - t0, 2), "tool_calls": [(c["name"], c["args"]) for c in (run.calls if run else [])],
            "error": error, "stopped": run.stopped if run else None,
            "final": (run.final_text[:200] if run else None)})
        return run

    # ===================================================================== plan -> executor
    def _speeds(self, plan: dict) -> np.ndarray:
        """Planned speed at every racing-line point: factor x grip limit, then the braking pass."""
        p = self.phys
        factor = np.full(len(self.line), float(plan["aggression"]))
        for k, c in enumerate(self.corners):
            factor[self.corner_at_point == k] = plan["corners"].get(c["id"], plan["aggression"])
        v = np.clip(self.grip * factor, 5.0, p["max_speed"])
        return dk.braking_pass(v, self.line, p["brake"])

    def estimate_lap(self, plan: dict) -> float:
        """Rough lap time of a plan from the map alone (planned speeds + zones, no walls): the
        Steward's sanity check. Not a prediction - only good for comparing two plans."""
        v = self._speeds(plan)
        segments = dk.segment_ids(len(self.line), dk.waypoint_indices(self.track, self.line))
        n_seg = int(segments.max()) + 1
        for z in plan["zones"]:
            for k in range((z["to_wp"] - z["from_wp"]) % n_seg or n_seg):
                inside = segments == (z["from_wp"] + k) % n_seg
                v[inside] = np.minimum(v[inside], max(z["max_speed"] - 0.5, 1.0))
        v = dk.braking_pass(v, self.line, self.phys["brake"])
        ds = np.linalg.norm(np.roll(self.line, -1, axis=0) - self.line, axis=1)
        return float(np.sum(ds / np.maximum(v, 1.0)))

    def _apply(self, plan: dict) -> None:
        """Turn a plan into speeds along the racing line and hand them to the executor."""
        speeds = self._speeds(plan)
        margin = float(plan["margin"])
        self.follower = dk.PathFollower.from_track(self.track, line=self.line, speeds=speeds, safety_margin=margin,
                                                   blind_factor=min(1.0, margin / 3.0))
        for z in plan["zones"]:
            self.follower.set_speed_limit(z["from_wp"], z["to_wp"], z["max_speed"])
        self.plan_now = plan
        self.last_say = None

    def _plan_summary(self, plan: dict) -> str:
        special = ", ".join(f"{k} {v:.2f}" for k, v in sorted(plan["corners"].items())[:4])
        zones = " | yellow " + ", ".join(f"WP{z['from_wp']}-{z['to_wp']}<={z['max_speed']:g}" for z in plan["zones"]) \
            if plan["zones"] else ""
        return (f"Max v{plan['version']} ({plan['source']}): aggr {plan['aggression']:.2f}, margin "
                f"{plan['margin']:.1f}" + (f", {special}" if special else "") + zones)


# ========================================================================= the STEWARD (code)
class Steward:
    """Code verifier between the LLM and the car. Every tool call goes through here.

    Checks: bounds (PARAMS: factor 0.6..1.6, margin 0.8..4), max change per debrief (10 % per knob,
    1 unit of margin), zones (grounded: only zones whose numbers are in the briefing; every briefing
    zone stays, limits can only get stricter) and a sanity check of the strategist's first plan (an
    estimated lap > 5 % slower than the code plan is refused). Verdicts are returned to the model as
    the tool result, so it can fix a rejected proposal in its next reply.
    """

    def __init__(self, max_: Driver, base: dict, first_plan: bool) -> None:
        self.max, self.P = max_, max_.PARAMS
        self.base = copy.deepcopy(base)          # the starting point of this step
        self.plan = copy.deepcopy(base)          # the working copy the tools change
        self.first_plan = first_plan             # the strategist's first plan: no 10 % rule yet
        self.accepted = 0
        self.rejected = 0
        self.refused = False                     # the whole plan was refused by the sanity check

    def _log(self, tool: str, args: dict, verdict: str) -> str:
        self.max.crew_log.append({"role": "Steward", "tool": tool, "args": args, "verdict": verdict})
        if verdict.startswith("REJECTED"):
            self.rejected += 1
        else:
            self.accepted += 1
        return verdict

    def _check_factor(self, value: object, old: float) -> tuple[float | None, str]:
        f = _number(value)
        if f is None:
            return None, "REJECTED: factor must be a number"
        if not self.P["min_factor"] <= f <= self.P["max_factor"]:
            return None, f"REJECTED: factor {f} outside {self.P['min_factor']}..{self.P['max_factor']}"
        if not self.first_plan:
            lo, hi = old * (1 - self.P["max_step"]), old * (1 + self.P["max_step"])
            if not lo - 1e-9 <= f <= hi + 1e-9:
                trimmed = round(min(max(f, lo), hi), 3)
                return trimmed, f"TRIMMED to {trimmed} (max {self.P['max_step']:.0%} change per debrief, was {old:.3f})"
        return round(f, 3), "OK"

    def tools(self, max_: Driver, strategist: bool) -> list:
        """The Python functions the model may call (their docstrings are the tool descriptions)."""
        steward, car = self, max_

        def set_aggression(factor: float, reason: str = "") -> str:
            """Speed factor (x grip limit) for every corner that has no factor of its own.

            Args:
                factor: 0.6 .. 1.6 (Steward bounds, PARAMS min_factor..max_factor)
                reason: why, in a few words
            """
            f, verdict = steward._check_factor(factor, steward.base["aggression"])
            if f is not None:
                steward.plan["aggression"] = f
            return steward._log("set_aggression", {"factor": factor, "reason": reason}, verdict)

        def set_corner_factor(corner: str, factor: float, reason: str = "") -> str:
            """Speed factor (x grip limit on our line) for one corner, e.g. corner="T5", factor=0.95.

            Args:
                corner: corner id from the table, e.g. "T3"
                factor: 0.6 .. 1.6
                reason: why, in a few words
            """
            if corner not in car.corner_by_id:
                return steward._log("set_corner_factor", {"corner": corner, "factor": factor},
                                    f"REJECTED: unknown corner {corner!r}; ids: {', '.join(car.corner_by_id)}")
            old = steward.base["corners"].get(corner, steward.base["aggression"])
            f, verdict = steward._check_factor(factor, old)
            if f is not None:
                steward.plan["corners"][corner] = f
            return steward._log("set_corner_factor", {"corner": corner, "factor": factor, "reason": reason}, verdict)

        def set_safety_margin(units: float, reason: str = "") -> str:
            """Free space (units) the executor keeps in front of the car; lower = later braking.

            Args:
                units: 0.8 .. 4 (default 3)
                reason: why, in a few words
            """
            m = _number(units)
            if m is None or not steward.P["min_margin"] <= m <= steward.P["max_margin"]:
                return steward._log("set_safety_margin", {"units": units},
                                    f"REJECTED: margin must be {steward.P['min_margin']}..{steward.P['max_margin']}")
            verdict = "OK"
            step = steward.P["max_margin_step"]
            if not steward.first_plan and abs(m - steward.base["margin"]) > step:
                m = round(steward.base["margin"] + math.copysign(step, m - steward.base["margin"]), 2)
                verdict = f"TRIMMED to {m} (max {step} units change per debrief)"
            steward.plan["margin"] = round(m, 2)
            return steward._log("set_safety_margin", {"units": units, "reason": reason}, verdict)

        def add_speed_limit(from_wp: int, to_wp: int, max_speed: float, reason: str = "") -> str:
            """Respect a speed-limit zone from the briefing: between waypoints from_wp and to_wp.

            Args:
                from_wp: first waypoint of the zone
                to_wp: last waypoint of the zone
                max_speed: limit in u/s
            """
            a, b, v = _number(from_wp), _number(to_wp), _number(max_speed)
            n = car.n_wp
            if None in (a, b, v) or not (0 <= a < n and 0 <= b < n) or not 3 <= v <= 40:
                return steward._log("add_speed_limit", {"from_wp": from_wp, "to_wp": to_wp, "max_speed": max_speed},
                                    f"REJECTED: need waypoints 0..{n - 1} and max_speed 3..40")
            if not {a, b, v} <= _numbers_in(car.track.get("briefing")):
                return steward._log("add_speed_limit", {"from_wp": from_wp, "to_wp": to_wp, "max_speed": max_speed},
                                    "REJECTED: this zone is not in the briefing (only copy zones race control "
                                    f"announced; the Steward found: {json.dumps(car.briefing_zones) or 'none'})")
            zone = {"from_wp": int(a), "to_wp": int(b), "max_speed": float(v), "source": "llm"}
            same = [z for z in steward.plan["zones"] if (z["from_wp"], z["to_wp"]) == (zone["from_wp"], zone["to_wp"])]
            if same:
                same[0]["max_speed"] = min(same[0]["max_speed"], zone["max_speed"])   # only ever stricter
                verdict = f"OK (zone already known; limit {same[0]['max_speed']:g})"
            else:
                steward.plan["zones"].append(zone)
                verdict = "OK"
            return steward._log("add_speed_limit", {"from_wp": from_wp, "to_wp": to_wp, "max_speed": max_speed}, verdict)

        def remove_speed_limit(from_wp: int, to_wp: int) -> str:
            """Remove a zone WE added by mistake. Zones from the briefing can never be removed.

            Args:
                from_wp: first waypoint of the zone
                to_wp: last waypoint of the zone
            """
            a, b = _number(from_wp), _number(to_wp)
            hit = [z for z in steward.plan["zones"] if a is not None and b is not None
                   and (z["from_wp"], z["to_wp"]) == (int(a), int(b))]
            official = {(z["from_wp"], z["to_wp"]) for z in car.briefing_zones}
            if not hit:
                verdict = "REJECTED: no such zone in the plan"
            elif (hit[0]["from_wp"], hit[0]["to_wp"]) in official:
                verdict = "REJECTED: this zone is in the briefing - it stays"
            else:
                steward.plan["zones"] = [z for z in steward.plan["zones"] if z is not hit[0]]
                verdict = "OK (zone removed)"
            return steward._log("remove_speed_limit", {"from_wp": from_wp, "to_wp": to_wp}, verdict)

        def corner_report() -> list:
            """The corner table: id, direction, waypoints, min radius, grip-limit speed, our current factor."""
            return [{"id": c["id"], "dir": c["direction"], "wp": f"{c.get('from_wp')}-{c.get('to_wp')}",
                     "min_R": c["min_radius"], "v_max": c["v_max"],
                     "factor": steward.plan["corners"].get(c["id"], steward.plan["aggression"])} for c in car.corners]

        def max_corner_speed(radius: float) -> float:
            """Grip-limit speed (u/s) for a corner radius: sqrt(20 * R), capped at 40."""
            return round(min(40.0, math.sqrt(20.0 * max(float(radius), 0.0))), 2)

        def get_lessons() -> list:
            """The lessons notebook: what earlier laps of this session taught us."""
            return car.notebook[-10:]

        def corner_stats(corner: str) -> dict:
            """Last lap in one corner: speeds from race control + our telemetry (net/plan-limited ticks)."""
            report_rows = {c["id"]: c for c in car.last_report.get("corners") or []}
            tele = car.history[-1]["telemetry"].get(corner) if car.history else None
            if corner not in car.corner_by_id:
                return {"error": f"unknown corner {corner}"}
            return {"corner": car.corner_by_id[corner], "lap": report_rows.get(corner), "telemetry": tele,
                    "factor": steward.plan["corners"].get(corner, steward.plan["aggression"])}

        def segment_stats(segment: int) -> dict:
            """Last lap in one waypoint segment (time, entry/min/max speed, crashes)."""
            for s in car.last_report.get("segments") or []:
                if s.get("segment") == int(segment):
                    return s
            return {"error": f"no segment {segment}"}

        def crash_detail(index: int = 0) -> dict:
            """One crash of the last lap (0 = first): where, how fast, controls and wall rays."""
            log = car.last_report.get("crash_log") or []
            if not log:
                return {"error": "no crashes on the last lap"}
            return log[max(0, min(int(index), len(log) - 1))]

        def add_lesson(text: str) -> str:
            """Write one short lesson into the notebook (read again at the next debrief)."""
            car.notebook.append(f"Lap {len(car.history)}: {str(text)[:160]}")
            return "noted"

        if strategist:
            return [set_aggression, set_corner_factor, set_safety_margin, add_speed_limit, corner_report,
                    max_corner_speed, get_lessons]
        return [corner_stats, segment_stats, crash_detail, set_aggression, set_corner_factor, set_safety_margin,
                add_speed_limit, remove_speed_limit, add_lesson]

    def finish(self, reference: dict | None = None) -> dict:
        """Final Steward check before the plan goes to the car: every briefing zone must be there, and
        (first plan) the plan must not be estimated much slower than the `reference` code plan."""
        for z in self.max.briefing_zones:
            if not any((y["from_wp"], y["to_wp"]) == (z["from_wp"], z["to_wp"]) and y["max_speed"] <= z["max_speed"]
                       for y in self.plan["zones"]):
                self.plan["zones"].append(dict(z, source="steward"))
                self._log("finish", z, "ADDED missing briefing zone")
        if reference is not None and self.plan != self.base:
            mine, ref = self.max.estimate_lap(self.plan), self.max.estimate_lap(reference)
            if mine > ref * (1 + self.P["sanity_slack"]):
                self._log("finish", {"estimate": round(mine, 2), "code_plan": round(ref, 2)},
                          f"REFUSED: plan estimated {mine:.1f}s vs {ref:.1f}s for the code plan - racing the code plan")
                self.refused = True
                return copy.deepcopy(reference)
        if self.plan != self.base:
            self.plan["version"] = self.base["version"] + (1 if self.first_plan else 0)
        return self.plan


# ========================================================================= helpers
def parse_briefing(text: str, n_wp: int) -> list[dict]:
    """The Steward's cross-check of the briefing: "between waypoints 9 and 11: max speed 15"."""
    zones = []
    for m in re.finditer(r"waypoints?\s+(\d+)\s*(?:and|to|-|–)\s*(\d+)[^.\d]*?(?:max(?:imum)?\s+speed|limit)\D{0,12}"
                         r"(\d+(?:\.\d+)?)", text, re.I):
        a, b, v = int(m.group(1)), int(m.group(2)), float(m.group(3))
        if 0 <= a < n_wp and 0 <= b < n_wp:
            zones.append({"from_wp": a, "to_wp": b, "max_speed": v})
    return zones


def _without_segments(text: str) -> str:
    """The per-segment table is long and rarely needed: the engineer can ask segment_stats() instead."""
    out, skipping = [], False
    for line in text.splitlines():
        if line.startswith("Segments"):
            skipping = True
            out.append("(per-segment table omitted - call segment_stats(segment) if you need it)")
            continue
        if skipping and line.startswith("  "):
            continue
        skipping = False
        out.append(line)
    return "\n".join(out)


def _same_plan(a: dict, b: dict) -> bool:
    """Same knobs and the same zones (the version number and the radio text do not count)."""
    def zones(plan: dict) -> list[tuple]:
        return [(z["from_wp"], z["to_wp"], z["max_speed"]) for z in plan["zones"]]

    same_knobs = all(a.get(k) == b.get(k) for k in ("aggression", "corners", "margin"))
    return same_knobs and zones(a) == zones(b)


def _numbers_in(text: str | None) -> set[float]:
    """Every number written in a text (the Steward's grounding check for zones)."""
    return {float(x) for x in re.findall(r"\d+(?:\.\d+)?", text or "")}


def _radio_line(text: str) -> str | None:
    m = re.search(r"RADIO:\s*(.+)", text or "")
    line = (m.group(1) if m else "").strip().strip('"')
    return line[:120] or None


def _number(x: object) -> float | None:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _corner_mask(n: int, corner: dict, pad: int = 0) -> np.ndarray:
    a, b = int(corner["start_index"]) - pad, int(corner["end_index"]) + pad
    idx = np.arange(n)
    a, b = a % n, b % n
    return (idx >= a) & (idx <= b) if a <= b else (idx >= a) | (idx <= b)
