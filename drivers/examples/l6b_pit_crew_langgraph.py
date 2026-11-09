"""L6b Pit Crew (LangGraph) - the debrief as a multi-agent graph: analyst -> strategist -> safety officer.

Agent type (Russell & Norvig): learning agent; its learning element is a small MULTI-AGENT system
orchestrated by LangGraph (the only LangGraph file of the ladder). Agentic concepts: explicit
orchestration (nodes, edges, shared state), a conditional edge (retry loop with feedback), and a
hard loop limit = an LLM budget you can read off the graph.

    reflect(report):  LangGraph StateGraph (shared state: report, findings, proposal, verdict, plan)

        START ─> analyst ─> strategist ─> safety_officer ──ok──> commit ─> END
                 (code)     (MiniMax)     (code)       │
                                ^                      │ rejected and loops < 2
                                └──── feedback ────────┘

    plan(track): code plan (racing line, grip-limit speeds, zones from the briefing)
    drive(obs):  PathFollower on the plan (no LLM), logs which limit held the car back per corner

How it works
- analyst (code): turns the lap report + the car's own telemetry into findings, one line per corner
  ("T5: min 14.8 vs v_max 16.5, safety net limited 60 % of ticks") plus crashes and the lap time.
- strategist (LLM via llm.get_langchain_model()): proposes JSON {"changes": [{"corner", "factor",
  "why"}], "safety_margin", "lesson", "say"}.
- safety_officer (code): rejects unknown corners, factors outside [0.6, 1.15], changes above 10 %
  per debrief and margins outside [1.5, 4]. Any rejection -> back to the strategist WITH the reasons,
  at most 2 strategist visits (= the official limit of 2 LLM calls per reflect).
- commit (code): keep-best / revert, crash guardrail, then the accepted changes become the next plan.
- Lock-in: next_phase qualifying/race -> no graph, drive the best plan. Mock LLM: the graph still
  runs, the safety officer rejects the empty proposals and the safe plan stays.

Run it:
    python race.py drive drivers/examples/l6b_pit_crew_langgraph.py --attempts 3
    LLM_PROVIDER=mock python race.py drive drivers/examples/l6b_pit_crew_langgraph.py --attempts 3

Expected on grand_prix: 49.95 s on lap 1 (the code plan = racing line), then ~45-48 s with MiniMax
after 2-3 debriefs; with the mock LLM 49.95 s every lap, clean.

Try this
1. print(self.graph.get_graph().draw_mermaid()) - paste it into mermaid.live to see the crew.
2. Read self.graph_trace after a session: which node said what, how many strategist loops?
3. Add a "critic" node between strategist and safety_officer that asks the LLM to double-check.
   How does that interact with the 2-calls-per-reflect budget?
4. Compare with Max (l6_llm_crew_chief.py): same roles, written as plain Python + tool calls.
"""

from __future__ import annotations

import copy
import math
import re
from typing import Any, TypedDict

import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from agentic_gp import driverkit as dk
from agentic_gp import llm

try:
    _MODEL = llm.describe_llm()
except Exception:
    _MODEL = "unknown"

STRATEGIST_SYSTEM = """You are the strategist of a racing pit crew. The analyst gives you findings about the last
lap. Propose small changes to the plan for the next lap. Knobs: a speed factor per corner (x grip limit,
allowed 0.6 .. 1.15, at most 10 % change per debrief) and the safety margin (free space the executor
keeps in front of the car, 1.5 .. 4, at most 1 unit change; lower = brakes later). High "net %" in a
corner = the safety margin is the limit there; high "plan %" = the corner factor is.
Reply with ONE JSON object only:
{"changes": [{"corner": "T3", "factor": 1.08, "why": "..."}], "safety_margin": 2.5,
 "lesson": "<one sentence>", "say": "<radio message, max 80 characters>"}"""


class CrewState(TypedDict, total=False):
    """The shared blackboard every node reads and writes."""
    report: dict
    plan: dict                 # current plan (knobs)
    findings: list[str]
    proposal: dict | None
    feedback: list[str]        # safety officer -> strategist
    accepted: list[dict]
    margin: float | None
    loops: int                 # strategist visits (= LLM calls)
    trace: list[dict]


class Driver:
    NAME = "Pit Crew"
    AUTHOR = "Agentic GP"
    COLOR = "#06d6a0"
    KIND = "llm"
    MODEL = _MODEL
    MOTTO = "Many hands, one car."
    ANATOMY = ["perceive", "act", "map", "plan", "memory", "reflect", "llm", "roles"]
    PARAMS = {
        "min_factor": 0.6,
        "max_factor": 1.15,
        "max_step": 0.10,         # max relative factor change per debrief
        "min_margin": 1.5,
        "max_margin": 4.0,
        "max_loops": 2,           # strategist visits per debrief (= LLM calls)
        "crash_cut": 0.9,
    }

    def __init__(self) -> None:
        self.follower: dk.PathFollower | None = None
        self.best: dict | None = None
        self.lessons: list[str] = []
        self.graph_trace: list[dict] = []
        self.locked = False
        self.last_say: str | None = None
        self.graph = self._build_graph()

    # ===================================================================== the graph
    def _build_graph(self) -> Any:
        g = StateGraph(CrewState)
        g.add_node("analyst", self.analyst)
        g.add_node("strategist", self.strategist)
        g.add_node("safety_officer", self.safety_officer)
        g.add_node("commit", self.commit)
        g.add_edge(START, "analyst")
        g.add_edge("analyst", "strategist")
        g.add_edge("strategist", "safety_officer")
        g.add_conditional_edges("safety_officer", self.route, {"retry": "strategist", "commit": "commit"})
        g.add_edge("commit", END)
        return g.compile()

    def analyst(self, state: CrewState) -> dict:
        """Code: telemetry -> findings (one line per corner, crashes, lap time)."""
        rep, plan = state["report"], state["plan"]
        lap = rep["race_time"] if rep.get("completed") else None
        findings = [f"Lap {rep.get('attempt')}: " + (f"{lap:.2f}s" if lap else f"DNF ({rep.get('ended_by')})")
                    + f", {rep.get('crashes', 0)} crash(es)"
                    + (f"; best so far {self.best['time']:.2f}s" if self.best else "")
                    + f"; safety margin {plan['margin']:.2f}"]
        lap_corners = {c["id"]: c for c in rep.get("corners") or []}
        for c in self.corners:
            t = self.ticks.get(c["id"], {})
            n = max(t.get("ticks", 0), 1)
            seen = lap_corners.get(c["id"], {})
            findings.append(f"{c['id']} (R {c['min_radius']}, v_max {c['v_max']}): factor "
                            f"{plan['corners'].get(c['id'], 1.0):.2f}, min speed {seen.get('min_speed', '?')}, "
                            f"net % {100 * t.get('net', 0) // n}, plan % {100 * t.get('plan', 0) // n}, "
                            f"crashes {seen.get('crashes', 0)}")
        for crash in (rep.get("crash_log") or [])[:5]:
            findings.append(f"crash at {crash.get('corner') or 'segment ' + str(crash.get('segment'))}: "
                            f"speed {crash.get('speed_before')}")
        return {"findings": findings, "trace": state.get("trace", []) + [{"node": "analyst", "lines": len(findings)}]}

    def strategist(self, state: CrewState) -> dict:
        """LLM: findings (+ feedback) -> a JSON proposal. One LLM call per visit."""
        loops = state.get("loops", 0) + 1
        text = "FINDINGS:\n" + "\n".join(state["findings"])
        if self.lessons:
            text += "\nLESSONS SO FAR:\n" + "\n".join(f"- {x}" for x in self.lessons[-6:])
        if state.get("feedback"):
            text += ("\nTHE SAFETY OFFICER REJECTED PARTS OF YOUR LAST PROPOSAL:\n- " + "\n- ".join(state["feedback"])
                     + "\nSend a corrected proposal.")
        proposal, error = None, None
        try:
            if not llm.llm_allowed():
                raise llm.LLMNotAllowed("no LLM call allowed now")
            reply = self.model.invoke([SystemMessage(STRATEGIST_SYSTEM), HumanMessage(text)])
            proposal = llm.extract_json(str(reply.content))
        except Exception as exc:                       # LLMError, bad JSON, ...: no proposal
            error = f"{type(exc).__name__}: {exc}"[:160]
        step = {"node": "strategist", "loop": loops, "proposal": proposal, "error": error}
        return {"proposal": proposal, "loops": loops, "trace": state["trace"] + [step]}

    def safety_officer(self, state: CrewState) -> dict:
        """Code: accept or reject every proposed change, with reasons."""
        P, plan, prop = self.PARAMS, state["plan"], state.get("proposal")
        accepted, feedback, margin = [], [], None
        if not isinstance(prop, dict):
            feedback.append("no valid JSON proposal received")
            prop = {}
        changes = prop.get("changes") or []
        if not isinstance(changes, list):
            feedback.append(f"'changes' must be a list of {{corner, factor, why}}, got {type(changes).__name__}")
            changes = []
        for ch in changes:
            cid = ch.get("corner") if isinstance(ch, dict) else None
            f = _number(ch.get("factor")) if isinstance(ch, dict) else None
            if cid not in self.corner_ids:
                feedback.append(f"unknown corner {cid!r} (ids: {', '.join(self.corner_ids)})")
                continue
            old = plan["corners"].get(cid, 1.0)
            if f is None or not P["min_factor"] <= f <= P["max_factor"]:
                feedback.append(f"{cid}: factor {ch.get('factor')} outside {P['min_factor']}..{P['max_factor']}")
            elif abs(f - old) > P["max_step"] * old + 1e-9:
                feedback.append(f"{cid}: {old:.2f} -> {f} is more than {P['max_step']:.0%} in one debrief")
            else:
                accepted.append({"corner": cid, "factor": round(f, 3), "why": str(ch.get("why", ""))[:80]})
        if prop.get("safety_margin") is not None:
            m = _number(prop.get("safety_margin"))
            if m is None or not P["min_margin"] <= m <= P["max_margin"] or abs(m - plan["margin"]) > 1.0 + 1e-9:
                feedback.append(f"safety_margin {prop.get('safety_margin')} must be {P['min_margin']}..{P['max_margin']} "
                                f"and within 1 unit of {plan['margin']:.2f}")
            else:
                margin = m
        step = {"node": "safety_officer", "accepted": accepted, "margin": margin, "rejected": feedback}
        return {"accepted": accepted, "margin": margin, "feedback": feedback, "trace": state["trace"] + [step]}

    def route(self, state: CrewState) -> str:
        """Conditional edge: retry with feedback while the budget allows, else commit."""
        if state.get("feedback") and state.get("loops", 0) < self.PARAMS["max_loops"] and llm.llm_allowed():
            return "retry"
        return "commit"

    def commit(self, state: CrewState) -> dict:
        """Code: the accepted changes become the next plan."""
        plan = copy.deepcopy(state["plan"])
        for ch in state.get("accepted") or []:
            plan["corners"][ch["corner"]] = ch["factor"]
        if state.get("margin") is not None:
            plan["margin"] = round(float(state["margin"]), 2)
        prop = state.get("proposal") if isinstance(state.get("proposal"), dict) else {}
        if prop.get("lesson"):
            self.lessons.append(f"Lap {state['report'].get('attempt')}: {str(prop['lesson'])[:160]}")
        say = prop.get("say")
        plan["say"] = (say.strip()[:80] or None) if isinstance(say, str) else None
        step = {"node": "commit", "changes": len(state.get("accepted") or []), "margin": plan["margin"]}
        return {"plan": plan, "trace": state["trace"] + [step]}

    # ===================================================================== plan (code)
    def plan(self, track: dict) -> None:
        self.track = track
        self.phys = dk.physics_of(track)
        self.corners = dk.track_corners(track)
        self.corner_ids = [c["id"] for c in self.corners]
        self.line = dk.racing_line(track, margin=3.0)
        radii = np.minimum(dk.curvature_radii(self.line), 1e9)
        self.grip = np.minimum(self.phys["max_speed"], np.sqrt(self.phys["max_lateral_accel"] * radii))
        self.corner_at_point = np.full(len(self.line), -1)
        for k, c in enumerate(self.corners):
            self.corner_at_point[_corner_mask(len(self.line), c, pad=3)] = k
        self.model = llm.get_langchain_model()
        zones = parse_briefing(track.get("briefing") or "", len(track["waypoints"]))
        self.ticks: dict = {}
        self._apply({"corners": {}, "margin": 3.0, "zones": zones, "say": "Pit crew: code plan on the racing line"})

    def _apply(self, plan: dict) -> None:
        factor = np.ones(len(self.line))
        for k, c in enumerate(self.corners):
            factor[self.corner_at_point == k] = plan["corners"].get(c["id"], 1.0)
        v = dk.braking_pass(np.clip(self.grip * factor, 5.0, self.phys["max_speed"]), self.line, self.phys["brake"])
        m = plan["margin"]
        self.follower = dk.PathFollower.from_track(self.track, line=self.line, speeds=v, safety_margin=m,
                                                   blind_factor=min(1.0, m / 3.0))
        for z in plan["zones"]:
            self.follower.set_speed_limit(z["from_wp"], z["to_wp"], z["max_speed"])
        self.plan_now = plan
        self.last_say = None

    # ===================================================================== drive (code)
    def drive(self, obs: dict) -> dict:
        if self.follower is None:
            return {"throttle": 0.3, "steering": dk.clamp(obs["angle_to_next_waypoint"] / 30.0)}
        throttle, steering = self.follower.control(obs)
        if obs["tick"] == 0:
            self.ticks = {c["id"]: {"ticks": 0, "net": 0, "plan": 0} for c in self.corners}
        idx, d = self.follower.idx, self.follower.debug
        if idx is not None and self.corner_at_point[idx] >= 0 and "v_plan" in d:
            row = self.ticks[self.corner_ids[self.corner_at_point[idx]]]
            row["ticks"] += 1
            row["net" if d["v_target"] < d["v_plan"] - 0.5 else "plan"] += 1
        say = None
        if obs["tick"] == 0 and self.plan_now.get("say") and self.plan_now["say"] != self.last_say:
            say = self.last_say = self.plan_now["say"]           # announce each new plan once
        return {"throttle": throttle, "steering": steering, "say": say}

    # ===================================================================== reflect (the graph)
    def reflect(self, report: dict) -> None:
        if self.follower is None or self.locked or report.get("phase") in ("qualifying", "race"):
            return
        P = self.PARAMS
        lap = report["race_time"] if report.get("completed") else None
        if lap is not None and (self.best is None or lap < self.best["time"]):
            self.best = {"plan": copy.deepcopy(self.plan_now), "time": lap}
        if report.get("next_phase") in ("qualifying", "race", None):           # lock-in
            if self.best:
                self._apply(dict(copy.deepcopy(self.best["plan"]), say=f"Locked in: {self.best['time']:.2f}s plan"))
            self.locked = report.get("next_phase") is not None
            return
        start = copy.deepcopy(self.best["plan"] if self.best else self.plan_now)       # revert to the best
        for crash in report.get("crash_log") or []:                                     # crash guardrail
            cid = (crash.get("corner") or "").replace("before ", "")
            if cid in self.corner_ids:
                start["corners"][cid] = round(max(P["min_factor"], start["corners"].get(cid, 1.0) * P["crash_cut"]), 3)
        out = self.graph.invoke({"report": report, "plan": start, "loops": 0, "trace": [], "feedback": []})
        self.graph_trace.append({"attempt": report.get("attempt"), "steps": out["trace"]})
        self._apply(out["plan"])


# ========================================================================= helpers
def parse_briefing(text: str, n_wp: int) -> list[dict]:
    """Speed-limit zones from the briefing: "between waypoints 9 and 11: max speed 15"."""
    zones = []
    for m in re.finditer(r"waypoints?\s+(\d+)\s*(?:and|to|-)\s*(\d+)[^.\d]*?(?:max(?:imum)?\s+speed|limit)\D{0,12}"
                         r"(\d+(?:\.\d+)?)", text, re.I):
        a, b = int(m.group(1)), int(m.group(2))
        if 0 <= a < n_wp and 0 <= b < n_wp:
            zones.append({"from_wp": a, "to_wp": b, "max_speed": float(m.group(3))})
    return zones


def _number(x: object) -> float | None:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _corner_mask(n: int, corner: dict, pad: int = 0) -> np.ndarray:
    a, b = (int(corner["start_index"]) - pad) % n, (int(corner["end_index"]) + pad) % n
    idx = np.arange(n)
    return (idx >= a) & (idx <= b) if a <= b else (idx >= a) | (idx <= b)
