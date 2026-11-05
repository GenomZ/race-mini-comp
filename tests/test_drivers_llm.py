"""LLM drivers with LLM_PROVIDER=mock (conftest): fallbacks, scripted plans, garbage answers, budgets.

(a) default mock (no useful replies)      -> the code fallbacks keep every LLM driver racing clean
(b) scripted mock responder (valid plans) -> the plan is applied (changed factor, zone, lesson, ...)
(c) a responder that raises / talks rubbish -> still finishes
Live MiniMax checks are opt-in: AGP_LIVE_LLM=1 python -m pytest -m llm tests/test_drivers_llm.py
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import os
from pathlib import Path

import pytest

from agentic_gp import SessionConfig, llm, run_session

REPO = Path(__file__).resolve().parent.parent
EX = REPO / "drivers" / "examples"
_ids = itertools.count()


def make(name: str, **params):
    """A fresh Driver instance from an example file (so the test can inspect its memory)."""
    spec = importlib.util.spec_from_file_location(f"_drv_{next(_ids)}", EX / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    drv = mod.Driver()
    if params:
        drv.PARAMS = {**drv.PARAMS, **params}
    return drv


def gp_config(track: str, practice: int = 2, **kw) -> SessionConfig:
    """plan -> practice x N -> qualifying, with the official LLM rules (as in a grand prix)."""
    return SessionConfig(track=track, attempts=practice + 1, phases=["practice"] * practice + ["qualifying"],
                         llm_policy="official", **kw)


def system_of(messages: list) -> str:
    return next((str(m.get("content")) for m in messages if m.get("role") == "system"), "")


def tool_round_done(messages: list) -> bool:
    return any(m.get("role") == "tool" for m in messages)


L2_GRAND_PRIX = 52.5          # tests/test_ladder_core.py measures it; the LLM drivers must not be slower


# ================================================================== (a) default mock: fallbacks
@pytest.mark.parametrize("name", ["l5_llm_strategist.py", "l6_llm_crew_chief.py", "l6b_pit_crew_langgraph.py"])
def test_default_mock_falls_back_to_a_clean_code_plan(name):
    res = run_session(make(name), gp_config("grand_prix"))
    assert res.plan_error is None
    for a in res.attempts:
        assert a.result.completed and a.result.crashes == 0, (name, a.result)
    assert res.best_lap <= L2_GRAND_PRIX


@pytest.mark.parametrize("track", ["oval", "sprint", "speedway", "random:4"])
def test_max_fallback_is_clean_everywhere(track):
    res = run_session(make("l6_llm_crew_chief.py"), gp_config(track, practice=1))
    assert all(a.result.completed and a.result.crashes == 0 for a in res.attempts)


def test_max_steward_reads_the_briefing_even_without_llm():
    """Mock LLM adds no zone, the Steward's own briefing check does: 0 zone violations."""
    max_ = make("l6_llm_crew_chief.py")
    res = run_session(max_, gp_config("briefing_demo", practice=1))
    assert [(z["from_wp"], z["to_wp"], z["max_speed"]) for z in max_.plan_now["zones"]] == [(9, 11, 15.0)]
    assert all(a.result.crashes == 0 and not a.zone_violations for a in res.attempts)


def test_max_obeys_the_official_llm_budget():
    max_ = make("l6_llm_crew_chief.py")
    res = run_session(max_, gp_config("oval", practice=3))
    by_role = [(c["role"], c["calls"]) for c in max_.crew_log if "calls" in c]
    assert by_role[0][0] == "Strategist" and all(n <= 2 for _, n in by_role)
    assert [r for r, _ in by_role].count("Engineer") == 2          # practice 1 + 2; practice 3 -> lock-in
    assert res.llm["calls"] <= 8
    assert max_.locked


# ================================================================== (b) scripted plans are applied
def test_l5_scripted_plan_is_validated_and_applied():
    def responder(messages, tools):
        return {"cap": 60, "corners": [{"id": "T1", "speed_factor": 1.1, "note": "flat"},
                                       {"id": "T5", "speed_factor": 0.3, "note": "hairpin"},
                                       {"id": "T99", "speed_factor": 1.0}],
                "zones": [{"from_wp": 9, "to_wp": 11, "max_speed": 15}], "say": "Yellow 9-11, careful at T5"}

    llm.set_mock_responder(responder)
    drv = make("l5_llm_strategist.py")
    res = run_session(drv, SessionConfig(track="briefing_demo"))
    plan = drv.plan_used
    assert plan["factors"]["T1"] == 1.1 and plan["factors"]["T5"] == 0.6          # 0.3 clamped to 0.6
    assert plan["cap"] == 40.0 and plan["zones"] == [{"from_wp": 9, "to_wp": 11, "max_speed": 15.0}]
    assert any("T99" in p for p in drv.problems) and any("clamped" in p for p in drv.problems)
    a = res.attempts[0]
    assert a.result.completed and a.result.crashes == 0 and not a.zone_violations
    assert a.messages[0]["text"].startswith("Yellow 9-11")


def test_l5_retries_once_with_the_validation_errors():
    seen = []

    def responder(messages, tools):
        seen.append(messages[-1]["content"])
        if len(seen) == 1:
            return {"corners": [{"id": "T1", "speed_factor": "fast"}], "zones": []}
        return {"corners": [{"id": c, "speed_factor": 1.0} for c in [f"T{i}" for i in range(1, 11)]],
                "zones": [{"from_wp": 9, "to_wp": 11, "max_speed": 15}]}

    llm.set_mock_responder(responder)
    drv = make("l5_llm_strategist.py", use_tools=0)
    run_session(drv, SessionConfig(track="briefing_demo", max_ticks_per_lap=5))
    assert len(seen) == 2 and "Problems found by the validator" in seen[1]
    assert drv.problems == [] and drv.source == "llm"


def _max_responder(strategist_calls: list, engineer_calls: list, radio: str = "RADIO: Push!"):
    def responder(messages, tools):
        system = system_of(messages)
        if tool_round_done(messages):
            return radio
        if "STRATEGIST" in system:
            return strategist_calls
        return engineer_calls
    return responder


def test_max_scripted_strategist_and_engineer_change_the_plan():
    llm.set_mock_responder(_max_responder(
        [{"tool": "set_aggression", "args": {"factor": 1.05, "reason": "base"}},
         {"tool": "set_corner_factor", "args": {"corner": "T5", "factor": 0.9, "reason": "hairpin"}},
         {"tool": "set_corner_factor", "args": {"corner": "T2", "factor": 1.9, "reason": "too much"}},
         {"tool": "set_safety_margin", "args": {"units": 2.0}},
         {"tool": "add_speed_limit", "args": {"from_wp": 9, "to_wp": 11, "max_speed": 15}}],
        [{"tool": "corner_stats", "args": {"corner": "T5"}},
         {"tool": "set_corner_factor", "args": {"corner": "T5", "factor": 1.15, "reason": "plan-limited"}},
         {"tool": "add_lesson", "args": {"text": "T5 is plan-limited: raise it slowly"}}]))
    max_ = make("l6_llm_crew_chief.py")
    res = run_session(max_, gp_config("briefing_demo", practice=2))

    first = max_.history[0]["plan"]                                       # the strategist's plan (lap 1)
    assert first["source"] == "llm" and first["aggression"] == 1.05 and first["corners"]["T5"] == 0.9
    assert first["margin"] == 2.0 and "T2" not in first["corners"]       # 1.9 rejected by the Steward
    second = max_.history[1]["plan"]                                      # after the engineer's debrief
    assert second["corners"]["T5"] == pytest.approx(0.99)                # 0.9 -> 1.15 trimmed to +10 %
    assert any("raise it slowly" in x for x in max_.notebook)            # Reflexion notebook
    verdicts = [c["verdict"] for c in max_.crew_log if c.get("role") == "Steward"]
    assert any(v.startswith("REJECTED") for v in verdicts) and any(v.startswith("TRIMMED") for v in verdicts)
    assert all(a.result.completed and a.result.crashes == 0 and not a.zone_violations for a in res.attempts)
    assert any(m["text"] == "Push!" for a in res.attempts for m in a.messages)          # radio from the LLM


def test_max_reverts_to_the_best_plan_when_a_change_made_it_slower():
    # The engineer slows everything down: lap 2 is slower -> qualifying must use the lap-1 plan.
    llm.set_mock_responder(_max_responder(
        [{"tool": "set_aggression", "args": {"factor": 1.1}}],
        [{"tool": "set_aggression", "args": {"factor": 0.6}}, {"tool": "set_safety_margin", "args": {"units": 4}}]))
    max_ = make("l6_llm_crew_chief.py", max_step=1.0)
    res = run_session(max_, gp_config("oval", practice=2))
    t = [a.result.race_time for a in res.attempts]
    assert t[1] > t[0] and t[2] == t[0], t
    assert max_.best["attempt"] == 1 and any("reverted" in x for x in max_.notebook)


def test_l6b_graph_retries_after_a_rejection_and_commits():
    answers = iter([
        json.dumps({"changes": [{"corner": "T1", "factor": 1.9}], "safety_margin": 9}),             # rejected
        json.dumps({"changes": [{"corner": "T1", "factor": 1.08}, {"corner": "T3", "factor": 1.05}],
                    "safety_margin": 2.5, "lesson": "small steps", "say": "Pit crew: T1 up"}),
    ])

    def responder(messages, tools):
        return next(answers, "{}")

    llm.set_mock_responder(responder)
    crew = make("l6b_pit_crew_langgraph.py")
    res = run_session(crew, gp_config("grand_prix", practice=2))
    steps = crew.graph_trace[0]["steps"]
    assert [s["node"] for s in steps] == ["analyst", "strategist", "safety_officer", "strategist",
                                          "safety_officer", "commit"]
    assert steps[2]["rejected"] and not steps[4]["rejected"]
    assert crew.lessons and "small steps" in crew.lessons[0]
    assert all(a.result.completed and a.result.crashes == 0 for a in res.attempts)
    assert res.attempts[1].messages[0]["text"] == "Pit crew: T1 up"


# ================================================================== (c) exceptions and garbage
def _raiser(messages, tools):
    raise RuntimeError("provider exploded")


def _garbage(messages, tools):
    return "banana { not json"


@pytest.mark.parametrize("responder", [_raiser, _garbage], ids=["raises", "garbage"])
@pytest.mark.parametrize("name", ["l5_llm_strategist.py", "l6_llm_crew_chief.py", "l6b_pit_crew_langgraph.py",
                                  "l4_llm_rookie.py"])
def test_llm_drivers_survive_a_broken_llm(name, responder):
    llm.set_mock_responder(responder)
    res = run_session(make(name), SessionConfig(track="oval", attempts=2))
    assert res.plan_error is None and not res.driver_errors
    assert all(a.result.completed and a.result.crashes == 0 for a in res.attempts)


def test_max_tool_errors_go_back_to_the_model():
    """Unknown tools / bad arguments become error text for the model, never an exception."""
    llm.set_mock_responder(_max_responder([{"tool": "warp_drive", "args": {}},
                                           {"tool": "set_corner_factor", "args": {"corner": "T1"}}], []))
    max_ = make("l6_llm_crew_chief.py")
    res = run_session(max_, SessionConfig(track="oval", max_ticks_per_lap=10))
    assert res.plan_error is None and max_.plan_now["source"] == "code"


WRONG_TYPES = [
    {"cap": "fast", "corners": 5, "zones": 7, "say": {"a": 1}},
    {"corners": [{"id": ["T1"], "speed_factor": 1}, 3, None], "zones": [None, 3, {"from_wp": []}]},
    {"corners": {"T1": 1.1}, "zones": "9-11"},
    ["not", "a", "dict"],
]


@pytest.mark.parametrize("raw", WRONG_TYPES, ids=["ints", "nested", "dicts", "list"])
def test_l5_validate_never_raises_on_wrong_types(raw):
    drv = make("l5_llm_strategist.py")
    run_session(drv, SessionConfig(track="briefing_demo", max_ticks_per_lap=2))     # plan() with the mock
    plan, problems = drv.validate(raw)
    assert plan["factors"] == {} and plan["zones"] == [] and plan["say"] is None
    if isinstance(raw, dict):
        assert problems


def test_l5_wrong_types_from_the_llm_race_the_physics_plan():
    """Valid JSON with the wrong types used to crash validate() -> the car crawled (146 s, 6 crashes)."""
    llm.set_mock_responder(lambda m, t: WRONG_TYPES[0])
    drv = make("l5_llm_strategist.py", use_tools=0)
    res = run_session(drv, gp_config("grand_prix", practice=1))
    assert res.plan_error is None and drv.source == "physics" and drv.follower is not None
    assert all(a.result.completed and a.result.crashes == 0 for a in res.attempts)
    assert res.attempts[0].result.race_time < L2_GRAND_PRIX


def test_l5_keeps_its_first_plan_when_the_retry_fails():
    answers = iter([{"cap": 40, "corners": [{"id": "T1", "speed_factor": 1.0, "note": "ok"}],
                     "zones": [{"from_wp": 9, "to_wp": 11, "max_speed": 15}], "say": "first plan"}])
    llm.set_mock_responder(lambda m, t: next(answers, "sorry, still thinking..."))   # the retry is not JSON
    drv = make("l5_llm_strategist.py", use_tools=0)
    res = run_session(drv, gp_config("briefing_demo", practice=1))
    assert drv.source == "llm (fixed)" and drv.plan_used["zones"] == [{"from_wp": 9, "to_wp": 11, "max_speed": 15.0}]
    assert any("first plan kept" in p for p in drv.problems)
    assert not res.attempts[0].zone_violations and res.attempts[0].messages[0]["text"].startswith("first plan")


def test_l5_drops_a_zone_that_is_not_in_the_briefing():
    llm.set_mock_responder(lambda m, t: {"corners": [], "zones": [{"from_wp": 1, "to_wp": 4, "max_speed": 8}]})
    drv = make("l5_llm_strategist.py", use_tools=0)
    res = run_session(drv, gp_config("grand_prix", practice=1))
    assert drv.plan_used["zones"] == [] and any("not in the briefing" in p for p in drv.problems)
    assert res.attempts[0].result.race_time < L2_GRAND_PRIX


def test_l6b_survives_a_proposal_with_wrong_types():
    llm.set_mock_responder(lambda m, t: json.dumps({"changes": 5, "safety_margin": [1], "lesson": ["x"],
                                                    "say": {"a": 1}}))
    crew = make("l6b_pit_crew_langgraph.py")
    res = run_session(crew, gp_config("oval", practice=2))
    assert all(a.reflect_error is None for a in res.attempts) and not res.driver_errors
    rejected = [s for t in crew.graph_trace for s in t["steps"] if s["node"] == "safety_officer"]
    assert rejected and any("must be a list" in r for s in rejected for r in s["rejected"])


def test_max_rejects_an_invented_zone():
    """One hallucinated zone used to cost the whole weekend (83 s instead of 50 s on grand_prix)."""
    llm.set_mock_responder(_max_responder([{"tool": "add_speed_limit", "args": {"from_wp": 1, "to_wp": 4,
                                                                                "max_speed": 8}}], []))
    max_ = make("l6_llm_crew_chief.py")
    res = run_session(max_, gp_config("grand_prix", practice=2))
    assert max_.plan_now["zones"] == []
    verdicts = [c["verdict"] for c in max_.crew_log if c.get("tool") == "add_speed_limit"]
    assert verdicts and verdicts[0].startswith("REJECTED") and "not in the briefing" in verdicts[0]
    assert all(a.result.race_time < L2_GRAND_PRIX for a in res.attempts)


def test_max_steward_zone_tools():
    max_ = make("l6_llm_crew_chief.py")
    run_session(max_, SessionConfig(track="briefing_demo", max_ticks_per_lap=2))   # code plan (default mock)
    Steward = type(max_).plan.__globals__["Steward"]                              # the module's Steward class
    steward = Steward(max_, max_.plan_now, first_plan=False)
    tools = {f.__name__: f for f in steward.tools(max_, strategist=False)}
    assert tools["add_speed_limit"](1, 4, 8).startswith("REJECTED")               # not in the briefing
    assert tools["add_speed_limit"](9, 15, 11) == "OK"                            # numbers from the briefing
    assert tools["remove_speed_limit"](9, 11).startswith("REJECTED")              # the briefing zone stays
    assert tools["remove_speed_limit"](9, 15) == "OK (zone removed)"              # our own mistake can go
    assert [(z["from_wp"], z["to_wp"]) for z in steward.finish()["zones"]] == [(9, 11)]


def test_max_steward_refuses_a_first_plan_much_slower_than_the_code_plan():
    llm.set_mock_responder(_max_responder([{"tool": "set_aggression", "args": {"factor": 0.6}}], []))
    max_ = make("l6_llm_crew_chief.py")
    res = run_session(max_, gp_config("oval", practice=1))
    assert max_.history[0]["plan"]["source"] == "code" and max_.history[0]["plan"]["aggression"] == 1.0
    assert any(str(c.get("verdict", "")).startswith("REFUSED") for c in max_.crew_log)
    assert res.attempts[0].result.race_time == pytest.approx(25.0)                 # = the code plan (L2b)


# ================================================================== L4 rookie: a few dozen ticks
def test_l4_actions_come_from_the_llm_then_the_fallback_takes_over():
    calls = []

    def responder(messages, tools):
        calls.append(json.loads(messages[-1]["content"]))               # the sensor JSON the model sees
        return {"throttle": 0.7, "steering": 0.1, "say": "LLM driving"}

    llm.set_mock_responder(responder)
    rookie = make("l4_llm_rookie.py", MAX_LLM_TICKS=5)
    res = run_session(rookie, SessionConfig(track="grand_prix", max_ticks_per_lap=30))
    frames = res.attempts[0].frames
    assert len(calls) == 5 and rookie.llm_ticks == 5 and "angle_to_next_waypoint" in calls[0]
    assert [(f["throttle"], f["steering"]) for f in frames[1:6]] == [(0.7, 0.1)] * 5   # frame 0 = the start
    assert (frames[10]["throttle"], frames[10]["steering"]) != (0.7, 0.1)          # code fallback now
    texts = [m["text"] for m in res.attempts[0].messages]
    assert "LLM driving" in texts and any("reflexes drive" in t for t in texts)


def test_l4_decide_every_n_ticks_holds_the_action():
    n_calls = []
    llm.set_mock_responder(lambda m, t: n_calls.append(1) or {"throttle": 0.5, "steering": -0.2})
    rookie = make("l4_llm_rookie.py", DECIDE_EVERY_N_TICKS=4, MAX_LLM_TICKS=20)
    res = run_session(rookie, SessionConfig(track="oval", max_ticks_per_lap=16))
    assert len(n_calls) == 4
    assert all((f["throttle"], f["steering"]) == (0.5, -0.2) for f in res.attempts[0].frames[1:17])


def test_l4_bad_answers_fall_back_per_tick():
    llm.set_mock_responder(lambda m, t: {"throttle": "full", "steering": None})
    rookie = make("l4_llm_rookie.py", MAX_LLM_TICKS=3)
    res = run_session(rookie, SessionConfig(track="oval"))
    assert rookie.failures == 3 and res.attempts[0].result.completed


def test_l4_respects_official_rules():
    rookie = make("l4_llm_rookie.py")
    res = run_session(rookie, SessionConfig(track="oval", llm_policy="official", phases=["race"]))
    assert rookie.llm_ticks == 0 and res.attempts[0].result.completed
    assert any("not allowed" in m["text"] for m in res.attempts[0].messages)


# ================================================================== optional live check
@pytest.mark.llm
@pytest.mark.skipif(os.getenv("AGP_LIVE_LLM") != "1", reason="live MiniMax test: set AGP_LIVE_LLM=1")
def test_live_max_plans_with_tools(monkeypatch):
    monkeypatch.delenv("LLM_NO_DOTENV", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    llm.load_env(REPO / ".env")
    max_ = make("l6_llm_crew_chief.py")
    res = run_session(max_, gp_config("oval", practice=2))
    assert max_.crew_log and all(c.get("calls", 0) <= 2 for c in max_.crew_log)
    assert all(a.result.completed for a in res.attempts)
