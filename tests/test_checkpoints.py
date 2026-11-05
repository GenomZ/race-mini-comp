"""The lesson's handrail files (lesson/checkpoints/) keep the promises the chapters make.

Every handrail: passes the static check, has the metadata, and lands in its checkpoint band
(lesson/instructor/CALIBRATION.md). LLM handrails run on the mock provider (conftest.py), and the
scripted mock answers prove the LLM paths: validation, clamping, zones, the Steward, budgets.
"""

from __future__ import annotations

import ast
import difflib
import importlib.util
import json
import re
import subprocess
import sys

import pytest

from agentic_gp import llm
from agentic_gp.driver import ANATOMY_TAGS, KINDS
from agentic_gp.race import SessionConfig, check_driver_source, load_driver, run_session
from agentic_gp.sim.track import PUBLIC_TRACKS

CP = "lesson/checkpoints"
HANDRAILS = ["ch0_pit_lane.py", "ch1_reflex.py", "ch2_planner.py", "ch3_learner.py", "ch4_pilot.py",
             "ch5_strategist.py", "ch6_crew_chief.py", "ch7_fly_fixed.py"]


def _path(repo, name):
    return repo / CP / name


def _run(repo, name, track="grand_prix", **kw):
    return run_session(str(_path(repo, name)), SessionConfig(track=track, **kw))


def _module(repo, name):
    spec = importlib.util.spec_from_file_location(f"cp_{name[:-3]}", _path(repo, name))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _times(res):
    return [a.result.race_time if a.result.completed else None for a in res.attempts]


# ------------------------------------------------------------------ contract + metadata
@pytest.mark.parametrize("name", HANDRAILS)
def test_static_check_passes(repo, name):
    issues = check_driver_source(_path(repo, name))
    assert [i.message for i in issues if i.level == "error"] == []


@pytest.mark.parametrize("name", HANDRAILS)
def test_race_check_command_passes(repo, name):
    out = subprocess.run([sys.executable, "race.py", "check", f"{CP}/{name}"], cwd=repo, capture_output=True,
                         text=True, encoding="utf-8", errors="replace")
    assert out.returncode == 0 and "PASSED" in out.stdout, out.stdout + out.stderr


def test_metadata(repo):
    names, colors = set(), set()
    for name in HANDRAILS:
        drv = load_driver(_path(repo, name))
        assert drv.author == "your_handle", name                     # the student puts their handle here
        assert drv.kind in KINDS and drv.motto and set(drv.anatomy) <= set(ANATOMY_TAGS), name
        assert all(isinstance(v, (int, float, type(None))) for v in drv.params.values()), name
        names.add(drv.name)
        colors.add(drv.color.lower())
        drv.close()
    assert len(names) == len(colors) == len(HANDRAILS)                # distinct cars in a ghost race


@pytest.mark.parametrize("name", HANDRAILS)
def test_docstring_says_how_to_use_it(repo, name):
    doc = ast.get_docstring(ast.parse(_path(repo, name).read_text(encoding="utf-8")))
    assert doc.startswith("HANDRAIL CP"), name
    assert "git diff --no-index" in doc and "race.py" in doc, name


def test_never_imports_the_sim(repo):
    for name in HANDRAILS:
        tree = ast.parse(_path(repo, name).read_text(encoding="utf-8"))
        mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | \
               {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        assert not any(m and m.startswith(("agentic_gp.sim", "agentic_gp.race", "agentic_gp.cli")) for m in mods)


# ------------------------------------------------------------------ grown from the template (diffable)
def _code(text: str) -> list[str]:
    """Source lines without the module docstring."""
    tree = ast.parse(text)
    start = tree.body[1].lineno if isinstance(tree.body[0], ast.Expr) else 1
    return text.splitlines()[start - 1:]


def test_ch0_is_the_template_plus_metadata(repo):
    tpl = _code((repo / "drivers" / "template.py").read_text(encoding="utf-8"))
    ch0 = _code(_path(repo, "ch0_pit_lane.py").read_text(encoding="utf-8"))
    changed = [ln for ln in difflib.unified_diff(tpl, ch0, lineterm="", n=0)
               if ln[:1] in "+-" and not ln.startswith(("+++", "---"))]
    assert changed and all(re.match(r"[+-](NAME|COLOR|MOTTO) = ", ln) for ln in changed), changed


@pytest.mark.parametrize("name", ["ch1_reflex.py", "ch4_pilot.py"])
def test_function_style_keeps_the_template_names(repo, name):
    text = _path(repo, name).read_text(encoding="utf-8")
    for marker in ("# --------------------------------------------------------------------------- TODO 0: who are you?",
                   "PARAMS = {", "_memory: dict", "dk.recover_if_stuck(obs, _memory)", "dk.steer_towards(angle",
                   "dk.wall_repulsion(obs, P[\"wall_gain\"])", "v_stop = math.sqrt(2 * 25.0 * max(front - P[\"margin\"]"):
        assert marker in text, (name, marker)


@pytest.mark.parametrize("name", ["ch2_planner.py", "ch3_learner.py", "ch5_strategist.py", "ch6_crew_chief.py"])
def test_class_style_keeps_the_ch1_reflexes_as_fallback(repo, name):
    text = _path(repo, name).read_text(encoding="utf-8")
    assert "class Driver" in text and "def plan(self, track" in text and "def reflexes(obs: dict, memory: dict)" in text
    assert "v[i] = min(v[i], math.sqrt(v[nxt] ** 2 + 2 * " in text          # the Ch2 backward pass, by hand


@pytest.mark.parametrize("name", ["ch2_planner.py", "ch3_learner.py", "ch5_strategist.py", "ch6_crew_chief.py"])
def test_fallback_reflexes_drive_when_plan_failed(repo, name):
    obs = _run(repo, "ch1_reflex.py", "oval").attempts[0].frames[3]["obs"]
    ch1 = _module(repo, "ch1_reflex.py").drive(obs)
    a = _module(repo, name).Driver().drive(obs)               # plan() never ran: follower is None
    assert (a["throttle"], a["steering"]) == pytest.approx(ch1)               # = the Ch1 reflexes


# ------------------------------------------------------------------ CP0 - CP3: code handrails, exact bands
def test_cp0_pit_lane_finishes(repo):
    r = _run(repo, "ch0_pit_lane.py").attempts[0].result
    assert r.completed and r.crashes == 0 and 80 < r.race_time < 95           # measured 85.85


@pytest.mark.parametrize("track", PUBLIC_TRACKS)
def test_cp1_reflex_clean_on_every_public_track(repo, track):
    r = _run(repo, "ch1_reflex.py", track).attempts[0].result
    assert r.completed and r.crashes == 0, (track, r.crashes)


def test_cp1_band_on_grand_prix(repo):
    t = _run(repo, "ch1_reflex.py").best_lap
    assert 60.0 <= t <= 80.0                                                   # measured 66.70


@pytest.mark.parametrize("track", list(PUBLIC_TRACKS) + ["random:7", "random:12", "random:21"])
def test_cp2_planner_clean_everywhere(repo, track):
    r = _run(repo, "ch2_planner.py", track).attempts[0].result
    assert r.completed and r.crashes == 0, (track, r.crashes)


def test_cp2_band_and_faster_than_ch1(repo):
    t2 = _run(repo, "ch2_planner.py").best_lap
    assert 48.0 <= t2 <= 55.0                                                  # measured 52.50
    assert t2 < _run(repo, "ch1_reflex.py").best_lap - 5.0


def test_cp2_aggression_param_reaches_the_plan(repo):
    slow = run_session(load_driver(_path(repo, "ch2_planner.py"), params={"aggression": 0.8}),
                       SessionConfig(track="grand_prix")).best_lap
    assert slow > _run(repo, "ch2_planner.py").best_lap


def test_cp3_learning_curve_on_sprint(repo):
    res = _run(repo, "ch3_learner.py", "sprint", attempts=6)
    times = _times(res)
    assert None not in times
    assert (times[0] - min(times)) / times[0] >= 0.03                          # measured 41.40 -> 39.40 (4.8 %)
    best = res.attempts[res.best_attempt].result
    assert best.crashes == 0


def test_cp3_locks_in_the_best_knobs(repo):
    res = _run(repo, "ch3_learner.py", attempts=4, phases=["practice", "practice", "practice", "qualifying"])
    practice = [a.result.race_time for a in res.attempts[:3]]
    q = res.attempts[3].result
    assert q.completed and q.race_time == min(practice)                        # deterministic: the best plan again


def test_cp3_crash_slows_that_segment(repo):
    mod = _module(repo, "ch3_learner.py")
    drv = mod.Driver()
    res = run_session(drv, SessionConfig(track="sprint", attempts=1, reflect_after_last=False))
    report = res.attempts[0].to_reflect_dict()
    report.update(crashes=1, completed=False, crash_log=[{"segment": 5}], next_phase="practice")
    before = list(drv.seg_factor)
    drv.reflect(report)
    assert drv.seg_factor[5] == pytest.approx(before[5] * 0.85)
    assert drv.seg_factor[4] == pytest.approx(before[4] * 0.85)
    assert drv.seg_factor[7] == before[7]


# ------------------------------------------------------------------ CP4: the pilot (mock LLM)
def test_cp4_pilot_finishes_with_the_mock_llm(repo):
    res = _run(repo, "ch4_pilot.py")
    ch1 = _run(repo, "ch1_reflex.py").best_lap
    assert res.attempts[0].result.completed and res.best_lap == ch1          # mock answers "{}": reflexes drive


def test_cp4_pilot_uses_at_most_20_llm_ticks(repo):
    llm.set_mock_responder(lambda messages, tools: {"throttle": 0.6, "steering": 0.0, "say": "flat out"})
    llm.reset_llm_stats()
    res = _run(repo, "ch4_pilot.py", track="oval")
    assert res.attempts[0].result.completed
    assert res.llm["calls"] == 20
    assert any(m["text"] == "flat out" for m in res.attempts[0].messages)


def test_cp4_pilot_rejects_bad_answers(repo):
    llm.set_mock_responder(lambda messages, tools: {"throttle": "fast", "steering": None})
    mod = _module(repo, "ch4_pilot.py")
    obs = run_session(str(_path(repo, "ch1_reflex.py")), SessionConfig(track="oval")).attempts[0].frames[5]["obs"]
    assert mod.ask_llm(obs) is None
    llm.set_mock_responder(lambda messages, tools: {"throttle": 7, "steering": -3})
    assert mod.ask_llm(obs) == {"throttle": 1.0, "steering": -1.0, "say": None}    # clipped to -1..1


# ------------------------------------------------------------------ CP5: the strategist
def _strategist(plan: dict):
    def responder(messages, tools):
        system = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
        if "race strategist" in system:
            return plan
        return "OK"
    return responder


def test_cp5_mock_llm_races_the_physics_plan(repo):
    res5 = _run(repo, "ch5_strategist.py", attempts=4)
    res3 = _run(repo, "ch3_learner.py", attempts=4)
    assert _times(res5) == _times(res3)                                        # no LLM = your Ch3 car


def test_cp5_reads_the_briefing_zone(repo):
    plan = {"cap": 40, "corners": [{"id": "T5", "speed_factor": 0.9, "note": "hairpin"}],
            "zones": [{"from_wp": 9, "to_wp": 11, "max_speed": 15}], "say": "Yellow flag 9-11"}
    llm.set_mock_responder(_strategist(plan))
    res = _run(repo, "ch5_strategist.py", "briefing_demo")
    a = res.attempts[0]
    assert a.result.completed and a.result.crashes == 0 and a.zone_violations == []
    assert res.llm["calls"] <= 3
    llm.set_mock_responder(None)
    assert _run(repo, "ch5_strategist.py", "briefing_demo").attempts[0].zone_violations   # mock: no reader


def test_cp5_validator_clamps_drops_and_grounds(repo):
    mod = _module(repo, "ch5_strategist.py")
    drv = mod.Driver()
    llm.set_mock_responder(_strategist({
        "cap": 99, "corners": [{"id": "T1", "speed_factor": 2.0}, {"id": "T99", "speed_factor": 1.0},
                               {"id": "T2", "speed_factor": "fast"}],
        "zones": [{"from_wp": 9, "to_wp": 11, "max_speed": 15}, {"from_wp": 2, "to_wp": 3, "max_speed": 12}]}))
    run_session(drv, SessionConfig(track="briefing_demo", attempts=1, reflect_after_last=False))
    s = drv.strategy
    assert s["factors"] == {"T1": 1.15} and s["cap"] == 40.0
    assert s["zones"] == [{"from_wp": 9, "to_wp": 11, "max_speed": 15.0}]   # the invented zone is dropped
    assert any("not in the briefing" in p for p in drv.problems)
    assert drv.source == "llm (fixed)"


def test_cp5_respects_the_official_plan_budget(repo):
    llm.set_mock_responder(_strategist({"corners": "banana"}))              # always invalid -> wants a retry
    res = _run(repo, "ch5_strategist.py", attempts=1, phases=["practice"], llm_policy="official")
    assert res.attempts[0].result.completed and res.llm["calls"] <= 2


# ------------------------------------------------------------------ CP6: the crew chief
def test_cp6_mock_llm_learns_with_the_ch3_rules(repo):
    times = _times(_run(repo, "ch6_crew_chief.py", "random:7", attempts=4))
    assert None not in times and min(times) < times[0]                        # measured 44.45 -> 42.25


def _engineer(patch: dict, tool_first: bool = True):
    state = {"n": 0}

    def responder(messages, tools):
        system = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
        if "RACE ENGINEER" not in system:
            return {}                                                          # strategist: physics plan
        state["n"] += 1
        if tool_first and messages[-1]["role"] == "user":
            return [{"tool": "segment_stats", "args": {"segment": 3}}, {"tool": "crash_detail", "args": {}}]
        return patch
    return responder


def test_cp6_engineer_patch_goes_through_the_steward(repo):
    llm.set_mock_responder(_engineer({"segments": {"3": 1.5, "4": 0.97, "99": 1.0, "5": 9.0}, "margin": 0.2,
                                      "lesson": "push 3", "say": "Pushing segment 3"}))
    mod = _module(repo, "ch6_crew_chief.py")
    drv = mod.Driver()
    run_session(drv, SessionConfig(track="sprint", attempts=2, phases=["practice", "practice"]))
    entry = drv.crew_log[0]
    assert entry["calls"] == 2 and [t[0] for t in entry["tool_calls"]] == ["segment_stats", "crash_detail"]
    verdicts = " | ".join(entry["verdicts"])
    assert "TRIMMED segment 3 to 1.100" in verdicts                            # check 2: <= 10 % per debrief
    assert "OK segment 4 1.000 -> 0.970" in verdicts
    assert "REJECTED segment 99" in verdicts and "REJECTED segment 5=9.0" in verdicts   # check 1: bounds
    assert "OK margin 3.00 -> 2.00 (trimmed)" in verdicts                     # margin: <= 1 unit per debrief
    assert "Lap 1: push 3" in drv.notebook and any(n.startswith("Steward:") for n in drv.notebook)


def test_cp6_steward_never_pushes_inside_a_zone(repo):
    mod = _module(repo, "ch6_crew_chief.py")
    drv = mod.Driver()
    llm.set_mock_responder(lambda messages, tools: {"corners": [], "zones": [
        {"from_wp": 9, "to_wp": 11, "max_speed": 15}]} if "race strategist" in messages[0]["content"] else "OK")
    run_session(drv, SessionConfig(track="briefing_demo", attempts=1, reflect_after_last=False))
    assert drv.zone_segments() == {9, 10}
    drv.crew_log.append({})
    verdicts = drv.steward({"segments": {"9": 1.05, "10": 0.95, "12": 1.05}})
    assert verdicts[0].startswith("REJECTED segment 9") and drv.seg_factor[9] == 1.0
    assert drv.seg_factor[10] == pytest.approx(0.95) and drv.seg_factor[12] == pytest.approx(1.05)


def test_cp6_official_weekend_stays_in_budget(repo):
    llm.set_mock_responder(_engineer({"segments": {"2": 1.05}, "lesson": "ok"}))
    res = _run(repo, "ch6_crew_chief.py", "oval", attempts=4,
               phases=["practice", "practice", "practice", "qualifying"], llm_policy="official")
    assert all(a.result.completed and a.result.crashes == 0 for a in res.attempts)
    calls = [a.llm_usage.get("calls", 0) for a in res.attempts]                # cumulative per attempt
    per_reflect = [b - a for a, b in zip(calls, calls[1:])]
    assert max(per_reflect) <= 2 and res.llm["calls"] <= 8
    assert calls[-1] == calls[-2]                                              # lock-in: no LLM before qualifying


# ------------------------------------------------------------------ CP7: the fixed fly
def test_cp7_fixed_fly_is_the_toy_fly(repo):
    fixed = _run(repo, "ch7_fly_fixed.py").attempts[0].result
    assert fixed.completed and fixed.crashes == 0 and fixed.race_time == pytest.approx(63.3)
    mod = _module(repo, "ch7_fly_fixed.py")
    assert mod.mirror_check()


def test_cp7_one_wiring_row_differs_from_the_broken_fly(repo):
    broken = (repo / "drivers" / "examples" / "fly_broken.py").read_text(encoding="utf-8")
    fixed = _path(repo, "ch7_fly_fixed.py").read_text(encoding="utf-8")
    rows = lambda text: [ln.split("#")[0].strip() for ln in text.splitlines() if ln.strip().startswith('("')]
    diff = [(a, b) for a, b in zip(rows(broken), rows(fixed)) if a != b]
    assert diff == [('("WALL_L20",  "DNa02_L",   3.0,  "wall ahead-left"),',
                     '("WALL_L20",  "DNa02_R",   3.0,  "wall ahead-left"),')]


def test_cp7_fly_class_lint(repo):
    out = subprocess.run([sys.executable, "-m", "agentic_gp.fly.rules", f"{CP}/ch7_fly_fixed.py"], cwd=repo,
                         capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert out.returncode == 0 and "Fly-class OK" in out.stdout, out.stdout + out.stderr


# ------------------------------------------------------------------ the calibration report
def test_calibration_report_renders_and_every_band_holds(repo):
    spec = importlib.util.spec_from_file_location("agp_calibrate", repo / "lesson" / "instructor" / "calibrate.py")
    cal = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cal)
    data = json.loads((repo / "lesson" / "instructor" / "calibration.json").read_text(encoding="utf-8"))
    md = cal.render(data)
    assert "## Checkpoint bands (CP0-CP8)" in md
    bands = cal.bands(data)
    assert [b["cp"] for b in bands] == [f"CP{i}" for i in range(9)]
    assert all(b["ok"] for b in bands), [b for b in bands if not b["ok"]]
