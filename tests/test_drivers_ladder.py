"""The code drivers of the ladder keep the lesson's promises (no LLM involved).

The "ladder test" on grand_prix: L2 clean; L2b faster than L2; L3 best-of-4 faster than L2 with no DNF;
L1 and the student template clean on every public track. Plus: every example file passes the static
check and carries the metadata the viewers and Streamlit need.
"""

from __future__ import annotations

import ast

import pytest

from agentic_gp import SessionConfig, load_driver, run_session
from agentic_gp.driver import ANATOMY_TAGS, KINDS
from agentic_gp.race import check_driver_source
from agentic_gp.sim.track import PUBLIC_TRACKS

EX = "drivers/examples"
LADDER = ["l0_dummy.py", "l1_reflex.py", "l2_planner.py", "l2b_racing_line.py", "l3_learner.py",
          "l4_llm_rookie.py", "l5_llm_strategist.py", "l6_llm_crew_chief.py", "l6b_pit_crew_langgraph.py"]


def _best(repo, name: str, track: str = "grand_prix", **kw):
    return run_session(str(repo / EX / name), SessionConfig(track=track, **kw))


# ------------------------------------------------------------------ files + metadata
@pytest.mark.parametrize("name", LADDER + ["../template.py"])
def test_static_check_has_no_errors(repo, name):
    issues = check_driver_source(repo / EX / name)
    assert [i.message for i in issues if i.level == "error"] == []


@pytest.mark.parametrize("name", LADDER)
def test_metadata_and_docstring(repo, name):
    drv = load_driver(repo / EX / name)
    assert drv.name and drv.author in ("Agentic GP", "House")
    assert drv.kind in KINDS and drv.color.startswith("#")
    assert drv.motto and drv.anatomy and set(drv.anatomy) <= set(ANATOMY_TAGS)
    assert isinstance(drv.params, dict) and drv.params, "PARAMS (numeric knobs) are required"
    assert all(isinstance(v, (int, float)) for v in drv.params.values())
    if drv.kind == "llm":
        assert drv.model
    doc = ast.get_docstring(ast.parse((repo / EX / name).read_text(encoding="utf-8")))
    for part in ("How it works", "Run it", "Expected", "Try this"):
        assert part in doc, f"{name}: docstring misses '{part}'"
    drv.close()


def test_colors_are_distinct(repo):
    colors = [load_driver(repo / EX / n).color.lower() for n in LADDER]
    assert len(set(colors)) == len(colors)


def test_examples_never_import_the_sim(repo):
    for name in LADDER + ["../template.py"]:
        tree = ast.parse((repo / EX / name).read_text(encoding="utf-8"))
        mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | \
               {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        assert not any(m and m.startswith(("agentic_gp.sim", "agentic_gp.race", "agentic_gp.cli")) for m in mods), name


# ------------------------------------------------------------------ reflex + template: clean everywhere
@pytest.mark.parametrize("track", PUBLIC_TRACKS)
@pytest.mark.parametrize("name", ["l1_reflex.py", "../template.py"])
def test_reflex_and_template_clean_on_every_public_track(repo, name, track):
    r = _best(repo, name, track).attempts[0].result
    assert r.completed and r.crashes == 0, (name, track, r.ended_by, r.crashes)


def test_l1_reflex_is_slower_than_the_planner(repo):
    assert 55.0 < _best(repo, "l1_reflex.py").best_lap < 80.0          # measured 66.7 s


# ------------------------------------------------------------------ the ladder test on grand_prix
@pytest.fixture(scope="module")
def l2_time(request):
    from pathlib import Path
    repo = Path(__file__).resolve().parent.parent
    res = run_session(str(repo / EX / "l2_planner.py"), SessionConfig(track="grand_prix"))
    assert res.attempts[0].result.crashes == 0
    return res.best_lap                                                # measured 52.5 s


def test_l2b_racing_line_beats_l2_and_is_clean(repo, l2_time):
    res = _best(repo, "l2b_racing_line.py")
    assert res.attempts[0].result.crashes == 0
    assert res.best_lap < l2_time                                      # measured 49.95 vs 52.5


@pytest.mark.parametrize("track", ["oval", "sprint", "speedway", "random:3"])
def test_l2b_clean_on_other_tracks(repo, track):
    r = _best(repo, "l2b_racing_line.py", track).attempts[0].result
    assert r.completed and r.crashes == 0


def test_l3_learner_best_of_4_beats_l2_without_dnf(repo, l2_time):
    res = _best(repo, "l3_learner.py", attempts=4)
    times = [a.result.race_time for a in res.attempts]
    assert all(a.result.completed for a in res.attempts), times
    assert res.best_lap < l2_time, (times, l2_time)                    # measured 52.5 -> 48.3
    assert min(times) <= 0.97 * times[0]                               # a real learning curve (>= 3 %)
    assert sum(a.result.crashes for a in res.attempts) == 0


def test_l3_learner_locks_in_for_qualifying(repo):
    res = run_session(str(repo / EX / "l3_learner.py"),
                      SessionConfig(track="oval", attempts=4, phases=["practice"] * 3 + ["qualifying"]))
    practice = [a.result.race_time for a in res.attempts[:3]]
    assert res.attempts[3].result.race_time == min(practice)          # deterministic sim: best plan replayed
    assert any("Locked in" in m["text"] for m in res.attempts[3].messages)


def test_l3_learner_on_an_unseen_random_track(repo):
    res = _best(repo, "l3_learner.py", "random:11", attempts=3)
    assert all(a.result.completed and a.result.crashes == 0 for a in res.attempts)
    assert res.best_lap < res.attempts[0].result.race_time


def test_l0_dummy_never_finishes(repo):
    r = _best(repo, "l0_dummy.py", max_ticks_per_lap=100).attempts[0].result
    assert not r.completed


# ------------------------------------------------------------------ `race.py check` (40-tick smoke test + reflect)
@pytest.mark.parametrize("name", LADDER + ["../template.py"])
def test_race_py_check_passes(repo, name, capsys):
    """What a student runs first: the contract check must be green for every example and the template.
    (It calls reflect() after an UNFINISHED 40-tick lap - L3 used to crash right there.)"""
    from agentic_gp import cli
    code = cli.main(["check", str(repo / EX / name)])
    out = capsys.readouterr().out
    assert code == 0 and "PASSED" in out, out


def _l3_after_plan(track: str = "oval"):
    import importlib.util

    from agentic_gp.sim.track import make_track
    spec = importlib.util.spec_from_file_location("_l3_unit", EX + "/l3_learner.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    drv = mod.Driver()
    trk = run_session(drv, SessionConfig(track=track, max_ticks_per_lap=2))   # plan() with the real track dict
    assert trk.plan_error is None
    return drv


def _report(attempt: int, time=None, crashes: int = 0, next_phase="practice", ended_by="max_ticks") -> dict:
    return {"attempt": attempt, "completed": time is not None, "race_time": time, "crashes": crashes,
            "crash_log": [], "segments": [{"segment": 0, "time": 1.0}], "next_phase": next_phase,
            "ended_by": None if time is not None else ended_by}


def test_l3_reflect_survives_unfinished_laps_without_a_crash():
    drv = _l3_after_plan()
    drv.history.clear()
    drv.reflect(_report(1))                                       # no best lap yet, no crash, no time
    assert "did not finish" in drv.say_next and drv.best is None
    drv.reflect(_report(2, time=30.0))                            # a best lap
    knobs_best = drv.best["knobs"]
    tried, step = drv.experiment, drv.steps[drv.experiment]
    drv.reflect(_report(3))                                       # DNF after a best lap: revert + learn
    assert "did not finish" in drv.say_next and "reverted to the best (30.00 s)" in drv.say_next
    assert drv.steps[tried] == pytest.approx(step * 0.5)          # that experiment hurt: smaller step
    assert drv.best["knobs"] == knobs_best and len(drv.history) == 3
    assert all("crash_segments" in h for h in drv.history)


@pytest.mark.parametrize("track", ["random:13", "random:41", "random:91", "random:139", "random:145"])
def test_template_finishes_the_former_crash_loop_tracks(repo, track):
    """Hairpins where the gate hides behind the inside wall: the line-of-sight rule gets the template round."""
    r = _best(repo, "../template.py", track).attempts[0].result
    assert r.completed and r.crashes <= 2, (track, r.ended_by, r.crashes)
