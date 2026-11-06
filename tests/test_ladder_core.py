"""The two core example drivers behave as the lesson promises (no LLM involved).

l0_dummy: never finishes. l2_planner: clean (0 crashes) on every public track and on random tracks,
with no per-track tuning, and deterministic.
"""

from __future__ import annotations

import pytest

from agentic_gp import SessionConfig, run_session
from agentic_gp.race import check_driver_source
from agentic_gp.sim.track import PUBLIC_TRACKS

EXAMPLES = "drivers/examples"


def _run(repo, name: str, track: str, **kw):
    return run_session(str(repo / EXAMPLES / name), SessionConfig(track=track, **kw))


@pytest.mark.parametrize("name", ["l0_dummy.py", "l2_planner.py"])
def test_examples_pass_the_static_check(repo, name):
    assert [i for i in check_driver_source(repo / EXAMPLES / name) if i.level == "error"] == []


def test_l0_dummy_does_not_finish(repo):
    res = _run(repo, "l0_dummy.py", "grand_prix", max_ticks_per_lap=120)
    r = res.attempts[0].result
    assert not r.completed and r.crashes > 0 and r.ended_by == "max_ticks"


@pytest.mark.parametrize("track", [t for t in PUBLIC_TRACKS] + [f"random:{s}" for s in range(1, 6)])
def test_l2_planner_clean_everywhere(repo, track):
    r = _run(repo, "l2_planner.py", track).attempts[0].result
    assert r.completed, (track, r.ended_by)
    assert r.crashes == 0, (track, r.crashes)


def test_l2_planner_lap_time_and_determinism(repo):
    a = _run(repo, "l2_planner.py", "grand_prix").attempts[0]
    b = _run(repo, "l2_planner.py", "grand_prix").attempts[0]
    assert 45.0 < a.result.race_time < 58.0          # measured 52.5 s (dense centerline, aggression 1.0)
    assert a.result.race_time == b.result.race_time
    assert [f["x"] for f in a.frames] == [f["x"] for f in b.frames]
    assert any(m["text"].startswith("Plan:") for m in a.messages)   # radio works


def test_l2_planner_params_change_the_plan(repo):
    slow = run_session(str(repo / EXAMPLES / "l2_planner.py"), SessionConfig(track="oval"))
    from agentic_gp import load_driver
    drv = load_driver(repo / EXAMPLES / "l2_planner.py", params={"aggression": "0.8"})
    assert drv.params["aggression"] == 0.8
    slower = run_session(drv, SessionConfig(track="oval"))
    assert slower.best_lap > slow.best_lap
