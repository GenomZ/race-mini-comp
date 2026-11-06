import json
import textwrap
import threading
import time

import pytest

from agentic_gp import llm
from agentic_gp.driver import REPORT_FIELDS
from agentic_gp.race import SessionConfig, iter_session, run_session
from agentic_gp.sim import PhysicsConfig
from agentic_gp.sim.track import PUBLIC_TRACKS

PLANNER = """
from agentic_gp import driverkit as dk

NAME = "Planner"
AUTHOR = "tests"
COLOR = "#2a9d8f"

class Driver:
    PARAMS = {"aggression": 1.0}

    def plan(self, track):
        self.follower = dk.PathFollower.from_track(track, aggression=self.PARAMS["aggression"])

    def drive(self, obs):
        throttle, steering = self.follower.control(obs)
        return {"throttle": throttle, "steering": steering, "say": "lights out" if obs["tick"] == 0 else None,
                "debug": {"target": self.follower.debug.get("target")}}
"""


@pytest.fixture(autouse=True)
def mock_llm(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("LLM_QUIET", "1")
    llm.set_mock_responder(None)


@pytest.fixture
def planner(tmp_path):
    p = tmp_path / "planner.py"
    p.write_text(textwrap.dedent(PLANNER), encoding="utf-8")
    return p


class Recorder:
    """Class-style driver used as an instance (so the test can look inside)."""

    NAME = "Recorder"

    def __init__(self, throttle=0.6, steering=0.0):
        self.throttle, self.steering = throttle, steering
        self.reports, self.phases, self.planned = [], [], 0

    def plan(self, track):
        self.planned += 1
        self.track = track

    def drive(self, obs):
        self.phases.append(obs["phase"])
        return self.throttle, self.steering

    def reflect(self, report):
        self.reports.append(report)


# ------------------------------------------------------------------ real driving
@pytest.mark.parametrize("track", list(PUBLIC_TRACKS) + ["random:3"])
def test_planner_finishes_clean_laps_everywhere(planner, track):
    res = run_session(planner, SessionConfig(track=track, attempts=1))
    att = res.attempts[0]
    assert att.result.completed and att.result.crashes == 0, (track, att.status, att.summary_text)
    assert res.best_lap is not None and res.best_attempt == 0 and res.status == ""
    assert res.driver["name"] == "Planner" and res.driver["color"] == "#2a9d8f"
    if track == "briefing_demo":           # the planner ignores the briefing: yellow-flag penalties
        assert att.zone_violations and att.zone_violations[0]["limit"] == 15.0
        assert att.result.penalty_time == pytest.approx(len(att.zone_violations))
    else:
        assert att.zone_violations == []


def test_report_contents_and_reflect_dict(planner):
    rec = Recorder(throttle=1.0, steering=0.35)           # crashes into walls a lot
    res = run_session(rec, SessionConfig(track="oval", attempts=3, max_ticks_per_lap=60))
    assert rec.planned == 1 and len(rec.reports) == 3     # same instance for every attempt (memory!)
    for i, rep in enumerate(rec.reports):
        assert set(rep) == set(REPORT_FIELDS)
        json.dumps(rep, allow_nan=False)
        assert rep["attempt"] == i + 1 and rep["phase"] == "session"
        assert rep["next_phase"] == ("session" if i < 2 else None)
        assert rep["crashes"] > 0 and len(rep["crash_log"]) == rep["crashes"]
        assert rep["summary_text"].startswith(f"Attempt {i + 1}")
        assert len(rep["telemetry"]) == rep["ticks"] + 1       # frame 0 = the start
    crash = rec.reports[0]["crash_log"][0]
    assert set(crash) >= {"tick", "t", "lap", "segment", "progress", "position", "speed_before", "throttle",
                          "steering", "rays", "corner"}
    assert crash["speed_before"] > 0 and crash["throttle"] == 1.0 and "distance_to_wall_front" in crash["rays"]
    seg = rec.reports[0]["segments"]
    att = res.attempts[0]
    assert sum(s["ticks"] for s in seg) == att.result.ticks
    assert sum(s["time"] for s in seg) == pytest.approx(att.frames[-1]["t"], abs=0.05)
    assert sum(s["crashes"] for s in seg) == att.result.crashes
    assert att.status == "DNF (max ticks)"


def test_schedule_injects_phases_and_laps():
    rec = Recorder(throttle=0.0)
    cfg = SessionConfig(track="oval", attempts=3, phases=["practice", "qualifying", "race"], attempt_laps=[1, 1, 2],
                        max_ticks_per_lap=4)
    res = run_session(rec, cfg)
    assert [a.phase for a in res.attempts] == ["practice", "qualifying", "race"]
    assert [a.result.laps for a in res.attempts] == [1, 1, 2]
    assert [a.result.ticks for a in res.attempts] == [4, 4, 8]
    assert rec.phases == ["practice"] * 4 + ["qualifying"] * 4 + ["race"] * 8
    assert [r["next_phase"] for r in rec.reports] == ["qualifying", "race", None]
    assert res.report_for("race") is res.attempts[2]
    with pytest.raises(ValueError):
        SessionConfig(phases=["warmup"]).schedule()


def test_iter_session_yields_live_events(planner):
    events = list(iter_session(planner, SessionConfig(track="oval", attempts=1)))
    types = [e.type for e in events]
    assert types[0] == "plan" and types[1] == "attempt_start" and types[-2:] == ["attempt_end", "session_end"]
    ticks = [e.data for e in events if e.type == "tick"]
    assert len(ticks) == events[-1].data.attempts[0].result.ticks
    first = ticks[0]
    assert first["obs"]["tick"] == 0 and first["tick"] == 1          # obs = what the driver saw before the tick
    assert first["action"].say == "lights out" and first["say"] == "lights out"
    assert first["debug"]["target"] is not None and first["think_seconds"] >= 0
    report = events[-2].data
    assert report.messages == [{"tick": 0, "t": 0.0, "text": "lights out"}]
    assert "obs" in report.frames[1] and report.frames[1]["obs"]["tick"] == 0


# ------------------------------------------------------------------ errors, stalls, stop, realtime
class Clumsy:
    NAME = "Clumsy"

    def plan(self, track):
        raise KeyError("centreline")

    def drive(self, obs):
        if obs["tick"] % 2:
            raise ZeroDivisionError("oops")
        return 1.0, 0.0

    def reflect(self, report):
        raise RuntimeError("no memory")


def test_driver_exceptions_coast_and_are_recorded():
    res = run_session(Clumsy(), SessionConfig(track="oval", attempts=2, max_ticks_per_lap=10))
    # the error names the line in the driver's code and suggests the right key
    assert res.plan_error.startswith("KeyError: 'centreline' (test_session.py:")
    assert res.plan_error.endswith("- did you mean 'centerline'?")
    att = res.attempts[0]
    line = Clumsy.drive.__code__.co_firstlineno + 2
    assert att.driver_error_count == 5 and att.driver_errors[0] == {
        "tick": 1, "phase": "session", "error": f"ZeroDivisionError: oops (test_session.py:{line})"}
    assert "Traceback" in res.first_error_traceback and "centreline" in res.first_error_traceback
    erred = [f for f in att.frames[1:] if f["driver_error"]]
    assert len(erred) == 5 and all(f["throttle"] == 0.0 for f in erred)
    assert all(f["throttle"] == 1.0 for f in att.frames[1:] if not f["driver_error"])
    assert "RuntimeError: no memory" in att.reflect_error
    assert len(res.attempts) == 2                        # the session went on
    assert any(e["phase"] == "plan" for e in res.driver_errors)
    assert res.to_replay().validate() == []
    kinds = {e.kind for e in res.to_replay().cars[0].events}
    assert "driver_error" in kinds


def test_in_process_stalls_end_the_attempt():
    class Slow(Recorder):
        def drive(self, obs):
            time.sleep(0.03)
            return 1.0, 0.0

    res = run_session(Slow(), SessionConfig(track="oval", attempts=1, drive_timeout_s=0.01,
                                            max_consecutive_stalls=3))
    att = res.attempts[0]
    assert att.stalls == 3 and att.result.ended_by == "stalled" and att.status == "DNF (stalled)"
    assert all(f["throttle"] == 0.0 and f["stalled"] for f in att.frames[1:])


def test_stop_event_ends_the_session_gracefully():
    stop = threading.Event()

    def on_event(ev):
        if ev.type == "tick" and ev.data["tick"] >= 5:
            stop.set()

    res = run_session(Recorder(), SessionConfig(track="oval", attempts=3), on_event=on_event, stop_event=stop)
    assert res.stopped and res.status == "DNF (stopped)"
    assert len(res.attempts) == 1 and res.attempts[0].result.ended_by == "stopped"
    assert res.attempts[0].result.ticks == 5


def test_realtime_mode_lets_the_world_move_while_thinking():
    class Thinker(Recorder):
        def drive(self, obs):
            if obs["tick"] in (2, 5):
                time.sleep(0.55)                      # > one tick of 0.5 s -> 1 late tick
            return 1.0, 0.0

    res = run_session(Thinker(), SessionConfig(track="oval", attempts=1, realtime=True, max_ticks_per_lap=12))
    att = res.attempts[0]
    late = [f for f in att.frames if f.get("late")]
    applied = [f for f in att.frames if f.get("late_ticks")]
    assert len(late) == 2 and [f["late_ticks"] for f in applied] == [1, 1]
    assert att.late_ticks == 2 and att.result.ticks == 12
    assert late[0]["tick"] == 3 and late[0]["throttle"] == 1.0          # previous command kept running
    assert [e.value for e in res.to_replay().cars[0].events if e.kind == "late"] == [1.0, 1.0]


# ------------------------------------------------------------------ LLM policy
class LLMHungry(Recorder):
    NAME = "Hungry"

    def plan(self, track):
        self.plan_log = []
        for _ in range(3):
            try:
                llm.chat("plan the lap")
                self.plan_log.append("ok")
            except llm.LLMError as exc:
                self.plan_log.append(type(exc).__name__)

    def drive(self, obs):
        if obs["tick"] == 0:
            llm.chat("steer?")                       # not allowed in official events
        return 0.0, 0.0

    def reflect(self, report):
        self.reports.append(report)
        try:
            llm.chat("what went wrong?")
            self.phases.append(("reflect ok", report["phase"]))
        except llm.LLMNotAllowed:
            self.phases.append(("reflect refused", report["phase"]))


def test_official_llm_policy_is_enforced_by_the_runner():
    d = LLMHungry()
    res = run_session(d, SessionConfig(track="oval", attempts=2, phases=["practice", "qualifying"],
                                       llm_policy="official", max_ticks_per_lap=3))
    assert d.plan_log == ["ok", "ok", "LLMBudgetExceeded"]                   # <= 2 calls in plan
    assert "LLMNotAllowed" in res.attempts[0].driver_errors[0]["error"]     # never in drive()
    assert ("reflect ok", "practice") in d.phases and ("reflect refused", "qualifying") in d.phases
    assert res.llm["calls"] == 3 and res.llm["mode"] == "official" and res.llm["calls_left"] == 5
    assert d.reports[0]["llm_usage"]["calls"] == 2


def test_open_policy_allows_llm_in_drive():
    d = LLMHungry()
    res = run_session(d, SessionConfig(track="oval", attempts=1, max_ticks_per_lap=2))
    assert d.plan_log == ["ok", "ok", "ok"] and res.attempts[0].driver_errors == []
    assert res.llm["calls"] == 5 and res.llm["calls_left"] is None


# ------------------------------------------------------------------ results + replays
def test_result_dict_and_replays(planner):
    res = run_session(planner, SessionConfig(track="oval", attempts=2, physics=PhysicsConfig(crash_rule="classic")))
    d = res.to_result_dict()
    json.dumps(d, allow_nan=False)
    assert d["name"] == "Planner" and d["track"] == "oval" and d["best_lap"] == res.best_lap
    assert d["completed_attempts"] == 2 and d["max_waypoints_passed"] == 8 and d["config"]["crash_rule"] == "classic"
    assert len(d["attempts"]) == 2 and d["attempts"][0]["phase"] == "session"
    json.dumps(res.to_dict(), allow_nan=False)
    for which, n in (("best", 1), ("last", 1), ("all", 2)):
        rep = res.to_replay(which)
        assert rep.validate() == [] and len(rep.cars) == n and rep.kind == "session"
    names = [c.name for c in res.to_replay("all").cars]
    assert sorted(names) == ["Planner #1", "Planner #2"]
    stopped = threading.Event()
    stopped.set()                                       # stopped before the first attempt
    empty = run_session(planner, SessionConfig(track="oval"), stop_event=stopped)
    assert empty.attempts == [] and empty.to_replay().validate() == []


# ------------------------------------------------------------------ review fixes
def test_sys_exit_in_driver_code_is_only_an_error():
    import sys

    class Quitter(Recorder):
        def plan(self, track):
            sys.exit("no plan")

        def drive(self, obs):
            if obs["tick"] == 2:
                sys.exit("bye")
            return 1.0, 0.0

    res = run_session(Quitter(), SessionConfig(track="oval", attempts=1, max_ticks_per_lap=6))
    assert res.plan_error.startswith("SystemExit: no plan")
    assert res.attempts[0].driver_error_count == 1 and "SystemExit: bye" in res.attempts[0].driver_errors[0]["error"]
    assert res.attempts[0].result.ticks == 6                  # the session went on


def test_official_rules_also_apply_to_threads_the_driver_starts():
    class Sneaky(Recorder):
        NAME = "Sneaky"

        def plan(self, track):
            self.log = [self._in_thread()]                    # inside plan(): counts against plan's 2 calls
            self.log.append(self._nested_open())

        def drive(self, obs):
            if obs["tick"] == 0:
                self.log += [self._in_thread(), self._nested_open()]
            return 0.0, 0.0

        @staticmethod
        def _in_thread():
            out = []

            def call():
                try:
                    llm.chat("from a thread")
                    out.append("ok")
                except llm.LLMError as exc:
                    out.append(type(exc).__name__)
            t = threading.Thread(target=call)
            t.start()
            t.join()
            return out[0]

        @staticmethod
        def _nested_open():
            try:
                with llm.llm_policy(mode="open"):
                    llm.chat("nested")
                return "ok"
            except llm.LLMError as exc:
                return type(exc).__name__

    d = Sneaky()
    res = run_session(d, SessionConfig(track="oval", attempts=1, phases=["race"], llm_policy="official",
                                       max_ticks_per_lap=3))
    assert d.log == ["ok", "ok", "LLMNotAllowed", "LLMNotAllowed"]
    assert res.llm["calls"] == 2                              # the thread call was counted too
    assert llm.llm_allowed() and llm.current_llm_policy()["mode"] == "open"     # rules end with the session


def test_reflect_gets_the_llm_deadline():
    class Timed(Recorder):
        def reflect(self, report):
            self.left = llm.current_llm_policy()["seconds_left"]

    d = Timed()
    run_session(d, SessionConfig(track="oval", attempts=1, max_ticks_per_lap=2, reflect_timeout_s=30))
    assert 26.0 < d.left <= 27.0                              # 30 s limit - 3 s margin
