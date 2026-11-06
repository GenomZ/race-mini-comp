"""Sessions: one driver, one track, a few attempts - plan -> (drive ... -> reflect) x attempts.

    from agentic_gp.race import run_session, SessionConfig
    result = run_session("drivers/examples/l2_planner.py", SessionConfig(track="sprint", attempts=3))
    result.best_lap, result.attempts[-1].summary_text
    result.to_replay("best").to_json("races/my_lap.json")

    for ev in iter_session(driver, SessionConfig(attempts=2)):      # live UIs: one event per tick
        if ev.type == "tick": draw(ev.data)                          # frame + action + say + obs

The runner owns the simulation; the driver only answers "what do I do now?". The SAME driver
instance drives every attempt of a session, so whatever it learns in reflect() carries over.
Exceptions in driver code never stop the session (sys.exit() included): drive() errors mean "coast
this tick" (and are recorded), plan()/reflect() errors are recorded in the result. Error texts name
the line in the driver file and suggest the right key for a misspelled obs/track/report key:
"KeyError: 'angle_to_waypoint' (my_driver.py:12) - did you mean 'angle_to_next_waypoint'?".
`SessionResult.first_error_traceback` keeps the full traceback of the first error.

Timing: a drive() answer slower than `drive_timeout_s` is a STALL (the car coasts, the late answer
is thrown away); 10 stalls in a row or 30 in total end the attempt as DNF "stalled". Realtime mode
(a teaching demo of LLM latency) keeps the world moving while the driver thinks: a 1.3 s answer
costs 2 ticks driven with the previous command.
"""

from __future__ import annotations

import difflib
import math
import os
import re
import sys
import sysconfig
import threading
import time
import traceback
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterator

from .. import driverkit, llm
from ..driver import COAST, OBS_FIELDS, PHASES, REPORT_FIELDS, TRACK_FIELDS, Action
from ..sim.config import DEFAULT_CONFIG, PhysicsConfig
from ..sim.env import AttemptResult, RaceEnv
from ..sim.track import Track, make_track
from .loader import DriverDied, DriverError, DriverLoadTimeout, DriverTimeout, load_driver
from .replay import Replay, build_replay, placeholder_car, replay_car_from_trace, utc_now

MAX_ERRORS_KEPT = 50          # driver_errors stored per attempt (the count keeps going)


@dataclass
class SessionConfig:
    track: str | Track = "grand_prix"
    attempts: int = 1
    laps: int = 1
    realtime: bool = False
    max_ticks_per_lap: int | None = None          # None = physics default (500)
    verbose: bool = False
    llm_policy: str = "open"                       # "open" (lessons) | "official" (grand prix rules)
    phases: list[str] | None = None                # phase of each attempt (default "session")
    attempt_laps: list[int] | None = None          # laps of each attempt (default `laps`)
    physics: PhysicsConfig | None = None           # e.g. PhysicsConfig(crash_rule="classic")
    drive_timeout_s: float | None = None           # None: 2 s for isolated drivers, no limit in-process
    plan_timeout_s: float = 120.0
    reflect_timeout_s: float = 120.0
    driver_budget_s: float | None = None           # total thinking time allowed on this track (GP: 300)
    max_consecutive_stalls: int = 10
    max_total_stalls: int = 30
    reflect_after_last: bool = True                # call reflect() after the final attempt too
    keep_obs: bool = True                          # store each tick's obs in the report frames
    label: str | None = None                       # LLM log label (default: driver name)

    def schedule(self) -> list[tuple[str, int]]:
        """[(phase, laps), ...] for every attempt."""
        n = max(1, int(self.attempts))
        phases = list(self.phases or [])
        laps = list(self.attempt_laps or [])
        out = []
        for i in range(n):
            phase = phases[i] if i < len(phases) else (phases[-1] if phases else "session")
            if phase not in PHASES:
                raise ValueError(f"phase {phase!r} is not one of {PHASES}")
            out.append((phase, int(laps[i] if i < len(laps) else self.laps)))
        return out

    def physics_config(self) -> PhysicsConfig:
        cfg = self.physics or DEFAULT_CONFIG
        if self.max_ticks_per_lap:
            cfg = replace(cfg, max_ticks=int(self.max_ticks_per_lap))
        return cfg

    def to_dict(self) -> dict:
        track = self.track.key if isinstance(self.track, Track) else str(self.track)
        return {"track": track, "attempts": self.attempts, "laps": self.laps, "realtime": self.realtime,
                "max_ticks_per_lap": self.max_ticks_per_lap, "llm_policy": self.llm_policy,
                "schedule": [list(s) for s in self.schedule()], "drive_timeout_s": self.drive_timeout_s,
                "driver_budget_s": self.driver_budget_s,
                "crash_rule": self.physics_config().crash_rule}


@dataclass
class SessionEvent:
    type: str            # "plan" | "attempt_start" | "tick" | "attempt_end" | "session_end"
    data: Any            # plan: dict; attempt_start: dict; tick: TickFrame dict; attempt_end: AttemptReport;
                         # session_end: SessionResult
    attempt: int = 0
    phase: str = ""


# A TickFrame is a plain dict: the env frame of the tick just driven (tick, t, x, y, heading_deg, speed,
# throttle, steering, lap, segment, progress, race_progress, offset_from_center, events, crashed, ...)
# + "action" (Action), "say", "think_seconds", "obs" (what the driver SAW before this tick),
# "driver_error", "stalled", "late_ticks", "debug", "attempt", "phase".
TickFrame = dict

TELEMETRY_KEYS = ("tick", "t", "lap", "segment", "progress", "x", "y", "speed", "throttle", "steering",
                  "offset_from_center")


@dataclass
class AttemptReport:
    result: AttemptResult
    frames: list[dict]
    crash_log: list[dict]
    segments: list[dict]
    messages: list[dict]
    driver_errors: list[dict]
    think_seconds_total: float = 0.0
    think_seconds_max: float = 0.0
    phase: str = "session"
    next_phase: str | None = None
    stalls: int = 0
    late_ticks: int = 0
    zone_violations: list[dict] = field(default_factory=list)
    corners: list[dict] = field(default_factory=list)
    llm_usage: dict = field(default_factory=dict)
    best_lap_so_far: float | None = None
    summary_text: str = ""
    status: str = ""                     # "" when finished, else "DNF (...)"
    crash_points: list = field(default_factory=list)
    driver_error_count: int = 0
    reflect_error: str | None = None
    reflect_seconds: float = 0.0

    @property
    def attempt(self) -> int:
        return self.result.attempt

    @property
    def telemetry(self) -> list[dict]:
        out = []
        for f in self.frames:
            row = {k: f[k] for k in TELEMETRY_KEYS}
            for k, nd in (("t", 2), ("progress", 4), ("x", 2), ("y", 2), ("speed", 2), ("throttle", 3),
                          ("steering", 3), ("offset_from_center", 2)):
                row[k] = round(float(row[k]), nd)
            out.append(row)
        return out

    def to_reflect_dict(self) -> dict:
        """The plain, JSON-serialisable dict reflect() receives (keys = driver.REPORT_FIELDS)."""
        r = self.result
        return {
            "attempt": r.attempt, "phase": self.phase, "next_phase": self.next_phase,
            "completed": r.completed, "race_time": r.race_time, "lap_times": list(r.lap_times),
            "best_lap": r.best_lap, "best_lap_so_far": self.best_lap_so_far, "sim_time": r.sim_time,
            "penalty_time": r.penalty_time, "crashes": r.crashes, "zone_violations": [dict(z) for z in
                                                                                      self.zone_violations],
            "waypoints_passed": r.waypoints_passed, "laps": r.laps, "laps_completed": r.laps_completed,
            "ticks": r.ticks, "ended_by": r.ended_by, "crash_log": [dict(c) for c in self.crash_log],
            "segments": [dict(s) for s in self.segments], "corners": [dict(c) for c in self.corners],
            "telemetry": self.telemetry, "stalls": self.stalls,
            "driver_errors": [dict(e) for e in self.driver_errors], "llm_usage": dict(self.llm_usage),
            "summary_text": self.summary_text,
        }

    def to_dict(self, frames: bool = False) -> dict:
        """JSON-friendly report (+ think times, messages, status; frames only if asked)."""
        d = self.to_reflect_dict()
        d.update({"messages": list(self.messages), "think_seconds_total": round(self.think_seconds_total, 4),
                  "think_seconds_max": round(self.think_seconds_max, 4), "late_ticks": self.late_ticks,
                  "status": self.status, "reflect_error": self.reflect_error,
                  "reflect_seconds": round(self.reflect_seconds, 3), "driver_error_count": self.driver_error_count})
        if frames:
            d["frames"] = [{k: v for k, v in f.items() if k != "obs"} for f in self.frames]
        return d


@dataclass
class SessionResult:
    driver: dict                     # name, author, color, kind, model, source (+ motto, anatomy, params, house)
    config: dict
    attempts: list[AttemptReport]
    best_lap: float | None
    best_attempt: int | None         # index into attempts (fastest finished attempt)
    wall_seconds: float
    think_seconds_mean: float
    think_seconds_max: float
    driver_errors: list[dict]        # plan/reflect errors + the first drive errors of every attempt
    llm: dict                        # calls, errors, prompt_tokens, completion_tokens, seconds, ...
    plan_error: str | None = None
    plan_seconds: float = 0.0
    status: str = ""                 # "" = ok; else e.g. "DNF (timeout)", "DNF (driver crashed: ...)"
    stopped: bool = False
    track_key: str = ""
    track_name: str = ""
    timestamp: str = field(default_factory=utc_now)
    track: Track | None = field(default=None, repr=False)
    first_error_traceback: str | None = field(default=None, repr=False)   # of the first driver error

    def report_for(self, phase: str) -> AttemptReport | None:
        """The last attempt of a phase (e.g. "qualifying", "race")."""
        return next((a for a in reversed(self.attempts) if a.phase == phase), None)

    def to_result_dict(self) -> dict:
        """What results/<name>.json stores (read by the leaderboard and the Learning Curves page)."""
        atts = []
        for a in self.attempts:
            d = a.result.to_dict()
            d.update({"phase": a.phase, "stalls": a.stalls, "driver_errors": a.driver_error_count,
                      "status": a.status})
            atts.append(d)
        return {
            "format": "agentic-gp-result", "version": 1,
            **{k: self.driver.get(k) for k in ("name", "author", "color", "kind", "model", "source", "motto")},
            "params": self.driver.get("params", {}),
            "track": self.track_key, "track_name": self.track_name, "timestamp": self.timestamp,
            "config": self.config, "attempts": atts, "best_lap": self.best_lap,
            "best_attempt": None if self.best_attempt is None else self.attempts[self.best_attempt].attempt,
            "total_crashes": sum(a.result.crashes for a in self.attempts),
            "completed_attempts": sum(1 for a in self.attempts if a.result.completed),
            "max_waypoints_passed": max((a.result.waypoints_passed for a in self.attempts), default=0),
            "wall_seconds": round(self.wall_seconds, 2), "think_seconds_mean": round(self.think_seconds_mean, 5),
            "think_seconds_max": round(self.think_seconds_max, 4), "llm": dict(self.llm),
            "llm_calls": int(self.llm.get("calls", 0)), "plan_error": self.plan_error,
            "driver_errors": len(self.driver_errors), "status": self.status,
        }

    def to_dict(self) -> dict:
        d = self.to_result_dict()
        d["reports"] = [a.to_dict() for a in self.attempts]
        return d

    def selected(self, which: str = "best") -> list[AttemptReport]:
        if not self.attempts:
            return []
        if which == "all":
            return list(self.attempts)
        if which == "last":
            return [self.attempts[-1]]
        if which != "best":
            raise ValueError("which must be 'best', 'last' or 'all'")
        if self.best_attempt is not None:
            return [self.attempts[self.best_attempt]]
        return [max(self.attempts, key=lambda a: a.frames[-1]["race_progress"] if a.frames else 0.0)]

    def to_replay(self, which: str = "best", title: str | None = None) -> Replay:
        """A replay with one car per selected attempt ("all" names them "<name> #<attempt>")."""
        if self.track is None:
            raise ValueError("this SessionResult has no track (was it loaded from JSON?)")
        chosen = self.selected(which)
        cars = []
        for a in chosen:
            meta = dict(self.driver)
            if which == "all":
                meta["name"] = f"{self.driver['name']} #{a.attempt}"
                meta["id"] = f"{_id(self.driver['name'])}-{a.attempt}"
            extra = {"llm_calls": int(a.llm_usage.get("calls", 0)),
                     "llm_tokens": int(a.llm_usage.get("prompt_tokens", 0) + a.llm_usage.get("completion_tokens", 0))}
            if a.status:
                extra["status"] = a.status
            cars.append(replay_car_from_trace(meta, a, extra))
        if not cars:                             # the driver never got to drive (load / plan crash)
            cars.append(placeholder_car(self.driver, self.track, self.status or "DNF"))
        laps = max((a.result.laps for a in chosen), default=1)
        cfg = self.config
        title = title or f"{self.driver['name']} on {self.track_name}"
        return build_replay(self.track, cars, title=title, kind="session", laps=laps,
                            tick_seconds=float(cfg.get("tick_seconds", 0.5)),
                            max_speed=float(cfg.get("max_speed", 40.0)))


def _id(name: str) -> str:
    from .replay import slug
    return slug(name)


# ============================================================================ the session loop
def run_session(driver: Any, config: SessionConfig | None = None, on_event: Callable[[SessionEvent], None] | None = None,
                stop_event: threading.Event | None = None) -> SessionResult:
    """Run a whole session and return its result (`on_event` sees every SessionEvent)."""
    result = None
    for ev in iter_session(driver, config, stop_event):
        if on_event is not None:
            on_event(ev)
        if ev.type == "session_end":
            result = ev.data
    return result


def iter_session(driver: Any, config: SessionConfig | None = None,
                 stop_event: threading.Event | None = None) -> Iterator[SessionEvent]:
    """Generator version of run_session for live UIs (see SessionEvent for what it yields).

    `driver` is anything load_driver accepts (a path is loaded in-process; pass an IsolatedDriver for
    isolation). A driver loaded here is closed at the end; one passed in stays open for the caller.
    """
    config = config or SessionConfig()
    owns = not hasattr(driver, "begin_session")
    drv = load_driver(driver) if owns else driver
    runner = _SessionRunner(drv, config, stop_event)
    try:
        yield from runner.run()
    finally:
        runner.env.close()
        end = getattr(drv, "end_session", None)
        if callable(end):
            end()                                # official LLM rules end with the session
        if owns:
            drv.close()


class _SessionRunner:
    """The state of one session (kept in an object so the loop stays readable)."""

    def __init__(self, drv: Any, config: SessionConfig, stop_event: threading.Event | None) -> None:
        self.drv = drv
        self.config = config
        self.stop_event = stop_event
        self.schedule = config.schedule()
        self.cfg = config.physics_config()
        self.track = make_track(config.track)
        self.env = RaceEnv(self.track, self.cfg, laps=self.schedule[0][1])
        self.isolated = bool(getattr(drv, "isolated", False))
        self.drive_timeout = config.drive_timeout_s if config.drive_timeout_s is not None else (
            2.0 if self.isolated else None)
        self.think_time = 0.0               # total driver time (plan + drive + reflect) for the budget
        self.drive_times: list[float] = []
        self.fatal: str | None = None       # set when the driver can no longer race
        self.stopped = False
        self.session_errors: list[dict] = []
        self.reports: list[AttemptReport] = []
        self._map: dict | None = None
        self.first_traceback: str | None = None

    # -- helpers
    def _timeout(self, limit: float | None) -> float | None:
        budget = self.config.driver_budget_s
        if budget is None:
            return limit
        left = max(0.0, budget - self.think_time)
        return left if limit is None else min(limit, left)

    def _budget_gone(self) -> bool:
        b = self.config.driver_budget_s
        return b is not None and self.think_time >= b

    def _stopping(self) -> bool:
        return self.stop_event is not None and self.stop_event.is_set()

    def _error(self, exc: BaseException, method: str) -> str:
        """Error text with the driver-file line and a key hint; keeps the first full traceback."""
        text, tb = describe_driver_error(exc, getattr(self.drv, "source_path", None), method)
        if self.first_traceback is None and tb:
            self.first_traceback = tb
        return text

    def _call(self, method: str, arg: Any, limit: float | None) -> tuple[Any, str | None, float, bool]:
        """Call plan/reflect: (result, error, seconds, timed_out). Fatal problems set self.fatal."""
        t0 = time.perf_counter()
        result, error, timed_out = None, None, False
        try:
            result = getattr(self.drv, method)(arg, timeout=self._timeout(limit))
        except DriverTimeout as exc:
            error, timed_out = f"timeout: {exc}", True
            self.fatal = "DNF (timeout)"
            _kill(self.drv)                      # a child stuck in plan/reflect cannot race any more
        except DriverDied as exc:
            if self._stopping():                 # the runner stopped the event (and the driver)
                self.stopped, error = True, "stopped"
            else:
                error = f"driver crashed: {exc}"
                self.fatal = f"DNF (driver crashed: {_short(exc)})"
        except DriverError as exc:               # raised inside an isolated driver: "Type: message"
            error = self._error(exc, method)
        except (Exception, SystemExit) as exc:   # noqa: BLE001 - driver code may raise anything
            error = self._error(exc, method)
        seconds = time.perf_counter() - t0
        self.think_time += seconds
        if not self.isolated and limit is not None and seconds > limit and error is None:
            error = f"timeout: {method}() took {seconds:.1f}s (limit {limit:g}s)"
        if self._budget_gone() and not self.fatal:
            self.fatal = "DNF (timeout)"
        return result, error, seconds, timed_out

    def _say(self, text: str) -> None:
        if self.config.verbose:
            print(text, flush=True)

    # -- the flow
    def run(self) -> Iterator[SessionEvent]:
        t_start = time.perf_counter()
        drv, config = self.drv, self.config
        if getattr(drv, "uses_llm", False) and not self.isolated:
            try:
                llm.warm_up()                    # in-process: pay the SDK import before plan() is timed
            except Exception:                    # noqa: BLE001 - the driver's own call will report it
                pass
        try:
            drv.begin_session(mode=config.llm_policy, label=config.label or drv.name)
        except (DriverDied, DriverTimeout) as exc:
            if self._stopping():
                self.stopped = True
            elif isinstance(exc, DriverLoadTimeout):     # still importing the LLM SDK (2 x load_timeout)
                self.fatal = "DNF (timeout: loading)"
                self.session_errors.append({"tick": None, "phase": "load", "error": str(exc)[:300]})
            elif isinstance(exc, DriverTimeout):
                self.fatal = "DNF (timeout)"
                self.session_errors.append({"tick": None, "phase": "begin_session", "error": str(exc)[:300]})
            else:
                self.fatal = f"DNF (driver crashed: {_short(exc)})"
        track_map = self.env.track_map()          # the driver's own copy (it may change it)
        plan_error, plan_seconds = None, 0.0
        if not self.fatal and not self.stopped:
            drv.phase = "session"
            _, plan_error, plan_seconds, _ = self._call("plan", track_map, config.plan_timeout_s)
            if plan_error:
                self.session_errors.append({"tick": None, "phase": "plan", "error": plan_error})
        self._say(f"[{drv.name}] plan: {plan_seconds:.2f}s" + (f" - ERROR {plan_error}" if plan_error else ""))
        yield SessionEvent("plan", {"track": track_map, "error": plan_error, "seconds": plan_seconds})

        for i, (phase, laps) in enumerate(self.schedule):
            if self.stop_event is not None and self.stop_event.is_set():
                self.stopped = True
            if self.fatal or self.stopped:
                break
            yield from self._attempt(i, phase, laps)

        result = self._result(t_start, plan_error, plan_seconds)
        yield SessionEvent("session_end", result)

    def _attempt(self, i: int, phase: str, laps: int) -> Iterator[SessionEvent]:
        drv, env, config = self.drv, self.env, self.config
        env.phase = phase
        drv.phase = phase
        obs = env.reset(laps)
        attempt_no = env.attempt_no
        yield SessionEvent("attempt_start", {"attempt": attempt_no, "phase": phase, "laps": laps,
                                             "track": self.track.key}, attempt_no, phase)
        stalls = consecutive = late_total = 0
        errors: list[dict] = []
        n_errors = 0
        think: list[float] = []
        messages: list[dict] = []
        last_say = None
        prev = COAST
        status = ""
        ended_by = None
        while not obs["attempt_over"]:
            if self.stop_event is not None and self.stop_event.is_set():
                env.stop()
                self.stopped, ended_by, status = True, "stopped", "DNF (stopped)"
                break
            if self._budget_gone():
                env.stop()
                self.fatal = status = "DNF (timeout)"
                ended_by = "timeout"
                break
            seen = obs
            action, error, stalled = COAST, None, False
            t0 = time.perf_counter()
            try:
                action = drv.drive(seen, timeout=self._timeout(self.drive_timeout))
            except DriverTimeout:
                stalled = True
            except DriverDied as exc:
                if self._stopping():
                    env.stop()
                    self.stopped, ended_by, status = True, "stopped", "DNF (stopped)"
                    break
                self.fatal = status = f"DNF (driver crashed: {_short(exc)})"
                error = f"driver crashed: {exc}"
            except DriverError as exc:
                error = self._error(exc, "drive")
            except (Exception, SystemExit) as exc:   # noqa: BLE001 - sys.exit() in drive() = an error
                error = self._error(exc, "drive")
            seconds = time.perf_counter() - t0
            self.think_time += seconds
            think.append(seconds)
            if not self.isolated and self.drive_timeout is not None and seconds > self.drive_timeout:
                stalled, action = True, COAST    # in-process: the late answer is discarded too
            if stalled:
                action = COAST
                stalls += 1
                consecutive += 1
            else:
                consecutive = 0
            if error:
                n_errors += 1
                if len(errors) < MAX_ERRORS_KEPT:
                    errors.append({"tick": seen["tick"], "phase": phase, "error": error})
            if self.fatal:
                env.stop()
                ended_by = "stopped"
                break

            # Realtime: the world kept moving while the driver was thinking.
            late = int(math.floor(seconds / self.cfg.tick_seconds)) if config.realtime else 0
            applied_late = 0
            for _ in range(late):
                if obs["attempt_over"]:
                    break
                obs = env.step(*prev.as_tuple())
                env.frames[-1].update({"late": True, "stalled": False, "driver_error": None})
                applied_late += 1
            late_total += applied_late
            if obs["attempt_over"]:
                break
            obs = env.step(action.throttle, action.steering)
            frame = env.frames[-1]
            frame.update({"say": action.say, "think_seconds": round(seconds, 5), "driver_error": error,
                          "stalled": stalled, "late_ticks": applied_late, "debug": action.debug})
            if config.keep_obs:
                frame["obs"] = seen
            if action.say and action.say != last_say:
                messages.append({"tick": int(seen["tick"]), "t": round(float(seen["race_time"]), 2),
                                 "text": action.say})
            last_say = action.say
            prev = action
            tick_frame = dict(frame, action=action, obs=seen, attempt=attempt_no, phase=phase)
            yield SessionEvent("tick", tick_frame, attempt_no, phase)
            too_many = consecutive >= config.max_consecutive_stalls or stalls >= config.max_total_stalls
            if too_many and not obs["attempt_over"]:
                env.stop()
                ended_by, status = "stalled", "DNF (stalled)"
                # A child that is STILL busy with an old call is hung: it cannot race on.
                wait_idle = getattr(drv, "wait_idle", None)
                if wait_idle is not None and not wait_idle(max(1.0, 2 * (self.drive_timeout or 1.0))):
                    _kill(drv)
                    self.fatal = "DNF (stalled)"
                break

        trace = env.attempt_traces[-1]
        result: AttemptResult = trace["result"]
        if ended_by and not result.completed:
            result.ended_by = ended_by
        if not result.completed and not status:
            if n_errors and n_errors >= max(1, result.ticks // 2):
                status = f"DNF (driver crashed: {errors[0]['error'][:60]})"
            else:
                status = {"max_ticks": "DNF (max ticks)", "stopped": "DNF (stopped)"}.get(result.ended_by,
                                                                                          f"DNF ({result.ended_by})")
        next_phase = self.schedule[i + 1][0] if i + 1 < len(self.schedule) else None
        report = self._report(trace, phase, next_phase, errors, n_errors, think, messages, stalls, late_total, status)
        self.reports.append(report)
        self.drive_times += think
        self._say(f"[{drv.name}] attempt {attempt_no} ({phase}): "
                  + (f"{result.race_time:.2f}s" if result.completed else status)
                  + f", {result.crashes} crash(es), think {report.think_seconds_max * 1000:.0f} ms max")
        yield SessionEvent("attempt_end", report, attempt_no, phase)

        last = i + 1 == len(self.schedule)
        if not self.fatal and not self.stopped and (not last or config.reflect_after_last):
            _, err, secs, _ = self._call("reflect", report.to_reflect_dict(), config.reflect_timeout_s)
            report.reflect_error, report.reflect_seconds = err, secs
            if err:
                self.session_errors.append({"tick": None, "phase": f"reflect {attempt_no}", "error": err})

    def _report(self, trace: dict, phase: str, next_phase: str | None, errors: list[dict], n_errors: int,
                think: list[float], messages: list[dict], stalls: int, late_total: int, status: str) -> AttemptReport:
        frames = trace["frames"]
        result = trace["result"]
        laps_so_far = [t for r in self.reports for t in r.result.lap_times] + list(result.lap_times)
        report = AttemptReport(
            result=result, frames=frames, crash_log=self._crash_log(frames, trace["crash_points"]),
            segments=_segments(frames, self.track.n_gates), messages=messages, driver_errors=errors,
            think_seconds_total=float(sum(think)), think_seconds_max=float(max(think, default=0.0)),
            phase=phase, next_phase=next_phase, stalls=stalls, late_ticks=late_total,
            zone_violations=self._zone_log(frames), llm_usage=_usage(self.drv),
            best_lap_so_far=min(laps_so_far) if laps_so_far else None, status=status,
            crash_points=list(trace["crash_points"]), driver_error_count=n_errors,
        )
        track_map = self._track_map()
        rd = report.to_reflect_dict()
        try:
            report.corners = driverkit.corner_stats(rd, track_map)
            rd["corners"] = report.corners
            report.summary_text = driverkit.summarize_report(rd, track_map)
        except Exception as exc:                 # noqa: BLE001 - a summary must never break a session
            report.summary_text = f"(summary unavailable: {type(exc).__name__}: {exc})"
        return report

    def _track_map(self) -> dict:
        if self._map is None:
            self._map = self.env.track_map()
        return self._map

    def _crash_log(self, frames: list[dict], points: list) -> list[dict]:
        out = []
        k = 0
        track_map = None
        for i, f in enumerate(frames):
            if not f.get("crashed"):
                continue
            before = frames[i - 1] if i > 0 else f
            pos = points[k] if k < len(points) else (f["x"], f["y"])
            k += 1
            # the penalty added in this tick (a crash and a zone violation never share a tick)
            penalty = max(0.0, float(f.get("penalty_time", 0.0)) - float(before.get("penalty_time", 0.0)))
            seen = f.get("obs") or {}
            entry = {
                "tick": int(f["tick"]), "t": round(float(f["t"]) - penalty, 2),
                "lap": int(f["lap"]), "segment": int(before["segment"]), "progress": round(float(before["progress"]), 4),
                "position": [round(float(pos[0]), 2), round(float(pos[1]), 2)],
                "speed_before": round(float(before["speed"]), 2), "throttle": round(float(f["throttle"]), 3),
                "steering": round(float(f["steering"]), 3),
                "rays": {key: v for key, v in seen.items() if key.startswith("distance_to_wall_")},
            }
            try:
                track_map = track_map or self._track_map()
                entry["corner"] = driverkit.corner_at(track_map, entry["position"])
            except Exception:                    # noqa: BLE001
                entry["corner"] = None
            out.append(entry)
        return out

    def _zone_log(self, frames: list[dict]) -> list[dict]:
        out = []
        for f in frames:
            if f.get("zone_violation"):
                zone = self.track.zone_for_segment(int(f["segment"]))
                out.append({"tick": int(f["tick"]), "t": round(float(f["t"]), 2), "lap": int(f["lap"]),
                            "segment": int(f["segment"]), "speed": round(float(f["speed"]), 2),
                            "limit": None if zone is None else float(zone.max_speed)})
        return out

    def _result(self, t_start: float, plan_error: str | None, plan_seconds: float) -> SessionResult:
        best_i, best = None, None
        for i, r in enumerate(self.reports):
            rt = r.result.race_time
            if r.result.completed and rt is not None and (best is None or rt < best):
                best_i, best = i, rt
        laps = [t for r in self.reports for t in r.result.lap_times]
        errors = list(self.session_errors) + [e for r in self.reports for e in r.driver_errors[:5]]
        status = self.fatal or ("DNF (stopped)" if self.stopped else "")
        cfg = self.config.to_dict()
        cfg.update({"tick_seconds": self.cfg.tick_seconds, "max_speed": self.cfg.max_speed})
        return SessionResult(
            driver=self.drv.meta(), config=cfg, attempts=list(self.reports),
            best_lap=min(laps) if laps else None, best_attempt=best_i,
            wall_seconds=time.perf_counter() - t_start,
            think_seconds_mean=(sum(self.drive_times) / len(self.drive_times)) if self.drive_times else 0.0,
            think_seconds_max=max(self.drive_times, default=0.0), driver_errors=errors, llm=_usage(self.drv),
            plan_error=plan_error, plan_seconds=plan_seconds, status=status, stopped=self.stopped,
            track_key=self.track.key, track_name=self.track.name, track=self.track,
            first_error_traceback=self.first_traceback,
        )


def _segments(frames: list[dict], n_gates: int) -> list[dict]:
    """Per waypoint segment over all laps: time (race clock, penalties included), speeds, crashes."""
    acc: dict[int, dict] = {}
    for j in range(len(frames) - 1):
        prev, f = frames[j], frames[j + 1]
        seg = int(prev["segment"])
        a = acc.setdefault(seg, {"time": 0.0, "speeds": [], "entries": [], "crashes": 0, "ticks": 0})
        if j == 0 or int(frames[j - 1]["segment"]) != seg:
            a["entries"].append(float(prev["speed"]))      # a new visit of this segment starts here
        a["time"] += float(f["t"]) - float(prev["t"])
        a["speeds"].append(float(f["speed"]))
        a["crashes"] += int(bool(f.get("crashed")))
        a["ticks"] += 1
    out = []
    for seg in sorted(acc):
        a = acc[seg]
        sp = a["speeds"]
        out.append({"segment": seg, "time": round(a["time"], 2),
                    "entry_speed": round(sum(a["entries"]) / len(a["entries"]), 2),
                    "min_speed": round(min(sp), 2), "max_speed": round(max(sp), 2),
                    "avg_speed": round(sum(sp) / len(sp), 2), "crashes": a["crashes"], "ticks": a["ticks"]})
    return out


def _usage(drv: Any) -> dict:
    try:
        return dict(drv.llm_usage())
    except Exception:                            # noqa: BLE001
        return {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0, "errors": 0}


def _kill(drv: Any) -> None:
    kill = getattr(drv, "kill", None)
    if callable(kill):
        kill()


def _short(exc: BaseException) -> str:
    return str(exc).split(":")[0][:60] or type(exc).__name__


# ============================================================================ friendly driver errors
_PKG_DIR = os.path.normcase(str(Path(__file__).resolve().parent.parent))
_LIB_DIRS = tuple(os.path.normcase(p) for p in {sysconfig.get_paths().get("stdlib", ""),
                                                 sysconfig.get_paths().get("purelib", ""),
                                                 sysconfig.get_paths().get("platlib", "")} if p)
_TB_FILE = re.compile(r'File "(.+?)", line (\d+)')
_FIELDS = {"drive": OBS_FIELDS, "plan": TRACK_FIELDS, "reflect": REPORT_FIELDS}


def describe_driver_error(exc: BaseException, source_path: str | None, method: str = "drive") -> tuple[str, str]:
    """('KeyError: 'x' (my_driver.py:12) - did you mean 'y'?', full traceback text) for driver code."""
    if isinstance(exc, DriverError):             # raised in an isolated child: text + traceback text
        text, tb = str(exc), exc.traceback_text or ""
        frames = [(f, int(n)) for f, n in _TB_FILE.findall(tb)]
    else:
        text = _plain(exc)
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        frames = [(fs.filename, fs.lineno or 0) for fs in traceback.extract_tb(exc.__traceback__)]
    where = _driver_frame(frames, source_path)
    if where:
        text += f" ({Path(where[0]).name}:{where[1]})"
    m = re.match(r"KeyError: '([^']+)'", text)
    fields = _FIELDS.get(method)
    if m and fields and m.group(1) not in fields:
        close = difflib.get_close_matches(m.group(1), list(fields), n=1, cutoff=0.6)
        if close:
            text += f" - did you mean '{close[0]}'?"
    return text, tb[-4000:]


def _plain(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _driver_frame(frames: list[tuple[str, int]], source_path: str | None) -> tuple[str, int] | None:
    """The innermost traceback frame in the driver file (else: in user code, not agentic_gp or a library)."""
    if source_path:
        want = os.path.normcase(os.path.abspath(source_path))
        mine = [f for f in frames if os.path.normcase(os.path.abspath(f[0])) == want]
        if mine:
            return mine[-1]
    user = [f for f in frames if not f[0].startswith("<")
            and not os.path.normcase(os.path.abspath(f[0])).startswith(_PKG_DIR + os.sep)
            and not any(os.path.normcase(os.path.abspath(f[0])).startswith(d) for d in _LIB_DIRS)]
    return user[-1] if user else None
