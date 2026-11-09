"""Run a driver in a BACKGROUND THREAD and watch it live from Streamlit.

    from app.lib.live_runner import LiveRun
    run = LiveRun("drivers/examples/l2_planner.py", SessionConfig(track="sprint", attempts=2), pace=4.0)
    run.start()
    ...
    @st.fragment(run_every=0.25)               # the page polls; the thread never touches Streamlit
    def live():
        run.heartbeat()                         # "someone is still watching"
        live_canvas(run.canvas_state(), key="live")
        st.line_chart(run.telemetry_frame(), x="t", y="speed")

Why a thread? A slow driver (an LLM thinks for seconds) must not freeze the page, and a click
on a widget restarts the page script - the race must survive that. So the session runs in its
own thread and only PUBLISHES snapshots (immutable, swapped under a lock). The page reads the
newest snapshot whenever it likes. Rule: the worker thread NEVER calls st.* (Streamlit's
documented multithreading rule); only the page script draws.

Lifecycle: start() -> pause()/resume()/step() -> stop(). The worker also stops by itself when
nobody called heartbeat() for `idle_timeout_s` seconds (the browser tab was closed).

BackgroundJob (below) is the same idea for any long function, e.g. run_grand_prix with its
progress callback.
"""

from __future__ import annotations

import threading
import time
import traceback
from dataclasses import dataclass, field, replace
from typing import Any, Callable

TRAIL_POINTS = 240


@dataclass(frozen=True)
class LiveSnapshot:
    """What the page needs to draw ONE moment of a live session (plain data, safe to share)."""

    seq: int = 0                      # +1 for every new tick (the canvas starts a tween)
    status: str = "loading"           # loading | planning | driving | thinking | reflecting | paused |
                                      # finished | stopped | error
    message: str = ""
    name: str = ""
    color: str = "#3987e5"
    kind: str = "code"
    attempt: int = 0
    attempts: int = 1
    phase: str = ""
    tick: int = 0
    t: float = 0.0                    # race clock of the attempt
    lap: int = 1
    laps: int = 1
    crashes: int = 0
    car: dict | None = None           # x, y, heading_deg, speed, throttle, steering (pose the driver SAW)
    obs: dict | None = None           # the observation of this tick (what drive() received)
    action: dict | None = None        # throttle, steering, say (what drive() answered)
    debug: dict | None = None         # the driver's debug dict (debug["target"] = [x, y])
    think_s: float | None = None      # how long drive() took
    say: tuple = ()                   # ({"t", "text", "attempt"}, ...) radio messages so far
    trail: tuple = ()                 # recent (x, y) of this attempt
    crash_points: tuple = ()          # (x, y) of this attempt's crashes
    done: bool = False
    error: str | None = None
    wall_s: float = 0.0               # wall seconds since start()


class LiveRun:
    """One driver, one session, in a background thread (see the module docstring).

    driver    a driver file path (or anything agentic_gp.race.load_driver accepts)
    config    SessionConfig (track, attempts, laps, realtime, physics, ...)
    params    PARAMS overrides, e.g. from app.lib.params_ui.params_sliders
    isolate   run the driver in its own process (can be killed; slower to start)
    pace      minimum wall time per tick = tick_seconds / pace. 1.0 = real time, 4.0 = 4x,
              None = as fast as the driver can go (code drivers: ~1000 ticks/s - too fast to watch)
    env       extra environment of the isolated driver process, e.g. {"LLM_PROVIDER": "mock",
              "LLM_LOG_DIR": ...} - per run, so one browser's choice never changes another's
              (ignored when isolate=False)
    """

    def __init__(self, driver: Any, config: Any = None, *, params: dict | None = None, isolate: bool = False,
                 pace: float | None = 4.0, idle_timeout_s: float = 120.0, name: str = "live-run",
                 env: dict | None = None) -> None:
        from agentic_gp.race.session import SessionConfig

        self.source = driver
        self.config = config or SessionConfig()
        self.params = params
        self.isolate = isolate
        self.env = dict(env or {})
        self.started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.pace = pace
        self.idle_timeout_s = idle_timeout_s
        self._lock = threading.Lock()
        self._snap = LiveSnapshot(attempts=len(self.config.schedule()))
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._step = threading.Event()
        self._last_seen = time.monotonic()
        self._t_start = 0.0
        self._telemetry: list[dict] = []
        self._reports: list[Any] = []          # AttemptReport objects (reflect fills in more later)
        self._events: list[str] = []
        self._driver = None
        self._track_payload: dict | None = None
        self.result = None                      # SessionResult when the session is over
        self.error: str | None = None
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)

    # ------------------------------------------------------------------ control (any thread)
    def start(self) -> "LiveRun":
        self._t_start = time.monotonic()
        self._thread.start()
        return self

    def stop(self, kill: bool = False) -> None:
        """Ask the session to stop after the current tick (kill=True also kills an isolated driver)."""
        self._stop.set()
        self._paused.clear()
        if kill and self._driver is not None and hasattr(self._driver, "kill"):
            try:
                self._driver.kill()
            except Exception:                    # noqa: BLE001
                pass

    def pause(self) -> None:
        self._paused.set()

    def resume(self) -> None:
        self._paused.clear()

    def toggle_pause(self) -> None:
        self.resume() if self._paused.is_set() else self.pause()

    def step(self) -> None:
        """While paused: drive exactly one more tick."""
        self._paused.set()
        self._step.set()

    def heartbeat(self) -> None:
        self._last_seen = time.monotonic()

    def join(self, timeout: float | None = None) -> bool:
        self._thread.join(timeout)
        return not self._thread.is_alive()

    @property
    def paused(self) -> bool:
        return self._paused.is_set()

    @property
    def done(self) -> bool:
        return self.snapshot().done

    @property
    def running(self) -> bool:
        return self._thread.is_alive()

    # ------------------------------------------------------------------ reading (any thread)
    def snapshot(self) -> LiveSnapshot:
        with self._lock:
            return self._snap

    def telemetry(self) -> list[dict]:
        """Every tick so far: attempt, phase, tick, t, x, y, speed, throttle, steering, lap, ..."""
        with self._lock:
            return list(self._telemetry)

    def telemetry_frame(self):
        import pandas as pd

        return pd.DataFrame(self.telemetry())

    def reports(self) -> list[dict]:
        """One summary per finished attempt (summary_text = what reflect() was told)."""
        with self._lock:
            reps = list(self._reports)
        out = []
        for r in reps:
            res = r.result
            out.append({"attempt": res.attempt, "phase": r.phase, "completed": res.completed,
                        "race_time": res.race_time, "lap_times": list(res.lap_times), "best_lap": res.best_lap,
                        "crashes": res.crashes, "zone_violations": len(r.zone_violations), "ticks": res.ticks,
                        "status": r.status or "finished", "think_ms_max": round(r.think_seconds_max * 1000, 1),
                        "reflect_seconds": round(r.reflect_seconds, 2), "reflect_error": r.reflect_error,
                        "summary_text": r.summary_text})
        return out

    def events(self) -> list[str]:
        with self._lock:
            return list(self._events)

    def replay(self, which: str = "all"):
        """The session as a Replay (None until the session is over)."""
        if self.result is None:
            return None
        try:
            return self.result.to_replay(which)
        except Exception:                        # noqa: BLE001
            return None

    def track_payload(self) -> dict:
        """The track for the canvases (computed once)."""
        if self._track_payload is None:
            from agentic_gp.viz.player import track_payload

            self._track_payload = track_payload(self.config.track)
        return self._track_payload

    def canvas_state(self, **kw: Any) -> dict:
        """live_state(...) for app.components.live_canvas from the newest snapshot (kw: extra options)."""
        from app.components.race_canvas import live_state

        s = self.snapshot()
        return live_state(self.track_payload(), obs=s.obs, car=s.car, action=s.action, debug=s.debug, name=s.name,
                          color=s.color, seq=s.seq, status=s.status, say=list(s.say), trail=list(s.trail),
                          crash_points=list(s.crash_points), attempt=s.attempt, attempts=s.attempts, phase=s.phase,
                          think_s=s.think_s, message=s.message or None, **kw)

    # ------------------------------------------------------------------ the worker thread
    def _publish(self, **changes: Any) -> None:
        with self._lock:
            self._snap = replace(self._snap, wall_s=round(time.monotonic() - self._t_start, 2), **changes)

    def _log(self, text: str) -> None:
        with self._lock:
            self._events.append(f"[{time.monotonic() - self._t_start:6.1f}s] {text}")

    def _idle(self) -> bool:
        return time.monotonic() - self._last_seen > self.idle_timeout_s

    def _wait(self, seconds: float) -> None:
        end = time.monotonic() + seconds
        while not self._stop.is_set() and time.monotonic() < end:
            time.sleep(min(0.02, max(0.0, end - time.monotonic())))

    def _hold_if_paused(self) -> None:
        if not self._paused.is_set():
            return
        prev = self.snapshot().status
        self._publish(status="paused")
        while self._paused.is_set() and not self._stop.is_set():
            if self._step.is_set():
                self._step.clear()
                break
            if self._idle():
                self._stop.set()
                break
            time.sleep(0.05)
        if self.snapshot().status == "paused":
            self._publish(status=prev)

    def _run(self) -> None:
        from agentic_gp.race.loader import load_driver
        from agentic_gp.race.session import iter_session

        trail: list = []
        crashes: list = []
        say: list = []
        last_say = None
        tick_seconds = 0.5
        try:
            self._publish(status="loading", message="Loading the driver...")
            if self.isolate:                     # Stop works while the child is still loading (cancel)
                self._driver = drv = load_driver(self.source, params=self.params, isolate=True, env=self.env,
                                                 cancel=self._stop)
            else:
                self._driver = drv = load_driver(self.source, params=self.params)
            if self._stop.is_set():
                raise RuntimeError("stopped while loading")
            meta = drv.meta()
            self._publish(name=str(meta.get("name", "car")), color=str(meta.get("color") or "#3987e5"),
                          kind=str(meta.get("kind") or "code"), status="planning", message="plan(track)...")
            self._log(f"loaded {meta.get('name')} ({meta.get('kind')})")
            last_wall = time.monotonic()
            seq = 0
            for ev in iter_session(drv, self.config, stop_event=self._stop):
                if ev.type == "plan":
                    err = ev.data.get("error")
                    tick_seconds = float(ev.data["track"].get("physics", {}).get("tick_seconds", 0.5))
                    self._log(f"plan() took {ev.data.get('seconds', 0):.2f}s" + (f" - ERROR {err}" if err else ""))
                    self._publish(message="" if not err else f"plan() failed: {err}")
                elif ev.type == "attempt_start":
                    trail, crashes = [], []
                    self._log(f"attempt {ev.attempt} ({ev.phase}) starts")
                    self._publish(attempt=ev.attempt, phase=ev.phase, laps=int(ev.data.get("laps", 1)), tick=0,
                                  crashes=0, trail=(), crash_points=(), status="driving", message="")
                elif ev.type == "tick":
                    seq += 1
                    f, obs = ev.data, ev.data["obs"]
                    act = f["action"]
                    think = float(f.get("think_seconds") or 0.0)
                    pos = obs.get("position", [f["x"], f["y"]])
                    trail.append((round(float(pos[0]), 2), round(float(pos[1]), 2)))
                    if f.get("crashed"):
                        crashes.append(_contact_guess(obs, tick_seconds))
                    if act.say and act.say != last_say:
                        say.append({"t": round(float(obs.get("race_time", 0.0)), 2), "text": act.say,
                                    "attempt": ev.attempt})
                    last_say = act.say
                    row = {"attempt": ev.attempt, "phase": ev.phase, "tick": int(f["tick"]), "t": round(float(f["t"]), 2),
                           "x": round(float(f["x"]), 2), "y": round(float(f["y"]), 2),
                           "speed": round(float(f["speed"]), 2), "throttle": round(float(act.throttle), 3),
                           "steering": round(float(act.steering), 3), "lap": int(f["lap"]),
                           "progress": round(float(f["progress"]), 4), "race_progress": round(float(f["race_progress"]), 4),
                           "offset": round(float(f["offset_from_center"]), 2), "crashed": bool(f.get("crashed")),
                           "think_ms": round(think * 1000, 2)}
                    with self._lock:
                        self._telemetry.append(row)
                    self._publish(
                        seq=seq, tick=int(obs["tick"]), t=float(obs.get("race_time", 0.0)), lap=int(obs.get("lap", 1)),
                        laps=int(obs.get("laps_total", 1)), crashes=int(obs.get("crashes", 0)) + int(bool(f.get("crashed"))),
                        car={"x": float(pos[0]), "y": float(pos[1]), "heading_deg": float(obs["heading_deg"]),
                             "speed": float(obs["speed"]), "throttle": act.throttle, "steering": act.steering},
                        obs=obs, action={"throttle": act.throttle, "steering": act.steering, "say": act.say},
                        debug=f.get("debug"), think_s=think, say=tuple(say[-20:]), trail=tuple(trail[-TRAIL_POINTS:]),
                        # a slow drive() (an LLM pilot) will be slow again: the canvas shows a "THINKING 1.3s" clock
                        crash_points=tuple(crashes), status="thinking" if think > 0.3 else "driving")
                    # pacing: never faster than `pace` x real time, so humans can follow
                    if self.pace:
                        self._wait(max(0.0, tick_seconds / self.pace - (time.monotonic() - last_wall)))
                    last_wall = time.monotonic()
                    self._hold_if_paused()
                elif ev.type == "attempt_end":
                    rep = ev.data
                    with self._lock:
                        self._reports.append(rep)
                    last = self._telemetry[-1] if self._telemetry else None
                    fr = rep.frames[-1] if rep.frames else None
                    car = None if fr is None else {"x": float(fr["x"]), "y": float(fr["y"]),
                                                   "heading_deg": float(fr["heading_deg"]), "speed": float(fr["speed"]),
                                                   "throttle": 0.0, "steering": 0.0}
                    res = rep.result
                    msg = (f"attempt {res.attempt}: {res.race_time:.2f} s, {res.crashes} crash(es)" if res.completed
                           else f"attempt {res.attempt}: {rep.status}")
                    self._log(msg)
                    seq += 1
                    self._publish(seq=seq, car=car, crash_points=tuple(tuple(p) for p in rep.crash_points),
                                  status="reflecting", message=msg + " - reflect(report)...",
                                  t=float(res.race_time if res.race_time is not None else (last or {}).get("t", 0.0)))
                elif ev.type == "session_end":
                    self.result = ev.data
                if self._idle():
                    self._log("no heartbeat - stopping (was the browser tab closed?)")
                    self._stop.set()
            res = self.result
            status = "stopped" if (res is not None and res.stopped) or self._stop.is_set() else "finished"
            best = f"best lap {res.best_lap:.2f} s" if res is not None and res.best_lap else "no lap completed"
            self._publish(status=status, message=f"Session {status}: {best}", done=True)
            self._log(f"session {status}")
        except Exception as exc:                 # noqa: BLE001 - report every failure on the page
            if self._stop.is_set() and self.result is None and not self._reports:
                self._log(f"stopped before the first attempt ({type(exc).__name__})")
                self._publish(status="stopped", message="Stopped.", done=True)
                return
            self.error = f"{type(exc).__name__}: {exc}"
            self._log("ERROR " + self.error + "\n" + traceback.format_exc(limit=4))
            self._publish(status="error", message=self.error, error=self.error, done=True)
        finally:
            drv = self._driver
            if drv is not None:
                try:
                    drv.close()
                except Exception:                # noqa: BLE001
                    pass


def _contact_guess(obs: dict, dt: float) -> tuple[float, float]:
    """Where the car probably hit the wall during this tick (exact points arrive with the report)."""
    import math

    x, y = obs["position"]
    h = math.radians(float(obs["heading_deg"]))
    d = min(float(obs.get("distance_to_wall_front", 0.0)), abs(float(obs.get("speed", 0.0))) * dt + 1.0)
    return round(x + d * math.cos(h), 2), round(y + d * math.sin(h), 2)


# ============================================================================ generic background job
@dataclass
class JobState:
    status: str = "idle"              # idle | running | done | error | stopped
    latest: dict = field(default_factory=dict)    # newest update per key (e.g. per driver)
    log: list = field(default_factory=list)       # every update (bounded)
    result: Any = None
    error: str | None = None
    elapsed_s: float = 0.0


class BackgroundJob:
    """Run any long function in a thread; it reports progress with `job.post(update)`.

        job = BackgroundJob(lambda job: run_grand_prix(files, cfg, progress=job.post, stop_event=job.stop_event))
        job.start()
        ...in a fragment: st.dataframe(list(job.state().latest.values())); if job.state().status == "done": ...

    `post(update)` keeps the newest update per `update[key_field]` (default "driver") and appends
    to a bounded log. The function's return value ends up in state().result.
    """

    def __init__(self, fn: Callable[["BackgroundJob"], Any], *, key_field: str = "driver", max_log: int = 2000,
                 name: str = "background-job") -> None:
        self.fn = fn
        self.key_field = key_field
        self.max_log = max_log
        self.stop_event = threading.Event()
        self._lock = threading.Lock()
        self._state = JobState()
        self._t0 = 0.0
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)

    def start(self) -> "BackgroundJob":
        self._t0 = time.monotonic()
        with self._lock:
            self._state.status = "running"
        self._thread.start()
        return self

    def stop(self) -> None:
        self.stop_event.set()

    def post(self, update: dict) -> None:
        with self._lock:
            k = update.get(self.key_field, "_")
            self._state.latest[k] = dict(update)
            self._state.log.append(dict(update))
            if len(self._state.log) > self.max_log:
                del self._state.log[: len(self._state.log) - self.max_log]

    def state(self) -> JobState:
        with self._lock:
            s = self._state
            return JobState(s.status, dict(s.latest), list(s.log[-200:]), s.result, s.error,
                            round(time.monotonic() - self._t0, 1) if self._t0 else 0.0)

    @property
    def done(self) -> bool:
        return self.state().status in ("done", "error", "stopped")

    def join(self, timeout: float | None = None) -> bool:
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def _run(self) -> None:
        try:
            result = self.fn(self)
            with self._lock:
                self._state.result = result
                self._state.status = "stopped" if self.stop_event.is_set() else "done"
        except Exception as exc:                 # noqa: BLE001
            with self._lock:
                self._state.error = f"{type(exc).__name__}: {exc}"
                self._state.status = "error"
