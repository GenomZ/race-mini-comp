"""Grand prix: many drivers, one track - plan, practice, qualifying, race - and the replays.

    from agentic_gp.race import run_grand_prix, GrandPrixConfig
    gp = run_grand_prix(["drivers/submissions", "drivers/examples/l2_planner.py"],
                        GrandPrixConfig(track="grand_prix", practice_attempts=2, race_laps=3),
                        progress=lambda u: print(u["driver"], u["phase"], u.get("race_progress")))
    gp.standings              # [{position, name, race_time, gap, laps_completed, best_lap, crashes, status, ...}]
    gp.save("races", "final") # final.json (race replay), final_qualifying.json, final_results.json

Every driver gets its own session: plan() once -> `practice_attempts` single laps with reflect()
-> one qualifying lap (grid order; reflect() is called with next_phase="race") -> the race of
`race_laps` laps. The SAME driver instance runs all of it, so practice learning carries into the race.
Sessions are independent (no collisions, ghost cars) and are merged by race clock in the replay.

How it runs: one THREAD per driver steps the simulation in this process, while the driver file
itself runs in its own child process (isolate) - a hanging or crashing driver costs only itself.
At most `jobs` drivers run at once, ramping up by one every `ramp_s` seconds so a class full of LLM
drivers does not hit the API in one burst. LLM calls of all drivers share one cross-process
semaphore (`llm_concurrency` in flight). `progress(update)` is called in the calling thread.
`inprocess=True` runs every driver sequentially in this process (tests, debugging, live UIs);
drivers cannot be interrupted there, so the per-call time limits are only checked afterwards, and
nothing a driver starts (threads, processes) is cleaned up: in-process is for TRUSTED code only.
Setting `stop_event` stops the event within about a second: running drivers are killed, even in
the middle of load/plan/reflect, and end as "DNF (stopped)".

Official rules (llm_policy="official"): no LLM calls in drive() or during qualifying/race,
<= 2 calls in plan, <= 2 per practice reflect, <= 8 per driver per track.
Classification: race time = sim time + 5 s per crash + zone penalties; ties -> fewer crashes;
DNFs by race progress.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..llm import log_file_name
from ..sim.config import PhysicsConfig
from ..sim.track import Track, make_track
from .loader import DriverLoadError, color_for, load_driver
from .replay import Replay, ReplayCar, build_replay, placeholder_car, replay_car_from_trace, slug, utc_now
from .session import AttemptReport, SessionConfig, SessionResult, iter_session

PROGRESS_INTERVAL_S = 0.25          # at most 4 progress updates per second per driver


@dataclass
class DriverEntry:
    """One car on the grid: a driver file (+ house flag and PARAMS overrides)."""

    source: str | Path
    house: bool = False                 # house cars race but are not scored in the championship
    params: dict | None = None


@dataclass
class GrandPrixConfig:
    track: str | Track = "grand_prix"
    practice_attempts: int = 2
    race_laps: int = 3
    jobs: int | None = None             # drivers at once (default min(cpu - 1, n))
    driver_timeout_s: float = 300.0     # total thinking time per driver on this track (plan + drive + reflect)
    tick_timeout_s: float = 120.0       # watchdog: a driver silent for this long is killed (never before
                                        #   its load/plan/reflect limit + 30 s: those have their own timeouts)
    title: str = "Agentic Grand Prix"
    start_method: str = "spawn"
    qualifying: bool = True
    drive_timeout_s: float = 2.0        # per drive() call (a slower answer = a stall: coast)
    plan_timeout_s: float = 120.0
    reflect_timeout_s: float = 120.0
    load_timeout_s: float = 300.0       # starting the child + importing the driver (a cold `import openai` can be slow)
    llm_policy: str = "official"
    llm_concurrency: int = 8            # LLM calls in flight across all drivers
    ramp_s: float = 2.0                 # one more concurrent driver every ramp_s seconds
    log_dir: str | None = None          # child stdout/stderr logs (default: a temp folder)
    llm_log_dir: str | None = None      # LLM_LOG_DIR for the drivers (every prompt + answer as JSONL)
    physics: PhysicsConfig | None = None

    def schedule(self) -> tuple[list[str], list[int]]:
        phases = ["practice"] * max(0, int(self.practice_attempts))
        laps = [1] * len(phases)
        if self.qualifying:
            phases.append("qualifying")
            laps.append(1)
        phases.append("race")
        laps.append(max(1, int(self.race_laps)))
        return phases, laps

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__ if k not in ("track", "physics")}
        d["track"] = self.track.key if isinstance(self.track, Track) else str(self.track)
        d["crash_rule"] = (self.physics.crash_rule if self.physics else "marshal")
        return d


@dataclass
class _Outcome:
    index: int
    entry: DriverEntry
    meta: dict
    session: SessionResult | None = None
    status: str = ""
    error: str | None = None
    log_path: str | None = None
    label: str | None = None             # LLM log label (= NAME, or "NAME-2" for a second car with that NAME)

    def report(self, phase: str) -> AttemptReport | None:
        return self.session.report_for(phase) if self.session else None


@dataclass
class GrandPrixResult:
    title: str
    track_key: str
    track_name: str
    replay: Replay                       # kind="race": the race, cars in official order
    qualifying_replay: Replay | None
    qualifying: list[dict]               # grid: position, name, qualifying_time, best_lap, crashes, attempts, status
    standings: list[dict]                # position, name, race_time, gap, laps_completed, best_lap, crashes, status, ...
    errors: list[dict]                   # {driver, status, error}
    drivers: list[dict]                  # per driver: meta, practice laps, log file, llm usage
    config: dict
    wall_seconds: float
    created_utc: str = field(default_factory=utc_now)
    sessions: dict = field(default_factory=dict, repr=False)   # name -> SessionResult ("Name-2" for a
                                                               #   second car with the same NAME)

    def to_dict(self) -> dict:
        """Everything except the replays (they are saved as their own files)."""
        return {"format": "agentic-gp-grand-prix", "version": 1, "title": self.title, "track": self.track_key,
                "track_name": self.track_name, "created_utc": self.created_utc, "config": self.config,
                "qualifying": self.qualifying, "standings": self.standings, "errors": self.errors,
                "drivers": self.drivers, "wall_seconds": round(self.wall_seconds, 2)}

    def save(self, directory: str | Path, stem: str = "grand_prix", extended: bool = True) -> dict[str, Path]:
        """Write <stem>.json (race replay), <stem>_qualifying.json and <stem>_results.json."""
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        paths = {"replay": out / f"{stem}.json", "results": out / f"{stem}_results.json"}
        self.replay.to_json(paths["replay"], extended=extended)
        if self.qualifying_replay is not None:
            paths["qualifying"] = out / f"{stem}_qualifying.json"
            self.qualifying_replay.to_json(paths["qualifying"], extended=extended)
        import json
        paths["results"].write_text(json.dumps(self.to_dict(), indent=1, allow_nan=False), encoding="utf-8")
        return paths


# ============================================================================ entry points
SIDE_SUFFIXES = ("_pilot", "_fly")       # <handle>_pilot.py (Ch4) and <handle>_fly.py (Ch7, Fly Cup)


def is_side_file(path: Path, folder_files: list[Path]) -> bool:
    """A Ch4 pilot is never a race car; a Ch7 fly races only in a folder of flies (the Fly Cup grid):
    next to a main file it is a side file."""
    stem = path.stem.lower()
    if stem.endswith("_pilot"):
        return True
    if stem.endswith("_fly"):
        return any(not f.stem.lower().endswith(SIDE_SUFFIXES) for f in folder_files)
    return False


def find_driver_files(paths: Any, skipped: list | None = None) -> list[Path]:
    """Files and folders -> driver files.

    A folder means all *.py in it not starting with '_', minus side files: <handle>_pilot.py, and
    <handle>_fly.py when the folder also holds main files (a folder of only *_fly.py files - the Fly Cup
    grid - races its flies). Skipped side files are appended to `skipped`. A file given by its path
    always races. Raises FileNotFoundError for a path that does not exist (a typo must not race - and
    overwrite the results - with zero drivers). An existing empty folder gives no files.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    out: list[Path] = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            here = sorted(f for f in p.glob("*.py") if not f.name.startswith("_"))
            side = [f for f in here if is_side_file(f, here)]
            out += [f for f in here if f not in side]
            if skipped is not None:
                skipped += side
        elif p.is_file():
            out.append(p)
        else:
            raise FileNotFoundError(f"not found: {p} (give driver .py files or folders that contain them)")
    return out


def _entries(sources: Any) -> list[DriverEntry]:
    if isinstance(sources, (str, Path, DriverEntry)):
        sources = [sources]
    out: list[DriverEntry] = []
    for s in sources:
        if isinstance(s, DriverEntry):
            out.append(s)
        elif isinstance(s, (str, Path)):
            out += [DriverEntry(f) for f in find_driver_files(s)]
        else:
            raise TypeError(f"a grand prix source is a driver file, a folder or a DriverEntry, not {type(s).__name__}")
    return out


def run_grand_prix(sources: Any, config: GrandPrixConfig | None = None,
                   progress: Callable[[dict], None] | None = None, stop_event: threading.Event | None = None,
                   inprocess: bool = False) -> GrandPrixResult:
    """Run a whole grand prix (see the module docstring) and return the result."""
    config = config or GrandPrixConfig()
    entries = _entries(sources)
    if not entries:
        raise ValueError("no driver files to race")
    labels = plan_labels(entries)
    track = make_track(config.track)
    t0 = time.perf_counter()
    log_dir = Path(config.log_dir) if config.log_dir else \
        Path(tempfile.gettempdir()) / "agentic_gp_driver_logs" / time.strftime("gp_%Y%m%d_%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    if inprocess:
        outcomes = _run_inprocess(entries, config, track, progress, stop_event, log_dir, labels)
    else:
        outcomes = _run_threads(entries, config, track, progress, stop_event, log_dir, labels)
    return _assemble(outcomes, config, track, time.perf_counter() - t0)


def _notify(progress: Callable[[dict], None] | None, update: dict) -> None:
    if progress is None:
        return
    try:
        progress(update)
    except Exception as exc:                     # noqa: BLE001 - a UI bug must not stop the race
        print(f"[grand prix] progress callback failed: {type(exc).__name__}: {exc}")


def _run_inprocess(entries: list[DriverEntry], config: GrandPrixConfig, track: Track, progress: Any,
                   stop_event: threading.Event | None, log_dir: Path, labels: dict | None = None) -> list[_Outcome]:
    old_log = os.environ.get("LLM_LOG_DIR")
    if config.llm_log_dir:
        os.environ["LLM_LOG_DIR"] = str(config.llm_log_dir)
    try:
        outcomes = []
        labels = labels if labels is not None else {}
        for i, entry in enumerate(entries):
            stop = stop_event or threading.Event()
            outcomes.append(_race_driver(i, entry, config, track, lambda u: _notify(progress, u), stop, None,
                                         True, log_dir, {}, labels))
        return outcomes
    finally:
        if config.llm_log_dir:
            if old_log is None:
                os.environ.pop("LLM_LOG_DIR", None)
            else:
                os.environ["LLM_LOG_DIR"] = old_log


def _run_threads(entries: list[DriverEntry], config: GrandPrixConfig, track: Track, progress: Any,
                 stop_event: threading.Event | None, log_dir: Path, labels: dict | None = None) -> list[_Outcome]:
    n = len(entries)
    jobs = max(1, min(config.jobs or max(1, (os.cpu_count() or 2) - 1), n))
    semaphore = mp.get_context(config.start_method).BoundedSemaphore(max(1, config.llm_concurrency))
    updates: queue.Queue = queue.Queue()
    outcomes: list[_Outcome | None] = [None] * n
    stops = [threading.Event() for _ in range(n)]
    activity = [time.monotonic()] * n
    holders: list[dict] = [{} for _ in range(n)]
    threads: dict[int, threading.Thread] = {}
    labels = labels if labels is not None else {}
    labels["lock"] = threading.Lock()
    pending = list(range(n))
    t_start = time.monotonic()
    # Loading may take two load windows with no other sign of life (the file, then the LLM SDK import;
    # each posts a "load" / "warm-up" progress event when it starts).
    silence_limit = max(config.tick_timeout_s, 30.0 + max(2 * config.load_timeout_s, config.plan_timeout_s,
                                                          config.reflect_timeout_s))

    def post(i: int) -> Callable[[dict], None]:
        def _post(update: dict) -> None:
            activity[i] = time.monotonic()
            updates.put(update)
        return _post

    def worker(i: int) -> None:
        try:
            outcomes[i] = _race_driver(i, entries[i], config, track, post(i), stops[i], semaphore, False,
                                       log_dir, holders[i], labels)
        except BaseException as exc:             # noqa: BLE001 - never lose a driver silently
            outcomes[i] = _Outcome(i, entries[i], _guess_meta(entries[i]),
                                   status=f"DNF (driver crashed: {type(exc).__name__})",
                                   error=traceback.format_exc()[-2000:])
            updates.put({"type": "error", "driver": outcomes[i].meta["name"], "index": i,
                         "traceback": outcomes[i].error})

    stopping = False
    try:
        while pending or threads:
            if stop_event is not None and stop_event.is_set() and not stopping:
                stopping = True
                for s in stops:
                    s.set()
                for i in pending:
                    outcomes[i] = _Outcome(i, entries[i], _guess_meta(entries[i]), status="DNF (not started)")
                pending = []
                for h in holders:               # do not wait for a driver busy in plan/reflect (<= 120 s)
                    drv = h.get("driver")
                    if drv is not None and hasattr(drv, "kill"):
                        drv.kill()
            ramp = 1 + int((time.monotonic() - t_start) / config.ramp_s) if config.ramp_s > 0 else jobs
            while pending and len(threads) < min(jobs, ramp):
                i = pending.pop(0)
                activity[i] = time.monotonic()
                th = threading.Thread(target=worker, args=(i,), name=f"gp-driver-{i}", daemon=True)
                threads[i] = th
                th.start()
            try:
                _notify(progress, updates.get(timeout=0.1))
                while True:
                    _notify(progress, updates.get_nowait())
            except queue.Empty:
                pass
            for i, th in list(threads.items()):
                if not th.is_alive():
                    th.join()
                    del threads[i]
                elif time.monotonic() - activity[i] > silence_limit:
                    stops[i].set()                       # watchdog: silent for too long
                    drv = holders[i].get("driver")
                    holders[i]["watchdog"] = True
                    if drv is not None and hasattr(drv, "kill"):
                        drv.kill()
                    activity[i] = time.monotonic()
        while True:
            _notify(progress, updates.get_nowait())
    except queue.Empty:
        pass
    finally:
        for s in stops:
            s.set()
        for h in holders:
            drv = h.get("driver")
            if drv is not None and hasattr(drv, "kill"):
                drv.kill()
        for th in threads.values():
            th.join(5.0)
    return [o if o is not None else _Outcome(i, entries[i], _guess_meta(entries[i]), status="DNF (not started)")
            for i, o in enumerate(outcomes)]


def _guess_meta(entry: DriverEntry) -> dict:
    name = Path(entry.source).stem
    return {"name": name, "author": "anonymous", "color": color_for(name), "kind": "code", "model": None,
            "source": str(entry.source), "motto": "", "anatomy": [], "params": {}, "house": entry.house}


def _static_name(entry: DriverEntry) -> str:
    """The NAME a driver file will load with, read without running it (the file stem if not a literal)."""
    from .loader import MAX_NAME_CHARS, static_metadata
    path = Path(entry.source)
    name = static_metadata(path).get("NAME")
    if not isinstance(name, str):
        name = path.stem
    return " ".join(name.split())[:MAX_NAME_CHARS].strip() or path.stem[:MAX_NAME_CHARS]


def plan_labels(entries: list[DriverEntry]) -> dict:
    """Decide every car's log label BEFORE the race, deterministically: house cars first (so the house
    car keeps "Max" and its log stays Max.jsonl), then the grid order; a second car with the same NAME
    gets "Name-2" - the same suffix its id gets in the results ("name-2")."""
    labels: dict = {"planned": {}}
    order = sorted(range(len(entries)), key=lambda i: (not entries[i].house, i))
    for i in order:
        name = _static_name(entries[i])
        labels["planned"][i] = (name, _unique_label(labels, name))
    return labels


def _label_for(labels: dict | None, i: int, name: str) -> str:
    """The planned label of car i (if it loaded with the NAME the plan expected), else a fresh unique one."""
    labels = labels if labels is not None else {}
    planned = labels.get("planned", {}).get(i)
    if planned is not None and planned[0] == name:
        return planned[1]
    return _unique_label(labels, name)


def _unique_label(labels: dict, name: str) -> str:
    """The driver's name, or "name-2", "name-3", ... when another car already uses it (LLM log files)."""
    lock = labels.get("lock")
    if lock is not None:
        lock.acquire()
    try:
        used = labels.setdefault("used", set())
        label, k = name, 1
        while label in used:
            k += 1
            label = f"{name}-{k}"
        used.add(label)
        return label
    finally:
        if lock is not None:
            lock.release()


def _race_driver(i: int, entry: DriverEntry, config: GrandPrixConfig, track: Track, post: Callable[[dict], None],
                 stop: threading.Event, semaphore: Any, inprocess: bool, log_dir: Path, holder: dict,
                 labels: dict | None = None) -> _Outcome:
    """Load one driver and run its whole weekend (plan, practice, qualifying, race)."""
    meta = _guess_meta(entry)
    log_path = str(log_dir / f"{i:02d}_{slug(meta['name'])}.log")
    post({"type": "progress", "driver": meta["name"], "index": i, "phase": "load", "attempt": 0, "tick": 0,
          "lap": 0, "race_progress": 0.0, "speed": 0.0, "t": 0.0})
    env = {"LLM_LOG_DIR": str(config.llm_log_dir)} if config.llm_log_dir else {}

    def stage(name: str) -> None:            # "warm-up": the child imports the LLM SDK (may take minutes)
        post({"type": "progress", "driver": meta["name"], "index": i, "phase": name, "attempt": 0, "tick": 0,
              "lap": 0, "race_progress": 0.0, "speed": 0.0, "t": 0.0})

    try:
        if inprocess:
            drv = load_driver(entry.source, params=entry.params, house=entry.house, llm_mode=config.llm_policy)
        else:
            drv = load_driver(entry.source, isolate=True, params=entry.params, house=entry.house,
                              log_path=log_path, llm_semaphore=semaphore, env=env,
                              load_timeout=config.load_timeout_s, llm_mode=config.llm_policy, cancel=stop,
                              on_stage=stage)
    except DriverLoadError as exc:
        if holder.get("watchdog"):
            msg = f"no sign of life for too long while loading: {exc}"
            post({"type": "error", "driver": meta["name"], "index": i, "phase": "load", "traceback": msg})
            return _Outcome(i, entry, meta, status="DNF (timeout: loading)", error=msg, log_path=log_path)
        if stop.is_set():
            return _Outcome(i, entry, meta, status="DNF (stopped)", log_path=log_path)
        post({"type": "error", "driver": meta["name"], "index": i, "phase": "load", "traceback": str(exc)})
        return _Outcome(i, entry, meta, status="DNF (load error)", error=str(exc), log_path=log_path)
    holder["driver"] = drv
    label = _label_for(labels, i, drv.name)
    meta = {**drv.meta(), "id": slug(label)}      # result id "max-2" <-> log file Max-2.jsonl
    phases, laps = config.schedule()
    sc = SessionConfig(
        track=track, attempts=len(phases), phases=phases, attempt_laps=laps, llm_policy=config.llm_policy,
        physics=config.physics, drive_timeout_s=None if inprocess else config.drive_timeout_s,
        plan_timeout_s=config.plan_timeout_s, reflect_timeout_s=config.reflect_timeout_s,
        driver_budget_s=config.driver_timeout_s, reflect_after_last=False, keep_obs=False,
        label=label,
    )
    session = None
    last_post = 0.0
    try:
        for ev in iter_session(drv, sc, stop_event=stop):
            if ev.type == "tick":
                f = ev.data
                now = time.monotonic()
                if now - last_post >= PROGRESS_INTERVAL_S or f["events"]:
                    last_post = now
                    post({"type": "progress", "driver": drv.name, "index": i, "phase": ev.phase,
                          "attempt": ev.attempt, "tick": int(f["tick"]), "lap": int(f["lap"]),
                          "race_progress": round(float(f["race_progress"]), 4), "speed": round(float(f["speed"]), 2),
                          "t": round(float(f["t"]), 2)})
            elif ev.type == "plan":
                post({"type": "progress", "driver": drv.name, "index": i, "phase": "plan", "attempt": 0, "tick": 0,
                      "lap": 0, "race_progress": 0.0, "speed": 0.0, "t": 0.0,
                      "seconds": round(ev.data["seconds"], 2), "error": ev.data["error"]})
            elif ev.type == "attempt_start":
                post({"type": "progress", "driver": drv.name, "index": i, "phase": ev.phase, "attempt": ev.attempt,
                      "tick": 0, "lap": 1, "race_progress": 0.0, "speed": 0.0, "t": 0.0})
            elif ev.type == "attempt_end":
                r = ev.data.result
                post({"type": "attempt", "driver": drv.name, "index": i, "phase": ev.phase, "attempt": ev.attempt,
                      "completed": r.completed, "race_time": r.race_time, "best_lap": r.best_lap,
                      "crashes": r.crashes, "status": ev.data.status})
            elif ev.type == "session_end":
                session = ev.data
    finally:
        drv.close()
    status = session.status if session else "DNF (driver crashed)"
    if holder.get("watchdog"):
        status = "DNF (timeout)"
    out = _Outcome(i, entry, meta, session=session, status=status, log_path=getattr(drv, "log_path", None),
                   label=label)
    race = out.report("race")
    post({"type": "done", "driver": drv.name, "index": i, "status": (race.status if race else status) or "finished",
          "race_time": race.result.race_time if race else None})
    return out


# ============================================================================ results
def _llm_numbers(o: _Outcome) -> tuple[int, int]:
    u = o.session.llm if o.session else {}
    return int(u.get("calls", 0)), int(u.get("prompt_tokens", 0)) + int(u.get("completion_tokens", 0))


def _car_status(o: _Outcome, report: AttemptReport | None) -> str:
    if report is not None and report.result.completed:
        return ""
    if o.status:
        return o.status
    return report.status if report is not None and report.status else "DNF"


def _assemble(outcomes: list[_Outcome], config: GrandPrixConfig, track: Track, wall: float) -> GrandPrixResult:
    cfg = config.physics or PhysicsConfig()
    by_car: dict[int, _Outcome] = {}

    # --- qualifying: the qualifying lap sets the grid (DNF -> back of the grid)
    quali_cars: list[ReplayCar] = []
    quali_time: dict[int, float | None] = {}
    for o in outcomes:
        rep = o.report("qualifying") if config.qualifying else None
        quali_time[o.index] = rep.result.race_time if rep is not None and rep.result.completed else None
        if config.qualifying:
            calls, tokens = _llm_numbers(o)
            extra = {"llm_calls": calls, "llm_tokens": tokens, "qualifying_time": quali_time[o.index],
                     "status": _car_status(o, rep)}
            car = replay_car_from_trace(o.meta, rep, extra) if rep is not None else \
                placeholder_car(o.meta, track, extra["status"] or "DNF (not started)", extra)
            by_car[id(car)] = o
            quali_cars.append(car)
    quali_replay = None
    grid: dict[int, int] = {}
    qualifying_rows: list[dict] = []
    if config.qualifying:
        quali_replay = build_replay(track, quali_cars, title=f"{config.title} - Qualifying", kind="race", laps=1,
                                    tick_seconds=cfg.tick_seconds, max_speed=cfg.max_speed)
        for car in quali_replay.cars:
            o = by_car[id(car)]
            grid[o.index] = car.position
            practice = [r for r in (o.session.attempts if o.session else []) if r.phase == "practice"]
            laps = [t for r in practice for t in r.result.lap_times]
            rep = o.report("qualifying")
            qualifying_rows.append({
                "position": car.position, "name": car.name, "id": car.id, "author": car.author, "kind": car.kind,
                "house": bool(o.entry.house or o.meta.get("house")), "qualifying_time": quali_time[o.index],
                "best_lap": min(laps + ([quali_time[o.index]] if quali_time[o.index] else []), default=None),
                "practice_times": [r.result.race_time for r in practice],
                "crashes": rep.result.crashes if rep else 0,
                "attempts": len(practice) + (1 if rep else 0), "status": car.status,
            })
    pole = qualifying_rows[0]["qualifying_time"] if qualifying_rows else None

    # --- the race
    race_cars: list[ReplayCar] = []
    for o in outcomes:
        rep = o.report("race")
        calls, tokens = _llm_numbers(o)
        extra = {"llm_calls": calls, "llm_tokens": tokens, "qualifying_time": quali_time.get(o.index),
                 "status": _car_status(o, rep)}
        car = replay_car_from_trace(o.meta, rep, extra) if rep is not None else \
            placeholder_car(o.meta, track, extra["status"] or "DNF (not started)", extra)
        car.house = bool(o.entry.house or o.meta.get("house"))
        by_car[id(car)] = o
        race_cars.append(car)
    race_replay = build_replay(track, race_cars, title=config.title, kind="race", laps=max(1, config.race_laps),
                               tick_seconds=cfg.tick_seconds, max_speed=cfg.max_speed)
    standings: list[dict] = []
    winner_time = race_replay.cars[0].race_time if race_replay.cars and race_replay.cars[0].finished else None
    for car in race_replay.cars:
        o = by_car[id(car)]
        rep = o.report("race")
        gap = None
        if car.finished and winner_time is not None and car.race_time is not None:
            gap = round(car.race_time - winner_time, 3)
        standings.append({
            "position": car.position, "id": car.id, "name": car.name, "author": car.author, "kind": car.kind,
            "model": car.model, "color": car.color, "house": car.house, "race_time": car.race_time, "gap": gap,
            "gap_text": ("" if gap is None or gap == 0 else f"+{gap:.2f}s") if car.finished else (car.status or "DNF"),
            "laps_completed": car.laps_completed, "race_progress": round(car.final_progress, 4),
            "waypoints_passed": rep.result.waypoints_passed if rep else 0, "best_lap": car.best_lap,
            "crashes": car.crashes, "penalty_time": car.penalty_time, "status": car.status,
            "llm_calls": car.llm_calls, "llm_tokens": car.llm_tokens, "qualifying_time": car.qualifying_time,
            "grid": grid.get(o.index), "pole": pole is not None and car.qualifying_time == pole,
        })

    errors, drivers, sessions = [], [], {}
    for o in outcomes:
        s = o.session
        first = o.error or (s.driver_errors[0]["error"] if s and s.driver_errors else None) or             (s.status if s else None)
        if first:
            errors.append({"driver": o.meta["name"], "status": _car_status(o, o.report("race")), "error": first})
        if s is not None:
            key, k = o.label or o.meta["name"], 1
            while key in sessions:              # two cars with the same NAME keep both sessions
                k += 1
                key = f"{o.meta['name']}-{k}"
            sessions[key] = s
        drivers.append({
            **{k: o.meta.get(k) for k in ("name", "author", "color", "kind", "model", "source", "motto", "anatomy")},
            "house": bool(o.entry.house or o.meta.get("house")), "status": o.status, "log": o.log_path,
            "id": o.meta.get("id"), "llm_log": log_file_name(o.label) if o.label else None,
            "plan_error": s.plan_error if s else None, "plan_seconds": round(s.plan_seconds, 3) if s else None,
            "llm": dict(s.llm) if s else {}, "think_seconds_mean": round(s.think_seconds_mean, 5) if s else None,
            "attempts": [{"attempt": r.attempt, "phase": r.phase, "completed": r.result.completed,
                          "race_time": r.result.race_time, "best_lap": r.result.best_lap, "crashes": r.result.crashes,
                          "stalls": r.stalls, "driver_errors": r.driver_error_count, "status": r.status,
                          "reflect_error": r.reflect_error} for r in (s.attempts if s else [])],
        })
    return GrandPrixResult(
        title=config.title, track_key=track.key, track_name=track.name, replay=race_replay,
        qualifying_replay=quali_replay, qualifying=qualifying_rows, standings=standings, errors=errors,
        drivers=drivers, config=config.to_dict(), wall_seconds=wall, sessions=sessions,
    )
