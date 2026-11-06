"""Demo data for the Streamlit lessons - tracks, sessions, replays - all cached.

Every lesson starts with `import _data`. That one line does four things for you:

1. it checks your Streamlit version (the canvases need a recent one) and says how to upgrade;
2. it puts the repo root on `sys.path`, so `import agentic_gp` and `import app` work;
3. it gives you ready-made, cached demo data read from `races/demo/` (made by
   `python races/demo/make_demo.py`), so no lesson depends on YOUR driver working;
4. if a demo file is missing, it computes the data on the fly with agentic_gp (code drivers run
   a lap in well under a second), so the lessons never break.

You do not need to read this file during the lesson - but it is a nice example of
`@st.cache_data` (data, copied out of the cache) and `@st.cache_resource` (shared objects).
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from pathlib import Path

import streamlit as st

# ------------------------------------------------------------------------------ version check
# Tested with 1.66. Older versions break some lessons (1.50: no st.components.v2 for the canvases,
# 1.51: no button shortcuts), so an old Streamlit stops here with a clear message instead.
MIN_STREAMLIT = (1, 66)
if tuple(int(n) for n in re.findall(r"\d+", st.__version__)[:2]) < MIN_STREAMLIT:
    raise RuntimeError(
        f"These lessons need Streamlit >= 1.66, you have {st.__version__}. Upgrade it in the "
        'course venv:  python -m pip install -U "streamlit>=1.66"  (uv users: uv pip install -U '
        '"streamlit>=1.66"), then start the app again.')

ROOT = Path(__file__).resolve().parents[1]  # streamlit_lessons/_data.py -> repo root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("LLM_PROVIDER", "mock")  # lessons never spend the class LLM budget

DEMO = ROOT / "races" / "demo"
RACES = ROOT / "races"
RESULTS = ROOT / "results"
CACHE = Path(__file__).resolve().parent / ".cache"  # replays generated when races/demo is empty
EXAMPLES = ROOT / "drivers" / "examples"
TRACKS = ["grand_prix", "sprint", "speedway", "oval", "briefing_demo", "random:7"]


def rel(path: str | Path) -> str:
    """'drivers/examples/l1_reflex.py' - a short, repo-relative name for selectboxes."""
    p = Path(path).resolve()
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return str(p)


def _read(name: str) -> dict | None:
    p = DEMO / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _session(driver: str | Path, track: str, attempts: int = 1, params: dict | None = None):
    """Run one driver in this process (a SessionResult). Code drivers: well under a second."""
    from agentic_gp import SessionConfig, load_driver, run_session

    drv = load_driver(str(driver), params=params or None)
    return run_session(drv, SessionConfig(track=track, attempts=attempts))


# ------------------------------------------------------------------------------ tracks
@st.cache_resource(show_spinner=False)
def track(spec: str = "grand_prix"):
    """A sim Track ("grand_prix", "sprint", "random:7", ...). Shared by every rerun - read only."""
    from agentic_gp import make_track

    return make_track(spec)


# ------------------------------------------------------------------------------ S01
@st.cache_data(show_spinner=False)
def reference_laps() -> dict:
    """Best lap on grand_prix per ladder level, from the class demo race (None = DNF)."""
    cars = {c["id"]: c for c in load_json(str(class_demo()))["cars"]}
    best = {cid: cars[cid].get("best_lap") if cid in cars else None
            for cid in ("reflex", "planner", "max")}
    return {"L0 Dummy": None, "L1 Reflex": best["reflex"], "L2 Planner": best["planner"],
            "Max (house)": best["max"]}


@st.cache_data(show_spinner=False)
def sample_obs(tick: int = 20) -> dict:
    """A real observation: what the planner saw at one tick on grand_prix (all of OBS_FIELDS)."""
    ticks = _read("l2_ticks.json")
    if ticks and len(ticks["ticks"]) > tick:
        return ticks["ticks"][tick]["obs"]
    from agentic_gp import RaceEnv

    env = RaceEnv("grand_prix")
    obs = env.reset()
    for _ in range(tick):
        obs = env.step(1.0, 0.0 if obs["speed"] < 20 else -0.1)
    return obs


# ------------------------------------------------------------------------------ S02
def half_circle(r: float, side: float = 1.0) -> tuple[list, list]:
    """x, y of a half circle of radius r from the car at (0, 0), heading "up" (side +1 = right)."""
    angles = [math.pi * k / 60 for k in range(61)]
    return [side * r * (1 - math.cos(a)) for a in angles], [r * math.sin(a) for a in angles]


# ------------------------------------------------------------------------------ S04 / S05
@st.cache_data(show_spinner="Running 4 practice laps ...")
def practice() -> dict:
    """A practice day: Reflex on sprint, 4 runs with different PARAMS - crashes first, then faster.

    One session dict (like SessionResult.to_dict()): "attempts" (one summary row per run) and
    "reports" (telemetry, segments, crash_log, messages per run). Computed fresh (< 1 s).
    """
    tries = [{"sight": 0}, {"sight": 0, "cap": 28}, {}, {"cap": 38, "turn_k": 1.0}]
    runs = [_session(EXAMPLES / "l1_reflex.py", "sprint", 1, p).to_dict() for p in tries]
    session = dict(runs[0], name="Reflex (practice)", attempts=[], reports=[])
    for n, (run, params) in enumerate(zip(runs, tries), start=1):
        note = ", ".join(f"{k}={v}" for k, v in params.items()) or "default PARAMS"
        session["attempts"].append(dict(run["attempts"][0], attempt=n, params=note))
        session["reports"].append(dict(run["reports"][0], attempt=n, params=note))
    session["best_lap"] = min(a["race_time"] for a in session["attempts"] if a["completed"])
    session["total_crashes"] = sum(a["crashes"] for a in session["attempts"])
    return session


def telemetry_rows(session: dict) -> list[dict]:
    """Every telemetry row of a session (one per tick, with its "attempt" number) - for pandas.

    A crash costs a time penalty with NO telemetry (the car stands still). So that a chart shows
    that, each crash adds a row at the moment of impact (speed 0, controls 0, crash=True) and one
    at the end of the penalty (the same, crash=False, just before the car drives on). Every
    normal row has crash=False, so lap[lap["crash"]] are the impacts.
    """
    rows = []
    for rep in session["reports"]:
        normal = [{"attempt": rep["attempt"], **row, "crash": False} for row in rep["telemetry"]]
        stops = []
        for c in rep["crash_log"]:
            after = [r["t"] for r in normal if r["t"] > c["t"]]  # the car drives on at after[0]
            for t, impact in [(c["t"], True)] + ([(after[0], False)] if after else []):
                stops.append({"attempt": rep["attempt"], "tick": c["tick"], "t": t, "lap": c["lap"],
                              "segment": c["segment"], "x": c["position"][0], "y": c["position"][1],
                              "speed": 0.0, "throttle": 0.0, "steering": 0.0, "crash": impact})
        rows += sorted(stops + normal, key=lambda r: r["t"])  # at equal t: the stop row first
    return rows


# ------------------------------------------------------------------------------ S06 / S08 / S09
def driver_files() -> list[Path]:
    """Code drivers you can run here: drivers/submissions first, then the examples, the template.

    LLM and fly drivers are left out on purpose (the lessons are about Streamlit, not latency).
    """
    from app.lib.drivers import discover_drivers

    found = [d for d in discover_drivers()
             if d.kind == "code" and d.style != "none" and not d.syntax_error]
    return [d.path for d in found]


def default_index(files: list[Path] | None = None, fallback: str = "l2_planner") -> int:
    """Index in driver_files() to preselect: your first submission, else the L2 planner, else 0."""
    files = driver_files() if files is None else files
    for k, f in enumerate(files):
        if f.parent.name == "submissions":
            return k
    return next((k for k, f in enumerate(files) if f.stem == fallback), 0)


def driver_info(path: str | Path):
    """Static facts about a driver file (NAME, COLOR, PARAMS, the comment next to each PARAM)."""
    from app.lib.drivers import inspect_driver

    return inspect_driver(Path(path))


@st.cache_data(show_spinner="Driving one lap ...", max_entries=16)
def frames(driver: str, track_spec: str = "sprint", mtime: float = 0.0) -> dict:
    """One lap of `driver`, tick by tick (pass the file's mtime so an edited driver drives again).

    {"name", "color", "completed", "race_time", "frames": [...]}. Frame k is decision k: "obs" is
    what the driver saw (tick k, race clock t), "throttle"/"steering"/"say"/"debug" what it
    answered, and "x"/"y"/"heading_deg"/"speed" where that answer took the car. The final frame
    also has "status": "finished" (lap completed) or "stopped" (the attempt ended without it).
    """
    result = _session(driver, track_spec)
    attempt = result.attempts[0]
    keep = ("x", "y", "heading_deg", "speed", "throttle", "steering", "lap", "say", "debug",
            "crashed", "obs", "think_seconds")
    rows = [{"tick": f["obs"]["tick"], "t": f["obs"]["race_time"], **{k: f.get(k) for k in keep}}
            for f in attempt.frames if f.get("obs")]  # frame 0 is the grid: no decision yet
    summary = attempt.result
    if rows:
        rows[-1]["status"] = "finished" if summary.completed else "stopped"
    return {"name": result.driver["name"], "color": result.driver["color"], "frames": rows,
            "completed": bool(summary.completed), "race_time": summary.race_time}


# ------------------------------------------------------------------------------ S10
@st.cache_data(show_spinner=False, max_entries=32)
def load_json(path: str, mtime: float | None = None) -> dict:
    """Any JSON file (cached; pass the mtime if the file can change while the app runs)."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def class_demo() -> Path:
    """races/demo/class_demo.json - or, if it is missing, the same race generated once (~3 s).

    The fallback races every example driver (no dummy, no flies) for 2 laps of grand_prix, with
    the crew chief as the house car "Max" - like races/demo/make_demo.py. The LLM drivers only
    join when LLM_PROVIDER=mock (offline, free); with a real provider the race is code-only.
    """
    path = DEMO / "class_demo.json"
    if path.exists():
        return path
    path = CACHE / "class_demo.json"
    if not path.exists():
        from agentic_gp.race.grand_prix import DriverEntry, GrandPrixConfig, run_grand_prix

        mock = os.environ.get("LLM_PROVIDER", "").lower() == "mock"
        files = [f for f in sorted(EXAMPLES.glob("l*.py"))
                 if f.stem != "l0_dummy" and (mock or "_llm_" not in f.stem)]
        entries = [DriverEntry(f, house="crew_chief" in f.stem) for f in files]
        cfg = GrandPrixConfig(track="grand_prix", practice_attempts=0, qualifying=False,
                              race_laps=2, title="Class demo (generated)")
        CACHE.mkdir(parents=True, exist_ok=True)
        run_grand_prix(entries, cfg, inprocess=True).replay.to_json(path)
    return path


def _starts_with(path: Path, marker: str) -> bool:
    try:
        with open(path, encoding="utf-8") as fh:
            return marker in fh.read(120)
    except OSError:
        return False


def replay_files() -> list[Path]:
    """Every replay JSON under races/ (the class demo first, then newest first)."""
    demo = class_demo()
    found = sorted((p for p in RACES.rglob("*.json")
                    if p != demo and _starts_with(p, '"agentic-gp-replay"')),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    return [demo] + found


def result_files() -> list[Path]:
    """results/*.json written by `python race.py drive <file> --track sprint --attempts 6 --save`.

    Newest first. Only files in the current format ("agentic-gp-result") count: old files from
    the v1 course are skipped. Nothing saved yet? You get the demo learner session, so the page
    still shows a curve.
    """
    files = sorted(RESULTS.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True) \
        if RESULTS.exists() else []
    files = [p for p in files if _starts_with(p, '"agentic-gp-result"')]
    if files:
        return files
    demo = DEMO / "learner_session.json"
    if not demo.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        demo = CACHE / "learner_session.json"
        if not demo.exists():
            result = _session(EXAMPLES / "l3_learner.py", "sprint", 5).to_result_dict()
            demo.write_text(json.dumps(result), encoding="utf-8")
    return [demo]
