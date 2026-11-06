"""Cached loaders for tracks, replays, sessions and results - so pages stay fast on every rerun.

    from app.lib.loaders import load_replay, list_replays, load_track, load_json, list_results

Streamlit reruns the whole page script on every click. Reading and parsing a 300 KB JSON file
each time would make the app sluggish, so every loader here is wrapped in @st.cache_data (data:
copied out of the cache, safe to modify) or @st.cache_resource (objects shared as-is: tracks).
File loaders take the file's modification time as part of the cache key, so a file that was
rewritten (a new race) is read again automatically.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st

from .paths import DEMO_DIR, RACES_DIR, RESULTS_DIR, rel


def _mtime(path: str | Path) -> float:
    try:
        return Path(path).stat().st_mtime
    except OSError:
        return 0.0


# ------------------------------------------------------------------------------ JSON files
@st.cache_data(show_spinner=False, max_entries=64)
def _load_json(path: str, mtime: float) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_json(path: str | Path) -> Any:
    """Any JSON file (cached until the file changes)."""
    return _load_json(str(Path(path).resolve()), _mtime(path))


def file_kind(data: Any) -> str:
    """What a loaded JSON is: "replay" | "session" | "result" | "grand_prix" | "ticks" | "unknown"."""
    if not isinstance(data, dict):
        return "unknown"
    fmt = data.get("format", "")
    if fmt == "agentic-gp-replay":
        return "replay"
    if fmt == "agentic-gp-grand-prix":
        return "grand_prix"
    if fmt == "agentic-gp-ticks":
        return "ticks"
    if fmt == "agentic-gp-result":
        return "session" if "reports" in data else "result"
    return "unknown"


@st.cache_data(show_spinner=False, max_entries=8, ttl=10)
def _scan(root: str, pattern: str) -> list[str]:
    return sorted(str(p) for p in Path(root).rglob(pattern) if p.is_file())


def list_json(root: str | Path = RACES_DIR, kind: str | None = None) -> list[Path]:
    """JSON files under `root` (newest first), optionally only one kind ("replay", "session", ...).

    The folder scan is cached for 10 s; the kind check reads each file (cached per file).
    """
    paths = [Path(p) for p in _scan(str(root), "*.json")]
    paths.sort(key=_mtime, reverse=True)
    if kind is None:
        return paths
    out = []
    for p in paths:
        try:
            if file_kind(load_json(p)) == kind:
                out.append(p)
        except (OSError, ValueError):
            continue
    return out


def list_replays(root: str | Path = RACES_DIR) -> list[Path]:
    """Every replay JSON under races/ (demo files included), newest first."""
    return list_json(root, "replay")


def replay_label(path: str | Path) -> str:
    """'demo/class_demo.json - Class demo (7 cars)' for selectboxes."""
    try:
        d = load_json(path)
        return f"{rel(path).removeprefix('races/')}  -  {d.get('title', '')} ({len(d.get('cars', []))} car(s))"
    except (OSError, ValueError):
        return rel(path)


# ------------------------------------------------------------------------------ replays
@st.cache_data(show_spinner=False, max_entries=32)
def _load_replay_dict(path: str, mtime: float) -> dict:
    from agentic_gp.race.replay import Replay

    return Replay.from_json(path).to_dict()


def load_replay(path: str | Path):
    """A Replay object (agentic_gp.race.replay.Replay) from a JSON file (cached)."""
    from agentic_gp.race.replay import Replay

    return Replay.from_dict(_load_replay_dict(str(Path(path).resolve()), _mtime(path)))


def replay_from_upload(data: bytes):
    """A Replay from an uploaded file's bytes (st.file_uploader(...).getvalue()).

    Raises ValueError with a readable message for anything that is not a replay JSON (the text is
    never taken for a file path).
    """
    from agentic_gp.race.replay import Replay

    try:
        d = json.loads(data.decode("utf-8-sig"))
    except UnicodeDecodeError:
        raise ValueError("the file is not text (a replay is a .json text file)") from None
    except ValueError as exc:
        raise ValueError(f"the file is not valid JSON ({exc})") from None
    if not isinstance(d, dict) or d.get("format") != "agentic-gp-replay":
        kind = file_kind(d)
        hint = {"session": " - it is a session file: open it on the Telemetry page",
                "result": " - it is a result file: open it on Learning Curves",
                "grand_prix": " - it is a grand prix results file: its replay is the file without _results",
                "ticks": " - it is a tick recording: open it on Sensor X-ray"}.get(kind, "")
        raise ValueError(f"this JSON is not a replay (format {d.get('format') if isinstance(d, dict) else type(d).__name__!r}){hint}")
    return Replay.from_dict(d)


@st.cache_data(show_spinner=False, max_entries=32)
def _payload(path: str, mtime: float) -> dict:
    from agentic_gp.viz.player import player_data

    return player_data(path)


def load_replay_payload(path: str | Path) -> dict:
    """What the replay canvas loads (replay dict + corners/v_max view data), cached per file."""
    return _payload(str(Path(path).resolve()), _mtime(path))


def standings_table(replay: Any, t: float | None = None) -> list[dict]:
    """Rows for st.dataframe: official classification, or the live order at race time `t`."""
    from agentic_gp.race.replay import Replay, standings_at

    rep = replay if isinstance(replay, Replay) else Replay.from_dict(replay)
    cars = {c.id: c for c in rep.cars}
    if t is None:
        lead = rep.cars[0].race_time if rep.cars and rep.cars[0].finished else None
        return [{"pos": c.position, "driver": c.name, "author": c.author, "kind": c.kind,
                 "time": round(c.race_time, 2) if c.finished and c.race_time is not None else None,
                 "gap": round(c.race_time - lead, 2) if c.finished and lead is not None and c.race_time else None,
                 "best lap": c.best_lap, "laps": c.laps_completed, "crashes": c.crashes,
                 "penalty s": c.penalty_time, "status": c.status or ("finished" if c.finished else "DNF")}
                for c in rep.cars]
    rows = []
    for k, (cid, progress, finished, ftime) in enumerate(standings_at(rep, t), start=1):
        c = cars[cid]
        rows.append({"pos": k, "driver": c.name, "race progress": round(progress, 3),
                     "lap": min(rep.laps, int(progress) + 1), "finished": finished,
                     "finish time": ftime})
    return rows


# ------------------------------------------------------------------------------ tracks
@st.cache_resource(show_spinner=False, max_entries=32)
def load_track(spec: str = "grand_prix"):
    """A sim Track for a spec ("grand_prix", "random:7", "file:my_track.py"). Shared, do not modify."""
    from agentic_gp.sim.track import make_track

    return make_track(spec)


@st.cache_data(show_spinner=False, max_entries=32)
def track_map(spec: str = "grand_prix") -> dict:
    """The plan() track dict a driver receives (Track.to_map()) - JSON-friendly."""
    return load_track(spec).to_map()


def track_choices(n_random: int = 3, saved: bool = True) -> list[str]:
    """Track specs for a selectbox: the public tracks, a few random seeds, then the Track Lab builds
    saved in races/tracks/*.py (as "file:races/tracks/<name>.py")."""
    from agentic_gp.sim.track import PUBLIC_TRACKS

    out = list(PUBLIC_TRACKS) + [f"random:{k}" for k in range(1, n_random + 1)]
    return out + (saved_track_specs() if saved else [])


def track_file_spec(path: str | Path) -> str:
    """"file:<path>" for make_track: repo-relative when the app runs from the repo root, else absolute."""
    from .paths import REPO_ROOT

    p = Path(path).resolve()
    try:
        if Path.cwd().resolve() == REPO_ROOT:
            return "file:" + p.relative_to(REPO_ROOT).as_posix()
    except (OSError, ValueError):
        pass
    return f"file:{p}"


def saved_track_specs(folder: str | Path | None = None) -> list[str]:
    """file: specs of the tracks saved by the Track Lab (races/tracks/*.py), newest first."""
    from . import paths

    folder = Path(folder) if folder is not None else paths.RACES_DIR / "tracks"
    if not folder.is_dir():
        return []
    files = sorted((f for f in folder.glob("*.py") if not f.name.startswith("_")), key=_mtime, reverse=True)
    return [track_file_spec(f) for f in files]


# ------------------------------------------------------------------------------ sessions / results
def load_session(path: str | Path) -> dict:
    """A saved session (SessionResult.to_dict(): result fields + "reports" with telemetry)."""
    d = load_json(path)
    if file_kind(d) not in ("session", "result"):
        raise ValueError(f"{rel(path)} is not a session/result file")
    return d


def list_results(root: str | Path = RESULTS_DIR) -> list[Path]:
    """results/*.json (driver results written by race.py --save), newest first."""
    if not Path(root).exists():
        return []
    return [p for p in list_json(root) if file_kind(load_json(p)) in ("result", "session")]


def attempts_table(result: dict) -> list[dict]:
    """One row per attempt of a session/result dict: the learning-curve data."""
    rows = []
    for a in result.get("attempts", []):
        rows.append({"driver": result.get("name"), "attempt": a.get("attempt"), "phase": a.get("phase"),
                     "race_time": a.get("race_time"), "best_lap": a.get("best_lap"), "crashes": a.get("crashes"),
                     "completed": a.get("completed"), "status": a.get("status") or a.get("ended_by")})
    return rows


def telemetry_rows(session: dict) -> list[dict]:
    """All telemetry rows of a session dict (+ "attempt"), ready for pandas.DataFrame."""
    rows = []
    for rep in session.get("reports", []):
        for r in rep.get("telemetry", []):
            rows.append({"attempt": rep.get("attempt"), **r})
    return rows


def demo_file(name: str) -> Path:
    """races/demo/<name> (raises a helpful error when the demo data was not generated yet)."""
    p = DEMO_DIR / name
    if not p.exists():
        raise FileNotFoundError(f"{rel(p)} is missing - run: python races/demo/make_demo.py")
    return p
