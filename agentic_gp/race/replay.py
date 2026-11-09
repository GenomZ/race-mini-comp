"""Replay schema v1: THE contract between the runner and every viewer (Streamlit, HTML player, Unity).

    from agentic_gp.race.replay import Replay, standings_at, car_state_at
    replay = session_result.to_replay("best")          # or GrandPrixResult.replay
    replay.to_json("races/my_lap.json")
    replay = Replay.from_json("races/final.json")
    problems = replay.validate()                        # [] = valid
    standings_at(replay, t=30.0)                        # [(car_id, race_progress, finished, finish_time), ...]
    car_state_at(replay.cars[0], t=30.0)                # interpolated x, y, heading_deg, speed, ...

The JSON is built for Unity's JsonUtility: no nested arrays, no dicts with dynamic keys, no
top-level array. Points are parallel float arrays, every number is finite, keys are fixed.

Time: `t` is the race clock (sim time + penalties). When a tick contains a crash, the car gets three
frames: a "contact" frame at the wall (the crash point, CONTACT_LEAD_S before the crash time), the
crash frame (the respawn pose, race clock BEFORE the penalty) and a "penalty hold" frame at the same
place with speed 0, `penalty` seconds later (the same `t` when the penalty is 0). Viewers that
interpolate by `t` therefore show the car hit the wall, jump back onto the track and stand still for
its penalty, and the on-screen order always equals the official order.

Additive fields (amendment A15, viewers may ignore them): track.zones, car motto / anatomy / house,
event kinds "zone" and "stall". `to_dict(extended=False)` leaves them out (strict v1 for old tools).
"""

from __future__ import annotations

import bisect
import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

FORMAT = "agentic-gp-replay"
VERSION = 1
EVENT_KINDS = ("crash", "lap", "finish", "dnf", "driver_error", "late")
EXTENDED_EVENT_KINDS = ("zone", "stall")
FRAME_KEYS = ("tick", "t", "x", "y", "heading_deg", "speed", "throttle", "steering", "lap", "race_progress")
_ROUND = {"tick": None, "t": 2, "x": 2, "y": 2, "heading_deg": 1, "speed": 2, "throttle": 3, "steering": 3,
          "lap": None, "race_progress": 4}
MAX_DRIVER_ERROR_EVENTS = 20
CONTACT_LEAD_S = 0.05         # the wall-contact frame comes this long before a crash frame


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _r(v: Any, nd: int | None) -> float | int:
    """Round + make finite (JSON must never contain NaN/inf)."""
    if v is None:
        return 0
    f = float(v)
    if not math.isfinite(f):
        f = 0.0
    return int(round(f)) if nd is None else round(f, nd)


def _opt(v: Any, nd: int = 3) -> float | None:
    if v is None:
        return None
    f = float(v)
    return round(f, nd) if math.isfinite(f) else None


# ============================================================================ dataclasses
@dataclass
class ReplayEvent:
    t: float
    tick: int
    kind: str              # crash | lap | finish | dnf | driver_error | late (+ zone | stall)
    lap: int = 1
    x: float = 0.0
    y: float = 0.0
    value: float = 0.0     # crash: penalty s; lap: lap time; finish: race time; dnf: race progress; late: ticks
    text: str = ""

    def to_dict(self) -> dict:
        return {"t": _r(self.t, 2), "tick": int(self.tick), "kind": self.kind, "lap": int(self.lap),
                "x": _r(self.x, 2), "y": _r(self.y, 2), "value": _r(self.value, 3), "text": str(self.text)}


@dataclass
class ReplayMessage:
    t: float
    tick: int
    text: str

    def to_dict(self) -> dict:
        return {"t": _r(self.t, 2), "tick": int(self.tick), "text": str(self.text)}


@dataclass
class ReplayTrack:
    key: str
    name: str
    half_width: float
    length: float
    center_x: list[float]
    center_y: list[float]
    left_x: list[float]
    left_y: list[float]
    right_x: list[float]
    right_y: list[float]
    gate_x1: list[float]
    gate_y1: list[float]
    gate_x2: list[float]
    gate_y2: list[float]
    start_x: float
    start_y: float
    start_heading_deg: float
    zones: list[dict] = field(default_factory=list)   # A15: {"from_index","to_index","max_speed","label"}

    @classmethod
    def from_track(cls, track: Any) -> "ReplayTrack":
        """From a sim Track (anything with centerline/left_wall/right_wall/gates/start_*)."""
        def col(a, j):
            return [round(float(v), 2) for v in a[:, j]]

        sx, sy = track.start_position
        zones = track.zone_ranges() if hasattr(track, "zone_ranges") else []
        return cls(
            key=str(getattr(track, "key", "custom")), name=str(track.name), half_width=float(track.half_width),
            length=round(float(track.length), 1),
            center_x=col(track.centerline, 0), center_y=col(track.centerline, 1),
            left_x=col(track.left_wall, 0), left_y=col(track.left_wall, 1),
            right_x=col(track.right_wall, 0), right_y=col(track.right_wall, 1),
            gate_x1=col(track.gates, 0), gate_y1=col(track.gates, 1),
            gate_x2=col(track.gates, 2), gate_y2=col(track.gates, 3),
            start_x=round(float(sx), 2), start_y=round(float(sy), 2),
            start_heading_deg=round(math.degrees(track.start_heading), 1),
            zones=[dict(z) for z in zones],
        )

    def to_dict(self, extended: bool = True) -> dict:
        d = {k: getattr(self, k) for k in (
            "key", "name", "half_width", "length", "center_x", "center_y", "left_x", "left_y", "right_x",
            "right_y", "gate_x1", "gate_y1", "gate_x2", "gate_y2", "start_x", "start_y", "start_heading_deg")}
        if extended:
            d["zones"] = [dict(z) for z in self.zones]
        return d


@dataclass
class ReplayCar:
    id: str
    name: str
    author: str = "anonymous"
    color: str = "#8d99ae"
    kind: str = "code"
    model: str | None = None
    finished: bool = False
    position: int = 0
    race_time: float | None = None
    laps_completed: int = 0
    crashes: int = 0
    penalty_time: float = 0.0
    best_lap: float | None = None
    lap_times: list[float] = field(default_factory=list)
    ended_by: str = ""
    qualifying_time: float | None = None
    llm_calls: int = 0
    llm_tokens: int = 0
    status: str = ""
    tick: list[int] = field(default_factory=list)
    t: list[float] = field(default_factory=list)
    x: list[float] = field(default_factory=list)
    y: list[float] = field(default_factory=list)
    heading_deg: list[float] = field(default_factory=list)
    speed: list[float] = field(default_factory=list)
    throttle: list[float] = field(default_factory=list)
    steering: list[float] = field(default_factory=list)
    lap: list[int] = field(default_factory=list)
    race_progress: list[float] = field(default_factory=list)
    events: list[ReplayEvent] = field(default_factory=list)
    messages: list[ReplayMessage] = field(default_factory=list)
    motto: str = ""                                 # A15
    anatomy: list[str] = field(default_factory=list)
    house: bool = False

    @property
    def n_frames(self) -> int:
        return len(self.t)

    @property
    def final_progress(self) -> float:
        return float(self.race_progress[-1]) if self.race_progress else 0.0

    def to_dict(self, extended: bool = True) -> dict:
        kinds = EVENT_KINDS + (EXTENDED_EVENT_KINDS if extended else ())
        d = {
            "id": self.id, "name": self.name, "author": self.author, "color": self.color, "kind": self.kind,
            "model": self.model, "finished": bool(self.finished), "position": int(self.position),
            "race_time": _opt(self.race_time), "laps_completed": int(self.laps_completed),
            "crashes": int(self.crashes), "penalty_time": _r(self.penalty_time, 3), "best_lap": _opt(self.best_lap),
            "lap_times": [_r(v, 3) for v in self.lap_times], "ended_by": self.ended_by,
            "qualifying_time": _opt(self.qualifying_time), "llm_calls": int(self.llm_calls),
            "llm_tokens": int(self.llm_tokens), "status": self.status,
        }
        for k in FRAME_KEYS:
            d[k] = [_r(v, _ROUND[k]) for v in getattr(self, k)]
        d["events"] = [e.to_dict() for e in self.events if e.kind in kinds]
        d["messages"] = [m.to_dict() for m in self.messages]
        if extended:
            d["motto"] = self.motto
            d["anatomy"] = list(self.anatomy)
            d["house"] = bool(self.house)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ReplayCar":
        kw = {k: d[k] for k in cls.__dataclass_fields__ if k in d and k not in ("events", "messages")}
        car = cls(**kw)
        car.events = [ReplayEvent(**{k: e[k] for k in ReplayEvent.__dataclass_fields__ if k in e})
                      for e in d.get("events", [])]
        car.messages = [ReplayMessage(**{k: m[k] for k in ReplayMessage.__dataclass_fields__ if k in m})
                        for m in d.get("messages", [])]
        return car


@dataclass
class Replay:
    title: str
    track: ReplayTrack
    cars: list[ReplayCar]
    kind: str = "race"                 # "race" | "session"
    laps: int = 1
    tick_seconds: float = 0.5
    max_speed: float = 40.0
    created_utc: str = field(default_factory=utc_now)

    def to_dict(self, extended: bool = True) -> dict:
        return {
            "format": FORMAT, "version": VERSION, "title": self.title, "created_utc": self.created_utc,
            "kind": self.kind, "laps": int(self.laps), "tick_seconds": float(self.tick_seconds),
            "max_speed": float(self.max_speed), "track": self.track.to_dict(extended),
            "cars": [c.to_dict(extended) for c in self.cars],
        }

    def to_json(self, path: str | Path | None = None, extended: bool = True) -> str:
        """JSON text (compact); also written to `path` if given (folders are created)."""
        text = json.dumps(self.to_dict(extended), separators=(",", ":"), allow_nan=False)
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(text, encoding="utf-8")
        return text

    @classmethod
    def from_dict(cls, d: dict) -> "Replay":
        if d.get("format") != FORMAT:
            raise ValueError(f"not an {FORMAT} file (format={d.get('format')!r})")
        tr = d["track"]
        track = ReplayTrack(**{k: tr[k] for k in ReplayTrack.__dataclass_fields__ if k in tr})
        return cls(title=d.get("title", ""), track=track, cars=[ReplayCar.from_dict(c) for c in d.get("cars", [])],
                   kind=d.get("kind", "race"), laps=int(d.get("laps", 1)),
                   tick_seconds=float(d.get("tick_seconds", 0.5)), max_speed=float(d.get("max_speed", 40.0)),
                   created_utc=d.get("created_utc", utc_now()))

    @classmethod
    def from_json(cls, path_or_text: str | Path) -> "Replay":
        p = Path(path_or_text) if not str(path_or_text).lstrip().startswith("{") else None
        text = p.read_text(encoding="utf-8") if p is not None else str(path_or_text)
        return cls.from_dict(json.loads(text))

    def validate(self, extended: bool = True) -> list[str]:
        return validate_replay_dict(self.to_dict(extended))

    def classify(self) -> "Replay":
        """Sort cars into official order and number them (see classify_cars)."""
        self.cars = classify_cars(self.cars)
        return self

    def car(self, car_id: str) -> ReplayCar:
        return next(c for c in self.cars if c.id == car_id)

    @property
    def duration(self) -> float:
        return max((c.t[-1] for c in self.cars if c.t), default=0.0)


# ============================================================================ builders
def classify_cars(cars: list[ReplayCar]) -> list[ReplayCar]:
    """Official order: finishers by race_time (ties: fewer crashes), then DNFs by race progress."""
    def key(c: ReplayCar):
        if c.finished and c.race_time is not None:
            return (0, c.race_time, c.crashes, 0.0)
        return (1, 0.0, 0, -c.final_progress)

    ordered = sorted(cars, key=key)
    for pos, c in enumerate(ordered, start=1):
        c.position = pos
    return ordered


_TRANSLIT = str.maketrans({"ł": "l", "Ł": "L", "đ": "d", "Đ": "D", "ø": "o", "Ø": "O", "ß": "ss", "æ": "ae",
                           "Æ": "AE", "œ": "oe", "Œ": "OE", "þ": "th", "Þ": "Th", "ı": "i"})


def slug(name: str) -> str:
    """A file/id-safe name: "Rocket 2" -> "rocket-2". Names with non-ASCII letters are transliterated
    where possible and get a short hash, so different names never share a slug ("Żółw" -> "zolw-3f2a9c",
    "鬼" -> "car-1b7e04")."""
    import unicodedata
    import zlib

    text = str(name)
    ascii_text = unicodedata.normalize("NFKD", text.translate(_TRANSLIT)).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9]+", "-", ascii_text.lower()).strip("-") or "car"
    if not text.isascii():
        s = f"{s}-{zlib.crc32(text.encode('utf-8')):08x}"[: len(s) + 7]
    return s


def unique_ids(cars: list[ReplayCar]) -> None:
    seen: dict[str, int] = {}
    for c in cars:
        base = c.id or slug(c.name)
        n = seen.get(base, 0)
        seen[base] = n + 1
        c.id = base if n == 0 else f"{base}-{n + 1}"


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _result_dict(result: Any) -> dict:
    if result is None:
        return {}
    return result if isinstance(result, dict) else result.to_dict()


def replay_car_from_trace(meta: dict, trace: Any, extra: dict | None = None) -> ReplayCar:
    """One car from an attempt trace: env.attempt_traces[i], an AttemptReport, or their dicts.

    `meta`: name, author, color, kind, model (+ id, motto, anatomy, house). `extra`: any ReplayCar
    field to set on top (qualifying_time, llm_calls, llm_tokens, status, ...).
    Adds a wall-contact frame before and a penalty-hold frame after every crash, and turns frame events
    into replay events.
    """
    res = _result_dict(_get(trace, "result"))
    frames = list(_get(trace, "frames", []) or [])
    crash_points = list(_get(trace, "crash_points", []) or [])
    messages = _get(trace, "messages", []) or []
    car = ReplayCar(
        id=str(meta.get("id") or slug(meta.get("name", "car"))), name=str(meta.get("name", "car")),
        author=str(meta.get("author") or "anonymous"), color=str(meta.get("color") or "#8d99ae"),
        kind=str(meta.get("kind") or "code"), model=meta.get("model"),
        motto=str(meta.get("motto") or ""), anatomy=list(meta.get("anatomy") or []), house=bool(meta.get("house")),
        finished=bool(res.get("completed", False)), race_time=res.get("race_time"),
        laps_completed=int(res.get("laps_completed", 0)), crashes=int(res.get("crashes", 0)),
        penalty_time=float(res.get("penalty_time", 0.0)), best_lap=res.get("best_lap"),
        lap_times=list(res.get("lap_times", [])), ended_by=str(res.get("ended_by") or ""),
    )
    lap_times = car.lap_times
    crash_i = 0
    errors_seen = 0
    last_error = None
    prev_pen = 0.0
    for k, f in enumerate(frames):
        pen = float(f.get("penalty_time", prev_pen))
        delta_pen = pen - prev_pen
        prev_pen = pen
        if f.get("crashed"):
            # A crash and a zone violation never share a tick (a crash sets the speed to 0), so the
            # penalty added in this tick is the crash penalty - also when it is configured as 0.
            penalty = max(0.0, delta_pen)
            t_crash = float(f["t"]) - penalty
            have_point = crash_i < len(crash_points)
            cx, cy = crash_points[crash_i] if have_point else (f["x"], f["y"])
            crash_i += 1
            if have_point:                                      # the car visibly reaches the wall ...
                before = frames[k - 1] if k > 0 else f
                t_contact = max(t_crash - CONTACT_LEAD_S, car.t[-1] if car.t else 0.0)
                _add_frame(car, f, t=t_contact, x=cx, y=cy, heading=float(before["heading_deg"]),
                           speed=float(before["speed"]), race_progress=float(before.get("race_progress", 0.0)))
            _add_frame(car, f, t=t_crash)                       # ... the crash: respawn pose, clock before the penalty
            _add_frame(car, f, t=float(f["t"]), speed=0.0)     # penalty hold: same place, speed 0
            car.events.append(ReplayEvent(t_crash, int(f["tick"]), "crash", int(f.get("lap", 1)), cx, cy,
                                          penalty, f"+{penalty:g}s"))
        else:
            _add_frame(car, f)
        t = float(f["t"])
        for ev in f.get("events", []) or []:
            if ev.startswith("lap_complete"):
                m = re.search(r"lap (\d+)", ev)
                n = int(m.group(1)) if m else len([e for e in car.events if e.kind == "lap"]) + 1
                value = lap_times[n - 1] if n - 1 < len(lap_times) else 0.0
                car.events.append(ReplayEvent(t, int(f["tick"]), "lap", n, f["x"], f["y"], value,
                                              f"lap {n} {value:.2f}s"))
            elif ev.startswith("zone_violation"):
                car.events.append(ReplayEvent(t, int(f["tick"]), "zone", int(f.get("lap", 1)), f["x"], f["y"],
                                              max(delta_pen, 0.0), ev.split(": ", 1)[-1][:120]))
        err = f.get("driver_error")
        if err and err != last_error and errors_seen < MAX_DRIVER_ERROR_EVENTS:
            errors_seen += 1
            car.events.append(ReplayEvent(t, int(f["tick"]), "driver_error", int(f.get("lap", 1)), f["x"], f["y"],
                                          0.0, str(err)[:160]))
        last_error = err
        if f.get("stalled"):
            car.events.append(ReplayEvent(t, int(f["tick"]), "stall", int(f.get("lap", 1)), f["x"], f["y"], 0.0,
                                          "drive() too slow - coasting"))
        if f.get("late_ticks"):
            car.events.append(ReplayEvent(t, int(f["tick"]), "late", int(f.get("lap", 1)), f["x"], f["y"],
                                          float(f["late_ticks"]), f"{f['late_ticks']} tick(s) late"))
    if frames:
        last = frames[-1]
        if car.finished:
            car.events.append(ReplayEvent(float(last["t"]), int(last["tick"]), "finish", int(last.get("lap", 1)),
                                          last["x"], last["y"], float(car.race_time or last["t"]), "finished"))
        else:
            car.events.append(ReplayEvent(float(last["t"]), int(last["tick"]), "dnf", int(last.get("lap", 1)),
                                          last["x"], last["y"], float(last.get("race_progress", 0.0)),
                                          f"DNF ({car.ended_by or 'did not finish'})"))
    for m in messages:
        car.messages.append(ReplayMessage(float(_get(m, "t", 0.0)), int(_get(m, "tick", 0)), str(_get(m, "text", ""))))
    for k, v in (extra or {}).items():
        setattr(car, k, v)
    if not car.status and not car.finished:
        car.status = f"DNF ({car.ended_by})" if car.ended_by and car.ended_by != "finished" else "DNF"
    return car


def _add_frame(car: ReplayCar, f: dict, t: float | None = None, speed: float | None = None,
               x: float | None = None, y: float | None = None, heading: float | None = None,
               race_progress: float | None = None) -> None:
    car.tick.append(int(f["tick"]))
    car.t.append(float(f["t"]) if t is None else t)
    car.x.append(float(f["x"]) if x is None else x)
    car.y.append(float(f["y"]) if y is None else y)
    car.heading_deg.append(float(f["heading_deg"]) if heading is None else heading)
    car.speed.append(float(f["speed"]) if speed is None else speed)
    car.throttle.append(float(f.get("throttle", 0.0)))
    car.steering.append(float(f.get("steering", 0.0)))
    car.lap.append(int(f.get("lap", 1)))
    car.race_progress.append(float(f.get("race_progress", 0.0)) if race_progress is None else race_progress)


def placeholder_car(meta: dict, track: Any, status: str, extra: dict | None = None) -> ReplayCar:
    """A car that never drove (load error, not started): one frame on the start grid."""
    sx, sy = track.start_position
    frame = {"tick": 0, "t": 0.0, "x": float(sx), "y": float(sy),
             "heading_deg": math.degrees(track.start_heading), "speed": 0.0, "lap": 1, "race_progress": 0.0}
    car = replay_car_from_trace(meta, {"result": {"ended_by": "not_started"}, "frames": [frame]},
                                {**(extra or {}), "status": status})
    car.events = [e for e in car.events if e.kind != "dnf"] + [
        ReplayEvent(0.0, 0, "dnf", 1, float(sx), float(sy), 0.0, status)]
    return car


def build_replay(track: Any, cars: list[ReplayCar], *, title: str, kind: str = "race", laps: int = 1,
                 tick_seconds: float = 0.5, max_speed: float = 40.0) -> Replay:
    """Assemble, de-duplicate car ids and classify."""
    rt = track if isinstance(track, ReplayTrack) else ReplayTrack.from_track(track)
    unique_ids(cars)
    rep = Replay(title=title, track=rt, cars=list(cars), kind=kind, laps=int(laps), tick_seconds=tick_seconds,
                 max_speed=max_speed)
    return rep.classify()


# ============================================================================ playback helpers
def _interp(a: float, b: float, w: float) -> float:
    return a + (b - a) * w


def _bracket(ts: list[float], t: float) -> tuple[int, int, float]:
    """Indices (i, j) around time t and the weight of j (the later frame wins on equal times)."""
    if not ts:
        raise ValueError("car has no frames")
    if t <= ts[0]:
        return 0, 0, 0.0
    if t >= ts[-1]:
        n = len(ts) - 1
        return n, n, 0.0
    j = bisect.bisect_right(ts, t)
    i = j - 1
    span = ts[j] - ts[i]
    w = 1.0 if span <= 0 else (t - ts[i]) / span
    return i, j, w


def car_state_at(car: ReplayCar | dict, t: float) -> dict:
    """Interpolated state of a car at race-clock time t (heading via the shortest angle)."""
    c = car if isinstance(car, ReplayCar) else ReplayCar.from_dict(car)
    i, j, w = _bracket(c.t, t)
    h0, h1 = c.heading_deg[i], c.heading_deg[j]
    dh = (h1 - h0 + 180.0) % 360.0 - 180.0
    heading = (h0 + dh * w + 180.0) % 360.0 - 180.0
    finished = bool(c.finished and c.race_time is not None and t >= c.race_time - 1e-9)
    return {
        "id": c.id, "t": float(t), "tick": c.tick[i] if w < 0.5 else c.tick[j],
        "x": _interp(c.x[i], c.x[j], w), "y": _interp(c.y[i], c.y[j], w), "heading_deg": heading,
        "speed": _interp(c.speed[i], c.speed[j], w), "throttle": c.throttle[i] if w < 1.0 else c.throttle[j],
        "steering": c.steering[i] if w < 1.0 else c.steering[j], "lap": c.lap[i] if w < 1.0 else c.lap[j],
        "race_progress": _interp(c.race_progress[i], c.race_progress[j], w), "finished": finished,
    }


def standings_at(replay: Replay | dict, t: float) -> list[tuple[str, float, bool, float | None]]:
    """Live order at time t: finished cars by finish time, then everybody else by race progress."""
    rep = replay if isinstance(replay, Replay) else Replay.from_dict(replay)
    rows = []
    for c in rep.cars:
        if not c.t:
            continue
        s = car_state_at(c, t)
        finish_time = float(c.race_time) if s["finished"] else None
        rows.append((c.id, round(s["race_progress"], 4), s["finished"], finish_time))
    rows.sort(key=lambda r: (0, r[3], 0.0) if r[2] else (1, 0.0, -r[1]))
    return rows


# ============================================================================ validation
ROOT_KEYS = ("format", "version", "title", "created_utc", "kind", "laps", "tick_seconds", "max_speed", "track",
             "cars")
TRACK_KEYS = ("key", "name", "half_width", "length", "center_x", "center_y", "left_x", "left_y", "right_x",
              "right_y", "gate_x1", "gate_y1", "gate_x2", "gate_y2", "start_x", "start_y", "start_heading_deg")
CAR_META_KEYS = ("id", "name", "author", "color", "kind", "model", "finished", "position", "race_time",
                 "laps_completed", "crashes", "penalty_time", "best_lap", "lap_times", "ended_by", "qualifying_time",
                 "llm_calls", "llm_tokens", "status")
EVENT_KEYS = ("t", "tick", "kind", "lap", "x", "y", "value", "text")


def _finite(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def validate_replay_dict(d: dict) -> list[str]:
    """Problems with a replay dict ([] = valid). Same rules as unity/tools/make_sample_replay.py,
    except that the A15 additive keys and event kinds are accepted."""
    errs: list[str] = []
    if not isinstance(d, dict):
        return ["root is not a JSON object"]
    for k in ROOT_KEYS:
        if k not in d:
            errs.append(f"root: missing key {k}")
    if d.get("format") != FORMAT or d.get("version") != VERSION:
        errs.append("root: format/version must be agentic-gp-replay / 1")
    if d.get("kind") not in ("race", "session"):
        errs.append("root.kind must be 'race' or 'session'")
    tr = d.get("track") or {}
    for k in TRACK_KEYS:
        if k not in tr:
            errs.append(f"track: missing key {k}")
    n = len(tr.get("center_x", []))
    if n < 3:
        errs.append("track: needs at least 3 centerline points")
    for k in ("center_y", "left_x", "left_y", "right_x", "right_y"):
        if len(tr.get(k, [])) != n:
            errs.append(f"track.{k} length differs from center_x")
    cars = d.get("cars")
    if not isinstance(cars, list) or not cars:
        return errs + ["cars must be a non-empty list"]
    kinds = set(EVENT_KINDS) | set(EXTENDED_EVENT_KINDS)
    positions = []
    for ci, c in enumerate(cars):
        where = f"cars[{ci}]({c.get('name', '?')})"
        for k in CAR_META_KEYS + FRAME_KEYS + ("events", "messages"):
            if k not in c:
                errs.append(f"{where}: missing key {k}")
        positions.append(c.get("position"))
        for k in ("position", "laps_completed", "crashes", "penalty_time", "llm_calls", "llm_tokens"):
            if not _finite(c.get(k)):
                errs.append(f"{where}.{k} must be a finite number")
        for k in ("race_time", "best_lap", "qualifying_time"):
            if c.get(k) is not None and not _finite(c.get(k)):
                errs.append(f"{where}.{k} must be a finite number or null")
        if not re.fullmatch(r"#[0-9a-fA-F]{6}", str(c.get("color", ""))):
            errs.append(f"{where}.color must be '#rrggbb'")
        lens = {k: len(c.get(k, [])) for k in FRAME_KEYS}
        if len(set(lens.values())) != 1:
            errs.append(f"{where}: frame arrays differ in length {lens}")
            continue
        if lens["t"] == 0:
            errs.append(f"{where}: no frames")
            continue
        for k in FRAME_KEYS:
            if not all(_finite(v) for v in c[k]):
                errs.append(f"{where}.{k} has non-finite numbers")
        t = c["t"]
        if any(b < a - 1e-9 for a, b in zip(t, t[1:])):
            errs.append(f"{where}.t is not non-decreasing")
        rp = c["race_progress"]
        if rp[0] > 0.05:
            errs.append(f"{where}.race_progress must start near 0")
        if any(b < a - 0.05 for a, b in zip(rp, rp[1:])):
            errs.append(f"{where}.race_progress drops by more than 0.05 lap")
        if d.get("kind") == "race" and c.get("finished") and abs(rp[-1] - d.get("laps", rp[-1])) > 0.02:
            errs.append(f"{where}: finished but race_progress ends at {rp[-1]:.3f}")
        for e in c.get("events", []):
            if any(k not in e for k in EVENT_KEYS):
                errs.append(f"{where}: event without all of {EVENT_KEYS}")
            if e.get("kind") not in kinds:
                errs.append(f"{where}: unknown event kind {e.get('kind')!r}")
            if e.get("kind") == "crash":
                i = bisect.bisect_left(t, e["t"] - 1e-6)
                ok = (i + 1 < len(t) and abs(t[i] - e["t"]) < 0.02 and abs(t[i + 1] - t[i] - e["value"]) < 0.02
                      and c["x"][i] == c["x"][i + 1] and c["y"][i] == c["y"][i + 1] and c["speed"][i + 1] == 0)
                if not ok:
                    errs.append(f"{where}: crash at t={e['t']} has no penalty hold frame")
    if sorted(p for p in positions if isinstance(p, int)) != list(range(1, len(cars) + 1)):
        errs.append("car positions must be 1..N")
    elif positions != sorted(positions):
        errs.append("cars must be sorted by position")
    return errs
