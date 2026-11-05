"""Small, cached experiments the lab pages run on demand (pure Python, no st.* calls here).

    from app.lib.experiments import record_ticks, latency_cliff, simulated_latencies, fly_race

record_ticks         one attempt of a driver, tick by tick (what drive() saw and answered) - Sensor X-ray
latency_cliff        the same driver with its answers arriving late (0, 0.5, 1, 2, 3 s) - Latency Lab
simulated_latencies  LLM-like call times for the offline mock (clearly labelled "simulated")
fly_race             the FlyWire brain drives a few ticks; keeps every neuron snapshot - Fly Lab

Each function returns plain JSON-friendly data so the pages can wrap it in @st.cache_data.
"""

from __future__ import annotations

import math
import random
import time
from pathlib import Path
from typing import Any

TICK_S = 0.5


# ------------------------------------------------------------------------------ Sensor X-ray
def record_ticks(driver: str, track: str = "grand_prix", params: dict | None = None,
                 max_ticks: int | None = None) -> dict:
    """One attempt, every tick: {"format": "agentic-gp-ticks", "ticks": [{tick, obs, action, debug, after}]}.

    Same format as races/demo/l2_ticks.json (written by races/demo/make_demo.py).
    """
    from agentic_gp.race.session import SessionConfig, iter_session

    cfg = SessionConfig(track=track, attempts=1, max_ticks_per_lap=max_ticks, reflect_after_last=False)
    ticks, meta = [], {}
    for ev in iter_session(str(driver) if isinstance(driver, Path) else driver, cfg):
        if ev.type == "tick":
            f = ev.data
            act = f["action"]
            ticks.append({"tick": int(f["obs"]["tick"]), "obs": f["obs"],
                          "action": {"throttle": round(act.throttle, 4), "steering": round(act.steering, 4),
                                     "say": act.say},
                          "debug": f.get("debug"), "think_seconds": f.get("think_seconds"),
                          "after": {k: round(float(f[k]), 3) for k in ("x", "y", "heading_deg", "speed", "t")}
                          | {"crashed": bool(f.get("crashed")), "events": list(f.get("events", []))}})
        elif ev.type == "session_end":
            meta = ev.data.driver
    return {"format": "agentic-gp-ticks", "version": 1, "track": track, "driver": meta, "attempt": 1,
            "ticks": ticks}


# ------------------------------------------------------------------------------ Track Lab
def build_track(track: Any):
    """A track spec ("oval", "random:7", "file:x.py"), a Track, or a JSON dict from track_job_spec()."""
    if isinstance(track, dict):
        from agentic_gp.sim.track import track_from_control_points

        return track_from_control_points([tuple(p) for p in track["control_points"]], name=track.get("name", "My Track"),
                                         key=track.get("key", "custom"), half_width=float(track["half_width"]),
                                         n_gates=int(track["n_gates"]))
    return track


def race_one_lap(driver: str, track: Any = "grand_prix", max_ticks: int | None = None) -> dict:
    """One attempt of one driver: {"replay": replay dict (best attempt), "result": {...}} (JSON-friendly).

    `track` may be a dict {"control_points", "name", "half_width", "n_gates"} (a Track Lab build), so
    the job can run in a child process (app.lib.sandbox).
    """
    from agentic_gp.race.session import SessionConfig, run_session

    cfg = SessionConfig(track=build_track(track), attempts=1, max_ticks_per_lap=max_ticks, reflect_after_last=False)
    res = run_session(str(driver), cfg)
    name = res.driver.get("name") if isinstance(res.driver, dict) else Path(driver).stem
    if not res.attempts:                                 # load / plan crash: nothing was driven
        return {"replay": res.to_replay("best").to_dict(),
                "result": {"name": name, "completed": False, "race_time": None, "crashes": 0, "waypoints_passed": 0,
                           "status": res.status or res.plan_error or "DNF"}}
    a = res.attempts[0]
    r = a.result
    return {"replay": res.to_replay("best").to_dict(),
            "result": {"name": name, "completed": bool(r.completed), "race_time": r.race_time,
                       "crashes": int(r.crashes), "waypoints_passed": int(r.waypoints_passed),
                       "status": a.status or res.status or ""}}


# ------------------------------------------------------------------------------ Latency Lab
def run_with_latency(driver: str, track: str, latency_s: float, max_ticks: int = 500,
                     params: dict | None = None) -> dict:
    """Drive one lap while every answer arrives `latency_s` late (exactly the runner's realtime rule).

    The driver sees the world at tick T, "thinks" for latency_s, and meanwhile the car keeps doing
    what it was told last time: floor(latency / 0.5) ticks with the OLD command. Then the new command
    (computed from the old picture) is applied. Returns the lap outcome.
    """
    from agentic_gp.race.loader import load_driver
    from agentic_gp.sim.config import PhysicsConfig
    from agentic_gp.sim.env import RaceEnv

    late = int(math.floor(latency_s / TICK_S + 1e-9))
    env = RaceEnv(track, PhysicsConfig(max_ticks=max_ticks))
    drv = load_driver(driver, params=params)
    try:
        obs = env.reset()
        drv.plan(env.track_map())
        prev = (0.0, 0.0)
        while not obs["attempt_over"]:
            a = drv.drive(obs)
            for _ in range(late):                       # blind: the old command keeps driving
                if obs["attempt_over"]:
                    break
                obs = env.step(*prev)
            if obs["attempt_over"]:
                break
            obs = env.step(a.throttle, a.steering)
            prev = (a.throttle, a.steering)
    finally:
        drv.close()
    return {"latency_s": float(latency_s), "blind_ticks": late, "finished": bool(obs["finished"]),
            "race_time": round(float(obs["race_time"]), 2), "crashes": int(obs["crashes"]),
            "gates": int(obs["waypoints_passed"]), "gates_total": int(obs["waypoints_total"]),
            "ticks": int(obs["tick"])}


def latency_cliff(driver: str, track: str = "grand_prix", latencies=(0.0, 0.5, 1.0, 2.0, 3.0),
                  max_ticks: int = 500) -> list[dict]:
    """run_with_latency for several latencies (the "latency cliff" chart of Ch4)."""
    return [run_with_latency(driver, track, lat, max_ticks) for lat in latencies]


def simulated_latencies(n: int, *, p50: float = 1.3, spread: float = 0.45, seed: int | None = None) -> list[float]:
    """Log-normal call times that look like a real chat model (median p50 s). For the offline mock only."""
    rng = random.Random(seed)
    return [round(math.exp(math.log(p50) + spread * rng.gauss(0, 1)), 3) for _ in range(int(n))]


def timed_llm_calls(n: int, prompt: str, *, effort: str | None = "none", max_tokens: int = 200,
                    on_call=None) -> list[dict]:
    """n real calls through agentic_gp.llm (whatever provider is configured). Each row: seconds, text, error.

    on_call(i, row) is called after every call (progress bar). Errors are recorded, never raised.
    """
    from agentic_gp import llm

    rows = []
    for i in range(int(n)):
        t0 = time.perf_counter()
        try:
            text = llm.chat(prompt, effort=effort, max_tokens=max_tokens)
            row = {"call": i + 1, "seconds": round(time.perf_counter() - t0, 3), "answer": text[:300], "error": None}
        except Exception as exc:                 # noqa: BLE001 - budget / network errors are shown, not raised
            row = {"call": i + 1, "seconds": round(time.perf_counter() - t0, 3), "answer": "",
                   "error": f"{type(exc).__name__}: {exc}"[:300]}
        rows.append(row)
        if on_call is not None:
            on_call(i, row)
        if row["error"] and ("Budget" in row["error"] or "NotAllowed" in row["error"]):
            break
    return rows


# ------------------------------------------------------------------------------ Fly Lab
def fly_available() -> tuple[bool, str]:
    """(True, path) when the FlyWire circuit file exists, else (False, how to get it)."""
    try:
        from agentic_gp.fly.connectome import circuit_path

        p = circuit_path()
        if p.exists():
            return True, str(p)
        return False, f"circuit file not found: {p}"
    except Exception as exc:                     # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def fly_race(track: str = "oval", scramble: int | None = None, max_ticks: int = 120, lesion: str = "",
             seed: int = 0) -> dict:
    """Drive the FlyWire brain (the fly_connectome example driver) for up to max_ticks ticks.

    Returns {"frames": [{tick, t, x, y, heading_deg, speed, throttle, steering, crashed}],
    "snapshots": [FlyBrain snapshot per tick], "finished", "crashes", "gates", "race_time"}.
    """
    import importlib.util

    from agentic_gp.sim.config import PhysicsConfig
    from agentic_gp.sim.env import RaceEnv

    from .paths import DRIVERS_DIR

    path = DRIVERS_DIR / "examples" / "fly_connectome.py"
    spec = importlib.util.spec_from_file_location("agp_fly_connectome_lab", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    drv = mod.Driver()
    drv.PARAMS = dict(drv.PARAMS, SCRAMBLE=scramble, LESION=lesion, seed=seed)
    env = RaceEnv(track, PhysicsConfig(max_ticks=max_ticks))
    obs = env.reset()
    drv.plan(env.track_map())
    if getattr(drv, "brain", None) is None:
        raise FileNotFoundError(getattr(drv, "error", None) or "no fly circuit")
    frames = []
    while not obs["attempt_over"]:
        a = drv.drive(obs)
        px, py = obs["position"]
        frames.append({"tick": int(obs["tick"]), "t": float(obs["race_time"]), "x": float(px), "y": float(py),
                       "heading_deg": float(obs["heading_deg"]), "speed": float(obs["speed"]),
                       "throttle": float(a["throttle"]), "steering": float(a["steering"]), "say": a.get("say"),
                       "angle": float(obs["angle_to_next_waypoint"]), "crashes": int(obs["crashes"])})
        obs = env.step(a["throttle"], a["steering"])
    return {"frames": frames, "snapshots": list(drv.snapshots), "finished": bool(obs["finished"]),
            "crashes": int(obs["crashes"]), "gates": int(obs["waypoints_passed"]),
            "gates_total": int(obs["waypoints_total"]), "race_time": round(float(obs["race_time"]), 2),
            "crash_points": [list(map(float, p)) for p in env.crash_points]}


def json_safe(obj: Any) -> Any:
    """numpy scalars/arrays -> plain Python (for st.json / downloads)."""
    import numpy as np

    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj
