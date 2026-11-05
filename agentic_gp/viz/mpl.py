"""Static matplotlib figures (headless, Agg): tracks, attempts, sessions, replays, telemetry.

    from agentic_gp.viz.mpl import plot_track, plot_attempt, plot_telemetry
    plot_track(make_track("sprint"), "sprint.png")                  # track + gates + corners + zones
    plot_attempt(env.track, env.attempt_traces[-1], "lap.png")      # trajectory coloured by speed
    plot_telemetry(env.attempt_traces[-1], "telemetry.png")        # speed + controls vs distance
    plot_replay(replay, "race.png")                                 # every car's line in one figure

Inputs are duck-typed so the same functions work on live objects and on JSON:
a track may be a `Track`, a track map dict (`Track.to_map()`), a replay track
(object or dict with center_x/left_x/...) or a whole replay; an attempt may be an
env trace dict ({"frames": [...]}) , a report object with `.frames`, or a report
dict with "frames" or "telemetry".
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

PALETTE = ["#e63946", "#457b9d", "#2a9d8f", "#f4a261", "#8338ec", "#ff006e",
           "#3a86ff", "#fb5607", "#6a994e", "#bc4749", "#118ab2", "#9c6644"]


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """obj[key] for dicts, obj.key for objects."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


# ------------------------------------------------------------------ inputs
def _track_arrays(src: Any) -> dict:
    """Normalise any track-like input to plain numpy arrays."""
    from ..sim.track import Track

    if isinstance(src, Track):
        return {
            "name": src.name, "center": src.centerline, "left": src.left_wall, "right": src.right_wall,
            "gates": src.gates, "gate_centers": src.gate_centers, "length": src.length,
            "half_width": src.half_width, "zones": src.zone_ranges(), "corners": src.corners(),
        }
    if _get(src, "track") is not None and _get(src, "center_x") is None and _get(src, "centerline") is None:
        src = _get(src, "track")                                  # a whole replay
    if _get(src, "center_x") is not None:                         # replay track
        xy = lambda a, b: np.column_stack([np.asarray(_get(src, a), float), np.asarray(_get(src, b), float)])
        gates = np.column_stack([np.asarray(_get(src, k), float) for k in ("gate_x1", "gate_y1", "gate_x2", "gate_y2")])
        return {
            "name": _get(src, "name", ""), "center": xy("center_x", "center_y"), "left": xy("left_x", "left_y"),
            "right": xy("right_x", "right_y"), "gates": gates,
            "gate_centers": np.column_stack([(gates[:, 0] + gates[:, 2]) / 2, (gates[:, 1] + gates[:, 3]) / 2]),
            "length": _get(src, "length", 0.0), "half_width": _get(src, "half_width", 0.0),
            "zones": list(_get(src, "zones", []) or []), "corners": [],
        }
    if _get(src, "centerline") is not None:                       # track map dict
        center = np.asarray(_get(src, "centerline"), float)
        left = np.asarray(_get(src, "left_wall"), float)
        right = np.asarray(_get(src, "right_wall"), float)
        idx = [w.get("centerline_index", 0) for w in _get(src, "waypoints", [])]
        gates = np.column_stack([left[idx], right[idx]]) if idx else np.zeros((0, 4))
        return {
            "name": _get(src, "name", ""), "center": center, "left": left, "right": right, "gates": gates,
            "gate_centers": center[idx] if idx else np.zeros((0, 2)), "length": _get(src, "lap_length", 0.0),
            "half_width": _get(src, "half_width", 0.0), "zones": [], "corners": list(_get(src, "corners", []) or []),
        }
    if isinstance(src, str):
        from ..sim.track import make_track

        return _track_arrays(make_track(src))
    raise TypeError(f"cannot plot a track from {type(src).__name__}")


def _trace_arrays(src: Any) -> dict:
    """Normalise an attempt (env trace, report object/dict, legacy trajectory) to arrays."""
    frames = _get(src, "frames") or _get(src, "telemetry")
    if frames:
        get = lambda k, d=0.0: np.array([float(f.get(k, d) if isinstance(f, dict) else getattr(f, k, d)) for f in frames])
        crashed = np.array([bool(_get(f, "crashed", False)) for f in frames])
        out = {k: get(k) for k in ("x", "y", "speed", "throttle", "steering", "t")}
        out["crashed"] = crashed
    elif _get(src, "trajectory"):                                  # legacy env trace
        traj = np.asarray(_get(src, "trajectory"), float)
        n = len(traj)
        out = {"x": traj[:, 0], "y": traj[:, 1], "speed": traj[:, 2], "throttle": np.zeros(n),
               "steering": np.zeros(n), "t": np.arange(n, dtype=float), "crashed": np.zeros(n, bool)}
    else:
        raise TypeError("attempt has no frames/telemetry")
    pts = _get(src, "crash_points")
    if not pts:
        pts = [] if "crashed" not in out else list(zip(out["x"][out["crashed"]], out["y"][out["crashed"]]))
    out["crash_points"] = np.asarray(pts, float).reshape(-1, 2)
    out["result"] = _get(src, "result")
    return out


def _save(fig, path: str | Path | None):
    if path is None:
        return fig
    plt = _mpl()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


# ------------------------------------------------------------------ drawing
def draw_track(ax, track: Any, labels: bool = True, corners: bool = True) -> None:
    """Walls, grey asphalt, gates (gate 0 red), zones (orange) and corner labels."""
    T = _track_arrays(track)
    L = np.vstack([T["left"], T["left"][:1]])
    R = np.vstack([T["right"], T["right"][:1]])
    ring = np.vstack([L, R[::-1]])
    ax.fill(ring[:, 0], ring[:, 1], color="#e8e8e8", zorder=0, lw=0)
    ax.plot(L[:, 0], L[:, 1], color="#333", lw=1.5)
    ax.plot(R[:, 0], R[:, 1], color="#333", lw=1.5)
    C = T["center"]
    for z in T["zones"]:
        a, b = int(_get(z, "from_index")), int(_get(z, "to_index"))
        idx = np.arange(a, b + 1) if b >= a else np.concatenate([np.arange(a, len(C)), np.arange(0, b + 1)])
        ax.plot(C[idx, 0], C[idx, 1], color="#f4a261", lw=6, alpha=0.7, solid_capstyle="butt", zorder=1)
        m = C[idx[len(idx) // 2]]
        ax.annotate(f"{_get(z, 'label', 'zone')} {_get(z, 'max_speed', 0):g}", m, fontsize=7, color="#9c4a00",
                    xytext=(6, 6), textcoords="offset points")
    for g, (x1, y1, x2, y2) in enumerate(T["gates"]):
        ax.plot([x1, x2], [y1, y2], color="#e63946" if g == 0 else "#2a9d8f", lw=2.5 if g == 0 else 1.2, alpha=0.9)
        if labels:
            cx, cy = T["gate_centers"][g]
            ax.annotate(str(g), (cx, cy), fontsize=7, ha="center", va="center", color="#264653",
                        bbox=dict(boxstyle="circle,pad=0.15", fc="white", ec="#264653", lw=0.6))
    if corners:
        for c in T["corners"]:
            p = C[int(c["apex_index"])]
            ax.annotate(c["id"], p, fontsize=7, color="#6d597a", xytext=(-4, -12), textcoords="offset points")
    sx, sy = C[-1]
    ax.plot(sx, sy, marker="o", color="#e63946", ms=5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def draw_trace(ax, trace: Any, cmap: str = "viridis", vmax: float | None = 40.0, label: str | None = None,
               color: str | None = None):
    """Trajectory of one attempt (dots coloured by speed, or a single colour) + crash crosses."""
    A = _trace_arrays(trace)
    if len(A["x"]) < 2:
        return None
    if color is None:
        sc = ax.scatter(A["x"], A["y"], c=A["speed"], cmap=cmap, s=9, vmin=0, vmax=vmax, zorder=3, label=label)
        ax.plot(A["x"], A["y"], color="#555", lw=0.6, alpha=0.6, zorder=2)
    else:
        sc = ax.plot(A["x"], A["y"], color=color, lw=1.4, alpha=0.9, zorder=3, label=label)[0]
    if len(A["crash_points"]):
        cp = A["crash_points"]
        ax.scatter(cp[:, 0], cp[:, 1], marker="x", color="#e63946", s=45, zorder=4, lw=1.5)
    return sc


def _result_title(res: Any) -> str:
    if res is None:
        return ""
    completed = _get(res, "completed")
    t = _get(res, "race_time")
    laps = _get(res, "laps", 1)
    head = f"attempt {_get(res, 'attempt', '?')}: "
    head += (f"{laps} lap{'s' if laps != 1 else ''} {t:.2f}s" if completed and t is not None else "DNF")
    best = _get(res, "best_lap")
    if best is not None and laps and laps > 1:
        head += f" (best lap {best:.2f}s)"
    return head + f"  |  crashes {_get(res, 'crashes', 0)}  |  waypoints {_get(res, 'waypoints_passed', 0)}"


# ------------------------------------------------------------------ figures
def plot_track(track: Any, path: str | Path | None = None, ax=None, title: str | None = None):
    """Track figure. With `ax` draws into it and returns ax; else saves to `path` (or returns the figure)."""
    if ax is not None:
        draw_track(ax, track)
        return ax
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(8, 8))
    draw_track(ax, track)
    T = _track_arrays(track)
    ax.set_title(title or f"{T['name']}  (lap {float(T['length']):.0f} units, width {2 * float(T['half_width']):g})")
    return _save(fig, path)


def plot_attempt(track: Any, trace: Any, path: str | Path | None = None, title: str | None = None,
                 max_speed: float = 40.0):
    """One attempt on its track, coloured by speed, crashes marked with red crosses."""
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(8, 8))
    draw_track(ax, track, corners=False)
    sc = draw_trace(ax, trace, vmax=max_speed)
    if sc is not None:
        cb = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
        cb.set_label("speed")
    ax.set_title(title or _result_title(_get(trace, "result")) or _track_arrays(track)["name"])
    return _save(fig, path)


def _session_track(session_result: Any) -> Any:
    for obj in (session_result, _get(session_result, "config")):
        tr = _get(obj, "track") if obj is not None else None
        if tr is not None:
            return tr
    return "grand_prix"


def _attempt_time(rep: Any) -> float:
    res = _get(rep, "result")
    t = _get(res, "race_time") if res is not None else None
    return float("inf") if t is None else float(t)


def plot_session(session_result: Any, path: str | Path | None = None, which: str = "best"):
    """Best (or last / all) attempt(s) of a session result (needs `.attempts` with frames)."""
    attempts = list(_get(session_result, "attempts", []) or [])
    if not attempts:
        raise ValueError("session has no attempts")
    track = _session_track(session_result)
    if which == "all":
        plt = _mpl()
        fig, ax = plt.subplots(figsize=(8, 8))
        draw_track(ax, track, corners=False)
        for k, rep in enumerate(attempts):
            draw_trace(ax, rep, color=PALETTE[k % len(PALETTE)], label=f"#{k + 1}")
        ax.legend(loc="best", fontsize=8)
        ax.set_title(f"{len(attempts)} attempts")
        return _save(fig, path)
    rep = attempts[-1] if which == "last" else min(attempts, key=_attempt_time)
    if which == "best" and _attempt_time(rep) == float("inf"):
        rep = attempts[-1]
    return plot_attempt(track, rep, path)


def plot_replay(replay: Any, path: str | Path | None = None, title: str | None = None):
    """Every car's trajectory of a replay (object or dict) in its own colour."""
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(8, 8))
    draw_track(ax, _get(replay, "track"), labels=False, corners=False)
    for k, car in enumerate(_get(replay, "cars", []) or []):
        x, y = np.asarray(_get(car, "x", []), float), np.asarray(_get(car, "y", []), float)
        if len(x) < 2:
            continue
        ax.plot(x, y, color=_get(car, "color") or PALETTE[k % len(PALETTE)], lw=1.3, alpha=0.85,
                label=_get(car, "name", f"car {k + 1}"))
    if _get(replay, "cars"):
        ax.legend(loc="best", fontsize=7)
    ax.set_title(title or _get(replay, "title", "") or "replay")
    return _save(fig, path)


def plot_telemetry(report: Any, path: str | Path | None = None, title: str | None = None):
    """Speed and controls (throttle, steering) against distance driven."""
    A = _trace_arrays(report)
    dist = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(A["x"]), np.diff(A["y"])))])
    plt = _mpl()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    ax1.plot(dist, A["speed"], color="#264653", lw=1.5)
    ax1.set_ylabel("speed")
    for d in dist[A["crashed"]]:
        ax1.axvline(d, color="#e63946", lw=1, alpha=0.7)
    ax2.step(dist, A["throttle"], where="pre", color="#2a9d8f", lw=1.2, label="throttle")
    ax2.step(dist, A["steering"], where="pre", color="#e76f51", lw=1.2, label="steering (+ = right)")
    ax2.axhline(0, color="#999", lw=0.6)
    ax2.set_ylim(-1.1, 1.1)
    ax2.set_xlabel("distance driven")
    ax2.legend(loc="lower right", fontsize=8)
    ax1.set_title(title or _result_title(A["result"]) or "telemetry")
    fig.tight_layout()
    return _save(fig, path)


def plot_env(env: Any, path: str | Path, which: str = "best", title: str | None = None):
    """Legacy helper: plot the best (or all) finished attempts of a RaceEnv."""
    traces = env.attempt_traces
    if not traces:
        return None
    if which == "all":
        plt = _mpl()
        fig, ax = plt.subplots(figsize=(8, 8))
        draw_track(ax, env.track, corners=False)
        for tr in traces:
            draw_trace(ax, tr, vmax=env.cfg.max_speed)
        ax.set_title(title or f"{env.track.name}: {len(traces)} attempts")
        return _save(fig, path)
    completed = [t for t in traces if t["result"].completed]
    best = min(completed, key=lambda t: t["result"].race_time) if completed else traces[-1]
    return plot_attempt(env.track, best, path, title=title, max_speed=env.cfg.max_speed)
