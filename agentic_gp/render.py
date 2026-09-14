"""Static visualisation of tracks and attempts (matplotlib, headless).

    from agentic_gp.render import plot_env
    plot_env(env, "results/my_agent.png")            # best attempt (or last if none completed)
    plot_env(env, "results/my_agent_all.png", which="all")
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .track import Track


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def draw_track(ax, track: Track) -> None:
    L = np.vstack([track.left_wall, track.left_wall[:1]])
    R = np.vstack([track.right_wall, track.right_wall[:1]])
    ax.plot(L[:, 0], L[:, 1], color="#333", lw=1.5)
    ax.plot(R[:, 0], R[:, 1], color="#333", lw=1.5)
    ax.fill(L[:, 0], L[:, 1], color="#e8e8e8", zorder=0)
    ax.fill(R[:, 0], R[:, 1], color="white", zorder=0)
    for g, (x1, y1, x2, y2) in enumerate(track.gates):
        ax.plot([x1, x2], [y1, y2], color="#2a9d8f" if g else "#e63946", lw=2.5 if g == 0 else 1.2, alpha=0.9)
        cx, cy = track.gate_centers[g]
        ax.annotate(str(g), (cx, cy), fontsize=7, ha="center", va="center", color="#264653",
                    bbox=dict(boxstyle="circle,pad=0.15", fc="white", ec="#264653", lw=0.6))
    sx, sy = track.start_position
    ax.plot(sx, sy, marker="o", color="#e63946", ms=5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def draw_trace(ax, trace: dict, cmap="viridis", vmax: float | None = None, label: str | None = None):
    traj = np.asarray(trace["trajectory"], dtype=float)
    if len(traj) < 2:
        return None
    sc = ax.scatter(traj[:, 0], traj[:, 1], c=traj[:, 2], cmap=cmap, s=9, vmin=0, vmax=vmax, zorder=3, label=label)
    ax.plot(traj[:, 0], traj[:, 1], color="#555", lw=0.6, alpha=0.6, zorder=2)
    if trace["crash_points"]:
        cp = np.asarray(trace["crash_points"], dtype=float)
        ax.scatter(cp[:, 0], cp[:, 1], marker="x", color="#e63946", s=45, zorder=4, lw=1.5)
    return sc


def plot_attempt(track: Track, trace: dict, path: str | Path, title: str | None = None, max_speed: float = 40.0) -> Path:
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(8, 8))
    draw_track(ax, track)
    sc = draw_trace(ax, trace, vmax=max_speed)
    if sc is not None:
        cb = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
        cb.set_label("speed")
    res = trace.get("result")
    if title is None and res is not None:
        title = (f"attempt {res.attempt}: " + (f"lap {res.lap_time:.2f}s" if res.completed else "DNF")
                 + f"  |  crashes {res.crashes}  |  waypoints {res.waypoints_passed}")
    ax.set_title(title or track.name)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_env(env, path: str | Path, which: str = "best", title: str | None = None) -> Path | None:
    """Plot the best (or all) finished attempts of an environment."""
    traces = env.attempt_traces
    if not traces:
        return None
    if which == "all":
        plt = _mpl()
        fig, ax = plt.subplots(figsize=(8, 8))
        draw_track(ax, env.track)
        for tr in traces:
            draw_trace(ax, tr, vmax=env.cfg.max_speed)
        ax.set_title(title or f"{env.track.name}: {len(traces)} attempts")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        return path
    completed = [t for t in traces if t["result"].completed]
    best = min(completed, key=lambda t: t["result"].lap_time) if completed else traces[-1]
    return plot_attempt(env.track, best, path, title=title, max_speed=env.cfg.max_speed)


def plot_track(track: Track, path: str | Path) -> Path:
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(8, 8))
    draw_track(ax, track)
    ax.set_title(f"{track.name}  (lap {track.length:.0f} units, width {2 * track.half_width:g})")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path
