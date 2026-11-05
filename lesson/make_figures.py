"""Regenerate every PNG figure of the lesson into lesson/assets/ (one command, deterministic).

    python lesson/make_figures.py                       # everything (about 1-2 minutes)
    python lesson/make_figures.py --only latency_cliff overfitting_matrix
    python lesson/make_figures.py --list                # figure names and what they show

Every number in every figure is measured here, now, by the real simulator (agentic_gp) - nothing is
typed in by hand except the few MiniMax lap times in LLM_MEASURED (real-key runs documented in
drivers/README.md; this script never calls a real LLM: LLM_PROVIDER is forced to "mock").
The measured numbers are also written to lesson/assets/figures_data.json, so the chapters can quote them.

The fly figures need the FlyWire circuit (data/fly/steering_circuit_783.json, not in git - see
agentic_gp/fly/README.md). Without it they are skipped and the committed PNGs stay as they are.

The SVG posters in lesson/assets/ are hand-made and NOT generated here.
"""

from __future__ import annotations

import argparse
import logging
import importlib.util
import json
import math
import os
import sys
import time as _time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ["LLM_PROVIDER"] = "mock"          # never spend the shared key on figures
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
ASSETS = ROOT / "lesson" / "assets"
EXAMPLES = ROOT / "drivers" / "examples"

import matplotlib                             # noqa: E402

matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)   # quiet font fallbacks
import matplotlib.pyplot as plt               # noqa: E402
import numpy as np                            # noqa: E402
from matplotlib import patheffects as pe      # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Polygon  # noqa: E402

from agentic_gp import driverkit as dk        # noqa: E402
from agentic_gp.race import session as session_mod  # noqa: E402
from agentic_gp.race.loader import load_driver  # noqa: E402
from agentic_gp.race.session import SessionConfig, run_session  # noqa: E402
from agentic_gp.sim import geometry as geo    # noqa: E402
from agentic_gp.sim.config import DEFAULT_CONFIG, PhysicsConfig  # noqa: E402
from agentic_gp.sim.env import RaceEnv        # noqa: E402
from agentic_gp.sim.physics import CarState, integrate  # noqa: E402
from agentic_gp.sim.track import make_track   # noqa: E402

# ============================================================================ style
# One palette for every lesson figure and poster (validated categorical order, light surface).
BG, PANEL = "#fcfcfb", "#f4f3ef"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
MAGENTA, GREEN, VIOLET, RED = "#e87ba4", "#008300", "#6250d6", "#e34948"
GOLD, MAX_BLACK = "#d4a017", "#1d1d1b"
ASPHALT, ASPHALT_EDGE, KERB = "#3b3a37", "#232321", "#f4f3ef"

# agent family -> colour (same mapping as ladder.svg / anatomy_cards.svg)
FAMILY = {
    "none": "#9a9891", "reflex": ORANGE, "model": BLUE, "learning": YELLOW,
    "llm_policy": RED, "llm_planner": VIOLET, "crew": MAX_BLACK, "spiking": MAGENTA,
}
# slow (hot, dark) -> fast (pale): one ramp, monotone in lightness
SPEED_CMAP = LinearSegmentedColormap.from_list(
    "agp_speed", ["#8e1b1b", "#d43d2f", "#ef7b3e", "#f7b55c", "#fde3a7"])
# good (pale) -> bad (dark): sequential blue ramp for "% slower than the best"
LOSS_CMAP = LinearSegmentedColormap.from_list("agp_loss", ["#eef5fd", "#b7d3f6", "#6da7ec", "#2a78d6", "#184f95"])

FONT = ["Segoe UI", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
DPI = 200
FOOT = "Agentic Grand Prix  ·  measured with agentic_gp by lesson/make_figures.py"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": FONT, "font.size": 10.5,
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 12, "axes.titleweight": "semibold", "axes.titlelocation": "left", "axes.titlepad": 10,
    "axes.labelsize": 10.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "xtick.major.size": 0, "ytick.major.size": 0, "legend.frameon": False, "legend.fontsize": 10,
    "lines.linewidth": 2.0, "lines.solid_capstyle": "round", "svg.fonttype": "none",
})

DATA: dict = {}                    # every measured number, written to figures_data.json
FIGURES: dict = {}                 # name -> (function, caption)
SAY = print


def figure(name: str, caption: str):
    def deco(fn):
        FIGURES[name] = (fn, caption)
        return fn
    return deco


def new_fig(w: float, h: float, title: str, subtitle: str = "", top: float | None = None):
    """A figure with a left-aligned title + subtitle band; returns (fig, top) for subplots_adjust."""
    fig = plt.figure(figsize=(w, h))
    fig.text(0.035, 1 - 0.30 / h, title, fontsize=17, fontweight="semibold", color=INK, va="top")
    if subtitle:
        fig.text(0.035, 1 - 0.68 / h, subtitle, fontsize=11, color=INK2, va="top", linespacing=1.45)
    fig.text(0.035, 0.14 / h, FOOT, fontsize=8, color=MUTED, va="bottom")
    lines = subtitle.count("\n") + 1 if subtitle else 0
    return fig, (top if top is not None else 1 - (0.95 + 0.24 * lines) / h)


def save(fig, name: str) -> Path:
    path = ASSETS / f"{name}.png"
    fig.savefig(path, dpi=DPI, metadata={"Software": None})
    plt.close(fig)
    SAY(f"  wrote {path.relative_to(ROOT)}")
    return path


def halo(color: str = BG, width: float = 3.5):
    return [pe.withStroke(linewidth=width, foreground=color)]


def secs(x) -> str:
    return "DNF" if x is None else f"{x:.2f} s"


# ============================================================================ running drivers
def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run(driver, track="grand_prix", attempts: int = 1, params: dict | None = None, **cfg) -> list:
    """run_session -> list of AttemptReports (driver: path, module object, instance or callable)."""
    drv = load_driver(str(driver) if isinstance(driver, Path) else driver, params=params)
    res = run_session(drv, SessionConfig(track=track, attempts=attempts, **cfg))
    return res.attempts


def outcome(rep) -> dict:
    r = rep.result
    return {"time": round(float(r.race_time), 2) if r.completed else None, "crashes": int(r.crashes),
            "gates": int(r.waypoints_passed), "ticks": int(r.ticks), "completed": bool(r.completed)}


def best_of(reps) -> dict:
    done = [outcome(r) for r in reps if r.result.completed]
    if not done:
        return outcome(reps[-1])
    return min(done, key=lambda o: o["time"])


def track_map(key: str) -> dict:
    return RaceEnv(key).track_map()


# ============================================================================ drawing helpers
def draw_track(ax, track, *, asphalt: str = ASPHALT, gates: bool = False, gate_labels: bool = False,
               start: bool = True, centerline: bool = True, zones: bool = True, lw_edge: float = 1.0,
               avoid=()):
    """Asphalt band between the walls, optional gate lines/labels, start line and zones."""
    L, R = track.left_wall, track.right_wall
    # fill the band: the outer wall loop in asphalt, the inner (infield) loop in the background colour
    outer, inner = (L, R) if abs(_poly_area(L)) > abs(_poly_area(R)) else (R, L)
    ax.add_patch(Polygon(outer, closed=True, fc=asphalt, ec="none", zorder=1))
    ax.add_patch(Polygon(inner, closed=True, fc=ax.get_facecolor(), ec="none", zorder=1.1))
    for wall in (L, R):
        w = np.vstack([wall, wall[:1]])
        ax.plot(w[:, 0], w[:, 1], color=ASPHALT_EDGE, lw=lw_edge, zorder=2)
    if centerline:
        C = np.vstack([track.centerline, track.centerline[:1]])
        ax.plot(C[:, 0], C[:, 1], color="#8f8d86", lw=0.8, ls=(0, (4, 5)), zorder=2.2)
    if zones:
        for z in track.zone_ranges():
            i0, i1 = z["from_index"], z["to_index"]
            idx = np.arange(i0, i1 + 1) % track.n_points if i1 >= i0 else \
                np.concatenate([np.arange(i0, track.n_points), np.arange(0, i1 + 1)])
            Lz, Rz = track.left_wall[idx], track.right_wall[idx]
            ax.add_patch(Polygon(np.vstack([Lz, Rz[::-1]]), closed=True, fc=YELLOW, ec="none", alpha=0.85,
                                 zorder=2.1, hatch="////"))
    if gates or gate_labels:
        n = track.n_gates
        for g in range(n):
            x1, y1, x2, y2 = track.gates[g]
            if gates:
                ax.plot([x1, x2], [y1, y2], color="#77756f", lw=0.8, zorder=2.3)
            if gate_labels:
                i = track.gate_indices[g]
                # gate numbers go on the INNER side (corner labels use the outer side), unless the infield
                # is too narrow there
                walls = np.vstack([track.left_wall, track.right_wall])
                sides = [w[i] + (w[i] - track.centerline[i]) / track.half_width * 8
                         for w in (track.left_wall, track.right_wall)]
                inner = [p for p in sides if _inside(track, p)]
                outer = [p for p in sides if not _inside(track, p)]
                p = inner[0] if inner else sides[0]
                near_label = any(np.linalg.norm(p - np.asarray(a)) < 36 for a in avoid)
                if (np.min(np.linalg.norm(walls - p, axis=1)) < 6 or near_label) and outer:
                    p = outer[0]
                ax.text(p[0], p[1], f"{g}", fontsize=7.5, color=MUTED, ha="center", va="center", zorder=5)
    if start:
        draw_start(ax, track, arrow=("white" if asphalt == ASPHALT else INK))


def _poly_area(P: np.ndarray) -> float:
    x, y = P[:, 0], P[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _inside(track, p) -> bool:
    """Is p inside the inner wall loop (the infield)?"""
    from matplotlib.path import Path as MPath
    L, R = track.left_wall, track.right_wall
    inner = L if abs(_poly_area(L)) < abs(_poly_area(R)) else R
    return bool(MPath(inner).contains_point(p))


def draw_start(ax, track, size: float = 1.0, arrow: str = "white"):
    """Chequered start/finish line on gate 0 and a direction arrow."""
    x1, y1, x2, y2 = track.gates[0]
    a, b = np.array([x1, y1]), np.array([x2, y2])
    t = track.tangents[track.gate_indices[0]]
    n_sq = 6
    for k in range(n_sq):
        for row in range(2):
            p0 = a + (b - a) * k / n_sq + t * (row * 2.2 - 2.2)
            p1 = a + (b - a) * (k + 1) / n_sq + t * (row * 2.2 - 2.2)
            quad = np.array([p0, p1, p1 + t * 2.2, p0 + t * 2.2])
            ax.add_patch(Polygon(quad, closed=True, fc="white" if (k + row) % 2 else "#111", ec="none", zorder=3))
    i0 = track.gate_indices[0]
    c = track.centerline
    p, q = c[(i0 + 3) % len(c)], c[(i0 + 9) % len(c)]
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=13 * size, lw=2.2, color=arrow, zorder=4))


def apex_label_pos(track, apex: int, dist: float) -> np.ndarray:
    """A point `dist` units outside the corner (on the outer side of the apex)."""
    c = track.centerline[apex]
    curv = track.curvature[apex]
    t = track.tangents[apex]
    left_n = np.array([-t[1], t[0]])
    outward = -left_n if curv > 0 else left_n           # left turn -> outside is on the right
    return c + outward * (track.half_width + dist)


def corner_label(ax, track, corner: dict, sub: str | None = None, *, gap: float = 9.0, fs: float = 9.5):
    """T-chip just outside the apex, with `sub` further out, both growing AWAY from the track."""
    apex = corner["apex_index"]
    edge = apex_label_pos(track, apex, 2.0)
    d = edge - track.centerline[apex]
    d = d / (np.linalg.norm(d) + 1e-9)
    ha = "left" if d[0] > 0.38 else "right" if d[0] < -0.38 else "center"
    va = "bottom" if d[1] > 0.38 else "top" if d[1] < -0.38 else "center"
    off = d * gap
    ax.annotate(corner["id"], edge, xytext=(off[0], off[1]), textcoords="offset points", ha=ha, va=va,
                fontsize=fs, fontweight="bold", color="white", zorder=7,
                bbox=dict(boxstyle="round,pad=0.28,rounding_size=0.6", fc=INK, ec="none"))
    if sub:
        chip_h = 15 * fs / 9.5
        if ha == "center":                       # stacked further out: chip, then sub text
            o2 = off + np.array([0, (chip_h + 3 if va == "bottom" else -chip_h - 3)])
            v2 = va
        else:                                    # side label: sub text right under the chip
            if va == "bottom":                   # label grows upwards: sub text above the chip
                o2, v2 = off + np.array([0, chip_h + 3]), "bottom"
            else:                                # sub text under the chip
                o2, v2 = off + np.array([0, -chip_h * (0.5 if va == "center" else 1.0) - 2]), "top"
        ax.annotate(sub, edge, xytext=(o2[0], o2[1]), textcoords="offset points", ha=ha, va=v2,
                    fontsize=fs - 1.3, color=INK2, zorder=7, path_effects=halo())


def frame_axes(ax, track, pad: float = 22):
    x0, y0, x1, y1 = track.bounds()
    ax.set_xlim(x0 - pad, x1 + pad)
    ax.set_ylim(y0 - pad, y1 + pad)
    ax.set_aspect("equal")
    ax.axis("off")


def track_band(ax, track, values: np.ndarray, cmap, vmin, vmax, zorder=1.5):
    """Paint the whole track surface (wall to wall), one quad per centerline step, coloured by `values`."""
    from matplotlib.collections import PolyCollection
    L, R = track.left_wall, track.right_wall
    L2, R2 = np.roll(L, -1, axis=0), np.roll(R, -1, axis=0)
    quads = np.stack([L, L2, R2, R], axis=1)
    pc = PolyCollection(quads, cmap=cmap, norm=plt.Normalize(vmin, vmax), zorder=zorder, linewidths=0.6,
                        antialiaseds=True)
    v = 0.5 * (values + np.roll(values, -1))
    pc.set_array(v)
    pc.set_edgecolor(cmap(plt.Normalize(vmin, vmax)(v)))       # hide hairline seams between quads
    ax.add_collection(pc)
    return pc


def grip_speed(track_or_points, a_lat=20.0, vmax=40.0) -> np.ndarray:
    pts = track_or_points.centerline if hasattr(track_or_points, "centerline") else track_or_points
    R = dk.curvature_radii(pts)
    return np.minimum(vmax, np.sqrt(a_lat * np.minimum(R, 1e9)))


def arc_of_xy(track, xy: np.ndarray) -> np.ndarray:
    idx = dk.nearest_index(track.centerline, xy)
    return track.arc[idx]


# ============================================================================ 1. track maps
PUBLIC = ["grand_prix", "oval", "sprint", "speedway", "briefing_demo"]


@figure("track_<key>", "Public track maps: walls, gates 0..n, start/finish, direction and corners T1..Tn "
                       "with radius and v_max (one PNG per public track).")
def fig_track_maps():
    DATA["tracks"] = {}
    for key in PUBLIC:
        tr = make_track(key)
        m = track_map(key)
        corners = m["corners"]
        DATA["tracks"][key] = {"name": tr.name, "length": round(tr.length, 1), "half_width": tr.half_width,
                               "gates": tr.n_gates, "corners": [{k: c[k] for k in ("id", "direction", "min_radius",
                                                                                     "v_max", "from_wp", "to_wp")}
                                                                for c in corners],
                               "zones": tr.zone_ranges()}
        x0, y0, x1, y1 = tr.bounds()
        aspect = (y1 - y0 + 70) / (x1 - x0 + 70)
        w = 10.0
        h = min(11.5, max(5.6, w * aspect * 0.86 + 1.6))
        sub = (f"{tr.length:.0f} units per lap  ·  {tr.n_gates} gates  ·  track width {2 * tr.half_width:g}  ·  "
               f"{len(corners)} corners (T1..T{len(corners)}, numbered from the start line)")
        if tr.zones:
            sub += f"\nYellow flag (hatched): WP {tr.zones[0].from_wp}-{tr.zones[0].to_wp}, max " \
                   f"{tr.zones[0].max_speed:g} u/s - announced ONLY in the briefing text"
        fig, top = new_fig(w, h, f"{tr.name}  ·  {key}", sub)
        ax = fig.add_axes([0.03, 0.06, 0.94, top - 0.07])
        draw_track(ax, tr, gates=True, gate_labels=True,
                   avoid=[apex_label_pos(tr, c["apex_index"], 12) for c in corners])
        for c in corners:
            corner_label(ax, tr, c, f"R {c['min_radius']:g} · {c['v_max']:g} u/s")
        for z in tr.zone_ranges():
            mid = (z["from_index"] + ((z["to_index"] - z["from_index"]) % tr.n_points) // 2) % tr.n_points
            p = tr.centerline[mid]
            ax.annotate(f"{z['label']}: max {z['max_speed']:g} u/s", p, xytext=(-150, 40), textcoords="offset points",
                        fontsize=10.5, fontweight="semibold", color=INK, ha="center",
                        bbox=dict(boxstyle="round,pad=0.4", fc=YELLOW, ec="none"),
                        arrowprops=dict(arrowstyle="-", color=INK, lw=1.0), zorder=8)
        frame_axes(ax, tr, pad=30)
        save(fig, f"track_{key}")


# ============================================================================ 2. v_max centerline
@figure("vmax_grand_prix", "Home track centerline coloured by the grip limit v_max = sqrt(20 R) (capped at 40): "
                           "the red spots are where the speed is decided.")
def fig_vmax_home():
    key = "grand_prix"
    tr = make_track(key)
    corners = track_map(key)["corners"]
    v = grip_speed(tr)
    fig, top = new_fig(10, 9.2, "Where is the speed decided?  v_max = √(20 · R)",
                       "Track coloured by the fastest speed the grip allows (max lateral accel 20 u/s², "
                       "top speed 40).\nTight corner = small R = low v_max. The tightest corner of the home "
                       f"track is {min(corners, key=lambda c: c['v_max'])['id']}.")
    ax = fig.add_axes([0.03, 0.05, 0.82, top - 0.06])
    draw_track(ax, tr, asphalt="#d9d8d2", centerline=False, start=True, lw_edge=1.3)
    track_band(ax, tr, v, SPEED_CMAP, 15, 40)
    for c in corners:
        corner_label(ax, tr, c, f"{c['v_max']:g} u/s", gap=8)
    frame_axes(ax, tr, pad=30)
    cax = fig.add_axes([0.90, 0.22, 0.02, 0.42])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(15, 40), cmap=SPEED_CMAP), cax=cax)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=9, colors=INK2)
    cb.set_label("v_max (u/s)", color=INK2)
    save(fig, "vmax_grand_prix")


@figure("vmax_tracks", "Small multiples: v_max-coloured centerline of every public track and two random tracks.")
def fig_vmax_all():
    keys = ["grand_prix", "oval", "sprint", "speedway", "random:1", "random:7"]
    fig, top = new_fig(13, 8.6, "Six tracks, one physics: v_max = √(20 · R)",
                       "Same colour scale everywhere. Speedway is fast and flowing; sprint is a string of "
                       "tight corners; random:<seed> tracks are new every seed.")
    for k, key in enumerate(keys):
        tr = make_track(key)
        col, row = k % 3, k // 3
        h = (top - 0.06) / 2
        x0, y0 = 0.03 + col * 0.31, 0.05 + (1 - row) * h
        ax = fig.add_axes([x0, y0 + 0.045, 0.29, h - 0.095])
        draw_track(ax, tr, asphalt="#d9d8d2", centerline=False, start=True, lw_edge=0.9)
        track_band(ax, tr, grip_speed(tr), SPEED_CMAP, 15, 40)
        frame_axes(ax, tr, pad=10)
        c = track_map(key)["corners"]
        slow = min(c, key=lambda x: x["v_max"]) if c else None
        fig.text(x0, y0 + h - 0.02, key, fontsize=12.5, fontweight="semibold", color=INK, va="top")
        if slow:
            fig.text(x0, y0 + 0.03, f"{tr.length:.0f} units  ·  {len(c)} corners  ·  slowest {slow['id']} "
                                    f"{slow['v_max']:g} u/s", fontsize=9.5, color=INK2, va="top")
    cax = fig.add_axes([0.955, 0.25, 0.012, 0.4])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(15, 40), cmap=SPEED_CMAP), cax=cax)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8.5)
    cb.set_label("v_max (u/s)", color=INK2, fontsize=9)
    save(fig, "vmax_tracks")


# ============================================================================ 3. turn radius
@figure("turn_radius", "Turn-radius explorer: steering is a fraction of grip, so R = v² / (20 · |steering|), "
                       "never tighter than v / (1.5 · |steering|). Steering 0.5 at 10 / 20 / 30 u/s gives "
                       "R = 13.3 / 40 / 90 - only the slow car makes the hairpin.")
def fig_turn_radius():
    cfg = DEFAULT_CONFIG

    def radius(v: float, s: float) -> float:                  # the sim's rule: grip limit AND yaw-rate limit
        return round(v / (abs(s) * min(cfg.max_yaw_rate, cfg.max_lateral_accel / v)), 1)

    corners = track_map("grand_prix")["corners"]
    hp = min(corners, key=lambda c: c["min_radius"])          # the home track's hairpin
    r_c, hw = hp["min_radius"], make_track("grand_prix").half_width
    # a hairpin with exactly that radius: straight in (heading +y), half circle to the LEFT, straight out
    ang = np.linspace(0, math.pi, 90)
    center = np.array([-r_c, 0.0])
    line = np.vstack([np.column_stack([np.zeros(30), np.linspace(-70, 0, 30, endpoint=False)]),
                      center + r_c * np.column_stack([np.cos(ang), np.sin(ang)]),
                      np.column_stack([np.full(30, -2 * r_c), np.linspace(0, -70, 31)[1:]])])
    t = np.gradient(line, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    nrm = np.column_stack([-t[:, 1], t[:, 0]])                # left normal
    left, right = line + nrm * hw, line - nrm * hw
    walls = np.vstack([np.column_stack([w[:-1], w[1:]]) for w in (left, right)])
    steer = -0.5                                              # half lock to the LEFT (+ = right)
    speeds = [(10, AQUA), (20, YELLOW), (30, RED)]
    DATA["turn_radius"] = {"corner": hp["id"], "corner_radius": r_c, "steering": steer, "arcs": []}

    fig, top = new_fig(13, 7.2, "Faster = wider: the turning circle grows with v²",
                       f"Same steering ({steer:+.1f} = half lock to the left) at three speeds, driven by the real physics "
                       f"(agentic_gp.sim.physics) into a hairpin as tight as the home track's {hp['id']} (R {r_c:g}).\n"
                       "Steering commands a fraction of the grip:  R = v² / (20 · |steering|),  never tighter than "
                       "v / 1.5.")
    ax = fig.add_axes([0.02, 0.07, 0.47, top - 0.08])
    ax.add_patch(Polygon(np.vstack([left, right[::-1]]), closed=True, fc=ASPHALT, ec="none", zorder=1))
    for w in (left, right):
        ax.plot(w[:, 0], w[:, 1], color=ASPHALT_EDGE, lw=1.0, zorder=2)
    ax.plot(line[:, 0], line[:, 1], color="#8f8d86", lw=0.8, ls=(0, (4, 5)), zorder=2.2)
    for v, col in speeds:
        st = CarState(0.0, -25.0, math.pi / 2, float(v))
        pts = [np.array([st.x, st.y])]
        hold = cfg.drag * v / cfg.accel                         # throttle that holds the speed
        turned = 0.0
        for _ in range(int(12.0 / 0.05)):                       # half lock until the car points back down
            h0 = st.heading
            turning = st.y >= 0.0 and turned < math.pi          # turn in where the hairpin starts
            st = integrate(st, hold, steer if turning else 0.0, cfg, 0.05)
            turned += abs(st.heading - h0)
            pts.append(np.array([st.x, st.y]))
            if st.y < -70:
                break
        P = np.array(pts)
        hit = geo.first_hit(P, walls)
        R = radius(v, steer)
        if hit is not None:
            k, _, point = hit
            ax.plot(np.r_[P[:k + 1, 0], point[0]], np.r_[P[:k + 1, 1], point[1]], color=col, lw=3.2, zorder=6)
            ax.plot(P[k:, 0], P[k:, 1], color=col, lw=1.5, ls=(0, (2, 3)), alpha=0.85, zorder=5)
            ax.plot(*point, marker="X", ms=14, color=RED, mec="white", mew=1.8, zorder=8)
        else:
            ax.plot(P[:, 0], P[:, 1], color=col, lw=3.2, zorder=6)
        DATA["turn_radius"]["arcs"].append({"speed": v, "radius": round(R, 1), "hits_wall": hit is not None})
    _car(ax, np.array([0.0, -25.0]), math.pi / 2, size=4.2, color=BLUE)
    ax.plot([-r_c, -r_c - r_c * math.cos(math.radians(120))], [0, r_c * math.sin(math.radians(120))],
            color="white", lw=1.0, zorder=7)
    ax.plot(-r_c, 0, "o", color="white", ms=3, zorder=7)
    mid = np.array([-r_c - 0.5 * r_c * math.cos(math.radians(120)), 0.5 * r_c * math.sin(math.radians(120))])
    ax.annotate(f"hairpin radius {r_c:g}\n(the line through the middle)", mid, xytext=(16, 40), fontsize=9.5,
                color=INK, ha="left", va="bottom", arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8), zorder=9)
    ax.text(hw + 2, 0, "turn in here", color=INK2, fontsize=9.5, ha="left", va="center")
    ax.plot([-hw, hw], [0, 0], color="white", lw=0.8, ls=(0, (2, 2)), zorder=3)
    ax.set_xlim(-100, 62)
    ax.set_ylim(-60, 95)
    ax.set_aspect("equal")
    ax.axis("off")
    for v, col in speeds:
        R = radius(v, steer)
        hits = next(a["hits_wall"] for a in DATA["turn_radius"]["arcs"] if a["speed"] == v)
        ax.plot([], [], color=col, lw=3, label=f"{v} u/s  →  R = {R:g}" + ("   wall!" if hits else "   makes it"))
    leg = ax.legend(loc="lower left", fontsize=10.5, handlelength=1.8, borderpad=0.8, frameon=True,
                    bbox_to_anchor=(0.0, 0.0))
    leg.get_frame().set(facecolor=BG, edgecolor=GRID)

    # right: R vs speed
    ax2 = fig.add_axes([0.57, 0.14, 0.40, top - 0.22])
    v = np.linspace(0.5, 40, 300)
    for s, col, lab in ((1.0, INK, "full lock (1.0)"), (0.5, BLUE, "half (0.5)"), (0.25, "#86b6ef", "quarter (0.25)")):
        yaw = s * np.minimum(cfg.max_yaw_rate, cfg.max_lateral_accel / v)
        Rv = v / yaw
        ax2.plot(v, Rv, color=col, lw=2.2)
        if Rv[-1] > 118:
            j = int(np.searchsorted(Rv, 118))
            ax2.text(v[j] - 0.6, 119, lab, fontsize=9.5, color=INK2, ha="right", va="bottom")
        else:
            ax2.text(39.5, Rv[-1] + 3, lab, fontsize=9.5, color=INK2, ha="right", va="bottom")
    for c in sorted(corners, key=lambda c: c["min_radius"])[:1]:
        ax2.axhline(c["min_radius"], color=RED, lw=1.2, ls=(0, (4, 3)))
        ax2.text(41.5, c["min_radius"] + 2.0, f"hairpin {c['id']}: R {c['min_radius']:g}", color=RED, fontsize=9.5,
                 ha="right", va="bottom")
    for v0, col in speeds:
        R = radius(v0, 0.5)
        ax2.plot(v0, R, "o", color=col, ms=9, mec=BG, mew=2, zorder=5)
        if v0 == 10:
            ax2.annotate(f"{v0} u/s → R {R:g}", (v0, R), xytext=(4.5, 34), ha="center", va="bottom", fontsize=9.5,
                         color=INK, fontweight="semibold", path_effects=halo(),
                         arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8, shrinkB=5))
        else:
            ax2.annotate(f"{v0} u/s → R {R:g}", (v0, R), xytext=(-10, 5), textcoords="offset points", ha="right",
                         va="bottom", fontsize=9.5, color=INK, fontweight="semibold", path_effects=halo())
    ax2.set_xlim(0, 42)
    ax2.set_ylim(0, 128)
    ax2.set_xlabel("speed (u/s)")
    ax2.set_ylabel("turn radius R (units)")
    ax2.set_title("Turn radius vs speed, for three steering commands", fontsize=11.5)
    save(fig, "turn_radius")


def _car(ax, xy, heading, size=4.0, color=BLUE, zorder=9):
    """A small top-down car (rounded triangle) at xy pointing along heading (radians)."""
    f = np.array([math.cos(heading), math.sin(heading)])
    l = np.array([-f[1], f[0]])
    pts = np.array([xy + f * size * 1.2, xy - f * size * 0.8 + l * size * 0.7, xy - f * size * 0.5,
                    xy - f * size * 0.8 - l * size * 0.7])
    ax.add_patch(Polygon(pts, closed=True, fc=color, ec="white", lw=1.5, zorder=zorder, joinstyle="round"))


# ============================================================================ 4. stopping distance
@figure("stopping_distance", "Stopping distance = blind distance (you only act every 0.5 s) + braking distance "
                             "v²/(2·25); at 30 u/s that is 15 + 18 = 33 units, nearly two track widths.")
def fig_stopping():
    cfg = DEFAULT_CONFIG
    v = np.linspace(0, 40, 200)
    blind = v * cfg.tick_seconds
    brake = v ** 2 / (2 * cfg.brake)
    fig, top = new_fig(13, 6.6, "How far do you need to stop?",
                       "Every answer from drive() is applied for 0.5 s before you are asked again: that part is "
                       "blind. Then the brakes (25 u/s²) need v² / 50.\nRight: an LLM that thinks for seconds "
                       "is blind for much longer.")
    ax = fig.add_axes([0.06, 0.14, 0.50, top - 0.22])
    ax.fill_between(v, 0, blind, color=ORANGE, alpha=0.9, lw=0, label="blind: v × 0.5 s (one tick)")
    ax.fill_between(v, blind, blind + brake, color=BLUE, alpha=0.85, lw=0, label="braking: v² / (2 × 25)")
    ax.plot(v, blind + brake, color=INK, lw=1.6)
    width = 2 * 9.0
    ax.axhline(width, color=INK2, lw=1, ls=(0, (4, 3)))
    ax.text(0.8, width + 1.2, "track width (18)", color=INK2, fontsize=9.5)
    rows = []
    for v0 in (10, 20, 30, 40):
        b, k = v0 * 0.5, v0 * v0 / 50
        rows.append({"speed": v0, "blind": b, "braking": round(k, 1), "total": round(b + k, 1)})
        ax.plot(v0, b + k, "o", color=INK, ms=7, mec=BG, mew=2, zorder=5)
        ax.annotate(f"{b + k:g}", (v0, b + k), xytext=(-6, 7), textcoords="offset points", ha="right", fontsize=10,
                    fontweight="semibold", color=INK, path_effects=halo())
    ax.annotate("at 30 u/s: 15 blind + 18 braking", (30, 33), xytext=(18, 46), fontsize=10, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.9), path_effects=halo())
    ax.set_xlim(0, 41)
    ax.set_ylim(0, 56 + 20)
    ax.set_xlabel("speed (u/s)")
    ax.set_ylabel("distance (units)")
    ax.legend(loc="upper left", fontsize=10)
    ax.set_title("One tick of reaction time + the brakes", fontsize=11.5)
    DATA["stopping"] = rows

    ax2 = fig.add_axes([0.66, 0.14, 0.31, top - 0.22])
    lat = [0.5, 1.0, 2.0, 3.0]
    d = [30 * x for x in lat]
    cols = [ORANGE, "#f09a6f", RED, "#a52c2b"]
    ax2.barh(range(len(lat)), d, color=cols, height=0.56, zorder=3)
    for i, (x, y) in enumerate(zip(lat, d)):
        ax2.text(y + 2, i, f"{y:g} units  ({y / 18:.1f} track widths)", va="center", fontsize=10, color=INK)
    ax2.set_yticks(range(len(lat)), [f"{x:g} s" for x in lat])
    ax2.invert_yaxis()
    ax2.set_xlim(0, 160)
    ax2.grid(axis="y", visible=False)
    ax2.set_xlabel("blind distance at 30 u/s (units)")
    ax2.set_title("Waiting for an answer at 30 u/s", fontsize=11.5)
    ax2.text(0.98, 0.98, "Measured: the L4 rookie's MiniMax\ncalls took p50 1.16 s, p95 1.95 s\n"
             "(turn-based run, drivers/README.md)", transform=ax2.transAxes, fontsize=9, color=INK2, va="top",
             ha="right")
    save(fig, "stopping_distance")


# ============================================================================ 5. speed profile
@figure("speed_profile", "Speed profile on the home track: the grip limit sqrt(20 R) per point, then the "
                         "backward braking pass, then what the L2 planner actually drove.")
def fig_speed_profile():
    key = "grand_prix"
    tr = make_track(key)
    corners = track_map(key)["corners"]
    pts = tr.centerline
    grip = grip_speed(tr)
    planned = dk.speed_profile(pts)
    rep = run(EXAMPLES / "l2_planner.py", key)[0]
    tel = rep.telemetry
    xy = np.array([[r["x"], r["y"]] for r in tel])
    sp = np.array([r["speed"] for r in tel])
    s_actual = arc_of_xy(tr, xy)
    s_actual = np.unwrap(s_actual * 2 * np.pi / tr.length) * tr.length / (2 * np.pi)   # continuous distance
    if s_actual[0] > tr.length / 2:                    # the grid is just behind the start line
        s_actual = s_actual - tr.length
    s = tr.arc[:-1]
    DATA["speed_profile"] = {"track": key, "l2_lap": outcome(rep), "grip_min": round(float(grip.min()), 1),
                             "planned_min": round(float(planned.min()), 1)}

    fig, top = new_fig(13, 7.4, "From map to speed plan: grip limit → backward pass → the actual lap",
                       "1) every point of the centerline gets v_max = √(20 R)      2) walk BACKWARDS: you can only be "
                       "as fast as you can still brake (25 u/s²) for what comes next\n3) the L2 planner drives that "
                       f"plan from a standing start: {secs(outcome(rep)['time'])}, {outcome(rep)['crashes']} crashes.")
    ax = fig.add_axes([0.06, 0.13, 0.91, top - 0.16])
    for c in corners:
        a0, a1 = tr.arc[c["start_index"]], tr.arc[c["end_index"]]
        if a1 < a0:
            a1 += tr.length
        ax.axvspan(a0, a1, color=PANEL, zorder=0, lw=0)
        ax.text((a0 + a1) / 2, 41.2, c["id"], ha="center", va="bottom", fontsize=9.5, fontweight="semibold", color=INK)
    brk = planned < grip - 0.3
    ax.fill_between(s, planned, grip, where=brk, color=RED, alpha=0.45, lw=0, zorder=1, interpolate=True)
    ax.plot(s, planned, color=BLUE, lw=3.0, zorder=3, label="2  after the backward braking pass = the plan")
    ax.plot(s, grip, color=INK, lw=1.2, ls=(0, (3, 2)), zorder=4, label="1  grip limit √(20 R), capped at 40")
    ax.plot(s_actual, sp, color=ORANGE, lw=1.8, zorder=5, marker="o", ms=2.6,
            label="3  actual speed of the L2 planner (one dot per 0.5 s tick)")
    ax.fill_between([], [], [], color=RED, alpha=0.45, lw=0, label="braking zone: the plan is below the grip limit")
    hp = min(corners, key=lambda c: c["v_max"])
    ax.annotate("standing start", (s_actual[2], sp[2]), xytext=(28, 3), fontsize=10, color=INK, ha="left",
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.9), path_effects=halo())
    # zoom on the hairpin approach: the three curves, large
    a0 = tr.arc[hp["start_index"]] - 70
    a1 = tr.arc[hp["apex_index"]] + 15
    ins = ax.inset_axes([0.135, 0.06, 0.20, 0.33], zorder=10)
    m = (s >= a0) & (s <= a1)
    ma = (s_actual >= a0) & (s_actual <= a1)
    ins.fill_between(s[m], planned[m], grip[m], where=brk[m], color=RED, alpha=0.45, lw=0, interpolate=True)
    ins.plot(s[m], planned[m], color=BLUE, lw=3.0)
    ins.plot(s[m], grip[m], color=INK, lw=1.2, ls=(0, (3, 2)))
    ins.plot(s_actual[ma], sp[ma], color=ORANGE, lw=1.8, marker="o", ms=3.5)
    ins.set_facecolor(BG)
    ins.set_title(f"zoom: braking for {hp['id']} (red = what the backward pass removes)", fontsize=9.5, pad=4, loc="left")
    ins.tick_params(labelsize=8)
    for sp_ in ins.spines.values():
        sp_.set_visible(True)
        sp_.set_color(AXIS)
    ax.set_xlim(-10, tr.length)
    ax.set_ylim(0, 44)
    ax.set_xlabel("distance along the lap (units from the start line)")
    ax.set_ylabel("speed (u/s)")
    ax.legend(loc="lower right", fontsize=10, ncol=1, frameon=True, facecolor=BG, edgecolor=GRID)
    save(fig, "speed_profile")


# ============================================================================ 6. overfitting matrix
OVERFIT_TRACKS = ["grand_prix", "oval", "sprint", "speedway", "random:1", "random:2", "random:3"]


class Memorizer:
    """Overfitting on purpose: one target speed per waypoint segment, copied from a home-track lap.

    Steering = the L1 reflex. Speed = TABLE[segment], no stopping-distance safety net.
    """
    NAME = "Memorizer"
    KIND = "code"

    def __init__(self, table: list[float], l1_module):
        self.table, self.l1 = list(table), l1_module

    def drive(self, obs):
        a = self.l1.drive(obs)
        if a["throttle"] < 0 and obs["speed"] < 1.0:          # L1 is reversing out of a wall
            return a
        target = self.table[obs["segment"] % len(self.table)]
        dv = target - obs["speed"]
        thr = (dv + 0.075 * obs["speed"]) / 6.0 if dv >= 0 else dv / 12.5
        return {"throttle": dk.clamp(thr), "steering": a["steering"]}


def home_speed_table() -> list[float]:
    rep = run(EXAMPLES / "l2b_racing_line.py", "grand_prix")[0]
    return [round(s["avg_speed"], 2) for s in sorted(rep.segments, key=lambda s: s["segment"])]


@figure("overfitting_matrix", "Overfitting matrix: example drivers (rows) x tracks (columns incl. random seeds). "
                              "Colour = % slower than the fastest car on that track; red = crashes. The Memorizer "
                              "(home speed table) is great at home and crashes everywhere else.")
def fig_overfitting():
    l1 = load_module(EXAMPLES / "l1_reflex.py", "agp_fig_l1_memo")
    table = home_speed_table()
    rows = [
        ("Template (starter)", lambda: ROOT / "drivers" / "template.py", None, 1),
        ("L1 Reflex", lambda: EXAMPLES / "l1_reflex.py", None, 1),
        ("L1 tuned at home\ncap=40, turn_k=1.0", lambda: EXAMPLES / "l1_reflex.py", {"cap": 40.0, "turn_k": 1.0}, 1),
        ("Memorizer\nhome speed table", lambda: Memorizer(table, l1), None, 1),
        ("L2 Planner", lambda: EXAMPLES / "l2_planner.py", None, 1),
        ("L2b Racing line", lambda: EXAMPLES / "l2b_racing_line.py", None, 1),
        ("L3 Learner\nbest of 4", lambda: EXAMPLES / "l3_learner.py", None, 4),
    ]
    M = {}
    for label, make, params, n in rows:
        for key in OVERFIT_TRACKS:
            M[(label, key)] = best_of(run(make(), key, attempts=n, params=params))
    DATA["overfitting_matrix"] = {"tracks": OVERFIT_TRACKS, "memorizer_table": table,
                                  "rows": {lab: {k: M[(lab, k)] for k in OVERFIT_TRACKS} for lab, *_ in rows}}
    best = {k: min(M[(lab, k)]["time"] for lab, *_ in rows if M[(lab, k)]["time"]) for k in OVERFIT_TRACKS}
    loss = np.array([[(M[(lab, k)]["time"] / best[k] - 1) * 100 if M[(lab, k)]["time"] else 100
                      for k in OVERFIT_TRACKS] for lab, *_ in rows])

    memo = [M[(rows[3][0], k)] for k in OVERFIT_TRACKS]
    reflex_home = sorted((M[(lab, "grand_prix")]["time"], lab) for lab, *_ in rows[:4])
    rank = [lab for _, lab in reflex_home].index(rows[3][0]) + 1
    away_crash = sum(1 for o in memo[1:] if o["crashes"])
    fig, top = new_fig(13.5, 8.4, "Overfitting matrix: fast at home ≠ fast everywhere",
                       "One lap per cell (the learner: best of 4 attempts on THAT track). Colour = how much slower than "
                       "the fastest car on that track. Red frame = crashes (+5 s each).\nThe Memorizer copies one speed "
                       "per waypoint segment from a home lap: " + ("the fastest" if rank == 1 else f"#{rank}") +
                       f" of the reflex drivers at home - and it crashes on {away_crash} of the {len(memo) - 1} "
                       "tracks it never saw.")
    ax = fig.add_axes([0.17, 0.08, 0.72, top - 0.16])
    im = ax.imshow(loss, cmap=LOSS_CMAP, vmin=0, vmax=60, aspect="auto")
    ax.set_xticks(range(len(OVERFIT_TRACKS)), [("grand_prix\n(home)" if k == "grand_prix" else k)
                                               for k in OVERFIT_TRACKS], fontsize=10.5, color=INK)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=10.5, color=INK)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    for i, (lab, *_) in enumerate(rows):
        for j, k in enumerate(OVERFIT_TRACKS):
            o = M[(lab, k)]
            dark = loss[i, j] > 32
            txt = secs(o["time"]).replace(" s", "")
            ax.text(j, i - 0.12, txt, ha="center", va="center", fontsize=11, fontweight="semibold",
                    color="white" if dark else INK)
            extra = f"+{loss[i, j]:.0f}%" if loss[i, j] >= 0.5 else "fastest"
            if o["crashes"]:
                extra = f"{o['crashes']} crash" + ("es" if o["crashes"] > 1 else "")
            ax.text(j, i + 0.22, extra, ha="center", va="center", fontsize=8.6,
                    color=("white" if dark else (RED if o["crashes"] else INK2)),
                    fontweight="semibold" if o["crashes"] else "normal")
            if o["crashes"]:
                ax.add_patch(plt.Rectangle((j - 0.47, i - 0.45), 0.94, 0.9, fill=False, ec=RED, lw=2.4, zorder=4))
    ax.set_xticks(np.arange(-0.5, len(OVERFIT_TRACKS)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows)), minor=True)
    ax.grid(which="minor", color=BG, lw=3)
    ax.tick_params(which="minor", length=0)
    ax.axvspan(-0.5, 0.5, ymin=0, ymax=1, fill=False, ec=INK, lw=1.2, ls=(0, (3, 2)))
    cax = fig.add_axes([0.915, 0.2, 0.014, 0.45])
    cb = fig.colorbar(im, cax=cax)
    cb.outline.set_visible(False)
    cb.set_label("% slower than the fastest on that track", color=INK2)
    cb.ax.tick_params(labelsize=9)
    save(fig, "overfitting_matrix")


# ============================================================================ 7. learning curves
@figure("learning_curve", "The L3 learner (drivers/examples) on three tracks: one experiment per attempt, keep-best, "
                          "revert - lap time goes down, never a crash.")
def fig_learning_curve():
    tracks = [("grand_prix", BLUE), ("sprint", ORANGE), ("speedway", VIOLET)]
    n = 6
    fig, top = new_fig(12, 6.6, "A learning agent: one experiment per attempt, keep the best",
                       f"drivers/examples/l3_learner.py, {n} attempts per track. Dots = every attempt; line = best "
                       "lap so far (what it would race after lock-in).")
    ax = fig.add_axes([0.07, 0.13, 0.56, top - 0.2])
    DATA["learning_curve"] = {}
    for key, col in tracks:
        reps = run(EXAMPLES / "l3_learner.py", key, attempts=n)
        t = [outcome(r)["time"] for r in reps]
        rel = [100 * (x / t[0] - 1) if x else None for x in t]
        best = np.minimum.accumulate([x for x in rel])
        xs = np.arange(1, n + 1)
        ax.plot(np.r_[xs, n + 0.3], np.r_[best, best[-1]], color=col, lw=2.4, drawstyle="steps-post", zorder=3)
        ax.plot(xs, rel, "o", color=col, ms=7, mec=BG, mew=1.8, zorder=4)
        ax.annotate(f"{key}\n{t[0]:.2f} → {min(t):.2f} s", (n + 0.3, best[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=10, color=INK)
        DATA["learning_curve"][key] = [outcome(r) for r in reps]
    ax.axhline(0, color=AXIS, lw=1)
    ax.set_xlim(0.7, n + 0.35)
    ax.set_xticks(range(1, n + 1))
    ax.set_xlabel("attempt")
    ax.set_ylabel("lap time vs attempt 1 (%)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:+.0f}%"))
    ax.text(1.2, 0.98, "Russell & Norvig, on the car", transform=ax.transAxes, fontsize=11, fontweight="semibold",
            color=INK, va="top")
    roles = [("performance element", "drive()"), ("critic", "the lap report"), ("learning element", "reflect()"),
             ("problem generator", "one change per attempt")]
    for i, (a, b) in enumerate(roles):
        ax.text(1.2, 0.86 - i * 0.12, a, transform=ax.transAxes, fontsize=9.5, color=INK2, va="top")
        ax.text(1.2, 0.81 - i * 0.12, b, transform=ax.transAxes, fontsize=10.5, color=INK, va="top",
                family=["Cascadia Mono", "Consolas", "DejaVu Sans Mono"] if "(" in b else FONT)
    save(fig, "learning_curve")


class TableLearner:
    """A speed-table learner (one target speed per segment, L1 reflex steering, no safety net).

    keep_best=True : keep the best clean plan, revert to it after a bad attempt, freeze crash segments,
                     halve the push after a slower clean lap.
    keep_best=False: always build on the LAST plan: crash -> that segment x0.85, otherwise push all x1.2.
    """
    KIND = "code"

    def __init__(self, keep_best: bool, l1_module, start: float = 18.0, push: float = 0.20, cut: float = 0.85):
        self.keep_best, self.l1, self.start, self.step, self.cut = keep_best, l1_module, start, push, cut
        self.NAME = "Keep-best" if keep_best else "No keep-best"

    def plan(self, track):
        self.n = len(track["waypoints"])
        self.table = [self.start] * self.n
        self.best, self.frozen = None, set()

    def drive(self, obs):
        a = self.l1.drive(obs)
        if a["throttle"] < 0 and obs["speed"] < 1.0:
            return a
        target = self.table[obs["segment"] % self.n]
        dv = target - obs["speed"]
        thr = (dv + 0.075 * obs["speed"]) / 6.0 if dv >= 0 else dv / 12.5
        return {"throttle": dk.clamp(thr), "steering": a["steering"]}

    def reflect(self, rep):
        t = rep["race_time"] if rep["completed"] else None
        crashed = {c["segment"] for c in rep["crash_log"]} | {(c["segment"] - 1) % self.n for c in rep["crash_log"]}
        if self.keep_best:
            if t is not None and not crashed and (self.best is None or t < self.best[0]):
                self.best = (t, list(self.table))
            else:
                self.frozen |= crashed
                self.table = list(self.best[1]) if self.best else [v * self.cut for v in self.table]
                if not crashed:
                    self.step *= 0.5
            self.table = [v if i in self.frozen else min(40.0, v * (1 + self.step)) for i, v in enumerate(self.table)]
        else:
            self.table = [v * self.cut if i in crashed else min(40.0, v * (1 + self.step))
                          for i, v in enumerate(self.table)]


@figure("learning_gone_wrong", "Learning gone wrong: the same speed-table learner with and without keep-best. "
                               "Without it the lap times wander and the plan you end with crashes.")
def fig_learning_wrong():
    key, n = "grand_prix", 10
    l1 = load_module(EXAMPLES / "l1_reflex.py", "agp_fig_l1_table")
    res = {}
    for keep in (True, False):
        reps = run(TableLearner(keep, l1), key, attempts=n)
        res[keep] = [outcome(r) for r in reps]
    DATA["learning_gone_wrong"] = {"track": key, "keep_best": res[True], "no_keep_best": res[False]}

    fig, top = new_fig(12.5, 7.0, "Learning gone wrong: forget what worked, and you wander",
                       "A speed-table learner (one target speed per segment, L1 steering, no safety net), home track, "
                       f"{n} attempts.\nClean lap → push every segment +20 %; crash → slow that segment. With keep-best "
                       "it restarts from its best clean plan;\nwithout, it builds on whatever it did last.")
    ax = fig.add_axes([0.07, 0.13, 0.70, top - 0.2])
    xs = np.arange(1, n + 1)
    for keep, col, lab in ((False, RED, "no keep-best"), (True, BLUE, "keep-best + revert")):
        t = np.array([o["time"] if o["time"] else np.nan for o in res[keep]])
        ax.plot(xs, t, color=col, lw=2.0, alpha=0.9, zorder=3)
        for x, o in zip(xs, res[keep]):
            if o["crashes"]:
                ax.plot(x, o["time"], "o", ms=10, mfc=BG, mec=col, mew=2.2, zorder=5)
                ax.annotate(f"{o['crashes']}", (x, o["time"]), xytext=(0, 9 if keep is False else -10),
                            textcoords="offset points", ha="center", va="bottom" if keep is False else "top",
                            fontsize=8.8, color=col, fontweight="semibold", path_effects=halo())
            else:
                ax.plot(x, o["time"], "o", ms=8, color=col, mec=BG, mew=1.8, zorder=5)
        if keep:
            clean = [o["time"] if (o["time"] and not o["crashes"]) else np.inf for o in res[keep]]
            bsf = np.minimum.accumulate(clean)
            ax.plot(np.r_[xs, n + 0.4], np.r_[bsf, bsf[-1]], color=BLUE, lw=1.2, ls=(0, (4, 3)),
                    drawstyle="steps-post", zorder=2)
        last = res[keep][-1]
        ax.annotate(f"{lab}\nlast plan: {secs(last['time'])}, {last['crashes']} crash"
                    + ("" if last["crashes"] == 1 else "es"),
                    (n, last["time"]), xytext=(14, 0), textcoords="offset points",
                    fontsize=10, color=INK, fontweight="semibold", va="center")
    ax.plot([], [], color=RED, lw=2, label="no keep-best")
    ax.plot([], [], color=BLUE, lw=2, label="keep-best + revert")
    ax.plot([], [], color=BLUE, lw=1.2, ls=(0, (4, 3)), label="keep-best: best clean plan so far")
    ax.plot([], [], "o", ms=9, mfc=BG, mec=INK2, mew=2, label="attempt with crashes (number = crashes)")
    ax.plot([], [], "o", ms=8, color=INK2, mec=BG, label="clean attempt")
    ax.legend(loc="upper right", fontsize=9.5)
    ax.set_xticks(xs)
    ax.set_xlim(0.6, n + 0.4)
    ax.set_xlabel("attempt")
    ax.set_ylabel("lap time (s, incl. +5 s per crash)")
    save(fig, "learning_gone_wrong")


# ============================================================================ 8. latency cliff
class _VirtualClock:
    """Stands in for the `time` module inside agentic_gp.race.session: perf_counter() only advances
    when a driver 'sleeps'. The runner's realtime rule (late ticks = floor(think / 0.5)) runs unchanged,
    but a 3-second think costs no wall time, so the chart is exact and takes a second."""

    def __init__(self):
        self.now = 0.0

    def perf_counter(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds

    def __getattr__(self, name):
        return getattr(_time, name)


class Sleepy:
    """Wraps a driver: every drive() 'thinks' for `latency` seconds before it answers."""
    KIND = "code"

    def __init__(self, path: Path, latency: float, clock):
        self.inner, self.latency, self.clock = load_driver(str(path)), latency, clock
        self.NAME = f"Sleepy {latency:g}s"

    def plan(self, track):
        self.inner.plan(track)

    def drive(self, obs):
        self.clock.sleep(self.latency) if self.clock else _time.sleep(self.latency)
        return self.inner.drive(obs)


def latency_runs(latencies, track="grand_prix", wall_clock: bool = False) -> list[dict]:
    rows = []
    clock = None if wall_clock else _VirtualClock()
    saved = session_mod.time
    if clock:
        session_mod.time = clock
    try:
        for lat in latencies:
            drv = Sleepy(EXAMPLES / "l2_planner.py", lat, clock)
            rep = run(drv, track, realtime=True)[0]
            o = outcome(rep)
            tel = rep.telemetry
            o.update({"latency": lat, "late_ticks": int(rep.late_ticks),
                      "xy": [[r["x"], r["y"]] for r in tel],
                      "crash_xy": [list(map(float, c["position"])) for c in rep.crash_log]})
            rows.append(o)
    finally:
        session_mod.time = saved
    return rows


@figure("latency_cliff", "Latency cliff: the L2 planner in realtime mode with answers 0 / 0.5 / 1 / 2 / 3 s late. "
                         "0 s: clean 52.5 s lap; at 0.5 s and above the old command keeps driving and the lap falls apart.")
def fig_latency_cliff(wall_clock: bool = False):
    lats = [0.0, 0.5, 1.0, 2.0, 3.0]
    rows = latency_runs(lats, wall_clock=wall_clock)
    DATA["latency_cliff"] = [{k: v for k, v in r.items() if k not in ("xy", "crash_xy")} for r in rows]
    tr = make_track("grand_prix")
    fig, top = new_fig(14, 6.5, "The latency cliff: the world does not wait for you",
                       "Same L2 planner, same home track, realtime mode: drive() sleeps before it answers. While it "
                       "thinks, the car keeps its OLD command for floor(latency / 0.5) ticks.\nRed crosses = wall hits "
                       "(+5 s each, the marshal puts the car back). DNF = still out there after 500 ticks.")
    n = len(rows)
    w = 0.94 / n
    for k, r in enumerate(rows):
        ax = fig.add_axes([0.03 + k * w, 0.24, w - 0.01, top - 0.26])
        draw_track(ax, tr, asphalt="#d9d8d2", centerline=False, start=True, lw_edge=0.7)
        P = np.array(r["xy"])
        col = BLUE if r["crashes"] == 0 else RED
        ax.plot(P[:, 0], P[:, 1], color=col if r["crashes"] == 0 else INK2, lw=1.0, alpha=0.9, zorder=4)
        if r["crash_xy"]:
            C = np.array(r["crash_xy"])
            ax.plot(C[:, 0], C[:, 1], "x", color=RED, ms=5.5, mew=1.6, zorder=5)
        frame_axes(ax, tr, pad=12)
        head = f"{r['latency']:g} s late"
        ax.set_title(head, fontsize=13, loc="center", pad=4)
        res = f"{r['time']:.1f} s" if r["time"] else f"DNF ({r['gates']}/16 gates)"
        ax.text(0.5, -0.04, res, transform=ax.transAxes, ha="center", va="top", fontsize=12.5, fontweight="semibold",
                color=INK)
        ax.text(0.5, -0.135, f"{r['crashes']} crashes  ·  {r['late_ticks']} blind ticks", transform=ax.transAxes,
                ha="center", va="top", fontsize=10, color=INK2)
    # bottom strip: crashes per latency
    ax = fig.add_axes([0.035, 0.065, 0.93, 0.06])
    ax.axis("off")
    ax.text(0, 0.5, "Where does the LLM belong?  Not on the TICK clock (0.5 s). On the SESSION clock (plan once) and "
                    "the LAP clock (reflect between attempts) it has seconds to think.",
            fontsize=11, color=INK, va="center", transform=ax.transAxes)
    save(fig, "latency_cliff")


# ============================================================================ 9. ladder lap times
# Real-key MiniMax runs (not re-run here: this script never calls a real LLM). Source: drivers/README.md,
# "With MiniMax (real LLM, MiniMax-M3)", measured 2026-10-21 on grand_prix.
LLM_MEASURED = {"l4_llm_rookie": (78.15, "turn-based, 20 LLM ticks"),
                "l5_llm_strategist": (49.4, "one plan"),
                "l6_llm_crew_chief": (44.65, "qualifying after 3 practice laps")}

LADDER = [  # file, label, agent type, family, attempts
    ("l0_dummy", "L0 Dummy", "no perception", "none", 1),
    ("../template", "Template", "simple reflex (starter)", "reflex", 1),
    ("l1_reflex", "L1 Reflex", "simple reflex", "reflex", 1),
    ("l2_planner", "L2 Planner", "model-based, goal-based", "model", 1),
    ("l2b_racing_line", "L2b Racing line", "model-based, better path", "model", 1),
    ("l3_learner", "L3 Learner", "learning (best of 4)", "learning", 4),
    ("l4_llm_rookie", "L4 Rookie", "LLM policy (mock = code fallback)", "llm_policy", 1),
    ("l5_llm_strategist", "L5 Strategist", "LLM planner (mock = physics plan)", "llm_planner", 1),
    ("l6_llm_crew_chief", "L6 Max", "tool-using multi-role (mock, best of 4)", "crew", 4),
    ("fly_toy", "Fly Toy", "spiking toy (57 LIF neurons)", "spiking", 1),
    ("fly_connectome", "Fly Connectome", "spiking, real FlyWire wiring", "spiking", 1),
]


@figure("ladder_laptimes", "The driver ladder on the home track (one lap; learners best of 4; LLM drivers with the "
                           "offline mock = their code fallback; hollow diamonds = documented MiniMax runs).")
def fig_ladder():
    rows = []
    for stem, label, kind, fam, n in LADDER:
        path = (EXAMPLES / f"{stem}.py").resolve()
        if not path.exists():
            continue
        if stem == "fly_connectome" and not _fly_ok():
            continue
        o = best_of(run(path, "grand_prix", attempts=n))
        rows.append({"stem": Path(stem).name, "label": label, "kind": kind, "family": fam, **o})
    DATA["ladder"] = rows
    fig, top = new_fig(13, 9.4, "The ladder on the home track",
                       "Lap time on grand_prix (lower is better), one lap; the learners show their best of 4 attempts. "
                       "LLM drivers ran with the offline mock here,\nso they race their code fallback. Outlined bars = the "
                       "same driver with the real MiniMax key (measured runs, drivers/README.md).")
    ax = fig.add_axes([0.25, 0.08, 0.66, top - 0.12])
    xmax = 135
    for i, r in enumerate(rows):
        t = r["time"] if r["time"] is not None else xmax
        col = FAMILY[r["family"]]
        clipped = t > xmax
        has_llm = r["stem"] in LLM_MEASURED
        y = i - (0.14 if has_llm else 0.0)
        h = 0.46 if has_llm else 0.6
        w = min(t, xmax - 6)
        ax.barh(y, w, height=h, color=col, zorder=3)
        if clipped:                                    # an axis break drawn on the bar
            for dx in (-2.2, 0.0):
                ax.plot([w - 6 + dx, w - 4 + dx], [y + h / 2 + 0.05, y - h / 2 - 0.05], color=BG, lw=3, zorder=4)
        lab = f"{t:.2f} s" + (f"  ·  {r['crashes']} crashes" if r["crashes"] else "")
        ax.text(w + 1.2, y, lab, va="center", fontsize=10, color=INK, zorder=4, path_effects=halo())
        ax.text(-1.5, i - 0.13, r["label"], ha="right", va="center", fontsize=11, fontweight="semibold", color=INK)
        ax.text(-1.5, i + 0.24, r["kind"], ha="right", va="center", fontsize=8.6, color=INK2)
        if has_llm:
            v, note = LLM_MEASURED[r["stem"]]
            ax.barh(i + 0.26, v, height=0.22, color=BG, ec=col if r["family"] != "crew" else GOLD, lw=1.8, zorder=3)
            ax.text(v + 1.2, i + 0.26, f"with MiniMax: {v:g} s  ({note})", va="center", fontsize=9, color=INK,
                    zorder=4, path_effects=halo())
    best_code = min(r["time"] for r in rows if r["time"] and r["family"] in ("model", "learning"))
    ax.axvline(best_code, color=INK2, lw=1, ls=(0, (3, 3)), zorder=2)
    ax.text(best_code, -0.62, f"best code lap: {best_code:g} s", fontsize=9, color=INK2, ha="center", zorder=6,
            bbox=dict(boxstyle="round,pad=0.25", fc=BG, ec="none"))
    ax.set_yticks([])
    ax.set_ylim(len(rows) - 0.45, -0.85)
    ax.set_xlim(0, xmax)
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("lap time on grand_prix (s, incl. +5 s per crash)")
    save(fig, "ladder_laptimes")


# ============================================================================ 10. crash loop
def naive_chaser(obs):
    """The obvious first driver: full throttle, steer at the next waypoint."""
    return 1.0, max(-1.0, min(1.0, obs["angle_to_next_waypoint"] / 30.0))


@figure("crash_loop", "The crash loop under the classic crash rule: a stopped car cannot turn, and its waypoint "
                      "is straight ahead behind a wall. The marshal rule (default) puts the car back on track.")
def fig_crash_loop():
    key = "grand_prix"
    tr = make_track(key)
    out = {}
    for rule in ("classic", "marshal"):
        rep = run(naive_chaser, key, physics=PhysicsConfig(crash_rule=rule))[0]
        out[rule] = (rep, outcome(rep))
    DATA["crash_loop"] = {r: o for r, (_, o) in out.items()}
    rep_c, oc = out["classic"]
    rep_m, om = out["marshal"]
    fig, top = new_fig(14, 7.8, "Why the obvious driver crash-loops",
                       "drive(obs) = full throttle, steering = angle_to_next_waypoint / 30.  "
                       "Left: the CLASSIC crash rule (stop + 2-unit nudge). Right: the default MARSHAL rule.")
    # left: zoom on where the classic car got stuck
    C = np.array([c["position"] for c in rep_c.crash_log])
    stuck = C[-1]
    ax = fig.add_axes([0.02, 0.07, 0.46, top - 0.08])
    draw_track(ax, tr, gates=True, gate_labels=False, start=False)
    tel = rep_c.telemetry
    P = np.array([[r["x"], r["y"]] for r in tel])
    ax.plot(P[:, 0], P[:, 1], color=ORANGE, lw=1.6, zorder=4)
    ax.plot(C[:, 0], C[:, 1], "x", color=RED, ms=6, mew=1.5, zorder=6)
    obs_last = rep_c.frames[-1]["obs"]
    nxt = int(obs_last["next_waypoint_index"])
    gate_c = tr.gate_centers[nxt % tr.n_gates]
    car = np.array(obs_last["position"], dtype=float)
    heading = math.radians(obs_last["heading_deg"])
    x1, y1, x2, y2 = tr.gates[nxt % tr.n_gates]
    ax.plot([x1, x2], [y1, y2], color=YELLOW, lw=3.5, zorder=3, solid_capstyle="butt")
    # the line of sight to the gate: which part of it is NOT on the track?
    ts = np.linspace(0, 1, 400)
    line = car[None, :] + ts[:, None] * (gate_c - car)[None, :]
    off = np.array([abs(tr.project(p)[1]) > tr.half_width for p in line]) if hasattr(tr, "project") else None
    ax.plot(line[:, 0], line[:, 1], color="white", lw=1.6, ls=(0, (4, 3)), zorder=7)
    if off is not None and off.any():
        ax.plot(line[off, 0], line[off, 1], color=RED, lw=4, zorder=7.5, solid_capstyle="butt")
    ax.annotate("", gate_c, line[-12], arrowprops=dict(arrowstyle="-|>", color="white", lw=1.6, mutation_scale=16),
                zorder=8)
    _car(ax, car, heading, size=2.6, color=ORANGE, zorder=9)
    ax.annotate(f"next waypoint: gate {nxt}\nangle_to_next_waypoint = {abs(obs_last['angle_to_next_waypoint']):.0f}°  "
                f"→  steering 0", gate_c, xytext=(70, 6), textcoords="offset points", fontsize=10.5, color=INK,
                zorder=9, bbox=dict(boxstyle="round,pad=0.4", fc=BG, ec=GRID),
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.9))
    ax.annotate(f"distance_to_wall_front = {obs_last['distance_to_wall_front']:g}\n"
                f"the wall is BETWEEN the car and its waypoint", car, xytext=(28, -34), textcoords="offset points",
                fontsize=10.5, color=INK, zorder=9, bbox=dict(boxstyle="round,pad=0.4", fc=BG, ec=GRID),
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.9))
    mid = 0.5 * (car + gate_c)
    span = max(26.0, 0.8 * float(np.linalg.norm(gate_c - car)))
    ax.set_xlim(mid[0] - span * 1.25, mid[0] + span * 1.25)
    ax.set_ylim(mid[1] - span, mid[1] + span)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"classic rule: {oc['crashes']} crashes, DNF after {oc['ticks']} ticks "
                 f"({oc['gates']}/16 gates)", fontsize=12.5, color=INK)
    ax.text(0.02, 0.03, "waypoint dead ahead → steering 0 → full throttle into the wall\n"
                        "classic rule: stop, nudge back 2 units, repeat (red × = every hit)",
            transform=ax.transAxes, fontsize=10.5, color=INK, va="bottom",
            bbox=dict(boxstyle="round,pad=0.5", fc=BG, ec=GRID))
    # right: the full lap under the marshal rule
    ax2 = fig.add_axes([0.53, 0.07, 0.45, top - 0.08])
    draw_track(ax2, tr, asphalt="#d9d8d2", centerline=False, start=True, lw_edge=0.7)
    P2 = np.array([[r["x"], r["y"]] for r in rep_m.telemetry])
    ax2.plot(P2[:, 0], P2[:, 1], color=BLUE, lw=1.4, zorder=4)
    C2 = np.array([c["position"] for c in rep_m.crash_log]) if rep_m.crash_log else np.zeros((0, 2))
    if len(C2):
        ax2.plot(C2[:, 0], C2[:, 1], "x", color=RED, ms=7, mew=1.8, zorder=6)
    frame_axes(ax2, tr, pad=14)
    ax2.set_title(f"marshal rule: finishes in {secs(om['time'])} with {om['crashes']} crashes", fontsize=12.5)
    ax2.text(0.5, -0.02, "after each hit the car is put back on the centerline, pointing along the track (+5 s)",
             transform=ax2.transAxes, ha="center", va="top", fontsize=10, color=INK2)
    save(fig, "crash_loop")


# ============================================================================ 11-12. fly figures
def _fly_ok() -> bool:
    try:
        from agentic_gp.fly.connectome import circuit_path
        return circuit_path().exists()
    except Exception:                                   # noqa: BLE001
        return False


@figure("fly_tuning_curve", "Fly 'electrophysiology': steering readout vs target bearing for the real FlyWire wiring "
                            "and 5 scrambled copies (same neurons, same connection counts, random wiring).")
def fig_fly_tuning():
    if not _fly_ok():
        SAY("  (skipped: FlyWire circuit not found - see agentic_gp/fly/README.md)")
        return
    from agentic_gp.fly import load_circuit
    from agentic_gp.fly.viz import bearing_sweep, lateralization
    angles = list(range(-90, 91, 10))
    sw = bearing_sweep(load_circuit(), angles=angles, scramble_seeds=range(5))
    real = np.clip(sw["real"]["steer_raw"], -1, 1)
    lat_real = lateralization(sw["real"], sw["angles"])
    lats = {s: lateralization(c, sw["angles"]) for s, c in sw["scrambled"].items()}
    DATA["fly_tuning"] = {"angles": angles, "real_steer": [round(float(x), 3) for x in real],
                          "lateralization_real": round(lat_real, 3),
                          "lateralization_scrambled": {str(k): round(v, 3) for k, v in lats.items()},
                          "dna02_real": {g: [round(x, 1) for x in sw["real"][g]] for g in ("DNa02_L", "DNa02_R")}}
    fig, top = new_fig(13.5, 7.0, "Does the wiring matter? Real fly vs scrambled fly",
                       "Put a target at each bearing (no walls), simulate 500 ms of 4,494 FlyWire neurons, read the "
                       "steering. Same input neurons, same readout, same decoder for every brain.\nScrambled = same "
                       "neurons and connection counts, wiring shuffled (5 seeds).")
    ax = fig.add_axes([0.07, 0.14, 0.50, top - 0.21])
    ax.axhline(0, color=AXIS, lw=1)
    ax.axvline(0, color=AXIS, lw=1)
    for s, c in sw["scrambled"].items():
        ax.plot(angles, np.clip(c["steer_raw"], -1, 1), color="#a9a7a0", lw=1.4, zorder=2)
    ax.plot(angles, real, color=MAGENTA, lw=3, zorder=4)
    ax.plot(angles, real, "o", color=MAGENTA, ms=6, mec=BG, mew=1.5, zorder=5)
    ax.text(62, 0.78, f"real wiring\nindex {lat_real:+.2f}", color=INK, fontsize=10.5, fontweight="semibold",
            ha="center")
    ax.text(-58, 0.33, f"5 scrambled\nindex {round(min(lats.values()), 2) + 0:+.2f} … {round(max(lats.values()), 2) + 0:+.2f}", color=INK2,
            fontsize=10, ha="center")
    ax.set_xlim(-95, 95)
    ax.set_ylim(-1.15, 1.15)
    ax.set_xticks(range(-90, 91, 30), [f"{a:+d}°" if a else "0°" for a in range(-90, 91, 30)])
    ax.set_xlabel("target bearing (deg, + = right)")
    ax.set_ylabel("steering readout (−1 left … +1 right)")
    ax.set_title("Steering follows the target only with the real wiring", fontsize=11.5)
    ax2 = fig.add_axes([0.65, 0.14, 0.32, top - 0.21])
    ax2.plot(angles, sw["real"]["DNa02_L"], color=VIOLET, lw=2.2, label="DNa02 left")
    ax2.plot(angles, sw["real"]["DNa02_R"], color=ORANGE, lw=2.2, label="DNa02 right")
    ax2.plot(angles, sw["real"]["AOTU019_L"], color=VIOLET, lw=1.4, ls=(0, (3, 2)), label="AOTU019 left")
    ax2.plot(angles, sw["real"]["AOTU019_R"], color=ORANGE, lw=1.4, ls=(0, (3, 2)), label="AOTU019 right")
    ax2.set_xticks(range(-90, 91, 45), [f"{a:+d}°" if a else "0°" for a in range(-90, 91, 45)])
    ax2.set_xlabel("target bearing")
    ax2.set_ylabel("firing rate (Hz)")
    ax2.set_title("Real wiring: who fires?", fontsize=11.5)
    for g, xa, dy, ha in (("DNa02_L", -85, 8, "left"), ("DNa02_R", 88, 8, "right"),
                          ("AOTU019_L", -20, 10, "right"), ("AOTU019_R", 20, 10, "left")):
        yv = float(np.interp(xa, angles, sw["real"][g]))
        ax2.annotate(g.replace("_L", " left").replace("_R", " right"), (xa, yv), xytext=(0, dy),
                     textcoords="offset points", ha=ha, va="bottom", fontsize=9.5, color=INK, path_effects=halo())
    ax2.set_ylim(0, 215)
    ax2.text(0.0, -0.155, "solid = DNa02 (descending neuron)   dashed = AOTU019 (hub)",
             transform=ax2.transAxes, fontsize=9, color=INK2, va="top")
    save(fig, "fly_tuning_curve")


def _fly_race(args) -> dict:
    """One fly_connectome lap in a worker process (real wiring or a scrambled seed)."""
    track, scramble = args
    os.environ["LLM_PROVIDER"] = "mock"
    from agentic_gp.race.loader import load_driver as _ld
    from agentic_gp.race.session import SessionConfig as _SC, run_session as _rs
    params = {} if scramble is None else {"SCRAMBLE": scramble}
    rep = _rs(_ld(str(EXAMPLES / "fly_connectome.py"), params=params), _SC(track=track)).attempts[0]
    r = rep.result
    return {"track": track, "scramble": scramble, "time": round(float(r.race_time), 2) if r.completed else None,
            "crashes": int(r.crashes), "gates": int(r.waypoints_passed), "ticks": int(r.ticks)}


@figure("fly_real_vs_scrambled", "Real vs scrambled FlyWire wiring racing the same car: one lap each on three "
                                 "tracks (scrambled seeds 0-4; seed 4 drives in circles).")
def fig_fly_laps():
    if not _fly_ok():
        SAY("  (skipped: FlyWire circuit not found - see agentic_gp/fly/README.md)")
        return
    tracks = ["oval", "sprint", "grand_prix"]
    brains = [None, 0, 1, 2, 3, 4]
    jobs = [(t, s) for t in tracks for s in brains]
    with ProcessPoolExecutor(max_workers=min(len(jobs), max(1, (os.cpu_count() or 2) - 1))) as pool:
        res = list(pool.map(_fly_race, jobs))
    DATA["fly_real_vs_scrambled"] = res
    fig, top = new_fig(13.5, 7.0, "Real vs scrambled wiring: same eyes, same readout, different brain",
                       "drivers/examples/fly_connectome.py, one lap per cell. Scrambled seeds 0-4 keep every neuron and "
                       "every neuron's number of connections, but shuffle who connects to whom.\nBars: race time "
                       "(incl. +5 s per crash). DNF = 500 ticks without finishing.")
    ax = fig.add_axes([0.07, 0.13, 0.90, top - 0.2])
    cols = [MAGENTA] + ["#b9b7b0"] * 5
    gw = 0.84 / len(brains)
    finished = [r["time"] for r in res if r["time"] is not None]
    dnf_h = 50 * math.ceil((max(finished) + 40) / 50)            # DNF bars reach above every finished lap
    for ti, t in enumerate(tracks):
        for bi, b in enumerate(brains):
            o = next(r for r in res if r["track"] == t and r["scramble"] == b)
            x = ti + (bi - (len(brains) - 1) / 2) * gw
            h = o["time"] if o["time"] is not None else dnf_h
            ax.bar(x, h, width=gw * 0.86, color=cols[bi] if o["time"] else BG, zorder=3,
                   hatch=None if o["time"] else "////", edgecolor="#a9a7a0" if o["time"] is None else "none", lw=1)
            lab = f"{o['time']:.0f} s" if o["time"] else f"DNF · {o['crashes']}×"
            ax.text(x, h + 5, lab, ha="center", va="bottom", fontsize=9.5 if bi else 11,
                    fontweight="semibold", color=INK, zorder=5)
            if o["crashes"] and o["time"]:
                ax.text(x, h - 6, f"{o['crashes']}×", ha="center", va="top", fontsize=8.6, zorder=5,
                        color="white" if o["time"] else INK2, fontweight="semibold")
        ax.text(ti, -18, t, ha="center", va="top", fontsize=12.5, fontweight="semibold", color=INK)
        r0 = next(r for r in res if r["track"] == t and r["scramble"] is None)
        if r0["time"]:
            sc = [r["time"] for r in res if r["track"] == t and r["scramble"] is not None and r["time"]]
            if sc:
                ax.text(ti, -44, f"scrambled: {min(sc) / r0['time']:.1f}-{max(sc) / r0['time']:.1f}× slower",
                        ha="center", va="top", fontsize=10, color=INK2)
    ax.set_xticks([])
    ax.set_xlim(-0.55, len(tracks) - 0.45)
    ax.set_ylim(0, dnf_h + 45)
    ax.spines["bottom"].set_visible(False)
    ax.set_ylabel("race time, one lap (s)")
    ax.bar(0, 0, color=MAGENTA, label="real FlyWire wiring")
    ax.bar(0, 0, color="#b9b7b0", label="scrambled wiring, seeds 0 1 2 3 4 (left to right)")
    ax.bar(0, 0, color=BG, hatch="////", edgecolor="#a9a7a0", label="did not finish (DNF)")
    ax.legend(loc="lower left", fontsize=10, ncol=4, bbox_to_anchor=(0.0, 1.0), handlelength=1.6)
    ax.text(1.0, 1.035, "N× = N crashes", transform=ax.transAxes, ha="right", va="center", fontsize=10, color=INK2)
    save(fig, "fly_real_vs_scrambled")


# ============================================================================ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", help="figure names (see --list); track_<key> = all track maps")
    ap.add_argument("--list", action="store_true", help="list the figures and exit")
    ap.add_argument("--wall-clock", action="store_true",
                    help="latency cliff with real time.sleep (about 10 minutes) instead of the virtual clock")
    args = ap.parse_args(argv)
    if args.list:
        for name, (_, cap) in FIGURES.items():
            print(f"{name:24s} {cap}")
        return 0
    ASSETS.mkdir(parents=True, exist_ok=True)
    data_path = ASSETS / "figures_data.json"
    if data_path.exists():
        try:
            DATA.update(json.loads(data_path.read_text(encoding="utf-8")))
        except ValueError:
            pass
    wanted = args.only or list(FIGURES)
    unknown = [w for w in wanted if w not in FIGURES]
    if unknown:
        ap.error(f"unknown figure(s) {unknown}; see --list")
    t0 = _time.perf_counter()
    for name in wanted:
        fn, _ = FIGURES[name]
        SAY(f"[{name}]")
        t1 = _time.perf_counter()
        fn(wall_clock=args.wall_clock) if name == "latency_cliff" else fn()
        SAY(f"  ({_time.perf_counter() - t1:.1f} s)")
    DATA["_about"] = ("Measured by lesson/make_figures.py with agentic_gp (LLM_PROVIDER=mock). "
                      "Times = race time (sim time + 5 s per crash + 1 s per tick too fast in a zone).")
    data_path.write_text(json.dumps(DATA, indent=1, sort_keys=True, default=float) + "\n", encoding="utf-8")
    SAY(f"wrote {data_path.relative_to(ROOT)}  -  done in {_time.perf_counter() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
