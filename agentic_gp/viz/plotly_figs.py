"""Plotly figures for the Streamlit app and lessons (interactive: hover, zoom, legend toggles).

    from agentic_gp.viz import plotly_figs as pf
    fig = pf.track_figure("grand_prix")                       # walls, gates, T1.., v_max colours, zones
    pf.add_car(fig, x, y, heading_deg, color="#e63946", name="me")
    pf.add_rays(fig, obs)                                     # the 7 wall rays, coloured by distance
    pf.add_target(fig, debug["target"], car=(x, y))
    st.plotly_chart(fig, theme=None)                          # theme=None keeps these dark colours

    pf.telemetry_figure(report)                               # speed / throttle / steering vs time
    pf.learning_curve_figure([session_dict, ...])            # lap time per attempt, per driver
    pf.segment_heatmap(session_dict)                          # attempts x segments: time lost vs best
    pf.latency_histogram([0.8, 1.2, ...])                     # LLM latency vs the 0.5 s tick
    pf.speed_profile_figure("grand_prix", actual=report)      # grip limit -> braking pass -> actual

Inputs are forgiving: a track can be a key ("sprint", "random:7"), a sim Track, a plan() track
map, a replay track or a whole replay; telemetry can be a list of rows, a DataFrame, a report
dict / AttemptReport, a session dict or a replay car. Colours follow the dark broadcast theme of
the canvases (agentic_gp/viz/assets/agp.css). Every function returns a go.Figure.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np

# ---------------------------------------------------------------------------- theme
SURFACE = "#1a1a19"
PAGE = "#0d0d0d"
INK = "#f0efec"
INK2 = "#c3c2b7"
MUTED = "#898781"
GRID = "#2c2c2a"
AXIS = "#383835"
ASPHALT = "#34373c"
WALL = "#e9e7e0"
ZONE = "#fab219"
GOOD, WARN, BAD = "#0ca30c", "#fab219", "#d03b3b"
ACCENT = "#3987e5"
SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]  # dark-mode slots
# one hue (orange): "how slow is this corner" / "how much time was lost" - dim = little, bright = a lot
ORANGE_SCALE = [[0.0, "#3a3532"], [0.35, "#7a4a2c"], [0.7, "#d95926"], [1.0, "#ffb27a"]]
# one hue (blue) for speed: slow = dark, fast = bright (dark background)
BLUE_SCALE = [[0.0, "#104281"], [0.5, "#3987e5"], [1.0, "#cde2fb"]]
FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'


def _go():
    import plotly.graph_objects as go

    return go


def style(fig, *, height: int | None = None, title: str | None = None, legend: bool = True):
    """Apply the dark theme (also handy for your own figures)."""
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=PAGE, plot_bgcolor=SURFACE,
        font=dict(family=FONT, color=INK2, size=13), title=dict(text=title, font=dict(color=INK, size=16)) if title else None,
        margin=dict(l=50, r=20, t=50 if title else 20, b=40), height=height, showlegend=legend,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=INK2)),
        hoverlabel=dict(bgcolor="#262624", bordercolor="#444", font=dict(color=INK, family=FONT)),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=AXIS, linecolor=AXIS, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=AXIS, linecolor=AXIS, tickfont=dict(color=MUTED))
    return fig


# ---------------------------------------------------------------------------- inputs
def _track(track: Any) -> dict:
    from .player import track_payload

    if isinstance(track, dict) and "view" in track and "center_x" in track:
        return track
    return track_payload(track)


def _rows(data: Any) -> list[dict]:
    """Telemetry-like input -> list of row dicts (t, speed, throttle, steering, x, y, ...)."""
    if data is None:
        return []
    if hasattr(data, "to_dict") and hasattr(data, "columns"):          # DataFrame
        return data.to_dict("records")
    if hasattr(data, "telemetry") and not isinstance(data, dict):      # AttemptReport
        return list(data.telemetry)
    if isinstance(data, dict):
        if "telemetry" in data:
            return list(data["telemetry"])
        if "reports" in data and data["reports"]:                        # session dict: last attempt
            return list(data["reports"][-1].get("telemetry", []))
        if "t" in data and isinstance(data["t"], list):                  # replay car
            keys = [k for k in ("t", "x", "y", "speed", "throttle", "steering", "lap", "race_progress") if k in data]
            return [dict(zip(keys, vals)) for vals in zip(*(data[k] for k in keys))]
        if "frames" in data:
            return list(data["frames"])
    return [dict(r) for r in data]


def _xy(track: dict, prefix: str) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(track[f"{prefix}_x"], float), np.asarray(track[f"{prefix}_y"], float)


def _closed(a: np.ndarray) -> list:
    return list(a) + [a[0]]


# ---------------------------------------------------------------------------- track
def track_figure(track: Any = "grand_prix", *, vmax: bool = True, corners: bool = True, gates: bool = True,
                 zones: bool = True, gate_numbers: bool = False, title: str | None = None, height: int = 560):
    """The circuit: asphalt between the walls, gates (gate 0 = start/finish), corner labels T1..,
    centerline coloured by the corner speed limit v_max = sqrt(20 R) (hover for numbers), zones."""
    go = _go()
    tr = _track(track)
    view = tr["view"]
    cx, cy = _xy(tr, "center")
    lx, ly = _xy(tr, "left")
    rx, ry = _xy(tr, "right")
    fig = go.Figure()
    # asphalt = the outer wall loop filled grey, then the inner loop filled with the background
    area = lambda x, y: abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))) / 2
    (ox, oy), (ix, iy) = ((lx, ly), (rx, ry)) if area(lx, ly) >= area(rx, ry) else ((rx, ry), (lx, ly))
    fig.add_trace(go.Scatter(x=_closed(ox), y=_closed(oy), fill="toself", fillcolor=ASPHALT, mode="lines",
                             line=dict(color=WALL, width=1.6), name="walls", hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=_closed(ix), y=_closed(iy), fill="toself", fillcolor=SURFACE, mode="lines",
                             line=dict(color=WALL, width=1.6), name="walls", hoverinfo="skip", showlegend=False))
    n_g = len(tr.get("gate_x1", []))
    seg_of = _segment_of_points(tr)
    if zones:
        for z in tr.get("zones", []):
            idx = _index_run(z["from_index"], z["to_index"], len(cx))
            fig.add_trace(go.Scatter(x=cx[idx], y=cy[idx], mode="lines", line=dict(color="rgba(250,178,25,0.35)",
                                     width=14), name=f"{z.get('label', 'zone')} <= {z['max_speed']:g}",
                                     hovertemplate=f"{z.get('label', 'zone')}: max {z['max_speed']:g} u/s<extra></extra>"))
    if gates and n_g:
        gx, gy = [], []
        for g in range(1, n_g):
            gx += [tr["gate_x1"][g], tr["gate_x2"][g], None]
            gy += [tr["gate_y1"][g], tr["gate_y2"][g], None]
        fig.add_trace(go.Scatter(x=gx, y=gy, mode="lines", line=dict(color="rgba(255,255,255,0.25)", width=1),
                                 name="gates", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[tr["gate_x1"][0], tr["gate_x2"][0]], y=[tr["gate_y1"][0], tr["gate_y2"][0]],
                                 mode="lines", line=dict(color=INK, width=5, dash="dot"), name="start / finish",
                                 hoverinfo="name"))
        if gate_numbers:
            ux = (np.asarray(tr["gate_x2"]) - np.asarray(tr["gate_x1"]))
            uy = (np.asarray(tr["gate_y2"]) - np.asarray(tr["gate_y1"]))
            d = np.hypot(ux, uy) + 1e-9
            fig.add_trace(go.Scatter(x=np.asarray(tr["gate_x2"]) + ux / d * 6, y=np.asarray(tr["gate_y2"]) + uy / d * 6,
                                     mode="text", text=[str(g) for g in range(n_g)], textfont=dict(color=MUTED, size=11),
                                     name="gate numbers", hoverinfo="skip"))
    if vmax:
        v = np.asarray(view["vmax"], float)
        top = float(view.get("max_speed", 40.0))
        fig.add_trace(go.Scatter(
            x=cx, y=cy, mode="markers", name="v_max (centerline)",
            marker=dict(size=5, color=top - v, colorscale=ORANGE_SCALE, cmin=0, cmax=max(1.0, top - 8.0),
                        colorbar=dict(title=dict(text="corner<br>speed", font=dict(color=MUTED)), thickness=12,
                                      tickvals=[0, (top - 8) / 2, top - 8],
                                      ticktext=[f"{top:.0f}", f"{(top + 8) / 2:.0f}", "<=8"], tickfont=dict(color=MUTED))),
            customdata=np.column_stack([np.arange(len(cx)), v, seg_of]),
            hovertemplate="point %{customdata[0]}<br>v_max %{customdata[1]:.1f} u/s<br>segment %{customdata[2]}<extra></extra>"))
    else:
        fig.add_trace(go.Scatter(x=_closed(cx), y=_closed(cy), mode="lines", line=dict(color="rgba(255,255,255,0.2)",
                                 width=1, dash="dash"), name="centerline", hoverinfo="skip"))
    if corners:
        for c in view.get("corners", []):
            a = int(c["apex_index"])
            wx, wy = (lx[a], ly[a]) if c.get("direction") == "right" else (rx[a], ry[a])
            dx, dy = wx - cx[a], wy - cy[a]
            d = math.hypot(dx, dy) or 1.0
            fig.add_annotation(x=wx + dx / d * 11, y=wy + dy / d * 11, text=f"<b>{c['id']}</b>", showarrow=False,
                               font=dict(color=INK, size=12), bgcolor="rgba(13,13,13,0.7)",
                               hovertext=f"{c['id']}: {c.get('direction', '')}, R {c.get('min_radius', '?')}, "
                                         f"v_max {c.get('v_max', '?')} u/s")
    style(fig, height=height, title=title or tr.get("name"))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
    fig.update_layout(plot_bgcolor=SURFACE, legend=dict(orientation="h", y=-0.02, x=0))
    return fig


def _index_run(i0: int, i1: int, n: int) -> list[int]:
    span = (i1 - i0) % n
    return [(i0 + k) % n for k in range(span + 1)]


def _segment_of_points(tr: dict) -> np.ndarray:
    """Waypoint segment of every centerline point (gate index the point comes after)."""
    cx, cy = _xy(tr, "center")
    n = len(cx)
    gx = (np.asarray(tr.get("gate_x1", []), float) + np.asarray(tr.get("gate_x2", []), float)) / 2
    gy = (np.asarray(tr.get("gate_y1", []), float) + np.asarray(tr.get("gate_y2", []), float)) / 2
    if not len(gx):
        return np.zeros(n, int)
    idx = [int(np.argmin((cx - x) ** 2 + (cy - y) ** 2)) for x, y in zip(gx, gy)]
    seg = np.zeros(n, int)
    order = sorted(range(len(idx)), key=lambda g: idx[g])
    for k, g in enumerate(order):
        start, end = idx[g], idx[order[(k + 1) % len(order)]]
        for i in _index_run(start, end, n)[:-1] if len(order) > 1 else range(n):
            seg[i] = g
    return seg


# ---------------------------------------------------------------------------- overlays
def car_polygon(x: float, y: float, heading_deg: float, length: float = 6.0, width: float = 3.0) -> tuple[list, list]:
    """Outline of a car (arrow shape) at (x, y) pointing along heading_deg (world, counter-clockwise)."""
    h = math.radians(heading_deg)
    c, s = math.cos(h), math.sin(h)
    pts = [(0.55, 0.0), (0.1, 0.5), (-0.5, 0.5), (-0.5, -0.5), (0.1, -0.5), (0.55, 0.0)]
    xs = [x + c * px * length - s * py * width for px, py in pts]
    ys = [y + s * px * length + c * py * width for px, py in pts]
    return xs, ys


def add_car(fig, x: float, y: float, heading_deg: float = 0.0, *, color: str = ACCENT, name: str = "car",
            length: float = 7.0, hover: str | None = None, legend: bool | None = None):
    """A filled car shape (true to scale-ish; zoom in to see it) + a marker that stays visible when zoomed out.
    legend: list the car in the legend (default: only when it has a name - no empty legend rows)."""
    go = _go()
    xs, ys = car_polygon(x, y, heading_deg, length, length * 0.5)
    fig.add_trace(go.Scatter(x=xs, y=ys, fill="toself", fillcolor=color, mode="lines",
                             line=dict(color="#111", width=1), name=name, hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers+text", text=[name], textposition="top center",
                             textfont=dict(color=INK, size=12),
                             marker=dict(size=11, color=color, line=dict(color="#111", width=1.5),
                                         symbol="triangle-up", angle=90 - heading_deg),
                             name=name, hovertext=hover or f"{name}<br>x {x:.1f}, y {y:.1f}, heading {heading_deg:.0f} deg",
                             hoverinfo="text", showlegend=bool(name) if legend is None else legend))
    return fig


def ray_color(dist: float, max_dist: float = 200.0) -> str:
    """Near = red (< 10), close = amber (< 25), clear = green."""
    return BAD if dist < 10 else WARN if dist < 25 else GOOD


def add_rays(fig, obs: dict, *, max_dist: float = 200.0, labels: bool = True):
    """The wall rays of an observation, drawn from the car, coloured by distance (driver frame: + = right)."""
    go = _go()
    x, y = obs["position"]
    h = float(obs["heading_deg"])
    for k, v in sorted(obs.items()):
        if not k.startswith("distance_to_wall_"):
            continue
        name = k[len("distance_to_wall_"):]
        side, _, deg = name.partition("_")
        angle = 0.0 if name == "front" else float(deg) * (1 if side == "right" else -1)
        a = math.radians(h - angle)
        d = min(float(v), max_dist)
        ex, ey = x + math.cos(a) * d, y + math.sin(a) * d
        col = ray_color(float(v), max_dist)
        fig.add_trace(go.Scatter(x=[x, ex], y=[y, ey], mode="lines+markers" if labels else "lines",
                                 line=dict(color=col, width=2.5), marker=dict(size=[0, 7], color=col),
                                 name=f"ray {angle:+.0f} deg", showlegend=False,
                                 hovertext=f"{k}: {float(v):.1f}", hoverinfo="text"))
        if labels:
            fig.add_annotation(x=ex, y=ey, text=f"{float(v):.0f}", showarrow=False, yshift=10,
                               font=dict(color=INK, size=11), bgcolor="rgba(13,13,13,0.7)")
    return fig


def add_target(fig, target: Any, *, car: Any = None, name: str = "target"):
    """The driver's target point (debug["target"]) with a dashed line from the car."""
    go = _go()
    if target is None:
        return fig
    tx, ty = float(target[0]), float(target[1])
    if car is not None:
        fig.add_trace(go.Scatter(x=[car[0], tx], y=[car[1], ty], mode="lines", line=dict(color="#e87ba4", dash="dot"),
                                 showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[tx], y=[ty], mode="markers", name=name,
                             marker=dict(symbol="circle-cross-open", size=16, color="#e87ba4", line=dict(width=2)),
                             hovertext=f"{name}: ({tx:.1f}, {ty:.1f})", hoverinfo="text"))
    return fig


def add_trajectory(fig, data: Any, *, name: str = "lap", color: str | None = None, by_speed: bool = True,
                   max_speed: float = 40.0):
    """A driven line (telemetry / replay car / frames); coloured by speed (hover shows t and speed)."""
    go = _go()
    rows = _rows(data)
    if not rows:
        return fig
    x = [r["x"] for r in rows]
    y = [r["y"] for r in rows]
    sp = [float(r.get("speed", 0.0)) for r in rows]
    t = [float(r.get("t", i)) for i, r in enumerate(rows)]
    if by_speed:
        fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name=name,
                                 marker=dict(size=6, color=sp, colorscale=BLUE_SCALE, cmin=0, cmax=max_speed,
                                             colorbar=dict(title=dict(text="speed", font=dict(color=MUTED)), thickness=12,
                                                           x=1.08, tickfont=dict(color=MUTED))),
                                 customdata=np.column_stack([t, sp]),
                                 hovertemplate=f"{name}<br>t %{{customdata[0]:.1f}} s<br>speed %{{customdata[1]:.1f}}<extra></extra>"))
    else:
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name, line=dict(color=color or ACCENT, width=2.5),
                                 customdata=t, hovertemplate=f"{name}<br>t %{{customdata:.1f}} s<extra></extra>"))
    return fig


def add_crashes(fig, points: Iterable, *, name: str = "crashes", labels: list[str] | None = None):
    go = _go()
    pts = [tuple(p) for p in points or []]
    if not pts:
        return fig
    fig.add_trace(go.Scatter(x=[p[0] for p in pts], y=[p[1] for p in pts], mode="markers", name=name,
                             marker=dict(symbol="x", size=12, color=BAD, line=dict(width=2.5, color=BAD)),
                             hovertext=labels or [f"crash {i + 1}" for i in range(len(pts))], hoverinfo="text"))
    return fig


def add_turning_circle(fig, x: float, y: float, heading_deg: float, speed: float, steering: float, *,
                       a_lat: float = 20.0, max_yaw_rate: float = 1.5, color: str = "#9085e9"):
    """The circle the car would drive with this speed and steering: R = v^2 / (a_lat * |s|),
    never tighter than v / max_yaw_rate (Physics Lab). Steering + = right."""
    go = _go()
    if abs(steering) < 1e-3 or speed <= 0:
        return fig
    r = max(speed ** 2 / (a_lat * abs(steering)), speed / max_yaw_rate)
    h = math.radians(heading_deg)
    side = -1.0 if steering > 0 else 1.0                       # right turn: centre on the right (-90 deg)
    ccx, ccy = x + r * math.cos(h + side * math.pi / 2), y + r * math.sin(h + side * math.pi / 2)
    a = np.linspace(0, 2 * math.pi, 181)
    fig.add_trace(go.Scatter(x=ccx + r * np.cos(a), y=ccy + r * np.sin(a), mode="lines",
                             line=dict(color=color, width=2, dash="dash"), name=f"turning circle R={r:.0f}",
                             hoverinfo="name"))
    return fig


# ---------------------------------------------------------------------------- telemetry
def telemetry_figure(data: Any, *, x: str = "t", compare: Any = None, names: tuple[str, str] = ("this lap", "compare"),
                     color: str = ACCENT, height: int = 560, title: str | None = None):
    """Speed, throttle/brake and steering, one panel each, shared x axis (x = "t" | "progress" | "tick").
    `compare`: a second telemetry drawn dashed (e.g. your best attempt)."""
    from plotly.subplots import make_subplots

    go = _go()
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.5, 0.25, 0.25],
                        subplot_titles=("speed (u/s)", "throttle (+) / brake (-)", "steering (+ = right)"))
    series = [(data, names[0], color, "solid")]
    if compare is not None:
        series.append((compare, names[1], "#d95926", "dash"))
    for k, (src, name, col, dash) in enumerate(series):
        rows = _rows(src)
        if not rows:
            continue
        xs = [r.get(x, i) for i, r in enumerate(rows)]
        for row, key in ((1, "speed"), (2, "throttle"), (3, "steering")):
            fig.add_trace(go.Scatter(x=xs, y=[r.get(key, 0.0) for r in rows], mode="lines", name=name,
                                     legendgroup=name, showlegend=row == 1, line=dict(color=col, width=2, dash=dash),
                                     hovertemplate=f"{name}<br>{x} %{{x}}<br>{key} %{{y:.2f}}<extra></extra>"),
                          row=row, col=1)
        crashes = [r for r in rows if r.get("crashed")]
        if crashes and k == 0:
            fig.add_trace(go.Scatter(x=[r.get(x) for r in crashes], y=[r.get("speed", 0) for r in crashes], mode="markers",
                                     marker=dict(symbol="x", color=BAD, size=10), name="crash"), row=1, col=1)
    for row in (2, 3):
        fig.add_hline(y=0, line=dict(color=AXIS, width=1), row=row, col=1)
    fig.update_yaxes(range=[-1.1, 1.1], row=2, col=1)
    fig.update_yaxes(range=[-1.1, 1.1], row=3, col=1)
    style(fig, height=height, title=title)
    fig.update_layout(margin=dict(t=60 if title else 36))
    fig.update_xaxes(title_text={"t": "race clock (s)", "progress": "lap progress"}.get(x, x), row=3, col=1)
    fig.update_annotations(font=dict(color=MUTED, size=12))
    return fig


def speed_profile_figure(track: Any = "grand_prix", *, actual: Any = None, aggression: float = 1.0,
                         a_lat: float = 20.0, brake: float = 25.0, max_speed: float = 40.0, height: int = 380):
    """Ch2's three speed curves along the lap: grip limit sqrt(a_lat R) -> after the backward braking pass ->
    (optional) what the car actually did."""
    from .. import driverkit as dk

    go = _go()
    tr = _track(track)
    pts = np.column_stack(_xy(tr, "center"))
    ds = np.hypot(*np.diff(np.vstack([pts, pts[:1]]), axis=0).T)
    dist = np.concatenate([[0.0], np.cumsum(ds)[:-1]])
    radii = dk.curvature_radii(pts)
    grip = np.minimum(max_speed, np.sqrt(a_lat * np.minimum(radii, 1e9))) * aggression
    planned = dk.braking_pass(np.clip(grip, 5.0, max_speed), pts, brake)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dist, y=grip, mode="lines", name="grip limit  sqrt(a_lat R)",
                             line=dict(color=MUTED, width=1.5, dash="dot")))
    fig.add_trace(go.Scatter(x=dist, y=planned, mode="lines", name="after the braking pass", line=dict(color=ACCENT, width=2.5)))
    rows = _rows(actual)
    if rows and "x" in rows[0]:
        xy = np.array([[r["x"], r["y"]] for r in rows])
        idx = dk.nearest_index(pts, xy)
        lap_d = dist[np.asarray(idx, int)]
        fig.add_trace(go.Scatter(x=lap_d, y=[r.get("speed", 0) for r in rows], mode="markers", name="actual",
                                 marker=dict(color="#d95926", size=5)))
    for c in tr["view"].get("corners", []):
        fig.add_annotation(x=dist[int(c["apex_index"])], y=float(c.get("v_max", 0)), text=c["id"], showarrow=False,
                           yshift=-14, font=dict(color=INK, size=11))
    style(fig, height=height)
    fig.update_xaxes(title_text="distance along the lap (units)")
    fig.update_yaxes(title_text="speed (u/s)", range=[0, max_speed * 1.05])
    return fig


# ---------------------------------------------------------------------------- learning
def _curves(results: Any) -> list[tuple[str, str | None, list[tuple[int, float | None, int]]]]:
    """-> [(driver name, colour, [(attempt, race_time or None, crashes)])]. Accepts session/result dicts,
    {"name": [time, None, ...]} mappings, or a list mixing both."""
    items = results if isinstance(results, list) else [results]
    out = []
    for r in items:
        if isinstance(r, dict) and "attempts" not in r:                  # {"me": [62.1, 60.3, None]}
            for name, times in r.items():
                out.append((str(name), None, [(i + 1, t, 0) for i, t in enumerate(times)]))
            continue
        pts = [(int(a.get("attempt", i + 1)), a.get("race_time") if a.get("completed", True) else None,
                int(a.get("crashes", 0))) for i, a in enumerate(r.get("attempts", []))]
        out.append((str(r.get("name", "driver")), r.get("color"), pts))
    return out


def learning_curve_figure(results: Any, *, best_so_far: bool = True, dnf_value: float | None = None,
                          height: int = 420, title: str | None = None):
    """Race time per attempt for each driver (lower = better). DNFs are drawn as X at the top.

    results: a session/result dict, a list of them, or {"name": [time, time, None, ...]}."""
    go = _go()
    curves = _curves(results)
    fig = go.Figure()
    finite = [t for _, _, pts in curves for _, t, _ in pts if t is not None]
    top = dnf_value or ((max(finite) * 1.08) if finite else 100.0)
    for k, (name, color, pts) in enumerate(curves):
        col = color or SERIES[k % len(SERIES)]
        xs = [a for a, _, _ in pts]
        ys = [t for _, t, _ in pts]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name=name, line=dict(color=col, width=2),
                                 marker=dict(size=8), connectgaps=False,
                                 customdata=[c for _, _, c in pts],
                                 hovertemplate=f"{name}<br>attempt %{{x}}<br>%{{y:.2f}} s<br>crashes %{{customdata}}<extra></extra>"))
        dnf = [a for a, t, _ in pts if t is None]
        if dnf:
            fig.add_trace(go.Scatter(x=dnf, y=[top] * len(dnf), mode="markers", name=f"{name} DNF", legendgroup=name,
                                     showlegend=False, marker=dict(symbol="x", size=11, color=col),
                                     hovertemplate=f"{name}<br>attempt %{{x}}: DNF<extra></extra>"))
        if best_so_far and len([y for y in ys if y is not None]) > 1:
            best, run = [], math.inf
            for y in ys:
                run = min(run, y) if y is not None else run
                best.append(run if math.isfinite(run) else None)
            fig.add_trace(go.Scatter(x=xs, y=best, mode="lines", name=f"{name} best so far", legendgroup=name,
                                     showlegend=False, line=dict(color=col, width=1, dash="dot"), hoverinfo="skip"))
    style(fig, height=height, title=title)
    fig.update_xaxes(title_text="attempt", dtick=1)
    fig.update_yaxes(title_text="race time (s)  - lower is better")
    return fig


def segment_heatmap(session: Any, *, value: str = "time_lost", height: int | None = None, title: str | None = None):
    """Attempts (rows) x waypoint segments (columns). value="time_lost": seconds slower than the best
    attempt in that segment (0 = your best); value="time" | "crashes" | "avg_speed" for the raw numbers."""
    go = _go()
    reports = session.get("reports", []) if isinstance(session, dict) else list(session)
    reports = [r if isinstance(r, dict) else r.to_dict() for r in reports]
    segs = sorted({s["segment"] for r in reports for s in r.get("segments", [])})
    if not reports or not segs:
        return style(go.Figure(), height=height or 300, title="no segment data")
    key = "time" if value == "time_lost" else value
    grid = np.full((len(reports), len(segs)), np.nan)
    crash = np.zeros_like(grid)
    for i, r in enumerate(reports):
        for s in r.get("segments", []):
            j = segs.index(s["segment"])
            grid[i, j] = s.get(key, np.nan)
            crash[i, j] = s.get("crashes", 0)
    z = grid - np.nanmin(grid, axis=0) if value == "time_lost" else grid
    labels = [f"#{r.get('attempt', i + 1)}" + (" DNF" if not r.get("completed", True) else "") for i, r in enumerate(reports)]
    def cell(i: int, j: int) -> str:                    # quiet cells stay empty: the eye goes to the losses
        v = z[i, j]
        num = "" if not np.isfinite(v) or (value == "time_lost" and abs(v) < 0.05) else f"{v:.1f}"
        return (("X " if crash[i, j] else "") + num).strip()
    text = [[cell(i, j) for j in range(len(segs))] for i in range(len(reports))]
    fig = go.Figure(go.Heatmap(
        z=z, x=[f"S{s}" for s in segs], y=labels, colorscale=ORANGE_SCALE if value in ("time_lost", "time", "crashes") else "Blues",
        text=text, texttemplate="%{text}", textfont=dict(size=11, color=INK), xgap=2, ygap=2,
        colorbar=dict(title=dict(text={"time_lost": "s lost", "time": "s"}.get(value, value), font=dict(color=MUTED)),
                      tickfont=dict(color=MUTED)),
        hovertemplate="attempt %{y}, segment %{x}<br>%{z:.2f}<extra></extra>"))
    style(fig, height=height or max(260, 60 + 34 * len(reports)), title=title, legend=False)
    fig.update_yaxes(autorange="reversed", title_text="attempt")
    fig.update_xaxes(title_text="waypoint segment  (X = crash)")
    return fig


# ---------------------------------------------------------------------------- latency
def latency_histogram(seconds: Iterable[float], *, tick: float = 0.5, height: int = 360, title: str | None = None,
                      color: str = ACCENT):
    """How long LLM calls took, against one sim tick: every call right of the red line means the car
    drove blind for (latency / tick) ticks in realtime mode."""
    go = _go()
    xs = [float(s) for s in seconds if s is not None]
    fig = go.Figure()
    if xs:
        size = max(0.05, round(max(max(xs), tick * 2) / 24, 2))
        fig.add_trace(go.Histogram(x=xs, xbins=dict(start=0, size=size), marker=dict(color=color, line=dict(color=SURFACE, width=2)), name="calls",
                                   hovertemplate="%{x} s: %{y} call(s)<extra></extra>"))
        q = np.percentile(xs, [50, 95])
        for v, lab in zip(q, ("p50", "p95")):
            fig.add_vline(x=float(v), line=dict(color=INK2, width=1, dash="dot"),
                          annotation=dict(text=f"{lab} {v:.2f}s", font=dict(color=INK2), yanchor="top"))
    fig.add_vline(x=tick, line=dict(color=BAD, width=2.5),
                  annotation=dict(text=f"one tick = {tick:g} s", font=dict(color=BAD), yanchor="bottom", y=1))
    style(fig, height=height, title=title, legend=False)
    fig.update_xaxes(title_text="seconds per LLM call", rangemode="tozero")
    fig.update_yaxes(title_text="calls")
    return fig


def blind_distance_figure(track: Any, x: float, y: float, heading_deg: float, latency_s: float, speed: float = 30.0,
                          **kw):
    """Track figure + the stretch the car covers 'blind' while waiting latency_s for an answer."""
    go = _go()
    fig = track_figure(track, vmax=False, **kw)
    d = speed * latency_s
    h = math.radians(heading_deg)
    fig.add_trace(go.Scatter(x=[x, x + d * math.cos(h)], y=[y, y + d * math.sin(h)], mode="lines",
                             line=dict(color=BAD, width=6), name=f"blind: {d:.0f} units in {latency_s:.1f}s"))
    add_car(fig, x, y, heading_deg)
    return fig
