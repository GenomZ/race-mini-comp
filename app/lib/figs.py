"""Extra Plotly figures for the Race Control pages (the shared ones live in agentic_gp.viz.plotly_figs).

    from app.lib.figs import driver_frame_figure, stopping_figure, vmax_figure, ...

Every function returns a go.Figure in the same dark theme (plotly_figs.style); show it with
st.plotly_chart(fig, theme=None).
"""

from __future__ import annotations

import math
from typing import Any, Iterable

from agentic_gp.viz import plotly_figs as pf

RAYS = [(0, "front"), (-20, "left_20"), (20, "right_20"), (-45, "left_45"), (45, "right_45"),
        (-90, "left_90"), (90, "right_90")]


def _go():
    import plotly.graph_objects as go

    return go


# ------------------------------------------------------------------------------ Sensor X-ray
def driver_frame_figure(obs: dict, action: dict | None = None, debug: dict | None = None, *,
                        view: float = 60.0, height: int = 520):
    """What the driver 'sees', drawn in ITS OWN frame: the car points up, + angles are to the RIGHT.

    The 7 rays with their lengths (clipped to the view, the label tells the true distance), and
    arrows to the next and after-next waypoints labelled with their signed angles.
    """
    go = _go()
    fig = go.Figure()

    def to_xy(angle_deg: float, dist: float) -> tuple[float, float]:
        a = math.radians(angle_deg)                          # 0 = straight ahead (up), + = right
        return dist * math.sin(a), dist * math.cos(a)

    # sight circles (distance guides)
    for r, lab in ((10, "10"), (25, "25"), (50, "50")):
        if r > view:
            continue
        ts = [k * math.pi / 60 for k in range(121)]
        fig.add_trace(go.Scatter(x=[r * math.cos(t) for t in ts], y=[r * math.sin(t) for t in ts], mode="lines",
                                 line=dict(color=pf.GRID, width=1), hoverinfo="skip", showlegend=False))
        fig.add_annotation(x=r * 0.71, y=-r * 0.71, text=lab, showarrow=False, font=dict(color=pf.MUTED, size=10))
    # rays
    for ang, key in RAYS:
        d = float(obs.get(f"distance_to_wall_{key}", 200.0))
        shown = min(d, view)
        x, y = to_xy(ang, shown)
        col = pf.ray_color(d)
        fig.add_trace(go.Scatter(x=[0, x], y=[0, y], mode="lines", line=dict(color=col, width=3),
                                 hovertemplate=f"ray {ang:+d} deg: {d:.1f} units<extra></extra>", showlegend=False))
        lx, ly = to_xy(ang, min(shown + 4, view * 0.98))
        fig.add_annotation(x=lx, y=ly, text=f"<b>{d:.0f}</b>" + ("+" if d > view else ""), showarrow=False,
                           font=dict(color=col, size=13), bgcolor="rgba(13,13,13,.75)")
    # waypoint arrows
    for k, (a_key, d_key, name, col) in enumerate((
            ("angle_to_next_waypoint", "distance_to_next_waypoint", "next waypoint", "#f0efec"),
            ("angle_to_waypoint_after_next", "distance_to_waypoint_after_next", "after next", "#9085e9"))):
        if a_key not in obs:
            continue
        ang, dist = float(obs[a_key]), float(obs.get(d_key, 0.0))
        x, y = to_xy(ang, min(dist, view * 0.9))
        fig.add_annotation(x=x, y=y, ax=0, ay=0, xref="x", yref="y", axref="x", ayref="y", showarrow=True,
                           arrowhead=3, arrowsize=1.4, arrowwidth=2.5 - k, arrowcolor=col, opacity=0.95)
        fig.add_annotation(x=x, y=y, text=f"{'next' if k == 0 else 'after next'} {ang:+.0f}°", showarrow=False,
                           yshift=12, font=dict(color=col, size=12), bgcolor="rgba(13,13,13,.8)")
    # the car (pointing up)
    xs, ys = pf.car_polygon(0, 0, 90, length=6, width=3)
    fig.add_trace(go.Scatter(x=xs, y=ys, fill="toself", fillcolor=pf.ACCENT, mode="lines",
                             line=dict(color="#111", width=1), hoverinfo="skip", showlegend=False))
    if action:
        s = float(action.get("steering", 0.0))
        xa, ya = to_xy(30 * s, 14)
        fig.add_trace(go.Scatter(x=[0, xa], y=[0, ya], mode="lines", line=dict(color="#d55181", width=3, dash="dot"),
                                 hovertemplate=f"steering {s:+.2f}<extra></extra>", name="steering command",
                                 showlegend=False))
    if debug and isinstance(debug.get("target"), (list, tuple)) and "position" in obs:
        # world target -> driver frame
        tx, ty = debug["target"]
        px, py = obs["position"]
        h = math.radians(float(obs.get("heading_deg", 0.0)))
        dx, dy = tx - px, ty - py
        fwd = dx * math.cos(h) + dy * math.sin(h)
        right = dx * math.sin(h) - dy * math.cos(h)
        if math.hypot(fwd, right) < view * 1.2:
            fig.add_trace(go.Scatter(x=[right], y=[fwd], mode="markers", name="debug target",
                                     marker=dict(symbol="x-thin", size=16, line=dict(width=3, color="#d55181")),
                                     hovertemplate="debug['target']<extra></extra>", showlegend=False))
    fig.add_annotation(x=-view * 0.97, y=view * 1.02, xanchor="left", text="LEFT  (- angles, - steering)",
                       showarrow=False, font=dict(color=pf.MUTED, size=12))
    fig.add_annotation(x=view * 0.97, y=view * 1.02, xanchor="right", text="RIGHT  (+ angles, + steering)",
                       showarrow=False, font=dict(color=pf.MUTED, size=12))
    pf.style(fig, height=height, legend=False)
    fig.update_xaxes(range=[-view, view], showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(range=[-view * 0.35, view * 1.08], showgrid=False, zeroline=False, showticklabels=False,
                     scaleanchor="x", scaleratio=1)
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    return fig


# ------------------------------------------------------------------------------ Physics Lab
def stopping_figure(speed: float, *, brake: float = 25.0, tick: float = 0.5, front: float | None = None,
                    max_speed: float = 40.0, height: int = 380):
    """Stopping distance v^2 / (2 * brake) and the extra 'blind' distance v * tick before you can react."""
    go = _go()
    vs = [v * 0.5 for v in range(int(max_speed * 2) + 1)]
    braking = [v * v / (2 * brake) for v in vs]
    total = [b + v * tick for b, v in zip(braking, vs)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=vs, y=braking, mode="lines", name=f"braking only  v^2 / {2 * brake:g}",
                             line=dict(color=pf.ACCENT, width=2.5)))
    fig.add_trace(go.Scatter(x=vs, y=total, mode="lines", name=f"+ blind {tick:g} s tick  (v x {tick:g})",
                             line=dict(color=pf.BAD, width=2.5), fill="tonexty", fillcolor="rgba(208,59,59,0.12)"))
    b = speed * speed / (2 * brake)
    t = b + speed * tick
    fig.add_trace(go.Scatter(x=[speed, speed], y=[b, t], mode="markers+text", text=[f"{b:.0f}", f"{t:.0f} u"],
                             textposition="middle left", marker=dict(color=pf.INK, size=9), showlegend=False,
                             textfont=dict(color=pf.INK)))
    if front is not None:
        fig.add_hline(y=front, line=dict(color=pf.WARN, dash="dot"),
                      annotation=dict(text=f"wall ahead: {front:.0f} u", font=dict(color=pf.WARN)))
    pf.style(fig, height=height)
    fig.update_xaxes(title_text="speed (u/s)")
    fig.update_yaxes(title_text="distance to stop (units)", rangemode="tozero")
    fig.update_layout(legend=dict(x=0.02, y=0.98))
    return fig


def vmax_figure(corners: list[dict], *, a_lat: float = 20.0, max_speed: float = 40.0, radius: float | None = None,
                height: int = 380):
    """The corner speed law v_max = sqrt(a_lat * R), with this track's corners on the curve."""
    go = _go()
    rs = [r for r in range(2, 151)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rs, y=[min(max_speed, math.sqrt(a_lat * r)) for r in rs], mode="lines",
                             name=f"v_max = sqrt({a_lat:g} R)", line=dict(color=pf.ACCENT, width=2.5)))
    if corners:
        fig.add_trace(go.Scatter(
            x=[c["min_radius"] for c in corners], y=[min(max_speed, c["v_max"]) for c in corners],
            mode="markers+text", text=[c["id"] for c in corners], textposition="top center",
            marker=dict(color=pf.ZONE, size=10, line=dict(color="#111", width=1)), name="this track's corners",
            textfont=dict(color=pf.INK, size=11),
            hovertemplate="%{text}: R %{x:.0f} -> v_max %{y:.1f}<extra></extra>"))
    if radius is not None:
        v = min(max_speed, math.sqrt(a_lat * radius))
        fig.add_trace(go.Scatter(x=[radius], y=[v], mode="markers", marker=dict(color=pf.BAD, size=13, symbol="diamond"),
                                 name=f"R = {radius:.0f}: v_max {v:.1f}"))
    fig.add_hline(y=max_speed, line=dict(color=pf.MUTED, dash="dot"),
                  annotation=dict(text=f"top speed {max_speed:g}", font=dict(color=pf.MUTED)))
    pf.style(fig, height=height)
    fig.update_xaxes(title_text="corner radius R (units)")
    fig.update_yaxes(title_text="fastest safe speed (u/s)", range=[0, max_speed * 1.1])
    fig.update_layout(legend=dict(x=0.55, y=0.15))
    return fig


# ------------------------------------------------------------------------------ Latency Lab
def latency_cliff_figure(rows: list[dict], height: int = 360):
    """Crashes (bars) and lap outcome per answer latency: the cliff between 0 and 0.5 s."""
    go = _go()
    labels = [f"{r['latency_s']:g} s" for r in rows]
    colors = [pf.GOOD if r["crashes"] == 0 and r["finished"] else (pf.WARN if r["finished"] else pf.BAD) for r in rows]
    text = [(f"{r['race_time']:.0f} s" if r["finished"] else f"DNF {r['gates']}/{r['gates_total']}") for r in rows]
    fig = go.Figure(go.Bar(x=labels, y=[r["crashes"] for r in rows], marker=dict(color=colors), text=text,
                           textposition="outside", textfont=dict(color=pf.INK),
                           hovertemplate="latency %{x}: %{y} crashes<extra></extra>"))
    pf.style(fig, height=height, legend=False)
    fig.update_xaxes(title_text="answer arrives this late (blind ticks = latency / 0.5 s)")
    fig.update_yaxes(title_text="crashes in one lap", rangemode="tozero")
    top = max([r["crashes"] for r in rows] + [1])
    fig.update_yaxes(range=[0, top * 1.25])
    return fig


# ------------------------------------------------------------------------------ Fly Lab
def sankey_figure(data: dict, height: int = 560):
    """Cell type x side flow (synapses), feed-forward links only. Green = mostly excitatory, red = inhibitory."""
    go = _go()
    nodes = data["nodes"]
    links = data["links"]

    def node_color(n: str) -> str:
        if n.startswith("other"):
            return "#56544f"
        if n.startswith(("LC", "LPLC")):
            return pf.ACCENT                         # eyes (inputs)
        if n.startswith("AOTU"):
            return "#9085e9"                         # hubs
        return pf.ZONE                               # descending neurons (outputs)

    def column(n: str) -> float:                     # eyes -> hubs -> other -> descending neurons
        if n.startswith(("LC", "LPLC")):
            return 0.01
        if n.startswith("AOTU"):
            return 0.36
        if n.startswith("other"):
            return 0.62
        return 0.99

    xs = [column(n) for n in nodes]
    # feed-forward only (eyes -> hubs -> descending): backward/sideways links would form loops
    links = [l for l in links if xs[l["target"]] > xs[l["source"]]]
    # Plotly silently drops nodes without links, which would shift our x/y arrays: drop them ourselves
    used = sorted({l["source"] for l in links} | {l["target"] for l in links})
    remap = {old: new for new, old in enumerate(used)}
    nodes, xs = [nodes[i] for i in used], [xs[i] for i in used]
    links = [dict(l, source=remap[l["source"]], target=remap[l["target"]]) for l in links]
    ys = [0.0] * len(nodes)
    for x in set(xs):                                 # spread each column evenly from top to bottom
        idx = [i for i, v in enumerate(xs) if v == x]
        for k, i in enumerate(idx):
            ys[i] = 0.03 + 0.94 * (k + 0.5) / len(idx)
    fig = go.Figure(go.Sankey(
        arrangement="fixed",
        node=dict(label=nodes, color=[node_color(n) for n in nodes], pad=14, thickness=16, x=xs, y=ys,
                  line=dict(color="#111", width=0.5)),
        link=dict(source=[l["source"] for l in links], target=[l["target"] for l in links],
                  value=[l["value"] for l in links],
                  color=["rgba(64,190,64,0.45)" if l["sign"] > 0 else "rgba(230,80,80,0.55)" for l in links],
                  customdata=[f"{l['excitatory']} exc / {l['inhibitory']} inh" for l in links],
                  hovertemplate="%{source.label} -> %{target.label}<br>%{value} synapses (%{customdata})<extra></extra>"),
    ))
    pf.style(fig, height=height, legend=False)
    fig.update_layout(font=dict(size=12, color=pf.INK))
    return fig


def tuning_curve_figure(sweep: dict, key: str = "steer_raw", height: int = 400):
    """Steering readout vs target bearing: real wiring (bold) vs scrambled seeds (thin grey)."""
    go = _go()
    xs = sweep["angles"]
    fig = go.Figure()
    for seed, curve in sweep.get("scrambled", {}).items():
        fig.add_trace(go.Scatter(x=xs, y=curve[key], mode="lines+markers", name=f"scrambled seed {seed}",
                                 line=dict(color="#6e6c66", width=1.4), marker=dict(size=4)))
    fig.add_trace(go.Scatter(x=xs, y=sweep["real"][key], mode="lines+markers", name="real FlyWire wiring",
                             line=dict(color=pf.ZONE, width=3.5), marker=dict(size=7)))
    fig.add_hline(y=0, line=dict(color=pf.AXIS))
    fig.add_vline(x=0, line=dict(color=pf.AXIS))
    pf.style(fig, height=height)
    fig.update_xaxes(title_text="target bearing (deg, + = right)")
    fig.update_yaxes(title_text="steering readout (+ = right)")
    return fig


def raster_figure(raster: dict, groups: Iterable[str], cursor_ms: float | None = None, window_ms: float | None = None,
                  height: int = 380):
    """Spike raster of readout groups on the brain clock, one row band per group, with a cursor."""
    go = _go()
    fig = go.Figure()
    groups = [g for g in groups if g in raster]
    palette = {"DNa02": pf.ZONE, "AOTU019": "#9085e9", "MDN": pf.BAD, "DNp01": "#d55181"}
    for k, g in enumerate(groups):
        ts = raster[g]
        if window_ms is not None and cursor_ms is not None:
            ts = [t for t in ts if cursor_ms - window_ms <= t <= cursor_ms + window_ms]
        col = next((c for p, c in palette.items() if g.startswith(p)), pf.ACCENT)
        fig.add_trace(go.Scatter(x=ts, y=[k] * len(ts), mode="markers", name=g,
                                 marker=dict(symbol="line-ns", size=11, line=dict(width=1.6, color=col)),
                                 hovertemplate=f"{g}<br>%{{x:.0f}} ms<extra></extra>"))
    if cursor_ms is not None:
        fig.add_vline(x=cursor_ms, line=dict(color=pf.INK, width=2))
    pf.style(fig, height=height, legend=False)
    fig.update_yaxes(tickvals=list(range(len(groups))), ticktext=groups, range=[-0.6, len(groups) - 0.4])
    fig.update_xaxes(title_text="brain time (ms); the white line = the tick on the left")
    if window_ms is not None and cursor_ms is not None:
        fig.update_xaxes(range=[cursor_ms - window_ms, cursor_ms + window_ms])
    return fig


def progress_bars(rows: list[dict], height: int = 300):
    """Horizontal race-progress bars per driver (Grand Prix live table companion)."""
    go = _go()
    rows = sorted(rows, key=lambda r: -float(r.get("progress", 0)))
    fig = go.Figure(go.Bar(y=[r["driver"] for r in rows], x=[r.get("progress", 0) for r in rows], orientation="h",
                           marker=dict(color=[r.get("color", pf.ACCENT) for r in rows]),
                           text=[r.get("phase", "") for r in rows], textposition="auto"))
    pf.style(fig, height=height, legend=False)
    fig.update_yaxes(autorange="reversed")
    return fig


def blank(message: str, height: int = 240) -> Any:
    """An empty themed figure with a message (keeps layouts stable when data is missing)."""
    go = _go()
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(color=pf.MUTED, size=14), x=0.5, y=0.5,
                       xref="paper", yref="paper")
    pf.style(fig, height=height, legend=False)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig
