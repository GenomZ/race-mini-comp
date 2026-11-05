"""Track Lab: read a track like a race engineer (corner speeds, braking zones) and build your own."""

import json
import math
import re

import pandas as pd
import streamlit as st

from agentic_gp.race.replay import Replay
from agentic_gp.sim.track import GRAND_PRIX_CONTROL_POINTS, DEFAULT_SCALE, track_from_control_points
from agentic_gp.viz import plotly_figs as pf
from app.components import replay_canvas
from app.lib.loaders import load_track
from app.lib.paths import RACES_DIR, rel
from app.lib.sandbox import SandboxError, is_trusted, run_driver_job
from app.lib.ui import driver_picker, fast_mode, page_header, track_picker

page_header("Track Lab", "The track dict is a world model: from the centerline you can compute every corner's "
            "radius, its speed limit √(20·R), and where to start braking - before you ever drive it. "
            "Then build a track that breaks your neighbour's driver.", chapter="Ch2")

TRACKS_DIR = RACES_DIR / "tracks"
OVAL = [(0, 0), (200, 0), (280, 60), (200, 120), (0, 120), (-80, 60)]


def metrics_row(track) -> None:
    """Five headline numbers of a sim Track."""
    corners = track.corners()
    m = st.columns(5)
    m[0].metric("Lap length", f"{track.length:.0f} u")
    m[1].metric("Gates", track.n_gates)
    m[2].metric("Corners", len(corners))
    m[3].metric("Tightest radius", f"{track.min_turn_radius():.0f} u")
    m[4].metric("Slowest corner", f"{min((c['v_max'] for c in corners), default=40):.0f} u/s")


def corner_table(track) -> None:
    """One row per corner: radius and speed limit v_max = sqrt(20 R)."""
    corners = track.corners()
    if corners:
        st.dataframe([{"corner": c["id"], "turns": c["direction"], "radius": round(c["min_radius"], 1),
                       "v_max": round(c["v_max"], 1), "gates": f"{c['from_wp']}-{c['to_wp']}"} for c in corners],
                     hide_index=True, width="stretch",
                     column_config={"v_max": st.column_config.ProgressColumn("v_max (u/s)", min_value=0, max_value=40,
                                                                             format="%.1f")})


def race_here(track, key: str) -> None:
    """Pick a code driver and race one lap on `track` right here (code drivers take < 1 s).

    `track` is a spec string or a JSON dict of a Track Lab build. Files outside drivers/examples/ race
    in their own process with a time limit, so an endless loop cannot freeze the app.
    """
    with st.container(border=True):
        c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
        info = driver_picker("Race here with (code drivers)", key=f"{key}_drv", container=c1, only="code")
        go = c2.button("Race one lap", key=f"{key}_go", type="primary", icon=":material/flag:", width="stretch",
                       disabled=info is None)
        if info is not None and not is_trusted(info.path):
            st.caption(":material/shield: Races in its own process (stopped after 45 s) - a bug in the file "
                       "cannot freeze this app.")
        if go and info is not None:
            with st.spinner(f"{info.name} is driving..."):
                try:
                    out = run_driver_job("app.lib.experiments:race_one_lap", info.path,
                                         {"driver": str(info.path), "track": track,
                                          "max_ticks": 120 if fast_mode() else None}, timeout_s=45)
                    st.session_state[f"{key}_res"] = (info.name, Replay.from_dict(out["replay"]), out["result"])
                except SandboxError as exc:
                    st.session_state.pop(f"{key}_res", None)
                    st.error(f"{info.name} could not finish the lap: {exc}", icon=":material/error:")
                except Exception as exc:                 # noqa: BLE001 - show, don't crash
                    st.session_state.pop(f"{key}_res", None)
                    st.error(f"The driver failed: {type(exc).__name__}: {exc}")
        got = st.session_state.get(f"{key}_res")
        if got:
            name, replay, r = got
            st.markdown(f"**{name}**: " + (f"{r['race_time']:.2f} s" if r["completed"]
                                           else f"{r.get('status') or 'DNF'} ({r['waypoints_passed']} gates)")
                        + f", {r['crashes']} crash(es)")
            replay_canvas(replay, key=f"{key}_replay", height=420, autoplay=True, speed=2.0)


tab_explore, tab_build = st.tabs([":material/travel_explore: Explore a track", ":material/construction: Build your own"])

# ------------------------------------------------------------------------------ explore
with tab_explore:
    c1, c2 = st.columns([2, 3], vertical_alignment="bottom")
    with c1:
        spec = track_picker("Track", key="tl_track", default="grand_prix", n_random=5)
    with c2:
        st.caption("random:<seed> builds a brand-new track from a number - the same seed gives the same track on "
                   "every laptop. The mystery track of the final is one you have never seen.")
    try:
        tr = load_track(spec)
    except Exception as exc:                             # noqa: BLE001
        st.error(f"Cannot build {spec}: {exc}")
        st.stop()
    metrics_row(tr)
    left, right = st.columns([3, 2], gap="medium")
    with left:
        st.plotly_chart(pf.track_figure(spec, gate_numbers=True, height=560), theme=None, key="tl_fig")
    with right:
        corner_table(tr)
        brief = tr.briefing()
        if brief:
            with st.expander("The race briefing (plan() gets it as track['briefing'])", icon=":material/campaign:"):
                st.text(brief)
    aggr = st.slider("Aggression (x every corner speed)", 0.8, 1.2, 1.0, 0.05, key="tl_aggr",
                     help="Ch2 exercise: above 1.0 the lap time hardly changes (a plateau, not a cliff) - "
                          "the driver's stopping-distance safety net brakes where the plan is too brave.")
    st.plotly_chart(pf.speed_profile_figure(spec, aggression=aggr), theme=None, key="tl_profile")
    st.caption("Grey dotted = what grip allows at each point. Blue = after the backward braking pass: you must "
               "start braking BEFORE a slow corner, at 25 u/s². The area between them is braking.")
    race_here(spec, "tl_explore")

# ------------------------------------------------------------------------------ build
with tab_build:
    starts = {"Oval (6 points)": OVAL,
              "Grand Prix circuit (16 points)": [(x * DEFAULT_SCALE, y * DEFAULT_SCALE) for x, y in GRAND_PRIX_CONTROL_POINTS]}
    c1, c2, c3, c4 = st.columns([2, 2, 1, 1], vertical_alignment="bottom")
    start = c1.selectbox("Start from", list(starts), key="tl_start")
    name = c2.text_input("Track name", "My Track", key="tl_name", max_chars=30)
    half_width = c3.number_input("Half width", 6.0, 14.0, 9.0, 0.5, key="tl_hw")
    n_gates = c4.number_input("Gates", 6, 32, 16, key="tl_gates")
    if st.session_state.get("tl_pts_for") != start:
        st.session_state.tl_pts = pd.DataFrame(starts[start], columns=["x", "y"])
        st.session_state.tl_pts_for = start
    left, right = st.columns([1, 2], gap="medium")
    with left:
        st.markdown("**Control points** (the first point is the start line: put it mid-straight)")
        pts = st.data_editor(st.session_state.tl_pts, num_rows="dynamic", key="tl_editor", hide_index=False,
                             width="stretch", height=420,
                             column_config={"x": st.column_config.NumberColumn(format="%.0f"),
                                            "y": st.column_config.NumberColumn(format="%.0f")})
    pts = pts.dropna() if isinstance(pts, pd.DataFrame) else pd.DataFrame(pts).dropna()
    problems, track = [], None
    if len(pts) < 4:
        problems = ["need at least 4 control points"]
    else:
        try:
            track = track_from_control_points(pts[["x", "y"]].to_numpy(float), name=name or "My Track",
                                              key="custom", half_width=float(half_width), n_gates=int(n_gates))
            problems = track.validate(min_radius=12.0)
        except Exception as exc:                         # noqa: BLE001 - a weird table must not kill the page
            problems = [f"{type(exc).__name__}: {exc}"]
    with right:
        if track is not None:
            fig = pf.track_figure(track, gate_numbers=True, height=470)
            fig.add_scatter(x=pts["x"], y=pts["y"], mode="markers+text", text=[str(i) for i in range(len(pts))],
                            textposition="top right", marker=dict(color="#d55181", size=9), name="control points",
                            textfont=dict(color="#d55181"))
            st.plotly_chart(fig, theme=None, key="tl_build_fig")
        if problems:
            st.error("Not raceable yet:\n\n" + "\n".join(f"- {p}" for p in problems), icon=":material/block:")
        else:
            st.success("Valid track: corners wide enough (R >= 12), no overlapping walls, start on a straight.",
                       icon=":material/check_circle:")
    if track is not None and not problems:
        metrics_row(track)
        corner_table(track)
        safe = re.sub(r"[^a-z0-9_]+", "_", (name or "my_track").lower()).strip("_") or "my_track"
        body = (f'"""{name} - built in the Track Lab. Race it:  python race.py drive <driver.py> '
                f'--track-file races/tracks/{safe}.py"""\n\n'
                f"NAME = {json.dumps(name)}\nHALF_WIDTH = {float(half_width)}\nN_GATES = {int(n_gates)}\n"
                "CONTROL_POINTS = [\n" + "".join(f"    ({x:.1f}, {y:.1f}),\n" for x, y in pts[["x", "y"]].to_numpy(float))
                + "]\n")
        d1, d2 = st.columns(2)
        d1.download_button("Download track file (.py)", body, f"{safe}.py", "text/x-python", icon=":material/download:",
                           on_click="ignore", key="tl_dl", width="stretch")

        def _save() -> None:
            TRACKS_DIR.mkdir(parents=True, exist_ok=True)
            (TRACKS_DIR / f"{safe}.py").write_text(body, encoding="utf-8")
            st.session_state.tl_saved = rel(TRACKS_DIR / f"{safe}.py")

        d2.button("Save to races/tracks/", on_click=_save, icon=":material/save:", key="tl_save", width="stretch")
        if st.session_state.get("tl_saved"):
            st.success(f"Saved {st.session_state.tl_saved}. It is now in every Track box of the app (Live, Grand "
                       f"Prix, Track Lab) as file:{st.session_state.tl_saved} - and on the terminal: "
                       f"python race.py drive <driver.py> --track file:{st.session_state.tl_saved}",
                       icon=":material/check:")
        race_here({"control_points": pts[["x", "y"]].to_numpy(float).tolist(), "name": name or "My Track",
                   "half_width": float(half_width), "n_gates": int(n_gates)}, "tl_build")
