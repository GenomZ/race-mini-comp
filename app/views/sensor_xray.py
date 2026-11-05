"""Sensor X-ray: freeze any tick and see exactly what the driver saw and what it answered."""

import json

import streamlit as st

from app.components import live_canvas, live_state
from app.lib.figs import RAYS, driver_frame_figure
from app.lib.loaders import load_json
from app.lib.paths import DEMO_DIR
from app.lib.sandbox import is_trusted, run_driver_job
from app.lib.ui import driver_picker, fast_mode, page_header, track_picker

page_header("Sensor X-ray", "Your driver never sees the map on screen - only this dictionary, twice a second. "
            "One rule for every angle: + = RIGHT (rays, waypoint angles and steering). Step through the ticks "
            "and predict the answer before you look at it.", chapter="Ch1")

DEMO = DEMO_DIR / "l2_ticks.json"


@st.cache_data(show_spinner="Recording every tick...", max_entries=8)
def _record(path: str, mtime: float, track: str, max_ticks: int) -> dict:
    """Examples run here; any other file in its own process (an endless loop cannot freeze the app)."""
    return run_driver_job("app.lib.experiments:record_ticks", path,
                          {"driver": path, "track": track, "max_ticks": max_ticks}, timeout_s=45)


# ------------------------------------------------------------------------------ choose a recording
options = (["Demo recording: L2 planner on grand_prix"] if DEMO.exists() else []) + ["Record a driver now"]
mode = st.segmented_control("Recording", options, default=options[0], key="xr_mode", label_visibility="collapsed")
data = None
if mode == "Record a driver now":
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([3, 2, 1, 1], vertical_alignment="bottom")
        info = driver_picker("Driver (code drivers only: no LLM calls here)", key="xr_driver", container=c1,
                             default_stem="l1_reflex", only="code")
        with c2:
            track = track_picker("Track", key="xr_track", default="sprint")
        max_ticks = c3.number_input("Max ticks", 20, 500, 60 if fast_mode() else 240, step=20, key="xr_max")
        go = c4.button("Record", type="primary", key="xr_record", icon=":material/fiber_manual_record:",
                       disabled=info is None)
        if info is not None and not is_trusted(info.path):
            st.caption(":material/shield: Not a reference driver: it is recorded in its own process "
                       "(stopped after 45 s), so a bug in it cannot freeze this app.")
    if go and info is not None:
        try:
            st.session_state.xr_data = _record(str(info.path), info.path.stat().st_mtime, track, int(max_ticks))
            st.session_state.xr_tick = 0
        except Exception as exc:                        # noqa: BLE001 - a broken student file must not kill the page
            st.error(f"The driver failed: {type(exc).__name__}: {exc}")
    data = st.session_state.get("xr_data")
    if data is None:
        st.info("Pick a driver and press Record. Every tick of one attempt is kept, then you can step through it.",
                icon=":material/info:")
        st.stop()
elif mode:
    data = load_json(DEMO)
if not data or not data.get("ticks"):
    st.warning("The recording is empty. Run: python races/demo/make_demo.py", icon=":material/warning:")
    st.stop()

ticks = data["ticks"]
n = len(ticks)
track_spec = data.get("track", "grand_prix")
drv = data.get("driver") or {}
crash_ticks = [i for i, t in enumerate(ticks) if (t.get("after") or {}).get("crashed")]

# ------------------------------------------------------------------------------ the tick selector
if "xr_tick" not in st.session_state or st.session_state.xr_tick >= n:
    st.session_state.xr_tick = min(12, n - 1)


def _move(delta: int) -> None:
    st.session_state.xr_tick = max(0, min(n - 1, st.session_state.xr_tick + delta))


def _next_crash() -> None:
    later = [i for i in crash_ticks if i > st.session_state.xr_tick]
    st.session_state.xr_tick = later[0] if later else (crash_ticks[0] if crash_ticks else st.session_state.xr_tick)


c0, c1, c2, c3 = st.columns([6, 1.4, 1.4, 2], vertical_alignment="bottom")
c0.slider("Tick (0.5 s each)", 0, n - 1, key="xr_tick")
c1.button("Prev", icon=":material/chevron_left:", on_click=_move, args=(-1,), key="xr_prev", width="stretch",
          shortcut="Left", help="Previous tick (Left arrow)")
c2.button("Next", icon=":material/chevron_right:", on_click=_move, args=(1,), key="xr_next", width="stretch",
          shortcut="Right", help="Next tick (Right arrow)")
c3.button(f"Next crash ({len(crash_ticks)})", on_click=_next_crash, key="xr_crash", width="stretch",
          icon=":material/car_crash:", disabled=not crash_ticks)

k = st.session_state.xr_tick
rec = ticks[k]
obs, action, debug = rec["obs"], rec.get("action") or {}, rec.get("debug") or {}
trail = [t["obs"]["position"] for t in ticks[: k + 1]]
crashes = [[t["after"]["x"], t["after"]["y"]] for t in ticks[: k + 1] if (t.get("after") or {}).get("crashed")]

# ------------------------------------------------------------------------------ two views of one moment
left, right = st.columns([1, 1], gap="medium")
with left:
    st.markdown(f"**The world** - {drv.get('name', 'driver')} on `{track_spec}`, tick {obs['tick']}, "
                f"race clock {obs['race_time']:.1f} s")
    state = live_state(track_spec, obs=obs, action=action, debug=debug, name=drv.get("name", "car"),
                       color=drv.get("color", "#3987e5"), status="paused", trail=trail, crash_points=crashes,
                       show_waypoints=True, follow=True, tween_ms=250,
                       say=[{"t": obs["race_time"], "text": action["say"]}] if action.get("say") else None)
    live_canvas(state, key="xr_canvas", height=460)
with right:
    st.markdown("**The driver's frame** - the car points up; + angles are to the right")
    st.plotly_chart(driver_frame_figure(obs, action, debug, height=460), theme=None, key="xr_frame",
                    config={"displayModeBar": False})

# ------------------------------------------------------------------------------ the numbers
r1, r2, r3 = st.columns([5, 5, 4], gap="medium")
with r1:
    st.markdown("**7 wall rays** (distance to the wall along each direction)")
    rows = []
    for ang, key in RAYS:
        d = float(obs.get(f"distance_to_wall_{key}", 0.0))
        rows.append({"ray": key, "angle": f"{ang:+d}", "distance": round(d, 1), "how far": min(d, 60.0),
                     "verdict": "DANGER < 10" if d < 10 else "close < 25" if d < 25 else "clear"})
    st.dataframe(rows, hide_index=True, width="stretch",
                 column_config={"distance": st.column_config.NumberColumn(format="%.1f"),
                                "how far": st.column_config.ProgressColumn(min_value=0, max_value=60, format=" ")})
    st.caption("obs key = distance_to_wall_<ray>. The bar is capped at 60 units.")
with r2:
    st.markdown("**Waypoints** (where the track goes next)")
    a1, a2 = float(obs["angle_to_next_waypoint"]), float(obs.get("angle_to_waypoint_after_next", 0.0))

    def side(a: float) -> str:
        return "to the **right**" if a > 0.5 else "to the **left**" if a < -0.5 else "straight ahead"

    st.markdown(f"- next waypoint #{obs['next_waypoint_index']}: **{a1:+.1f} deg**, {side(a1)}, "
                f"{obs['distance_to_next_waypoint']:.0f} units away\n"
                f"- the one after: **{a2:+.1f} deg**, {side(a2)}, {obs.get('distance_to_waypoint_after_next', 0):.0f} units\n"
                f"- gates passed {obs['waypoints_passed']}/{obs['waypoints_total']}, lap {obs['lap']}/{obs['laps_total']}, "
                f"speed **{obs['speed']:.1f}** u/s, crashes {obs['crashes']}")
    st.markdown("**The answer** (what drive() returned)")
    thr, steer = float(action.get("throttle", 0)), float(action.get("steering", 0))
    m = st.columns(2)
    m[0].metric("throttle", f"{thr:+.2f}", "brake" if thr < 0 else ("full" if thr >= 0.99 else None),
                delta_color="off", delta_arrow="off")
    m[1].metric("steering", f"{steer:+.2f}", "right" if steer > 0.02 else "left" if steer < -0.02 else "straight",
                delta_color="off", delta_arrow="off")
    if action.get("say"):
        st.caption(f"radio: \"{action['say']}\"")
with r3:
    st.markdown("**debug** (the driver's own notes: the pink target)")
    st.json(debug or {"(empty)": "add debug={'target': [x, y]} to your Action"}, expanded=True)
    after = rec.get("after") or {}
    if after.get("crashed"):
        st.error(f"This tick ends in a CRASH (+5 s). Events: {', '.join(after.get('events', [])) or '-'}",
                 icon=":material/car_crash:")

with st.expander("The exact obs JSON the driver received this tick", icon=":material/data_object:", expanded=False):
    st.code(json.dumps(obs, indent=1), language="json")
st.caption("Exercise: cover the right half of the screen, read only the JSON, and say what YOU would do. "
           "Then compare with the answer.")
