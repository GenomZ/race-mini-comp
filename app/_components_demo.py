"""Components demo: every shared canvas on one page (for the page teams and for testing).

    streamlit run app/_components_demo.py

Tabs: Replay (multi-car player), Live (a driver in a background thread), Hand-drive (keyboard
pad + live canvas, one 0.5 s tick per 0.5 s), Export (standalone HTML player) and Figures (Plotly).
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:                      # make `import agentic_gp` / `import app` work
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from agentic_gp.race.session import SessionConfig  # noqa: E402
from agentic_gp.viz import plotly_figs as pf  # noqa: E402
from agentic_gp.viz.player import replay_html  # noqa: E402
from app.components import drive_pad, live_canvas, live_state, pad_controls, replay_canvas  # noqa: E402
from app.lib.checkpoints import render_where_am_i  # noqa: E402
from app.lib.drivers import discover_drivers  # noqa: E402
from app.lib.live_runner import LiveRun  # noqa: E402
from app.lib.loaders import list_replays, load_json, load_replay, replay_label, standings_table, track_choices  # noqa: E402
from app.lib.params_ui import params_sliders  # noqa: E402

st.set_page_config(page_title="Components demo", page_icon=":material/sports_motorsports:", layout="wide")
st.title("Race canvases - components demo")
st.caption("What you learn here: Python sends data, the browser draws. One keyed canvas per view; "
           "the replay plays with zero reruns, the live view glides between ticks.")
render_where_am_i()

tab_replay, tab_live, tab_drive, tab_export, tab_figs = st.tabs(
    ["Replay", "Live", "Hand-drive", "Export HTML", "Figures"])

# ------------------------------------------------------------------------------------- replay
with tab_replay:
    replays = list_replays()
    if not replays:
        st.warning("No replays in races/ yet - run `python races/demo/make_demo.py`.")
    else:
        default = next((i for i, p in enumerate(replays) if p.name == "class_demo.json"), 0)
        choice = st.selectbox("Replay", replays, index=default, format_func=replay_label, key="demo_replay_file")
        c1, c2, c3 = st.columns(3)
        speed = c1.select_slider("Start speed", [0.5, 1, 2, 4, 8], value=2, key="demo_speed")
        vmax = c2.toggle("Colour the centerline by corner speed", key="demo_vmax")
        report = c3.toggle("Report time back to Python", value=True, key="demo_report")
        res = replay_canvas(choice, key="demo_replay", height=560, speed=speed, vmax=vmax, report=report)
        t = res.t if res is not None and res.t is not None else None
        if t is None:
            st.caption("Official classification (pause the player to see the standings at that moment):")
            st.dataframe(standings_table(load_replay(choice)), hide_index=True)
        else:
            st.caption(f"Paused at t = {t:.2f} s - standings at that moment (follow: {res.follow or 'overview'}):")
            st.dataframe(standings_table(load_replay(choice), t), hide_index=True)

# ------------------------------------------------------------------------------------- live
with tab_live:
    drivers = [d for d in discover_drivers() if d.style != "none"]
    names = [d.label for d in drivers]
    pick = next((i for i, d in enumerate(drivers) if d.stem == "l2_planner"), 0)
    c1, c2, c3, c4 = st.columns([3, 2, 1, 2])
    label = c1.selectbox("Driver", names, index=pick, key="live_driver")
    info = drivers[names.index(label)]
    track = c2.selectbox("Track", track_choices(), key="live_track")
    attempts = c3.number_input("Attempts", 1, 6, 2, key="live_attempts")
    pace = c4.select_slider("Pace (x real time)", [0.5, 1.0, 2.0, 4.0, 8.0, 16.0], value=4.0, key="live_pace")
    with st.expander(f"PARAMS of {info.name}", expanded=False):
        params = params_sliders(info.params, key=f"live_{info.stem}", help=info.param_help)

    def _start() -> None:
        old = st.session_state.get("live_run")
        if old is not None:
            old.stop()
        cfg = SessionConfig(track=st.session_state.live_track, attempts=int(st.session_state.live_attempts))
        st.session_state.live_run = LiveRun(str(info.path), cfg, params=params or None,
                                            pace=float(st.session_state.live_pace)).start()

    def _pause() -> None:
        if (r := st.session_state.get("live_run")) is not None:
            r.toggle_pause()

    def _step() -> None:
        if (r := st.session_state.get("live_run")) is not None:
            r.step()

    def _stop() -> None:
        if (r := st.session_state.get("live_run")) is not None:
            r.stop()

    b1, b2, b3, b4 = st.columns(4)
    b1.button("Start", key="live_start", on_click=_start, type="primary", width="stretch", icon=":material/play_arrow:")
    b2.button("Pause / resume", key="live_pause", on_click=_pause, width="stretch", icon=":material/pause:")
    b3.button("Step one tick", key="live_step", on_click=_step, width="stretch", icon=":material/skip_next:")
    b4.button("Stop", key="live_stop", on_click=_stop, width="stretch", icon=":material/stop:")

    run = st.session_state.get("live_run")
    polling = run is not None and not run.done

    @st.fragment(run_every=0.25 if polling else None)
    def live_view() -> None:
        run = st.session_state.get("live_run")
        if run is None:
            live_canvas(live_state(st.session_state.live_track, message="Press Start to send the car out"),
                        key="demo_live", height=480)
            return
        run.heartbeat()
        snap = run.snapshot()
        live_canvas(run.canvas_state(), key="demo_live", height=480)
        m = st.columns(5)
        m[0].metric("Status", snap.status)
        m[1].metric("Attempt", f"{snap.attempt}/{snap.attempts}")
        m[2].metric("Tick", snap.tick)
        m[3].metric("Race clock", f"{snap.t:.1f} s")
        m[4].metric("Crashes", snap.crashes)
        tele = run.telemetry_frame()
        if len(tele) > 1:
            tele["attempt"] = "attempt " + tele["attempt"].astype(str)
            st.line_chart(tele, x="t", y="speed", color="attempt", height=200)
        for rep in run.reports():
            with st.expander(f"Attempt {rep['attempt']}: "
                             + (f"{rep['race_time']:.2f} s" if rep["completed"] else rep["status"])
                             + f", {rep['crashes']} crash(es) - what reflect() was told"):
                st.code(rep["summary_text"] or "(no summary)", language=None)
        if snap.done and polling:
            st.rerun()                                   # full rerun: polling off

    live_view()
    run = st.session_state.get("live_run")
    if run is not None and run.done and run.result is not None:
        st.subheader("Session replay")
        replay_canvas(run.replay("all"), key="demo_live_replay", height=420)

# ------------------------------------------------------------------------------------- hand-drive
with tab_drive:
    st.caption("Click the pad, then hold the arrow keys (or WASD). One sim tick every 0.5 s, like a driver.")
    hd_track = st.selectbox("Track", track_choices(0), key="hd_track")

    def _new_env() -> None:
        from agentic_gp.sim.env import RaceEnv

        env = RaceEnv(st.session_state.hd_track)
        st.session_state.hd_env = env
        st.session_state.hd_obs = env.reset()
        st.session_state.hd_last = 0.0
        st.session_state.hd_trail = []

    c1, c2 = st.columns([1, 3])
    c1.button("New car / reset", key="hd_reset", on_click=_new_env, icon=":material/restart_alt:")
    engine = c2.toggle("Engine on (the page polls the pad 10x per second)", value=True, key="hd_on")
    if "hd_env" not in st.session_state:
        _new_env()

    @st.fragment(run_every=0.1 if engine else None)
    def hand_drive() -> None:
        pressed = drive_pad(key="hd_pad")
        env, obs = st.session_state.hd_env, st.session_state.hd_obs
        throttle, steering = pad_controls(pressed)
        engaged = any(pressed.values())
        if engaged and not obs["attempt_over"] and time.monotonic() - st.session_state.hd_last >= 0.5:
            st.session_state.hd_last = time.monotonic()
            obs = st.session_state.hd_obs = env.step(throttle, steering)
            st.session_state.hd_trail.append(obs["position"])
        status = "finished" if obs["finished"] else ("driving" if engaged else "paused")
        live_canvas(live_state(st.session_state.hd_track, obs=obs, action=(throttle, steering), name="You", color="#e63946",
                               seq=obs["tick"], status=status, trail=st.session_state.hd_trail[-200:],
                               crash_points=list(env.crash_points), tween_ms=500, follow=True, show_waypoints=True),
                    key="hd_canvas", height=440)
        c = st.columns(4)
        c[0].metric("Speed", f"{obs['speed']:.1f}")
        c[1].metric("Gates", f"{obs['waypoints_passed']}/{obs['waypoints_total']}")
        c[2].metric("Crashes", obs["crashes"])
        c[3].metric("Race clock", f"{obs['race_time']:.1f} s")

    hand_drive()

# ------------------------------------------------------------------------------------- export
with tab_export:
    replays = list_replays()
    if replays:
        src = st.selectbox("Replay to export", replays, format_func=replay_label, key="export_file")
        html = replay_html(load_json(src), height=None, autoplay=True)
        st.download_button("Download standalone HTML player", html, file_name=Path(src).with_suffix(".html").name,
                           mime="text/html", icon=":material/download:", on_click="ignore")
        st.caption(f"{len(html) // 1024} KB, no internet needed. The same JavaScript as the replay canvas.")

# ------------------------------------------------------------------------------------- figures
with tab_figs:
    ft = st.selectbox("Track", track_choices(2), key="fig_track")
    st.plotly_chart(pf.track_figure(ft, gate_numbers=True), theme=None, key="fig_trackmap")
    st.plotly_chart(pf.speed_profile_figure(ft), theme=None, key="fig_profile")
