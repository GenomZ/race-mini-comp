"""Live: run a driver file and watch every tick - rays, target point, gauges, radio, reflections."""

import json
import re
import time
from pathlib import Path

import streamlit as st

from agentic_gp.race.session import SessionConfig
from agentic_gp.viz.player import replay_html
from app.components import live_canvas, live_state, replay_canvas
from app.lib.experiments import json_safe
from app.lib.live_runner import LiveRun
from app.lib.llm_health import llm_failures
from app.lib.loaders import track_map
from app.lib.params_ui import changed_params, params_sliders
from app.lib.paths import RACES_DIR, rel
from app.lib.sandbox import is_trusted
from app.lib.ui import driver_picker, llm_run_env, page_header, presenter_mode, real_llm_possible, track_picker

page_header("Live", "Your driver is a function the runner calls every 0.5 s of race time. Watch what it sees "
            "(rays), where it aims (the pink target from debug), what it answers (gauges, radio) and what "
            "reflect() is told between attempts.", chapter="Ch1-Ch6")

LIVE_DIR = RACES_DIR / "live"
presenter = presenter_mode()
CANVAS_H = 520
run: LiveRun | None = st.session_state.get("live_run")
busy = run is not None and not run.done

# ------------------------------------------------------------------------------ setup
# Presenter mode (projector): the setup folds away once a run exists, so the canvas is on screen.
setup = (st.expander(f"Driver, track and options", icon=":material/tune:") if presenter
         else st.container(border=True))
with setup:
    c1, c2, c3, c4 = st.columns([3, 2, 1, 1])
    info = driver_picker("Driver file", key="live_driver", container=c1)
    with c2:
        track = track_picker("Track", key="live_track", default="grand_prix")
    attempts = c3.number_input("Attempts", 1, 8, 2, key="live_attempts",
                               help="reflect(report) runs between attempts: that is where learning happens (Ch3).")
    laps = c4.number_input("Laps", 1, 5, 1, key="live_laps")

    c5, c6, c7 = st.columns([2, 2, 3])
    realtime = c5.toggle("Realtime", key="live_realtime",
                         help="The world does not wait: while drive() thinks, the car keeps its last command "
                              "(an LLM answer 1.5 s late = 3 ticks driven blind). Pace is fixed to 1x.")
    pace = c6.select_slider("Pace (x real time)", [0.5, 1.0, 2.0, 4.0, 8.0, 16.0], value=4.0, key="live_pace",
                            disabled=realtime, help="Only for watching: the race result is the same at any pace.")
    # Student files (anything outside drivers/examples/) and LLM drivers ALWAYS run in their own process:
    # an endless loop or os._exit() there cannot freeze or kill the app, and the LLM choice below goes to
    # that process only (it is never changed for the whole server, i.e. for other browsers).
    forced = info is not None and (not is_trusted(info.path) or info.uses_llm)
    if forced:
        isolate = c7.toggle("Run the file in its own process", value=True, disabled=True, key="live_isolate_forced",
                            help="Always on for student files and LLM drivers: Stop can kill it and a bug in it "
                                 "cannot freeze the app. Takes a few seconds to start.")
    else:
        isolate = c7.toggle("Run the file in its own process", key="live_isolate",
                            help="Reference drivers are safe in the app process. On = run it like a student file "
                                 "(Stop can kill it), but it takes a few seconds to start.")

    llm_mode = None
    if info is not None and info.uses_llm:
        options = ["Mock (free, offline)"] + (["Real LLM (uses the shared key)"] if real_llm_possible() else [])
        llm_mode = st.radio("This driver calls an LLM in " + ("/".join(info.llm_in) or "its code"), options,
                            horizontal=True, key="live_llm_mode",
                            help="Mock answers instantly with canned text; real calls take 1-3 s each and count "
                                 "against the class budget. The choice applies to this run only.")

    params = {}
    if info is not None and info.params:
        with st.expander(f"PARAMS of {info.name} - your knobs, as sliders", icon=":material/tune:"):
            params = params_sliders(info.params, key=f"live_p_{info.stem}", help=info.param_help)
            changed = changed_params(info.params, params)
            if changed:
                st.caption("Changed: " + ", ".join(f"{k}={v}" for k, v in changed.items())
                           + f"  - on the terminal: python race.py drive {info.rel_path} "
                           + " ".join(f"--param {k}={v}" for k, v in changed.items()))


# ------------------------------------------------------------------------------ controls
def _start() -> None:
    old = st.session_state.get("live_run")
    if old is not None:
        old.stop(kill=True)
    if info is None:
        return
    real = bool(info.uses_llm and str(st.session_state.get("live_llm_mode", "Mock")).startswith("Real"))
    env = {"LLM_PROVIDER": "mock"}
    if info.uses_llm:
        LIVE_DIR.mkdir(parents=True, exist_ok=True)
        env = llm_run_env(real, LIVE_DIR / "llm_logs")                 # the Pit Wall page reads these logs
    run_isolated = bool(isolate) or forced
    # In its own process every drive() answer has a time limit (late = a stall: the car coasts). An LLM
    # pilot thinks for seconds per tick, so it gets a long one; a code driver needing > 5 s is stuck.
    drive_timeout = (60.0 if "drive" in info.llm_in else 5.0) if run_isolated else None
    cfg = SessionConfig(track=track, attempts=int(attempts), laps=int(laps), realtime=bool(realtime),
                        drive_timeout_s=drive_timeout)
    st.session_state.live_run = LiveRun(str(info.path), cfg, params=changed_params(info.params, params) or None,
                                        isolate=run_isolated, pace=1.0 if realtime else float(pace), env=env).start()
    st.session_state.live_llm = {"real": real, "log_dir": env.get("LLM_LOG_DIR")} if info.uses_llm else None
    st.session_state.live_saved = None


def _pause() -> None:
    if (r := st.session_state.get("live_run")) is not None:
        r.toggle_pause()


def _step() -> None:
    if (r := st.session_state.get("live_run")) is not None:
        if not r.paused:
            r.pause()
        r.step()


def _stop() -> None:
    if (r := st.session_state.get("live_run")) is not None:
        r.stop(kill=True)


b = st.columns([2, 2, 2, 2, 3])
b[0].button("Start" if not busy else "Restart", key="live_start", on_click=_start, type="primary", width="stretch",
            icon=":material/play_arrow:", disabled=info is None)
b[1].button("Resume" if (run is not None and run.paused) else "Pause", key="live_pause", on_click=_pause,
            width="stretch", icon=":material/pause:", disabled=not busy)
b[2].button("Step 1 tick", key="live_step", on_click=_step, width="stretch", icon=":material/skip_next:",
            disabled=not busy, help="Pauses, then advances exactly one tick per click.")
b[3].button("Stop", key="live_stop", on_click=_stop, width="stretch", icon=":material/stop:", disabled=not busy)
with b[4]:
    follow_cam = st.toggle("Chase camera", key="live_follow", help="Or click the canvas to switch.")


# ------------------------------------------------------------------------------ the live view (polled)
def chart_spec(field: str, title: str, domain: list) -> dict:
    """A FIXED Vega-Lite spec: only the data changes between polls, so the chart updates in place."""
    return {"mark": {"type": "line", "strokeWidth": 2},
            "encoding": {"x": {"field": "t", "type": "quantitative", "title": "race clock (s)"},
                         "y": {"field": field, "type": "quantitative", "title": title, "scale": {"domain": domain}},
                         "color": {"field": "attempt", "type": "nominal", "title": None,
                                   "legend": {"orient": "top"}}}}


@st.fragment(run_every=0.25 if busy else None)
def live_view() -> None:
    run = st.session_state.get("live_run")
    if run is None:
        try:
            tm = track_map(track)                          # the car waits on the grid
        except Exception as exc:                           # noqa: BLE001 - e.g. a broken saved track file
            st.error(f"Cannot build the track {track}: {type(exc).__name__}: {exc}", icon=":material/error:")
            return
        sx, sy = tm["start_position"]
        idle = live_state(track, car={"x": sx, "y": sy, "heading_deg": tm["start_heading_deg"], "speed": 0.0},
                          name=info.name if info else "car", color=(info.color if info and info.color else "#3987e5"),
                          status="paused", message="Press Start", vmax=True, seq=0)
        idle.update(tick=0, t=0.0, lap=1, laps=int(laps))   # (no obs yet: show zeros, not "null")
        live_canvas(idle, key="live_canvas", height=CANVAS_H, fit=presenter)
        return
    run.heartbeat()
    snap = run.snapshot()
    state = run.canvas_state(follow=follow_cam)
    if state.get("tick") is None:                          # loading / planning: no tick yet
        state.update(tick=0, t=0.0)
    live_canvas(state, key="live_canvas", height=CANVAS_H, fit=presenter)

    m = st.columns(6)
    m[0].metric("Status", snap.status)
    m[1].metric("Attempt", f"{snap.attempt}/{snap.attempts}" + (f" ({snap.phase})" if snap.phase not in ("", "session") else ""))
    m[2].metric("Lap", f"{snap.lap}/{snap.laps}")
    m[3].metric("Race clock", f"{snap.t:.1f} s")
    m[4].metric("Crashes", snap.crashes)
    m[5].metric("drive() took", "-" if snap.think_s is None else
                (f"{snap.think_s * 1000:.1f} ms" if snap.think_s < 1 else f"{snap.think_s:.2f} s"))
    if snap.error:
        st.error(snap.error, icon=":material/error:")
    elif snap.message and snap.status in ("loading", "planning", "reflecting"):
        st.caption(snap.message)

    tele = run.telemetry_frame()
    if len(tele) > 1:
        tele = tele.assign(attempt="attempt " + tele["attempt"].astype(str))[["t", "speed", "steering", "attempt"]]
        g1, g2 = st.columns(2)
        g1.vega_lite_chart(tele, chart_spec("speed", "speed (u/s)", [0, 40]), height=230, key="live_speed_chart")
        g2.vega_lite_chart(tele, chart_spec("steering", "steering (+ = right)", [-1, 1]), height=230,
                            key="live_steer_chart")

    reports = run.reports()
    if reports:
        st.markdown("**After each attempt** - the lap report reflect() received:")
        for rep in reversed(reports):
            outcome = f"{rep['race_time']:.2f} s" if rep["completed"] else rep["status"]
            with st.expander(f"Attempt {rep['attempt']}: {outcome}, {rep['crashes']} crash(es), "
                             f"slowest drive() {rep['think_ms_max']:.0f} ms", expanded=rep is reports[-1] and len(reports) == 1):
                st.code(rep["summary_text"] or "(no summary)", language=None, wrap_lines=True)
                if rep.get("reflect_error"):
                    st.error(f"reflect() raised: {rep['reflect_error']}")
    if snap.done and busy:
        st.rerun()                                        # one full rerun: polling off, replay below


live_view()


def llm_health_note(run: LiveRun) -> None:
    """Warn when this run's LLM calls failed: the driver then falls back and still finishes, which hides it."""
    llm_info = st.session_state.get("live_llm")
    if not llm_info or not llm_info.get("log_dir"):
        return
    bad = llm_failures(llm_info["log_dir"], since_utc=getattr(run, "started_utc", None),
                       label=run.snapshot().name or None)
    usage = (run.result.llm if run.result is not None else None) or {}
    n_err = max(bad["failed"], int(usage.get("errors", 0) or 0))
    n_calls = max(bad["calls"], int(usage.get("calls", 0) or 0), n_err)
    where = "real LLM" if llm_info.get("real") else "mock LLM"
    if n_err:
        st.warning(f"**{n_err} of {n_calls} LLM call(s) failed** ({where}) - the driver fell back to its non-LLM "
                   f"code, so the lap time says nothing about the LLM. First error: "
                   f"{bad['first_error'] or 'see the Pit Wall page'}", icon=":material/cloud_off:")
    elif n_calls:
        st.caption(f":material/cloud_done: {n_calls} LLM call(s) in this run ({where}), none failed. "
                   "Read the conversation on the Pit Wall page.")


# ------------------------------------------------------------------------------ after the session
run = st.session_state.get("live_run")
if run is not None and run.done:
    events = run.events()
    llm_health_note(run)
    if run.result is None:
        st.warning("The session ended without a result (stopped or the driver failed).", icon=":material/warning:")
    else:
        res = run.result
        st.divider()
        st.subheader("Session replay - every attempt as a ghost car", anchor=False)
        replay = run.replay("all")
        if replay is not None:
            replay_canvas(replay, key="live_replay", height=460, autoplay=True, speed=2.0, fit=presenter)
        rows = [{"attempt": r.result.attempt, "phase": r.phase,
                 "time": r.result.race_time if r.result.completed else None,
                 "laps": ", ".join(f"{x:.2f}" for x in r.result.lap_times), "crashes": r.result.crashes,
                 "zone violations": r.result.zone_violations, "status": r.status or r.result.ended_by}
                for r in res.attempts]
        if rows:
            st.dataframe(rows, hide_index=True, width="stretch",
                         column_config={"time": st.column_config.NumberColumn(format="%.2f s")})
        spec = str(run.config.track)
        track_tag = re.sub(r"[^A-Za-z0-9_-]+", "-", Path(spec[5:]).stem if spec.startswith("file:") else spec)
        stem = f"{info.stem if info else 'driver'}_{track_tag.strip('-') or 'track'}"
        session_json = json.dumps(json_safe(res.to_dict()), default=str)
        d = st.columns(4)
        d[0].download_button("Session JSON", session_json, f"{stem}_session.json", "application/json",
                             icon=":material/download:", on_click="ignore", width="stretch", key="live_dl_session")
        if replay is not None:
            d[1].download_button("Replay JSON", replay.to_json(), f"{stem}_replay.json", "application/json",
                                 icon=":material/download:", on_click="ignore", width="stretch", key="live_dl_replay")
            d[2].download_button("HTML player", lambda: replay_html(replay, height=None), f"{stem}.html", "text/html",
                                 icon=":material/download:", on_click="ignore", width="stretch", key="live_dl_html")

        def _save() -> None:
            LIVE_DIR.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%H%M%S")
            p1 = LIVE_DIR / f"{stem}_{stamp}_session.json"
            p1.write_text(session_json, encoding="utf-8")
            if replay is not None:
                replay.to_json(LIVE_DIR / f"{stem}_{stamp}_replay.json")
            st.session_state.live_saved = rel(p1)

        d[3].button("Save to races/live/", on_click=_save, icon=":material/save:", width="stretch", key="live_save",
                    help="Then open it on the Telemetry and Learning Curves pages.")
        if st.session_state.get("live_saved"):
            st.success(f"Saved {st.session_state.live_saved} (+ replay). Open it on Telemetry / Learning Curves.",
                       icon=":material/check:")
    if events:
        with st.expander("Event log", icon=":material/list:"):
            st.code("\n".join(events[-300:]), language=None)
