"""S06 Run My Driver - buttons, st.status, cache keys, toasts.                     (12 min)
SOLUTION  python -m streamlit run streamlit_lessons/solutions/s06_run_my_driver.py

Concept:  st.button is True for ONE rerun, so keep results in st.session_state. @st.cache_data
          keyed on (file mtime, track, params) drives again only when something really changed.
Run:      python -m streamlit run streamlit_lessons/s06_run_my_driver.py
You should see: a driver picker, a slider per PARAMS knob (made from the file!), a Race button,
          then 2 attempts as ghost cars. Race again without changes: instant (cache).
Exercise: compare mode - keep the PREVIOUS run in st.session_state, show both runs as ghosts in
          one replay and a metric with the time difference.
"""

import streamlit as st

import _data
from agentic_gp import SessionConfig, load_driver, run_session
from app.components import replay_canvas

s = st.session_state
st.title("Run my driver")
path = st.selectbox("Driver", _data.driver_files(), _data.default_index(), format_func=_data.rel)
track = st.selectbox("Track", _data.TRACKS)
info = _data.driver_info(path)  # reads NAME, PARAMS, ... WITHOUT running the file
knobs = {k: v for k, v in info.params.items() if type(v) in (int, float)}  # the number knobs
params = {k: st.slider(k, min(type(v)(0), 2 * v), max(type(v)(1), 2 * v), v,  # int knob: int slider
                       help=info.param_help.get(k)) for k, v in knobs.items()}


@st.cache_data(show_spinner=False)  # same file contents + track + params -> answer from memory
def race(mtime: float, path: str, track: str, params: dict) -> dict:
    result = run_session(load_driver(path, params=params), SessionConfig(track=track, attempts=2))
    knobs = ", ".join(f"{k}={v:g}" for k, v in params.items())  # SOLUTION: what this run was
    return {"best": result.best_lap, "replay": result.to_replay("all").to_dict(),
            "track": track, "label": f"{result.driver['name']} {knobs}"}  # SOLUTION


if st.button("Race!", type="primary", icon=":material/sports_score:"):
    with st.status(f"{info.name}: 2 attempts on {track}"):  # "complete" when the block ends
        mtime = path.stat().st_mtime  # save your file -> new mtime -> new cache key -> drives again
        s.s06_prev = s.get("s06_run")  # SOLUTION: keep the old run before overwriting it
        s.s06_run = race(mtime, str(path), track, params)
    st.toast(f"Best lap: {s.s06_run['best']} s  (None = no finished lap)")
if "s06_run" in s:  # a replay dict: {"track": ..., "cars": [{"id", "name", "color", "t", ...}]}
    run, prev = s.s06_run, s.get("s06_prev")  # SOLUTION: compare mode
    replay = dict(run["replay"])  # a copy: we add cars to it
    if prev and prev["track"] == run["track"]:  # same track -> race the ghosts together
        replay["cars"] = replay["cars"] + [
            dict(c, id="prev-" + c["id"], name="prev " + c["name"], color="#9a9a9a")
            for c in prev["replay"]["cars"]]
        now, before = run["best"], prev["best"]
        delta = f"{now - before:+.2f} s vs previous" if now and before else None
        st.metric(f"Best lap now ({run['label']})", f"{now} s", delta, delta_color="inverse")
        st.caption(f"Previous run: {prev['label']} - best {before} s (the grey cars).")
    replay_canvas(replay, key="s06_replay", height=480, autoplay=True, speed=4)
