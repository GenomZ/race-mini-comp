"""S06 Run My Driver - buttons, st.status, cache keys, toasts.                     (12 min)

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
    return {"best": result.best_lap, "replay": result.to_replay("all").to_dict()}


if st.button("Race!", type="primary", icon=":material/sports_score:"):
    with st.status(f"{info.name}: 2 attempts on {track}"):  # "complete" when the block ends
        mtime = path.stat().st_mtime  # save your file -> new mtime -> new cache key -> drives again
        s.s06_run = race(mtime, str(path), track, params)  # TODO(exercise): keep the old run
    st.toast(f"Best lap: {s.s06_run['best']} s  (None = no finished lap)")
if "s06_run" in s:  # a replay dict: {"track": ..., "cars": [{"id", "name", "color", "t", ...}]}
    replay_canvas(s.s06_run["replay"], key="s06_replay", height=480, autoplay=True, speed=4)
