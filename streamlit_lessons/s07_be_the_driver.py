"""S07 Be the Driver - st.session_state keeps a live simulation alive between reruns. (12 min)

Concept:  a rerun starts the script from scratch - except st.session_state, which survives: we
          park a RaceEnv there. on_click callbacks run BEFORE the rerun; st.rerun() reruns NOW.
Run:      python -m streamlit run streamlit_lessons/s07_be_the_driver.py
You should see: your car on sprint and 5 buttons. 1 click = 1 tick = 0.5 s. Arrows/Space too.
Exercise: (1) an "Autopilot" toggle: on each click, ask a driver file for the action instead;
          (2) an Undo button: store every action, rebuild the car and replay all but the last.
"""

import streamlit as st

import _data  # noqa: F401  (repo on sys.path)
from agentic_gp import RaceEnv
from app.components import live_canvas, live_state

TRACK, s = "sprint", st.session_state
BUTTONS = {"Left": (0.4, -0.8, "Left"), "Gas": (1.0, 0.0, "Up"), "Coast": (0.0, 0.0, "Space"),
           "Brake": (-1.0, 0.0, "Down"), "Right": (0.4, 0.8, "Right")}  # (throttle, steer, key)
if "s07_env" not in s:  # first run, or after "New car": a fresh car on the grid
    s.s07_env, s.s07_trail, s.s07_action = RaceEnv(TRACK), [], (0.0, 0.0)
    s.s07_obs = s.s07_env.reset()


def tick(throttle: float, steering: float) -> None:  # TODO(exercise): autopilot + undo stack
    if not s.s07_obs["attempt_over"]:
        s.s07_obs, s.s07_action = s.s07_env.step(throttle, steering), (throttle, steering)
        s.s07_trail.append(s.s07_obs["position"])


st.title("Be the driver")
for col, (label, (thr, steer, key)) in zip(st.columns(5), BUTTONS.items()):
    col.button(label, on_click=tick, args=(thr, steer), shortcut=key, width="stretch")
live_canvas(live_state(TRACK, obs=s.s07_obs, action=s.s07_action, name="You", color="#e63946",
                       trail=s.s07_trail, crash_points=s.s07_env.crash_points, follow=True,
                       show_waypoints=True), key="s07_car")
st.write(f"Gates passed: {s.s07_obs['waypoints_passed']} - finished: {s.s07_obs['finished']}")
if st.button("New car", icon=":material/restart_alt:"):
    del s["s07_env"]  # forget the old car ...
    st.rerun()  # ... and rerun right now: the `if` at the top builds a new one
