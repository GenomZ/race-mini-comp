"""S09 Track Lab - st.data_editor, st.form and clickable charts.      (10 min - optional)
SOLUTION  python -m streamlit run streamlit_lessons/solutions/s09_track_lab.py

Concept:  st.data_editor is an editable table; st.form holds every edit until you press Submit;
          st.plotly_chart(on_select="rerun") sends what you click on the chart back to Python.
Run:      python -m streamlit run streamlit_lessons/s09_track_lab.py
You should see: the Grand Prix control points as a table. Change some numbers (or add rows),
          press "Build track": validity checks + your circuit coloured by corner speed. Click
          the centreline to read a point; "Race here" sends the L2 planner around your track.
Exercise: design a track that breaks your neighbour's driver: pick their file, race it here and
          mark its crashes on the map.
"""

import pandas as pd
import streamlit as st

import _data
from agentic_gp import SessionConfig, run_session
from agentic_gp.sim.track import GRAND_PRIX_CONTROL_POINTS, track_from_control_points
from agentic_gp.viz import plotly_figs as pf

st.title("Track lab")
points = pd.DataFrame(GRAND_PRIX_CONTROL_POINTS, columns=["x", "y"]) * 0.8
with st.form("designer"):  # nothing reruns while you edit - only when you press the button
    table = st.data_editor(points, num_rows="dynamic")
    width = st.slider("Track width (units)", 14, 30, 18)
    st.form_submit_button("Build track", type="primary")
track = track_from_control_points(table.dropna().values, name="My track", half_width=width / 2)
problems = track.validate(min_radius=12)  # tight corners, crossing walls, start not straight ...
if problems:
    st.error("Not raceable: " + "; ".join(problems))
else:
    st.success(f"Valid! {track.length:.0f} units long, "
               f"tightest corner R = {track.min_turn_radius():.0f}")
fig = pf.track_figure(track, height=560)
clicked = st.plotly_chart(fig, theme=None, on_select="rerun", selection_mode="points")
if clicked.selection.points:
    st.write("You clicked:", clicked.selection.points[0])
victim = st.selectbox("Driver to break", _data.driver_files(), _data.default_index(),  # SOLUTION
                      format_func=_data.rel)
if st.button("Race here", type="primary", disabled=bool(problems)):
    result = run_session(str(victim), SessionConfig(track=track))
    report = result.to_dict()["reports"][0]  # a button is True ONE rerun: keep the result
    st.session_state.s09_result = report | {"driver": result.driver["name"]}
report = st.session_state.get("s09_result")
if report:  # SOLUTION: the result, its trajectory and red X crash markers
    lap = f"{report['race_time']:.2f} s" if report["completed"] else "DNF"
    st.metric(f"{report['driver']} on your track", lap, f"{report['crashes']} crash(es)",
              delta_color="off")
    fig = pf.add_trajectory(pf.track_figure(track, vmax=False, height=560), report)
    st.plotly_chart(pf.add_crashes(fig, [c["position"] for c in report["crash_log"]]), theme=None)
