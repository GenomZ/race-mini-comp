"""S09 Track Lab - st.data_editor, st.form and clickable charts.      (10 min - optional)

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
if st.button("Race here (L2 planner)", disabled=bool(problems)):  # TODO(exercise): any driver
    result = run_session(str(_data.EXAMPLES / "l2_planner.py"), SessionConfig(track=track))
    st.metric("L2 planner on your track", f"{result.best_lap} s")  # None = it did not finish
