"""S02 Corner Calculator - every widget is a variable.                            (9 min)

Concept:  st.slider / st.number_input / st.radio RETURN a value. Move one -> the script reruns.
Run:      python -m streamlit run streamlit_lessons/s02_corner_calculator.py
You should see: the corner speed for a radius, the free road you need to stop, and the arc
          your car drives for the chosen speed + steering. Drag things around!
Exercise: "Can I make it?" - add a number_input for the front-ray distance and show st.success
          if stopping + blind distance fits in it, else st.error.
"""

import math

import plotly.graph_objects as go
import streamlit as st

import _data

GRIP, BRAKE, TICK = 20.0, 25.0, 0.5  # sim/config.py: max_lateral_accel, brake, tick_seconds
st.title("Corner calculator")

radius = st.slider("Corner radius R (units)", 10, 200, 40)
st.metric("Fastest speed through it: v = sqrt(20 R)", f"{math.sqrt(GRIP * radius):.1f} u/s")

speed = st.number_input("Your speed v (u/s, max 40)", 0, 40, 30)
stop, blind = speed**2 / (2 * BRAKE), speed * TICK  # v^2/50 to brake + one tick to react
st.write(f"To stop you need **{stop:.0f}** units of braking + **{blind:.0f}** blind "
         f"= **{stop + blind:.0f}** units.")

# TODO(exercise): front = st.number_input(...); stop + blind <= front ? st.success : st.error

steer = st.radio("Steering (+ = right)", [-1.0, -0.5, -0.25, 0.25, 0.5, 1.0], index=4,
                 horizontal=True)
turn_r = max(speed**2 / GRIP, speed / 1.5) / abs(steer)  # half the steering = twice as wide
fig = go.Figure()
for name, r, color in [("your path", turn_r, "#ff4b4b"), ("the corner", radius, "#9a9a9a")]:
    xs, ys = _data.half_circle(r, side=math.copysign(1, steer))  # the car starts at (0, 0)
    fig.add_scatter(x=xs, y=ys, name=f"{name}, R = {r:.0f}", line=dict(color=color, width=3))
fig.update_layout(title="Your path wider than the corner? Then you meet the wall.", height=420)
st.plotly_chart(fig.update_yaxes(scaleanchor="x"))
