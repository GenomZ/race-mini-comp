"""S05 Driver Card - layout: sidebar, columns, tabs, expander.                    (10 min)
SOLUTION  python -m streamlit run streamlit_lessons/solutions/s05_driver_card.py

Concept:  st.sidebar holds controls, st.columns puts things side by side, st.tabs and
          st.expander hide detail until you ask for it.
Run:      python -m streamlit run streamlit_lessons/s05_driver_card.py
You should see: a run picker in the sidebar, three tiles, the lap on the map coloured by speed
          (hover it!), Speed / Controls / Radio tabs and the raw report folded in an expander.
Exercise: add a "Crashes" tab: a table of report["crash_log"] (tick, corner, speed_before) and
          red X markers on the map: pf.add_crashes(fig, [c["position"] for c in crashes]).
"""

import pandas as pd
import streamlit as st

import _data
from agentic_gp.viz import plotly_figs as pf

session = _data.practice()
with st.sidebar:
    st.header(session["name"])
    st.caption(f"{session['author']} - *{session['motto']}*")
    run = st.selectbox("Run", [a["attempt"] for a in session["attempts"]], index=2)
report = session["reports"][run - 1]  # all about this run: telemetry, segments, crash_log ...
telemetry = pd.DataFrame(report["telemetry"])
st.title(f"Driver card - run {run} ({report['params']})")
a, b, c = st.columns(3)
a.metric("Race time", f"{report['race_time']:.2f} s")
b.metric("Crashes", report["crashes"])
c.metric("Top speed", f"{telemetry['speed'].max():.1f} u/s")

left, right = st.columns([3, 2])
fig = pf.track_figure(session["track"], vmax=False, height=520)
crashes = report["crash_log"]  # SOLUTION: [{tick, t, position, corner, speed_before, ...}]
pf.add_crashes(fig, [c["position"] for c in crashes],  # SOLUTION: red X markers
               labels=[f"{c['corner']} at {c['t']:.1f} s" for c in crashes])
left.plotly_chart(pf.add_trajectory(fig, report), theme=None)  # dots coloured by speed
speed_tab, ctrl_tab, radio_tab, crash_tab = right.tabs(  # SOLUTION: a 4th tab
    ["Speed", "Controls", "Radio", f"Crashes ({len(crashes)})"])
if crashes:  # SOLUTION
    table = pd.DataFrame(crashes)[["tick", "t", "corner", "speed_before", "steering"]]
    crash_tab.dataframe(table, hide_index=True)
else:
    crash_tab.success("Clean run - no crashes.")
speed_tab.line_chart(telemetry, x="t", y="speed")
ctrl_tab.line_chart(telemetry, x="t", y=["throttle", "steering"], color=["#ff4b4b", "#3987e5"])
for msg in report["messages"] or [{"t": 0, "text": "(this run said nothing on the radio)"}]:
    radio_tab.chat_message("assistant").write(f"**{msg['t']:.1f} s** {msg['text']}")
with st.expander("Raw report - what reflect() receives after the run"):
    st.json({k: v for k, v in report.items() if k not in ("telemetry", "segments")}, expanded=False)
