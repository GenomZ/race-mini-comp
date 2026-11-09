"""S04 Telemetry - pandas + charts + a download button.                          (12 min)

Concept:  a list of dicts -> pd.DataFrame -> st.dataframe / st.line_chart / st.bar_chart.
          st.download_button hands the viewer a file (here: a CSV of the lap you picked).
Run:      python -m streamlit run streamlit_lessons/s04_telemetry.py
You should see: a practice day of the Reflex driver on sprint (4 runs with different PARAMS:
          two with crashes, then faster), speed and controls vs time, a per-segment table.
Exercise: (1) show only the CRASH ticks of the selected run (its report["crash_log"]);
          (2) add a column "time lost" = this run's segment time - the best run's time there.
"""

import pandas as pd
import streamlit as st

import _data

st.title("Telemetry")
session = _data.practice()  # one session dict: "attempts" (summary rows) + "reports" (details)
summary = pd.DataFrame(session["attempts"])
st.dataframe(summary[["attempt", "params", "race_time", "crashes"]], hide_index=True)

ticks = pd.DataFrame(_data.telemetry_rows(session))  # a row per tick: attempt, t, speed, ...
run = st.radio("Run", ticks["attempt"].unique(), horizontal=True, format_func=lambda n: f"Run {n}")
lap = ticks[ticks["attempt"] == run]

st.subheader("Speed and controls")
st.line_chart(lap, x="t", y="speed", height=220)  # a crash: speed 0, then 5 s standing still
st.line_chart(lap, x="t", y=["throttle", "steering"], color=["#ff4b4b", "#3987e5"], height=220)

st.subheader("Time per segment (gate to gate)")
segments = pd.DataFrame(session["reports"][run - 1]["segments"])
st.bar_chart(segments, x="segment", y="time", height=220)
st.dataframe(segments, hide_index=True)  # TODO(exercise): crash ticks + a "time lost" column

csv = lap.to_csv(index=False)
st.download_button("Download this run as CSV", csv, f"run_{run}.csv", "text/csv")
