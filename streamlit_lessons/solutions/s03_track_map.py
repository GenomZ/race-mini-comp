"""S03 Track Map - interactive Plotly charts and @st.cache_data.                  (12 min)
SOLUTION  python -m streamlit run streamlit_lessons/solutions/s03_track_map.py

Concept:  st.plotly_chart shows a zoomable, hoverable figure. @st.cache_data remembers a
          function's answer per input: the slow work happens ONCE per track, not on every rerun.
Run:      python -m streamlit run streamlit_lessons/s03_track_map.py
You should see: a track picker, the circuit (walls, gates, T1.. labels, centreline coloured by
          corner speed v_max - bright orange = slow corner), a timing caption and a corner table.
          Pick "sprint", then "grand_prix" again: the caption now says "from the CACHE".
Exercise: add layer checkboxes (v_max colours, corners, gates, zones) and pass them to
          pf.track_figure(...). Hover the centreline: it shows the segment and v_max.
"""

import time

import streamlit as st

import _data
from agentic_gp.viz import plotly_figs as pf

st.title("Track map")
spec = st.selectbox("Track", _data.TRACKS)
c1, c2, c3, c4 = st.columns(4)  # SOLUTION: one checkbox per layer
layers = {"vmax": c1.checkbox("v_max colours", True), "corners": c2.checkbox("Corners", True),
          "gates": c3.checkbox("Gates", True), "zones": c4.checkbox("Yellow-flag zones", True)}


@st.cache_data(show_spinner="Drawing the track ...")  # cache key = the arguments
def build_map(spec: str, layers: dict):  # SOLUTION: the layers are part of the cache key
    started = time.perf_counter()
    fig = pf.track_figure(_data.track(spec), title=_data.track(spec).name, height=600,
                          **layers)  # SOLUTION: vmax=..., corners=..., gates=..., zones=...
    return fig, time.perf_counter() - started, time.time()  # figure, build time, built when


asked = time.time()
fig, build_s, built_at = build_map(spec, layers)  # SOLUTION
got_ms = (time.time() - asked) * 1000
source = "from the CACHE" if built_at < asked else "built just now"
st.caption(f"Figure {source}: drawing it took {build_s * 1000:.0f} ms once, "
           f"getting it now took {got_ms:.0f} ms.")
st.plotly_chart(fig, theme=None)  # theme=None keeps the dark "broadcast" colours of the figure

corners = _data.track(spec).corners()  # [{id: "T1", min_radius, v_max, direction, ...}, ...]
st.dataframe(corners, column_order=["id", "direction", "min_radius", "v_max"], hide_index=True)
