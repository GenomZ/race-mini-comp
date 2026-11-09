"""S08 Live View - animation with @st.fragment(run_every=...).                    (14 min)

Concept:  a fragment reruns ON ITS OWN (here every 0.15 s); the rest of the page stays put. It
          replaces the old st.empty() + time.sleep loop. Compute the lap first (< 1 s), then play.
Run:      python -m streamlit run streamlit_lessons/s08_live_view.py
You should see: a car lapping sprint (rays, pink debug target, gauges, radio) + a progress bar.
Exercise: (1) a playback-speed slider (ticks per refresh); (2) a "realtime" toggle with a
          think-time slider: a command-age gauge + a ghost car where the driver was when it decided.
"""

import streamlit as st

import _data
from app.components import live_canvas, live_state

st.title("Live view")
path = st.selectbox("Driver", _data.driver_files(), _data.default_index(), format_func=_data.rel)
lap = _data.frames(str(path), "sprint", path.stat().st_mtime)  # the whole lap, tick by tick
frames, s = lap["frames"], st.session_state
s.s08_i = min(s.get("s08_i", 0), len(frames) - 1)  # which tick we show now (survives reruns)
playing = st.toggle("Play", value=True)
left, right = st.columns(2)
left.button("Step one tick", on_click=lambda: s.update(s08_i=min(s.s08_i + 1, len(frames) - 1)))
right.button("Restart", on_click=lambda: s.update(s08_i=0))


@st.fragment(run_every=0.15 if playing else None)  # only THIS function reruns, ~7 times a second
def live() -> None:
    if playing and s.s08_i < len(frames) - 1:
        s.s08_i += 1  # TODO(exercise): playback speed + realtime command-age gauge
    f, past = frames[s.s08_i], frames[: s.s08_i + 1]  # this tick (obs + answer), all ticks so far
    status = f.get("status") or ("driving" if playing else "paused")  # the last one: "finished"
    live_canvas(live_state("sprint", obs=f["obs"], action=(f["throttle"], f["steering"]),
                           debug=f["debug"], name=lap["name"], color=lap["color"], status=status,
                           say=[{"t": g["t"], "text": g["say"]} for g in past if g["say"]],
                           trail=[[g["x"], g["y"]] for g in past[:-1]]), key="s08_live", height=460)
    st.progress((s.s08_i + 1) / len(frames), f"tick {f['tick']} of {frames[-1]['tick']} - {status}")


live()
