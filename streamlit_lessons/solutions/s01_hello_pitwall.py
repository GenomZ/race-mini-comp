"""S01 Hello, Pit Wall - your first Streamlit app.                                (6 min)
SOLUTION  python -m streamlit run streamlit_lessons/solutions/s01_hello_pitwall.py

Concept:  a Streamlit app is a plain Python script. Any click or edit RERUNS it top to bottom.
Run:      python -m streamlit run streamlit_lessons/s01_hello_pitwall.py   (stop: Ctrl+C)
You should see: a title, four lap-time tiles (L0 ... Max) and the dict your driver gets each tick.
Try:      change the title text and save the file - the page updates by itself (runOnSave).
Exercise: add YOUR best lap as a 5th tile, with its gap to Max as the delta
          (delta_color="inverse": slower than Max = red).
"""

import streamlit as st

import _data  # demo data for every lesson (and it makes `import agentic_gp` work)

st.title("Pit Wall")
st.markdown("Best lap on the **Grand Prix** circuit at each level of the ladder you climb.")
st.caption("From the 2-lap class demo race: lap 2 starts at full speed, so these beat the 1-lap "
           "time `race.py drive` prints (L1 Reflex 66.70 s, L2 Planner 52.50 s).")

laps = _data.reference_laps()  # {"L0 Dummy": None, "L1 Reflex": 65.0, "L2 Planner": 50.8, ...}
tiles = st.columns(len(laps) + 1)  # SOLUTION: one more column for "Me"
for tile, (name, lap) in zip(tiles, laps.items()):
    tile.metric(name, f"{lap:.1f} s" if lap else "DNF", border=True)

my_lap = st.number_input("My best lap (s)", 30.0, 300.0, 58.0, step=0.1)  # SOLUTION
gap = my_lap - (laps["Max (house)"] or 46.9)  # SOLUTION: > 0 = slower than Max
tiles[-1].metric("Me", f"{my_lap:.1f} s", delta=f"{gap:+.1f} s vs Max", border=True,
                 delta_color="inverse")  # SOLUTION: for lap times, bigger = worse = red

st.subheader("What your driver sees")
obs = _data.sample_obs()
st.info(f"+ = RIGHT, - = LEFT. Here the car does {obs['speed']:.1f} u/s and the next gate is "
        f"{obs['angle_to_next_waypoint']:+.0f} deg away, "
        f"{obs['distance_to_next_waypoint']:.0f} units ahead.")

"Every 0.5 s the runner calls `drive(obs)` with this dict, and you answer `(throttle, steering)`:"
obs  # "magic": a value alone on a line is shown with st.write - try a dict, a list, a DataFrame
