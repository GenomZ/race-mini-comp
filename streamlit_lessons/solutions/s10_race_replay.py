"""S10 Race Replay - a multipage app with YOUR multi-car replay viewer.            (18 min)
SOLUTION  python -m streamlit run streamlit_lessons/solutions/s10_race_replay.py

Concept:  st.navigation + st.Page turn many scripts into ONE app (S01-S09 = pages in the menu).
          A time slider "scrubs" the race: each car is interpolated between its 0.5 s frames.
Run:      python -m streamlit run streamlit_lessons/s10_race_replay.py
You should see: a menu; "Race replay" = the class demo race: a race-clock slider, cars on the map,
          a timing tower, radio, Follow. Tick Play. The "Pro player" tab is the shared canvas.
Exercise: a "My learning curve" page: _data.result_files() -> pf.learning_curve_figure(...)
          (results from: python race.py drive <you>.py --track sprint --attempts 6 --save).
"""

import math
from pathlib import Path

import pandas as pd  # SOLUTION
import plotly.graph_objects as go
import streamlit as st

import _data
from agentic_gp.race.replay import car_state_at, standings_at
from agentic_gp.viz import plotly_figs as pf
from app.components import replay_canvas

st.set_page_config(page_title="My Race Control", layout="wide")


@st.cache_data(show_spinner="Drawing the track ...")
def track_drawing(path: str):  # the track never changes during a race: draw it once
    return pf.track_figure(_data.load_json(path), vmax=False, corners=False, height=580)


def race_map(path: str, rep: dict, t: float, follow: str | None) -> go.Figure:
    fig, cars = go.Figure(track_drawing(path)), {c["id"]: c for c in rep["cars"]}  # a copy
    named = []  # where names are written already: a car right next to one gets no label
    for cid, *_ in standings_at(rep, t):  # leader first
        car, p = cars[cid], car_state_at(cars[cid], t)  # x, y, heading interpolated at time t
        x, y = p["x"], p["y"]
        free = cid == follow or all(math.dist((x, y), q) > 18 for q in named)
        named += [(x, y)] if free else []
        pf.add_car(fig, x, y, p["heading_deg"], color=car["color"],
                   name=car["name"] if free else "", hover=car["name"])
        if cid == follow:  # zoom: a 160 x 120 window around the followed car
            fig.update_xaxes(range=[x - 80, x + 80]).update_yaxes(range=[y - 60, y + 60])
    return fig


def timing_tower(rep: dict, t: float) -> list[dict]:
    names = {c["id"]: c["name"] for c in rep["cars"]}
    order = standings_at(rep, t)  # [(car id, progress in laps, finished?, finish time), ...]
    return [{"P": k, "driver": names[cid], "lap": min(rep["laps"], int(prog) + 1),
             "gap": f"FIN {ftime:.1f}" if done else f"+{order[0][1] - prog:.2f}"}  # in laps
            for k, (cid, prog, done, ftime) in enumerate(order, start=1)]


def race_replay() -> None:
    st.title("Race replay")
    path = str(st.selectbox("Replay file", _data.replay_files(), format_func=_data.rel))
    rep = _data.load_json(path, Path(path).stat().st_mtime)  # 100+ KB of JSON, parsed once
    end, names = max(c["t"][-1] for c in rep["cars"]), {c["id"]: c["name"] for c in rep["cars"]}
    mine, pro = st.tabs(["My viewer", "Pro player"])
    with pro:
        st.caption("Space = play/pause, click a car to follow it, 1-9 = follow P1-P9.")
        replay_canvas(path, key="s10_pro", height=560)
    c1, c2, c3 = mine.columns([1, 2, 2])
    playing = c1.toggle("Play", key="s10_play")
    speed = c2.select_slider("Speed", [1, 2, 4, 8], value=4, key="s10_speed")
    follow = c3.selectbox("Follow", [None, *names], format_func=lambda c: names.get(c, "all cars"))
    st.session_state.s10_t = min(st.session_state.get("s10_t", 0.0), end)
    @st.fragment(run_every=0.5 if playing else None)  # playing: advance the clock twice a second
    def viewer() -> None:
        if playing:
            st.session_state.s10_t = min(end, st.session_state.s10_t + 0.5 * speed)
        t = st.slider("Race clock (s)", 0.0, end, step=0.5, key="s10_t")
        left, right = st.columns([3, 1])
        left.plotly_chart(race_map(path, rep, t, follow), theme=None, key="s10_map")
        right.dataframe(timing_tower(rep, t), hide_index=True)
        right.caption("gap = laps behind the leader, FIN = finish time (s)")
        radio = sorted((m["t"], c["name"], m["text"])
                       for c in rep["cars"] for m in c.get("messages", []))
        for when, who, text in [r for r in radio if r[0] <= t][-5:]:  # the last 5 until t
            right.caption(f"**{when:.0f} s {who}:** {text}")
    with mine:
        viewer()


def my_learning_curve() -> None:  # SOLUTION: the page you reuse in Chapter 3
    st.title("My learning curve")
    st.caption("Make one: `python race.py drive drivers/submissions/<you>.py --track sprint "
               "--attempts 6 --save`, then refresh. No results/ yet? You see the demo learner.")
    files = _data.result_files()
    picked = st.multiselect("Result files", files, default=files[:1], format_func=_data.rel)
    results = [_data.load_json(str(p), p.stat().st_mtime) for p in picked]
    if not results:
        st.info("Pick at least one result file.")
        return
    st.plotly_chart(pf.learning_curve_figure(results, height=460), theme=None)
    for res in results:
        times = [a.get("race_time") if a.get("completed") else None for a in res["attempts"]]
        done = [t for t in times if t is not None]
        gain = (times[0] - min(done)) / times[0] * 100 if times and times[0] and done else 0.0
        c1, c2, c3 = st.columns(3)
        best = f"best {min(done):.2f} s" if done else "DNF"
        c1.metric(f"{res['name']} on {res['track']}", best)
        c2.metric("Gain vs attempt 1", f"{gain:.1f} %", f"{gain:+.1f} %")
        c3.metric("DNFs", times.count(None))
        if gain >= 3 and None not in times:
            st.success("Checkpoint CP3 reached: at least 3 % faster than attempt 1, no DNF.")
        order = ["attempt", "phase", "race_time", "best_lap", "crashes", "completed", "status"]
        st.dataframe(pd.DataFrame(res["attempts"]), hide_index=True, column_order=order)


lessons = {f: f.stem[:3].upper() + " " + f.stem[4:].replace("_", " ").capitalize()  # S01 Hello..
           for f in sorted(Path(__file__).resolve().parent.glob("s0*.py"))}  # S01-S09, next to me
pages = {"My app": [st.Page(race_replay, title="Race replay", icon=":material/sports_score:"),
                    st.Page(my_learning_curve, title="My learning curve",  # SOLUTION
                            icon=":material/trending_down:")],
         "Lessons": [st.Page(f, title=title) for f, title in lessons.items()]}
st.navigation(pages).run()
