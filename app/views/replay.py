"""Replay: watch any race again - multi-car player, timing tower, standings at the paused moment."""

from pathlib import Path

import streamlit as st

from agentic_gp.viz.player import replay_html
from app.components import replay_canvas
from app.lib.loaders import list_replays, load_replay, replay_from_upload, replay_label, standings_table
from app.lib.paths import rel
from app.lib.ui import empty_state, hide_mystery, mark_dead_heats, mystery_note, page_header, presenter_mode, repo_path

page_header("Replay", "A replay is just data: every car's position twice a second. The browser animates it. "
            "Pause on a crash, follow one car, and read the timing tower like a race engineer.", chapter="Ch0")

NOTES = {   # what to look for in the demo files (shown under the player)
    "crash_loop.json": "Ch1 crash loop (classic crash rule, no marshals): the dummy drives straight into the first "
                       "corner wall and stays there. Every red X is a crash, every crash costs 5 s. Why does it never steer?",
    "class_demo.json": "The class demo: every example driver on the home track, 2 laps. Press 1-9 to follow a "
                       "position, click a car to follow it, Space to pause.",
    "l2_session_replay.json": "One driver, three attempts drawn as ghost cars: the planner on the home track.",
    "learner_session_replay.json": "The learner's 5 attempts on sprint as ghosts: is attempt 5 ahead of attempt 1?",
}

# ------------------------------------------------------------------------------ pick a replay
replays, n_hidden = hide_mystery(list_replays())
qp_file = st.query_params.get("file")
presenter = presenter_mode()
# Presenter mode (projector): the pickers fold into one line, so the player fills the screen.
options_box = (st.expander("Replay file and playback options", icon=":material/tune:") if presenter
               else st.container())
with options_box:
    source = st.segmented_control("Source", ["races/ folder", "Upload a file"], default="races/ folder",
                                  key="replay_source", label_visibility="collapsed")
    replay_obj, label, path = None, None, None
    if source == "Upload a file":
        up = st.file_uploader("Replay JSON (from race.py drive --replay, grand-prix --out, ...)", type=["json"],
                              key="replay_upload")
        if up is not None:
            try:
                replay_obj = replay_from_upload(up.getvalue())
                label = up.name
            except ValueError as exc:
                st.error(f"That is not a replay file: {exc}", icon=":material/error:")
            except Exception as exc:                   # noqa: BLE001 - a broken upload must not break the page
                st.error(f"That is not a replay file: {type(exc).__name__}: {exc}", icon=":material/error:")
    elif replays:
        default = 0
        wanted = repo_path(qp_file).resolve() if qp_file else None
        found = wanted is not None and any(p.resolve() == wanted for p in replays)
        if wanted is not None and not found:
            st.warning(f"Replay {qp_file} was not found (or is not a replay file) - showing your newest replay instead.",
                       icon=":material/warning:")
            wanted = None
        if wanted is not None:
            default = next(i for i, p in enumerate(replays) if p.resolve() == wanted)
        else:   # your newest replay (the list is newest first); the class demo when you have none yet
            demo = repo_path("races/demo").resolve()
            default = next((i for i, p in enumerate(replays) if demo not in p.resolve().parents),
                           next((i for i, p in enumerate(replays) if p.name == "class_demo.json"), 0))
        path = st.selectbox("Replay file", replays, index=default, format_func=replay_label, key="replay_file",
                            help="Newest first: your latest replay is picked for you. The demo races are in demo/.")
        mystery_note(n_hidden)
        replay_obj, label = load_replay(path), rel(path)

    if replay_obj is None:
        if source != "Upload a file":
            empty_state("No replay files in races/ yet. Make one (or generate the demo data):",
                        "python race.py drive drivers/examples/l1_reflex.py --replay races/me_l1.json\n"
                        "python races/demo/make_demo.py")
        else:
            st.caption("Drop a replay .json file above.")
        st.stop()

    cars = replay_obj.cars
    # ------------------------------------------------------------------------------ playback options
    c1, c2, c3 = st.columns([1, 2, 2], vertical_alignment="bottom")
    speed = c1.selectbox("Start speed", [0.5, 1, 2, 4, 8], index=2, key="replay_speed",
                         format_func=lambda v: f"{v:g}x")
    follow_names = ["(whole track)"] + [f"P{c.position or '-'} {c.name}" for c in cars]
    follow_pick = c2.selectbox("Follow", follow_names, key="replay_follow")
    vmax = c3.toggle("Colour corners by speed limit", key="replay_vmax",
                     help="Colour the centerline by corner speed v_max = sqrt(20 R): bright orange = slow corner.")
    follow = None if follow_pick == follow_names[0] else cars[follow_names.index(follow_pick) - 1].id

# ------------------------------------------------------------------------------ the player
res = replay_canvas(replay_obj if path is None else path, key="replay_canvas", height=600, speed=float(speed),
                    follow=follow, report=True, vmax=vmax, autoplay=False, fit=presenter)
note = NOTES.get(Path(str(label)).name)
if note:
    st.info(note, icon=":material/lightbulb:")
st.caption("Keys (click the player first): Space play/pause, Left/Right -5/+5 s, +/- speed, 1-9 follow position, 0 overview.")

# ------------------------------------------------------------------------------ standings + events
# report=True: when the viewer pauses, the browser sends the race time t back (one rerun).
t = getattr(res, "t", None) if res is not None else None
if t is None:
    st.subheader("Official classification", anchor=False)
    st.caption("Pause the player to see the order at that exact moment.")
    rows = mark_dead_heats(standings_table(replay_obj))
    st.dataframe(rows, hide_index=True, width="stretch",
                 column_config={"time": st.column_config.NumberColumn(format="%.2f s"),
                                "gap": st.column_config.NumberColumn(format="+%.2f s"),
                                "best lap": st.column_config.NumberColumn(format="%.2f s")})
else:
    st.subheader(f"Order at t = {t:.2f} s", anchor=False)
    rows = standings_table(replay_obj, t)
    st.dataframe(rows, hide_index=True, width="stretch",
                 column_config={"race progress": st.column_config.ProgressColumn(
                     min_value=0.0, max_value=float(replay_obj.laps), format="%.2f laps"),
                     "finish time": st.column_config.NumberColumn(format="%.2f s")})
st.subheader("Race control log", anchor=False)
events = [{"t": round(e.t, 2), "car": c.name, "event": e.kind,
           "detail": e.text or (f"{e.value:g} s" if e.kind in ("crash", "lap", "finish") else f"{e.value:g}")}
          for c in cars for e in c.events]
events += [{"t": round(m.t, 2), "car": c.name, "event": "radio", "detail": m.text}
           for c in cars for m in c.messages]
events.sort(key=lambda e: e["t"])
if t is not None:
    events = [e for e in events if e["t"] <= t]
if events:
    st.dataframe(events[-200:], hide_index=True, width="stretch", height=280,
                 column_config={"t": st.column_config.NumberColumn(format="%.1f s"),
                                "detail": st.column_config.TextColumn(width="large")})
else:
    st.caption("No events (crashes, laps, finishes) in this replay yet.")

with st.expander("Share it: a standalone HTML player (works offline, no Python needed)", icon=":material/share:"):
    st.download_button("Download the HTML player", data=lambda: replay_html(replay_obj, height=None),
                       file_name=Path(str(label)).with_suffix(".html").name, mime="text/html",
                       icon=":material/download:", on_click="ignore", key="replay_dl_html")
    st.code(f"python race.py watch {label if path else 'my_replay.json'} --html     # same, from the terminal\n"
            f"python race.py watch {label if path else 'my_replay.json'} --unity    # the 3D viewer", language="bash")
