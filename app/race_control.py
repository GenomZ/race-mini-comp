"""Race Control - the Agentic Grand Prix app (every page of the 2-day course in one place).

    streamlit run app/race_control.py            (from the repo root, so .streamlit/config.toml applies)

This file is only the "table of contents": st.navigation lists the pages (app/views/*.py), groups
them by when you use them in the course, and runs the one you clicked. Each page is a normal
Streamlit script you can read top to bottom.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # app/race_control.py -> repo root
if str(ROOT) not in sys.path:                        # `import agentic_gp` / `import app...` (not pip-installed)
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app.lib.checkpoints import render_where_am_i  # noqa: E402
from app.lib.ui import sidebar_status  # noqa: E402

st.set_page_config(page_title="Race Control", page_icon=":material/sports_motorsports:", layout="wide")


def page(file: str, title: str, icon: str, **kw) -> st.Page:
    return st.Page(f"views/{file}", title=title, icon=f":material/{icon}:", **kw)


PAGES = {
    "Race Control": [
        page("home.py", "Home", "home", default=True),
        page("live.py", "Live", "videocam"),
        page("replay.py", "Replay", "replay"),
        page("grand_prix.py", "Grand Prix", "emoji_events"),
    ],
    "Day 1 labs: reflexes and maps": [
        page("sensor_xray.py", "Sensor X-ray", "radar"),
        page("physics_lab.py", "Physics Lab", "speed"),
        page("track_lab.py", "Track Lab", "route"),
        page("telemetry.py", "Telemetry", "monitoring"),
        page("learning_curves.py", "Learning Curves", "trending_down"),
    ],
    "Day 2 labs: brains that think": [
        page("latency_lab.py", "Latency Lab", "timer"),
        page("pit_wall.py", "Pit Wall (LLM trace)", "forum"),
        page("fly_lab.py", "Fly Lab", "bug_report"),
    ],
    "Instructor": [
        page("inbox.py", "Inbox", "inbox"),
    ],
}

# Presenter mode for a 1280x720 projector: compact header, canvases sized to the window, setup folded.
PRESENTER_CSS = """<style>
[data-testid="stMainBlockContainer"] { padding-top: 1.1rem !important; padding-bottom: 1rem !important; }
[data-testid="stMainBlockContainer"] h1 { font-size: 1.55rem !important; padding: 0 0 0.15rem 0 !important; }
[data-testid="stMainBlockContainer"] [data-testid="stHeadingWithActionElements"] + div p { font-size: 0.82rem; }
[data-testid="stHeader"] { background: transparent !important; }
</style>"""

nav = st.navigation(PAGES, expanded=True)
if "presenter" not in st.session_state and str(st.query_params.get("presenter", "")).lower() in ("1", "true", "on"):
    st.session_state.presenter = True                # ?presenter=1 in the URL turns it on
st.sidebar.toggle("Presenter mode (projector)", key="presenter", help="Compact header, the race canvas sized to the "
                  "window and the setup panels folded away - for a 1280x720 projector. Or open the app with ?presenter=1.")
if st.session_state.get("presenter"):
    st.html(PRESENTER_CSS)
if "show_mystery" not in st.session_state and str(st.query_params.get("mystery", "")).lower() in ("1", "true", "on"):
    st.session_state.show_mystery = True             # ?mystery=1 in the URL reveals the Mystery GP files
st.sidebar.toggle("Show Mystery GP files", key="show_mystery",
                  help="Off (default): replays, results and LLM logs of the Mystery GP are hidden on every page - "
                       "their prompts describe the secret track. Turn on at the reveal (or open the app with ?mystery=1).")
render_where_am_i()                                  # "Where am I?" in the sidebar of every page
sidebar_status()
nav.run()
