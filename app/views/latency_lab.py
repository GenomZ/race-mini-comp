"""Latency Lab: why an LLM cannot drive the car tick by tick - measure it, then see the price on the track."""

import math
import os
import threading

import streamlit as st

from agentic_gp.viz import plotly_figs as pf
from app.lib.experiments import latency_cliff, simulated_latencies, timed_llm_calls
from app.lib.figs import blank, latency_cliff_figure
from app.lib.llm_trace import find_llm_logs, latency_stats, read_llm_log
from app.lib.loaders import load_json, track_choices, track_map
from app.lib.paths import DEMO_DIR, DRIVERS_DIR
from app.lib.ui import fast_mode, llm_status, page_header

page_header("Latency Lab", "The sim asks for a decision every 0.5 s. A chat model needs 1-3 s per answer, so in "
            "real time the car drives blind for several ticks - at 30 u/s that is 90 units, five track widths. "
            "Measure it yourself, then decide which clock the LLM belongs on.", chapter="Ch4")

BUDGET = int(os.environ.get("AGP_LATENCY_BUDGET", "30"))       # successful real calls per app server (CP4: <= 30)
TICK = 0.5
PROMPT = ("You drive a race car. Sensors: speed 28, wall ahead 35 units, next waypoint 24 degrees to the RIGHT "
          "(+ = right). Answer ONLY JSON: {\"throttle\": -1..1, \"steering\": -1..1}")

st.session_state.setdefault("lat_rows", [])


@st.cache_resource
def call_budget() -> dict:
    """ONE counter per app server process (shared by every browser; a page reload does not reset it)."""
    return {"used": 0, "lock": threading.Lock()}


budget = call_budget()

# ------------------------------------------------------------------------------ 1. measure
st.subheader("1. Measure: how long does one answer take?", anchor=False)
status = llm_status()
# Real calls are made from THIS app process, so they use the server's own provider: offered only when the
# app was started with a real provider and a key (the sidebar says which). Never switched per browser.
can_real = bool(status["real"] and status["available"])
with st.container(border=True):
    c1, c2, c3 = st.columns([3, 2, 2])
    modes = (["Real LLM"] if can_real else []) + ["Simulated (free, offline)", "Recorded LLM logs (free)"]
    mode = c1.radio("Where do the numbers come from?", modes, key="lat_mode",
                    help="Real: actual calls with the shared key (budget below). Simulated: realistic made-up "
                         "times for practice. Recorded: every call in races/**/llm_logs.")
    n = c2.slider("Calls", 1, 10, 3 if fast_mode() else 10, key="lat_n", disabled=mode != "Real LLM" and mode.startswith("Rec"))
    effort = c3.selectbox("Thinking effort", ["none", "low", "medium"], key="lat_effort",
                          disabled=mode != "Real LLM",
                          help="MiniMax-M3 thinks by default. 'none' turns thinking off: much faster.")
    if mode == "Real LLM":
        prompt = st.text_area("Prompt (the obs as text)", PROMPT, key="lat_prompt", height=80)
        left = BUDGET - budget["used"]
        budget_box = st.empty()                        # filled below, after this run's calls

        def show_budget() -> None:
            budget_box.progress(min(1.0, budget["used"] / BUDGET),
                                text=f"Budget: {budget['used']}/{BUDGET} successful real calls used on this app "
                                     f"server ({status['label']}); failed calls do not count. The class shares one "
                                     f"key - waves by row.")

        show_budget()
        run = st.button(f"Run {min(n, max(left, 0))} real call(s)", type="primary", key="lat_run",
                        icon=":material/bolt:", disabled=left <= 0)
        if run and left > 0:
            bar = st.progress(0.0, text="calling...")
            k = min(n, left)
            rows = timed_llm_calls(k, prompt, effort=effort,
                                   on_call=lambda i, r: bar.progress((i + 1) / k, text=f"call {i + 1}/{k}: {r['seconds']:.2f} s"))
            with budget["lock"]:
                budget["used"] += sum(1 for r in rows if not r["error"])
            show_budget()
            st.session_state.lat_rows += [dict(r, source=f"real ({effort})") for r in rows]
            errs = [r["error"] for r in rows if r["error"]]
            if errs:
                st.warning(f"{len(errs)} call(s) failed: {errs[0]}", icon=":material/warning:")
    elif mode.startswith("Simulated"):
        st.caption("No key needed: times drawn from a log-normal with a 1.3 s median, like a chat model "
                   "with thinking off. Labelled 'simulated' everywhere.")
        if st.button(f"Simulate {n} call(s)", type="primary", key="lat_sim", icon=":material/casino:"):
            st.session_state.lat_rows += [{"call": i + 1, "seconds": s, "answer": "(simulated)", "error": None,
                                           "source": "simulated"} for i, s in enumerate(simulated_latencies(n))]
    else:
        logs = find_llm_logs()
        secs = [r.get("seconds") for p in logs for r in read_llm_log(p) if r.get("seconds") and not r.get("cached")]
        st.caption(f"{len(logs)} log file(s), {len(secs)} timed call(s) - mock calls show up as ~0 s.")
        if st.button("Load recorded calls", key="lat_load", icon=":material/history:", disabled=not secs):
            st.session_state.lat_rows += [{"call": i + 1, "seconds": s, "answer": "(from log)", "error": None,
                                           "source": "recorded"} for i, s in enumerate(secs)]

rows = st.session_state.lat_rows
secs = [r["seconds"] for r in rows if not r.get("error")]
stats = latency_stats(secs)
if secs:                                               # same percentile method as the histogram's dotted lines
    import numpy as np
    stats["p50"], stats["p95"] = (float(v) for v in np.percentile(secs, [50, 95]))
left, right = st.columns([3, 2], gap="medium")
with left:
    fig = pf.latency_histogram(secs, tick=TICK) if secs else blank("No calls yet: the histogram of call times appears here", 360)
    st.plotly_chart(fig, theme=None, key="lat_hist")
with right:
    if stats["n"]:
        m = st.columns(3)
        m[0].metric("calls", stats["n"])
        m[1].metric("p50", f"{stats['p50']:.2f} s")
        m[2].metric("p95", f"{stats['p95']:.2f} s")
        blind = stats["p50"] / TICK
        st.metric("Ticks driven blind per call (p50 / 0.5 s)", f"{blind:.1f}",
                  help="In realtime mode the car keeps its last command for this many ticks.")
        st.caption(f"At 30 u/s the car covers **{30 * stats['p50']:.0f} units** before the answer arrives. "
                   "Sources: " + ", ".join(sorted({r['source'] for r in rows})))
        if st.button("Clear", key="lat_clear", icon=":material/delete:"):
            st.session_state.lat_rows = []
            st.rerun()
    else:
        st.info("No calls yet: run, simulate or load some above.", icon=":material/info:")
if rows:
    with st.expander(f"Every call ({len(rows)})", icon=":material/list:"):
        st.dataframe(rows, hide_index=True, width="stretch",
                     column_config={"seconds": st.column_config.NumberColumn(format="%.3f s")})

# ------------------------------------------------------------------------------ 2. blind distance
st.subheader("2. The price: distance driven blind", anchor=False)
c1, c2, c3, c4 = st.columns(4)
bt = c1.selectbox("Track", track_choices(0), key="lat_track")
lat = c2.slider("Latency (s)", 0.0, 5.0, float(round(stats["p50"], 1)) if stats["n"] else 3.0, 0.1, key="lat_lat")
spd = c3.slider("Speed (u/s)", 5.0, 40.0, 30.0, 1.0, key="lat_spd")
tm = track_map(bt)
corners = tm["corners"]
spots = [c["id"] for c in corners] or ["start"]
spot = c4.selectbox("Waiting for the answer before", spots, index=min(1, len(spots) - 1), key="lat_spot")
idx = max(0, corners[spots.index(spot)]["start_index"] - 10) if corners else 0
cl = tm["centerline"]
(x0, y0), (x1, y1) = cl[idx], cl[(idx + 1) % len(cl)]
heading = math.degrees(math.atan2(y1 - y0, x1 - x0))
d = spd * lat
left, right = st.columns([3, 1], gap="medium")
with left:
    fig = pf.blind_distance_figure(bt, x0, y0, heading, lat, speed=spd, height=480)
    span = max(80.0, d * 1.3)
    fig.update_xaxes(range=[x0 - span * 1.5, x0 + span * 1.5])
    fig.update_yaxes(range=[y0 - span, y0 + span])
    st.plotly_chart(fig, theme=None, key="lat_blind")
with right:
    st.metric("Blind distance", f"{d:.0f} units")
    st.metric("= track widths", f"{d / (2 * tm['half_width']):.1f}")
    st.metric("Ticks without a decision", f"{math.floor(lat / TICK + 1e-9)}")
    st.caption("The red line is where the car goes on its OLD command while the model is still thinking. "
               "The track turns; the car does not.")

# ------------------------------------------------------------------------------ 3. the cliff
st.subheader("3. The latency cliff", anchor=False)
st.caption("The same L2 planner, the same track - only its answers arrive late. Below one tick (0.5 s) nothing "
           "changes; above it, the lap falls apart.")
cliff_file = DEMO_DIR / "latency_cliff.json"


CLIFF_DRIVER = DRIVERS_DIR / "examples" / "l2_planner.py"


@st.cache_data(show_spinner="Driving 5 laps with late answers...", max_entries=4)
def _cliff(track: str, quick: bool) -> list[dict]:
    lats = (0.0, 0.5, 1.0) if quick else (0.0, 0.5, 1.0, 2.0, 3.0)
    return latency_cliff(str(CLIFF_DRIVER), track, lats, max_ticks=120 if quick else 500)


cliff = load_json(cliff_file).get("rows") if cliff_file.exists() else st.session_state.get("lat_cliff")
cc1, cc2 = st.columns([1, 3], vertical_alignment="bottom")
ctrack = cc1.selectbox("Track", track_choices(0), key="lat_cliff_track")
if cc2.button("Measure the cliff now (about 2 s, no LLM)", key="lat_cliff_go", icon=":material/science:",
              disabled=not CLIFF_DRIVER.exists(),
              help=None if CLIFF_DRIVER.exists() else "Needs the reference driver drivers/examples/l2_planner.py"):
    try:
        cliff = st.session_state.lat_cliff = _cliff(ctrack, fast_mode())
    except Exception as exc:                            # noqa: BLE001 - a missing / broken example must not crash
        st.error(f"Could not measure the cliff with {CLIFF_DRIVER.name}: {type(exc).__name__}: {exc}",
                 icon=":material/error:")
if not CLIFF_DRIVER.exists() and not cliff:
    st.warning("The cliff is measured with the reference driver drivers/examples/l2_planner.py, which is missing "
               "here. Restore it (git checkout drivers/examples/l2_planner.py) or ship races/demo/latency_cliff.json.",
               icon=":material/warning:")
if cliff:
    left, right = st.columns([3, 2], gap="medium")
    left.plotly_chart(latency_cliff_figure(cliff), theme=None, key="lat_cliff_fig")
    right.dataframe([{"latency": f"{r['latency_s']:g} s", "blind ticks": r["blind_ticks"],
                      "result": f"{r['race_time']:.1f} s" if r["finished"] else f"DNF ({r['gates']}/{r['gates_total']} gates)",
                      "crashes": r["crashes"]} for r in cliff], hide_index=True, width="stretch")
    right.info("Where does the LLM belong? Not on the TICK clock. On the SESSION clock (plan once, Ch5) "
               "and the LAP clock (reflect between attempts, Ch6), it has seconds to think.", icon=":material/schedule:")
else:
    st.info("Press the button to measure it (or ship races/demo/latency_cliff.json).", icon=":material/info:")
