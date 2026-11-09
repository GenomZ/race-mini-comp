"""Telemetry: speed, throttle and steering of every attempt - where the time goes, where the crashes happen."""

import json

import pandas as pd
import streamlit as st

from agentic_gp.viz import plotly_figs as pf
from app.lib.loaders import file_kind, list_json, load_json
from app.lib.paths import RACES_DIR, RESULTS_DIR, rel
from app.lib.ui import empty_state, page_header

page_header("Telemetry", "A lap is a table: one row per tick. Plot speed against the race clock and the corners "
            "appear as valleys; compare two attempts and the gap shows you where to brake later (or earlier).",
            chapter="Ch2-Ch3")


def sessions() -> list:
    """Session files (with per-attempt telemetry) under races/ and results/, newest first."""
    out = list_json(RACES_DIR, "session")
    if RESULTS_DIR.exists():
        out += list_json(RESULTS_DIR, "session")
    return out


def session_label(p) -> str:
    try:
        d = load_json(p)
        n = len(d.get("reports", []))
        return f"{rel(p)}  -  {d.get('name', '?')} on {d.get('track', '?')}, {n} attempt(s)"
    except (OSError, ValueError):
        return rel(p)


# ------------------------------------------------------------------------------ choose a session
files = sessions()
src = st.segmented_control("Source", ["Saved sessions", "Upload"], default="Saved sessions", key="tel_src",
                           label_visibility="collapsed")
session = None
if src == "Upload":
    up = st.file_uploader("Session JSON (Live page -> Save, or its 'Session JSON' download)", type=["json"], key="tel_up")
    if up is not None:
        try:
            session = json.loads(up.getvalue().decode("utf-8"))
            if file_kind(session) != "session":
                st.error("This file has no per-attempt telemetry (it is not a session file). Note: "
                         "results/<name>.json from race.py --save is a RESULT file (lap times only, for Learning "
                         "Curves) - session files come from the Live page.")
                session = None
        except ValueError as exc:
            st.error(f"Not JSON: {exc}")
elif files:
    # Default: the learner, whose attempts DIFFER (the planner's 3 attempts are identical: nothing to compare).
    default = next((i for i, p in enumerate(files) if p.name == "learner_session.json"),
                   next((i for i, p in enumerate(files) if p.name == "l2_session.json"), 0))
    pick = st.selectbox("Session", files, index=default, format_func=session_label, key="tel_file")
    session = load_json(pick)
if session is None:
    if src != "Upload":
        empty_state("No session files yet. A session file keeps every tick of every attempt: run a driver on the "
                    "Live page (2+ attempts), then press 'Save to races/live/' - it shows up here. "
                    "(race.py drive --save writes a RESULT file with lap times only: that one is for Learning "
                    "Curves.) Or rebuild the demo sessions:",
                    "python races/demo/make_demo.py --only l2_session learner_session")
    st.stop()

reports = session.get("reports", [])
if not reports:
    st.warning("This session has no attempts.")
    st.stop()


def att_label(r: dict) -> str:
    t = f"{r['race_time']:.2f} s" if r.get("completed") and r.get("race_time") is not None else (r.get("status") or "DNF")
    return f"Attempt {r['attempt']}: {t}, {r.get('crashes', 0)} crash(es)"


finished = [r for r in reports if r.get("completed") and r.get("race_time") is not None]
best = min(finished, key=lambda r: r["race_time"]) if finished else reports[0]

# ------------------------------------------------------------------------------ summary metrics
st.markdown(f"#### {session.get('name', '?')} :gray[on] {session.get('track_name') or session.get('track', '?')}")
m = st.columns(3)
m[0].metric("Best", f"{best['race_time']:.2f} s" if finished else "DNF", f"attempt {best['attempt']}", delta_color="off", delta_arrow="off")
m[1].metric("Crashes (all attempts)", sum(r.get("crashes", 0) for r in reports))
m[2].metric("Attempts finished", f"{len(finished)}/{len(reports)}")

c1, c2, c3 = st.columns([2, 2, 1])
labels = [att_label(r) for r in reports]
a_pick = c1.selectbox("Attempt", labels, index=len(labels) - 1, key="tel_att")
cmp_opts = ["(none)"] + labels
# Default comparison: your best attempt - or, when the last attempt IS the best, the first one (see the learning).
cmp_default = reports.index(best) if best is not reports[-1] or len(reports) == 1 else 0
cmp_pick = c2.selectbox("Compare with", cmp_opts, index=1 + cmp_default, key="tel_cmp",
                        help="Drawn dashed. Default: your best attempt (or attempt 1 when the last one is the best).")
xaxis = c3.radio("x axis", ["t", "progress"], horizontal=True, key="tel_x",
                 format_func=lambda v: {"t": "race clock", "progress": "lap distance"}[v])
rep = reports[labels.index(a_pick)]
cmp = None if cmp_pick == "(none)" else reports[cmp_opts.index(cmp_pick) - 1]
if cmp is rep:
    cmp = None

# ------------------------------------------------------------------------------ the plots
tab_plot, tab_map, tab_seg, tab_crash, tab_text = st.tabs(
    [":material/show_chart: Speed and controls", ":material/map: Where on the track", ":material/table: Segments",
     f":material/car_crash: Crashes ({rep.get('crashes', 0)})", ":material/notes: What reflect() read"])
with tab_plot:
    st.plotly_chart(pf.telemetry_figure(rep, x=xaxis, compare=cmp,
                                        names=(f"attempt {rep['attempt']}", f"attempt {cmp['attempt']}" if cmp else "")),
                    theme=None, key="tel_fig")
    st.caption("Valleys in the speed line are corners. Throttle below zero = braking. Steering + = right.")
with tab_map:
    fig = pf.track_figure(session.get("track", "grand_prix"), vmax=False, height=600)
    pf.add_trajectory(fig, rep, name=f"attempt {rep['attempt']}")
    pts = [c["position"] for c in rep.get("crash_log", []) if c.get("position")]
    if pts:
        pf.add_crashes(fig, pts, labels=[f"{c.get('corner') or 'seg ' + str(c.get('segment'))}: {c.get('speed_before')} u/s"
                                         for c in rep.get("crash_log", []) if c.get("position")])
    st.plotly_chart(fig, theme=None, key="tel_map")
    st.caption("The line is coloured by speed (bright = fast). Red X = crash (hover for the corner and speed).")
with tab_seg:
    segs = pd.DataFrame(rep.get("segments", []))
    if segs.empty:
        st.caption("No segment table in this file.")
    else:
        if cmp is not None and cmp.get("segments"):
            ref = pd.DataFrame(cmp["segments"]).set_index("segment")["time"]
            segs["vs compare (s)"] = (segs["time"] - segs["segment"].map(ref)).round(2)
        st.dataframe(segs, hide_index=True, width="stretch",
                     column_config={"vs compare (s)": st.column_config.NumberColumn(format="%+.2f"),
                                    "time": st.column_config.NumberColumn(format="%.1f s")})
        st.caption("One row per gate-to-gate segment. Positive 'vs compare' = slower than the compared attempt there.")
    if rep.get("corners"):
        st.markdown("**Corners**")
        st.dataframe(rep["corners"], hide_index=True, width="stretch")
with tab_crash:
    log = rep.get("crash_log", [])
    if not log:
        st.success("No crashes in this attempt.", icon=":material/check:")
    else:
        st.dataframe([{"t": c["t"], "corner": c.get("corner"), "segment": c["segment"], "speed before": c["speed_before"],
                       "throttle": c["throttle"], "steering": c["steering"],
                       "front ray": (c.get("rays") or {}).get("distance_to_wall_front")} for c in log],
                     hide_index=True, width="stretch")
        st.caption("Speed before the crash vs the corner's v_max (Track Lab) usually tells the whole story.")
with tab_text:
    st.code(rep.get("summary_text") or "(no summary)", language=None, wrap_lines=True)

tele = pd.DataFrame(rep.get("telemetry", []))
st.download_button("Telemetry of this attempt (CSV)", tele.to_csv(index=False), f"attempt_{rep['attempt']}.csv",
                   "text/csv", icon=":material/download:", on_click="ignore", key="tel_csv")
