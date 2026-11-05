"""Learning Curves: does practice pay? Lap time per attempt, where the time was won, and learning gone wrong."""

import streamlit as st

from agentic_gp.viz import plotly_figs as pf
from app.lib.loaders import attempts_table, file_kind, list_json, load_json
from app.lib.paths import RACES_DIR, REPO_ROOT, RESULTS_DIR, rel
from app.lib.ui import empty_state, page_header

page_header("Learning Curves", "A learning agent has a critic (the lap report), a learning element (reflect) and a "
            "problem generator (one change per attempt). The curve tells you if it works; keep-best stops one bad "
            "idea from ruining everything.", chapter="Ch3, Ch6")

FIGURES_DATA = REPO_ROOT / "lesson" / "assets" / "figures_data.json"   # written by lesson/make_figures.py


def gone_wrong() -> tuple[dict, str]:
    """The Ch3 'learning gone wrong' run (lesson/assets/figures_data.json, the same numbers as Ch3 and its
    figure): a greedy learner WITHOUT keep-best vs the same learner WITH it. Returns (curves, caption)."""
    try:
        data = load_json(FIGURES_DATA)["learning_gone_wrong"]
        runs = {"no keep-best": data["no_keep_best"], "keep-best": data["keep_best"]}
    except Exception:                                    # noqa: BLE001 - the figure data is optional
        return {}, ""
    curves = {f"{label} (measured)": [a.get("time") if a.get("completed") else None for a in rows]
              for label, rows in runs.items()}
    worst = max(runs["no keep-best"], key=lambda a: a.get("time") or 0)
    caption = (f"Red flag pattern without keep-best: one greedy change too far and the lap gets WORSE - up to "
               f"{worst['time']:.1f} s with {worst['crashes']} crashes - ending on "
               f"{runs['no keep-best'][-1]['time']:.1f} s, while keep-best ends on "
               f"{runs['keep-best'][-1]['time']:.1f} s (sprint, measured by lesson/make_figures.py, as in Ch3).")
    return curves, caption


def result_files() -> list:
    """Session and result files (one row per attempt) under races/ and results/, newest first."""
    out = []
    for root in (RACES_DIR, RESULTS_DIR):
        if root.exists():
            out += list_json(root, "session") + list_json(root, "result")
    return sorted(out, key=lambda p: p.stat().st_mtime, reverse=True)


def label(p) -> str:
    d = load_json(p)
    return f"{d.get('name', '?')} on {d.get('track', '?')}  ({len(d.get('attempts', []))} att.)  -  {rel(p)}"


files = result_files()
tab_curve, tab_heat, tab_wrong = st.tabs([":material/trending_down: Curves", ":material/grid_on: Segment heatmap",
                                          ":material/warning: Learning gone wrong"])

with tab_curve:
    if not files:
        empty_state("No results yet. Save a multi-attempt session:",
                    "python race.py drive drivers/examples/l3_learner.py --track sprint --attempts 6 --save")
    else:
        defaults = [p for p in files if p.name == "learner_session.json"][:1] or files[:1]
        picks = st.multiselect("Sessions / results to compare", files, default=defaults, format_func=label, key="lc_files")
        if picks:
            data = [load_json(p) for p in picks]
            st.plotly_chart(pf.learning_curve_figure(data, height=440), theme=None, key="lc_fig")
            st.caption("Solid = race time of each attempt (X at the top = DNF). Dotted = best so far: with keep-best, "
                       "that is what you would race with.")
            if len({d.get("track") for d in data}) > 1:
                st.warning("These files are from different tracks: compare the SHAPE of the curves, not the times.",
                           icon=":material/warning:")
            m = st.columns(len(data))
            for col, d in zip(m, data):
                times = [a["race_time"] for a in d.get("attempts", []) if a.get("completed") and a.get("race_time")]
                if len(times) >= 2:
                    gain = (times[0] - min(times)) / times[0] * 100
                    col.metric(f"{d.get('name', '?')} on {d.get('track', '?')}", f"{min(times):.2f} s",
                               f"{-gain:.1f} % vs attempt 1" if gain >= 0.05 else "no gain vs attempt 1",
                               delta_color="inverse", help="CP3 asks for at least 3 % better than attempt 1, no DNF.")
            rows = []
            for d in data:
                rows += attempts_table(d)
            st.dataframe(rows, hide_index=True, width="stretch",
                         column_config={"race_time": st.column_config.NumberColumn("race time", format="%.2f s"),
                                        "best_lap": st.column_config.NumberColumn("best lap", format="%.2f s")})
        else:
            st.info("Pick at least one file.")

with tab_heat:
    sessions = [p for p in files if file_kind(load_json(p)) == "session"]
    if not sessions:
        st.info("The heatmap needs a SESSION file (per-segment times of every attempt): run a driver on the Live "
                "page with 2+ attempts and press 'Save to races/live/'. (race.py drive --save writes a result file "
                "with lap times only - it feeds the Curves tab, not this heatmap.)")
    else:
        default = next((i for i, p in enumerate(sessions) if p.name == "learner_session.json"), 0)
        sp = st.selectbox("Session", sessions, index=default, format_func=label, key="lc_heat_file")
        value = st.radio("Show", ["time_lost", "time", "avg_speed", "crashes"], horizontal=True, key="lc_heat_val",
                         format_func=lambda v: {"time_lost": "seconds lost vs best", "time": "segment time",
                                                "avg_speed": "average speed", "crashes": "crashes"}[v])
        st.plotly_chart(pf.segment_heatmap(load_json(sp), value=value), theme=None, key="lc_heat")
        st.caption("Rows = attempts, columns = gate-to-gate segments. Bright = time lost against your own best "
                   "attempt in that segment. A good reflect() turns the bright cells dark, one corner at a time.")

with tab_wrong:
    st.markdown("A learner that always keeps its **latest** change: one greedy step too far and it never recovers. "
                "Keep-best (revert to the best PARAMS after a worse lap) is the fix.")
    series, caption = gone_wrong()
    learner = next((p for p in files if p.name == "learner_session.json"), None)
    curves = [series] if series else []
    if learner is not None and not series:
        curves.append(load_json(learner))
    if curves:
        st.plotly_chart(pf.learning_curve_figure(curves, height=420, best_so_far=False), theme=None, key="lc_wrong")
    st.caption(caption or "No figure data: run python lesson/make_figures.py --only learning_gone_wrong")
    st.code("def reflect(self, report):\n"
            "    t = report['race_time'] if report['completed'] else 999\n"
            "    if t < self.best_time:                       # better: keep it\n"
            "        self.best_time, self.best = t, dict(self.params)\n"
            "    else:                                        # worse: go back to the best\n"
            "        self.params = dict(self.best)\n"
            "    self.try_one_change(report)                   # problem generator", language="python")
