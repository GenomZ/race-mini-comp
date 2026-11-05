"""Grand Prix: every driver on one track - practice, qualifying, race - live progress, replays and exports."""

import os
import threading
import time
from pathlib import Path

import streamlit as st

from agentic_gp.viz.player import replay_html
from app.components import replay_canvas
from app.lib.drivers import discover_drivers
from app.lib.gp_worker import GrandPrixSpec, run_gp_job, unique_stem
from app.lib.live_runner import BackgroundJob
from app.lib.loaders import list_json, load_json, load_replay
from app.lib.paths import RACES_DIR, REPO_ROOT, rel
from app.lib.sandbox import is_trusted
from app.lib.paths import DRIVER_FOLDERS
from app.lib.ui import (fast_mode, hide_mystery, llm_run_env, mark_dead_heats, mystery_note, page_header,
                        presenter_mode, real_llm_possible, short_status, track_picker)

page_header("Grand Prix", "Your file is everything you bring: each driver gets plan() once, practice laps with "
            "reflect(), one qualifying lap for the grid, then the race - with fresh memory, official LLM rules "
            "(no LLM in drive() or in the race) and 5 s per crash.", chapter="Ch8")

HOUSE = ("drivers/examples/l6_llm_crew_chief.py", "drivers/examples/l2_planner.py", "drivers/examples/fly_toy.py")
FAST_DEFAULT = ("l1_reflex", "l2_planner", "l2b_racing_line", "l3_learner")
presenter = presenter_mode()


def merge_progress(post):
    """Grand prix updates come in several types; keep ONE row per driver with the newest of every field."""
    rows: dict = {}
    lock = threading.Lock()

    def on_update(u: dict) -> None:
        with lock:
            key = u.get("index", u.get("driver", "?"))          # the grid slot: the name changes after loading
            row = rows.setdefault(key, {"index": key, "status": "starting"})
            row["driver"] = u.get("driver", row.get("driver", "?"))
            kind = u.get("type")
            if kind == "progress":
                row.update({k: u[k] for k in ("phase", "attempt", "lap", "race_progress", "speed", "t") if k in u})
                row["status"] = "running"
            elif kind == "attempt":
                row["last"] = (f"{u['phase']} {u['race_time']:.2f} s" if u.get("completed") and u.get("race_time")
                               else f"{u['phase']} {short_status(u.get('status')) or 'DNF'}") + f", {u.get('crashes', 0)} crash(es)"
            elif kind == "error":
                row["status"] = "error"
                row["last"] = f"{u.get('phase', '')} error (details below)".strip()
                row["error"] = str(u.get("traceback", ""))[-600:]
            elif kind == "done":
                row["status"] = u.get("status") or "finished"
                row["phase"] = "done"
            post(dict(row))

    return on_update


# ------------------------------------------------------------------------------ 1. set up a race
job: BackgroundJob | None = st.session_state.get("gp_job")
running = job is not None and not job.done
GRID_DIR = RACES_DIR / "grid"          # grade.py scrutineer --grid races/grid: only files that PASSED
drivers = [d for d in discover_drivers([("grid", GRID_DIR), *DRIVER_FOLDERS])
           if d.group != "template" and d.style != "none"]
labels = [d.label for d in drivers]
grid = [d.label for d in drivers if d.group == "grid"]
# Without a scrutineered grid: the main files in drivers/submissions (never the Ch4 pilot / Ch7 fly side files).
subs = [d.label for d in drivers if d.group == "submissions" and not d.stem.lower().endswith(("_pilot", "_fly"))]
default = grid or subs or [d.label for d in drivers if d.stem in FAST_DEFAULT]

with st.expander("Set up a grand prix", icon=":material/tune:",
                 expanded=not running and not st.session_state.get("gp_view") and not presenter):
    picks = st.multiselect("Drivers on the grid", labels, default=default, key="gp_drivers",
                           help="Default: races/grid/ (the files that passed `grade.py scrutineer --grid races/grid`); "
                                "without it, the main files in drivers/submissions/ (the Inbox page puts them there).")
    if not grid and subs:
        st.caption(":material/warning: No races/grid/ yet: the default grid is drivers/submissions as uploaded "
                   "(not scrutineered). For a class race run `python lesson/instructor/grade.py scrutineer --grid "
                   "races/grid` first.")
    house = st.toggle("Add the house team (Max the MiniMax Crew Chief + reference cars; raced, not scored)",
                      key="gp_house")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        track = track_picker("Track", key="gp_track", default="grand_prix")
    practice = c2.number_input("Practice attempts", 0, 6, 0 if fast_mode() else 2, key="gp_practice",
                               help="Single laps with reflect() after each: learning time.")
    laps = c3.number_input("Race laps", 1, 10, 1 if fast_mode() else 3, key="gp_laps")
    qualifying = c4.toggle("Qualifying lap", value=not fast_mode(), key="gp_quali",
                           help="One lap that sets the grid order.")
    chosen = [drivers[labels.index(p)] for p in picks]
    # "All in one process" is only for our reference drivers: a student file always gets its own process.
    all_trusted = all(is_trusted(d.path) for d in chosen)
    c5, c6, c7 = st.columns([2, 1, 2])
    title = c5.text_input("Title", "Agentic Grand Prix", key="gp_title")
    jobs = c6.number_input("Cars at once", 1, max(1, (os.cpu_count() or 2)), min(4, max(1, (os.cpu_count() or 2) - 1)),
                           key="gp_jobs", help="Each car runs in its own process; more at once = faster, more CPU.")
    if all_trusted:
        isolate = c7.toggle("Each driver in its own process (official, safe)", value=not fast_mode(), key="gp_isolate",
                            help="Off = the reference drivers race one after another in ONE race process: faster "
                                 "for a demo. The app itself is never affected either way.")
    else:
        isolate = c7.toggle("Each driver in its own process (official, safe)", value=True, disabled=True,
                            key="gp_isolate_forced",
                            help="Always on with student files: a hanging or crashing file only loses its own car.")
    uses_llm = any(d.uses_llm for d in chosen) or house
    llm_real = False
    if uses_llm:
        opts = ["Mock (free, offline)"] + (["Real LLM (shared key)"] if real_llm_possible() else [])
        llm_real = st.radio("LLM drivers use", opts, horizontal=True, key="gp_llm",
                            help="For this race only (other browsers keep their own choice).").startswith("Real")
    stem = st.text_input("Save as races/<name>.json", "", key="gp_stem", placeholder="automatic: gp_<date>_<time>",
                         help="Also writes <name>_qualifying.json, <name>_results.json and the LLM logs. An existing "
                              "name is never overwritten: _2, _3 ... is added.")

    def _start() -> None:
        entries = [{"source": str(d.path)} for d in chosen]
        if house:
            have = {str(Path(e["source"]).resolve()) for e in entries}
            entries += [{"source": str(REPO_ROOT / h), "house": True} for h in HOUSE
                        if (REPO_ROOT / h).exists() and str((REPO_ROOT / h).resolve()) not in have]
        if not entries:
            st.session_state.gp_msg = "Pick at least one driver."
            return
        wanted = (stem or "").strip() or time.strftime("gp_%m%d_%H%M%S")      # built NOW, not when the page opened
        safe = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in wanted) or "gp"
        safe = unique_stem(RACES_DIR, safe)
        inprocess = not isolate and all(is_trusted(e["source"]) for e in entries)
        cfg = dict(track=track, practice_attempts=int(practice), race_laps=int(laps), jobs=int(jobs),
                   title=title or "Agentic Grand Prix", qualifying=bool(qualifying),
                   llm_log_dir=str(RACES_DIR / "llm_logs" / safe), ramp_s=0.5 if inprocess else 2.0)
        spec = GrandPrixSpec(entries=entries, config=cfg, out_dir=str(RACES_DIR), stem=safe, inprocess=inprocess,
                             llm_env=llm_run_env(llm_real) if uses_llm else {"LLM_PROVIDER": "mock"})
        st.session_state.gp_stem_run = safe
        st.session_state.gp_job = BackgroundJob(lambda j: run_gp_job(j, spec, merge_progress(j.post)),
                                                key_field="index", name="grand-prix").start()
        st.session_state.gp_saved = None
        st.session_state.gp_msg = (f"Saving as races/{safe}.json (the name {wanted} was taken)."
                                   if safe != "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in wanted) else None)

    st.button("Start the grand prix", type="primary", key="gp_start", on_click=_start, icon=":material/flag:",
              disabled=running)
if st.session_state.get("gp_msg"):
    st.info(st.session_state.gp_msg, icon=":material/info:")


# ------------------------------------------------------------------------------ 2. live progress
def _save(job: BackgroundJob) -> None:
    """When the job is done: the worker wrote the files; switch the results view to them (once)."""
    s = job.state()
    if s.status != "done" or not s.result or st.session_state.get("gp_saved"):
        return
    results = (s.result.get("paths") or {}).get("results")
    if results:
        st.session_state.gp_saved = results
        st.session_state.gp_view = results
        st.session_state.gp_select_new = True            # select it in the results box on the next full run


def progress_rows(s) -> list[dict]:
    """While racing: race leaders first. When done: the official finishing order."""
    rows = list(s.latest.values())
    standings = (s.result or {}).get("standings") if isinstance(s.result, dict) else None
    if standings:
        pos = {r.get("name"): r.get("position") or 999 for r in standings}
        rows.sort(key=lambda r: (pos.get(r.get("driver"), 999), r.get("index", 0)))
        for r in rows:
            r["pos"] = pos.get(r.get("driver"))
    else:
        rows.sort(key=lambda r: (-(r.get("phase") == "race"), -float(r.get("race_progress", 0) or 0), r.get("index", 0)))
    return rows


if job is not None:
    @st.fragment(run_every=0.5 if running else None)
    def progress_view() -> None:
        job = st.session_state.get("gp_job")
        s = job.state()
        h1, h2, h3 = st.columns([3, 1, 1], vertical_alignment="center")
        h1.markdown(f"**Grand prix {st.session_state.get('gp_stem_run', '')}: {s.status}** - {s.elapsed_s:.0f} s elapsed")
        if s.status == "running":
            h2.button("Stop", key="gp_stop", on_click=job.stop, icon=":material/stop:", width="stretch")
        rows = progress_rows(s)
        if rows:
            done = s.status == "done" and isinstance(s.result, dict)
            st.dataframe([({"pos": r.get("pos")} if done else {})
                          | {"driver": r["driver"], "phase": r.get("phase", "load"), "attempt": r.get("attempt"),
                             "lap": r.get("lap"), "progress": float(r.get("race_progress", 0.0) or 0.0),
                             "speed": r.get("speed"), "clock": r.get("t"), "status": short_status(r.get("status"), 24),
                             "last result": r.get("last", "")} for r in rows],
                         hide_index=True, width="stretch",
                         column_config={"progress": st.column_config.ProgressColumn(
                             "progress (laps)", min_value=0.0, max_value=float(max(1, int(laps))), format="%.2f"),
                             "clock": st.column_config.NumberColumn(format="%.1f s"),
                             "last result": st.column_config.TextColumn(width="large")})
            errs = [r for r in rows if r.get("error")]
            if errs:
                with st.expander(f"Driver problems ({len(errs)})", icon=":material/error:"):
                    for r in errs:
                        st.markdown(f"**{r['driver']}**")
                        st.code(r["error"], language=None, wrap_lines=True)
        elif s.status == "running":
            st.caption("Starting the race process and the drivers (a fresh process per car takes a few seconds)...")
        if s.status == "error":
            st.error(s.error)
        if s.status in ("done", "stopped") and not s.result:
            st.warning("The grand prix ended without a result (stopped)." if s.status == "stopped"
                       else "The grand prix ended without a result.")
        if job.done:
            _save(job)
            if running:
                st.rerun()

    progress_view()

if running:
    # One thing at a time on the projector: the old race's replay would autoplay under the new one.
    st.caption("Results and replays appear here when this grand prix is over.")
    st.stop()

# ------------------------------------------------------------------------------ 3. results
if not presenter:
    st.divider()
results, n_hidden = hide_mystery(list_json(RACES_DIR, "grand_prix"))
mystery_note(n_hidden)
view = st.session_state.get("gp_view")
if view and Path(view).exists() and all(p.resolve() != Path(view).resolve() for p in results):
    results.insert(0, Path(view))                       # just written: the folder scan is cached for 10 s
if not results:
    st.info("No grand prix results in races/ yet - start one above.", icon=":material/info:")
    st.stop()
idx = next((i for i, p in enumerate(results) if view and p.resolve() == Path(view).resolve()), 0)
if st.session_state.pop("gp_select_new", False):
    st.session_state.gp_result_file = results[idx]          # a widget keeps its old value unless we set it
rpath = st.selectbox("Results", results, index=idx, key="gp_result_file",
                     format_func=lambda p: f"{rel(p)}  -  {load_json(p).get('title', '')} on {load_json(p).get('track', '')}")
gp = load_json(rpath)
race_file = rpath.with_name(rpath.name.replace("_results.json", ".json"))
quali_file = rpath.with_name(rpath.name.replace("_results.json", "_qualifying.json"))
stand = gp.get("standings", [])

gp_title, gp_track = gp.get("title", "Grand prix"), gp.get("track_name", gp.get("track", ""))
if not presenter:
    st.subheader(gp_title if gp_track.lower() in gp_title.lower() else f"{gp_title} - {gp_track}", anchor=False)
if stand and presenter:                                 # projector: the podium in one line, the replay below it
    st.markdown("  ·  ".join(f"**P{r['position']} {r['name']}** "
                             + (f"{r['race_time']:.2f} s" if r.get("race_time") is not None else short_status(r.get("status")) or "DNF")
                             for r in stand[:3]))
elif stand:
    times = [round(r["race_time"], 2) for r in stand if r.get("race_time") is not None]
    pod = st.columns(3)
    for col, r in zip(pod, stand[:3]):
        full = r.get("status") or "DNF"
        if r.get("race_time") is not None:
            t = f"{r['race_time']:.2f} s"
            tie = times.count(round(r["race_time"], 2)) > 1
        else:
            t, tie = "DNF", False
        sub = (f"{r.get('crashes', 0)} crash(es), best lap {r['best_lap']:.2f} s" if r.get("best_lap") else
               f"{short_status(full) if t == 'DNF' else ''} {r.get('crashes', 0)} crash(es)".strip()) + (
               "  · dead heat" if tie else "")
        col.metric(f"P{r['position']}  {r['name']}" + (" (house)" if r.get("house") else ""), t, sub,
                   delta_color="off", delta_arrow="off", help=None if r.get("race_time") is not None else full)

tab_race, tab_quali, tab_table, tab_export = st.tabs([":material/sports_score: Race replay",
                                                     ":material/timer: Qualifying", ":material/leaderboard: Standings",
                                                     ":material/ios_share: Export"])
with tab_race:
    if race_file.exists():
        replay_canvas(race_file, key="gp_race_canvas", height=600, autoplay=True, speed=2.0, fit=presenter)
    else:
        st.caption(f"Race replay {rel(race_file)} not found.")
with tab_quali:
    if gp.get("qualifying"):
        st.dataframe([{"grid": q["position"], "driver": q["name"], "time": q.get("qualifying_time"),
                       "practice": ", ".join(f"{t:.2f}" if t else "DNF" for t in q.get("practice_times") or []),
                       "crashes": q.get("crashes"), "status": q.get("status")} for q in gp["qualifying"]],
                     hide_index=True, width="stretch",
                     column_config={"time": st.column_config.NumberColumn(format="%.2f s")})
    if quali_file.exists():
        replay_canvas(quali_file, key="gp_quali_canvas", height=480, speed=2.0)
    elif not gp.get("qualifying"):
        st.caption("No qualifying in this event.")
with tab_table:
    rows = mark_dead_heats([{"pos": r["position"], "driver": r["name"] + (" (house)" if r.get("house") else ""),
                             "author": r.get("author"), "kind": r.get("kind"), "grid": r.get("grid"),
                             "time": r.get("race_time"), "gap": r.get("gap"), "laps": r.get("laps_completed"),
                             "best lap": r.get("best_lap"), "crashes": r.get("crashes"), "LLM calls": r.get("llm_calls"),
                             "status": r.get("status") or "finished"} for r in stand])
    st.dataframe(rows, hide_index=True, width="stretch",
                 column_config={"time": st.column_config.NumberColumn(format="%.2f s"),
                                "gap": st.column_config.NumberColumn(format="+%.2f s"),
                                "best lap": st.column_config.NumberColumn(format="%.2f s"),
                                "status": st.column_config.TextColumn(width="medium")})
    if any("dead heat" in str(r["status"]) for r in rows):
        st.caption("Dead heat = the same race time to 0.01 s. With the offline mock LLM, LLM drivers fall back to "
                   "the same physics plan, so they often tie; with a real LLM their plans (and times) differ.")
    if gp.get("errors"):
        st.markdown("**Problems**")
        st.dataframe(gp["errors"], hide_index=True, width="stretch")
    st.caption(f"Wall time {gp.get('wall_seconds', 0):.0f} s · {gp.get('created_utc', '')[:16].replace('T', ' ')} UTC · "
               f"llm_policy {gp.get('config', {}).get('llm_policy', '?')}")
with tab_export:
    e = st.columns(4)
    if race_file.exists():
        e[0].download_button("Race replay JSON", race_file.read_bytes(), race_file.name, "application/json",
                             icon=":material/download:", on_click="ignore", width="stretch", key="gp_dl_replay")
        e[1].download_button("HTML player", lambda: replay_html(load_json(race_file), height=None),
                             race_file.with_suffix(".html").name, "text/html", icon=":material/download:",
                             on_click="ignore", width="stretch", key="gp_dl_html")
    e[2].download_button("Results JSON", rpath.read_bytes(), rpath.name, "application/json",
                         icon=":material/download:", on_click="ignore", width="stretch", key="gp_dl_results")
    md = rpath.with_name(rpath.name.replace("_results.json", "_standings.md"))
    if md.exists():
        e[3].download_button("Standings (markdown)", md.read_bytes(), md.name, "text/markdown",
                             icon=":material/download:", on_click="ignore", width="stretch", key="gp_dl_md")

    st.markdown("**Unity 3D viewer** (the big-screen show)")
    try:
        from agentic_gp.viz import unity
        proj, exe = unity.project_dir(REPO_ROOT), unity.find_viewer(REPO_ROOT)
    except Exception:                                     # noqa: BLE001
        unity, proj, exe = None, None, None
    if proj is None or not race_file.exists():
        st.caption("No unity/RaceViewer project in this repo (or no race replay).")
    else:
        strict = RACES_DIR / "unity" / race_file.name

        def _install() -> None:
            load_replay(race_file).to_json(strict, extended=False)        # strict schema v1 for the viewer
            written = unity.install_replay(strict, REPO_ROOT)
            st.session_state.gp_unity_msg = f"Installed as {rel(written[0])}" + (
                f" (+{len(written) - 1} build copy)" if len(written) > 1 else "") + ". An open viewer reloads within 2 s."

        def _launch() -> None:
            _install()
            unity.launch_viewer(strict, ["--title", gp.get("title", "Grand Prix")], REPO_ROOT)
            st.session_state.gp_unity_msg += " Viewer started."

        u = st.columns([1, 1, 2])
        u[0].button("Install for Unity", key="gp_unity_install", on_click=_install, icon=":material/view_in_ar:",
                    width="stretch")
        u[1].button("Launch the viewer", key="gp_unity_launch", on_click=_launch, icon=":material/rocket_launch:",
                    width="stretch", disabled=exe is None,
                    help=None if exe else f"Not built yet: {unity.BUILD_HINT}")
        u[2].caption(("Viewer: " + rel(exe)) if exe else f"Viewer not built - {unity.BUILD_HINT}")
        if st.session_state.get("gp_unity_msg"):
            st.success(st.session_state.gp_unity_msg, icon=":material/check:")
        st.caption("Keys in the viewer: Space pause, +/- speed, Left/Right scrub, C camera, 1-9 pick a car, H help.")
    st.code(f"python race.py watch {rel(race_file)} --html      # browser player\n"
            f"python race.py watch {rel(race_file)} --unity     # 3D viewer", language="bash")
