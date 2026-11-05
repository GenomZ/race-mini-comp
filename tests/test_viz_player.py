"""Shared visual layer: HTML replay player, plotly figures, Streamlit components and app/lib helpers.

Fast and offline: replays come from a tiny in-process session (a 60-tick lap on the oval), no
browser is needed (the real-browser check is a separate Playwright script, see app/components/README.md).
"""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import textwrap
import time
from pathlib import Path

import pytest

from agentic_gp.race.session import SessionConfig, run_session
from agentic_gp.viz import plotly_figs as pf
from agentic_gp.viz.player import ASSETS, asset, player_data, replay_html, save_replay_html, track_payload

REPO = Path(__file__).resolve().parent.parent


def _chaser(obs):
    """A tiny function driver: steer at the next waypoint, brake for walls ahead."""
    steer = max(-1.0, min(1.0, obs["angle_to_next_waypoint"] / 30.0))
    throttle = 0.6 if obs["distance_to_wall_front"] > 30 else -0.4
    say = "go" if obs["tick"] == 0 else ("</script><script>alert(1)</script>" if obs["tick"] == 5 else None)
    return {"throttle": throttle, "steering": steer, "say": say, "debug": {"target": obs["position"]}}


@pytest.fixture(scope="module")
def session():
    return run_session(_chaser, SessionConfig(track="oval", attempts=2, max_ticks_per_lap=60))


@pytest.fixture(scope="module")
def replay(session):
    return session.to_replay("all", title="Test <b>session</b>")


# --------------------------------------------------------------------------- player data
def test_track_payload_view_block():
    tp = track_payload("grand_prix")
    v = tp["view"]
    assert len(v["vmax"]) == len(tp["center_x"]) and max(v["vmax"]) <= 40.0
    assert v["corners"] and all(c["id"].startswith("T") for c in v["corners"])
    x0, y0, x1, y1 = v["bounds"]
    assert x0 < min(tp["left_x"]) and x1 > max(tp["right_x"] + tp["left_x"]) - 1e-6 and y0 < y1
    assert track_payload("grand_prix")["view"]["id"] == v["id"]          # stable id (the canvas caches by it)
    assert track_payload("random:3")["view"]["corners"]                   # random tracks work too


def test_track_payload_accepts_every_track_shape(replay):
    from agentic_gp.sim.track import make_track

    tr = make_track("sprint")
    for src in ("sprint", tr, tr.to_map(), replay.track, replay.track.to_dict(), replay.to_dict()):
        tp = track_payload(src)
        assert tp["center_x"] and "view" in tp


def test_player_data_is_the_replay_plus_view(replay):
    d = player_data(replay)
    assert d["format"] == "agentic-gp-replay" and len(d["cars"]) == 2
    assert d["view"]["duration"] == pytest.approx(max(c.t[-1] for c in replay.cars), abs=0.01)
    assert d["track"]["view"]["vmax"]
    assert player_data(replay.to_dict())["view"]["id"] == d["view"]["id"]


def test_player_data_from_path_and_session(tmp_path, replay, session):
    p = tmp_path / "r.json"
    replay.to_json(p)
    assert player_data(p)["title"] == replay.title
    assert len(player_data(session)["cars"]) == 2                        # SessionResult -> "all" attempts


# --------------------------------------------------------------------------- standalone HTML
def test_replay_html_is_self_contained_and_safe(replay):
    html = replay_html(replay, height=500)
    assert html.startswith("<!doctype html>")
    assert not re.search(r"<script[^>]+src=", html), "no external scripts"
    assert "http://" not in html and "https://" not in html.replace("https://docs", "")
    assert "class AgpReplayPlayer" in html and "class AgpTrack" in html
    assert "</script><script>alert(1)" not in html                       # radio text cannot break out
    assert "Test &lt;b&gt;session&lt;/b&gt;" in html                     # title escaped
    raw = re.search(r'<script id="agp-replay" type="application/json">(.*?)</script>', html, re.S).group(1)
    data = json.loads(raw)
    assert data["cars"][0]["messages"][1]["text"] == "</script><script>alert(1)</script>"


def test_save_replay_html(tmp_path, replay):
    out = save_replay_html(replay, tmp_path / "sub" / "r.html")
    assert out.exists() and out.stat().st_size > 20_000
    assert '"fill": true' in out.read_text(encoding="utf-8")             # default: fill the window


def test_cli_watch_html_uses_the_player(tmp_path, replay):
    from agentic_gp.cli import main

    src = tmp_path / "r.json"
    replay.to_json(src)
    assert main(["watch", str(src), "--html", str(tmp_path / "w.html"), "--no-open"]) == 0
    assert "AgpReplayPlayer" in (tmp_path / "w.html").read_text(encoding="utf-8")


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_javascript_parses_and_prepares_cars(tmp_path, replay):
    """The JS compiles, and the car preparation adds the wall-contact frame before each crash."""
    script = tmp_path / "check.js"
    data = json.dumps(player_data(replay))
    script.write_text(asset("agp_core.js") + asset("agp_replay.js") + asset("agp_live.js") + asset("agp_pad.js") + f"""
const d = {data};
const pc = agpPrepareCar(d.cars[0], 0);
const crashes = (d.cars[0].events || []).filter(e => e.kind === "crash").length;
const s = agpStateAt(pc, pc.end / 2);
console.log(JSON.stringify({{n: pc.n, frames: d.cars[0].t.length, crashes, ok: isFinite(s.x) && isFinite(s.h),
  gap: agpTimeAtProgress(pc, 0.1) }}));
""", encoding="utf-8")
    out = subprocess.run(["node", str(script)], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    r = json.loads(out.stdout.strip().splitlines()[-1])
    assert r["ok"] and r["n"] == r["frames"] + r["crashes"]
    assert r["gap"] is None or r["gap"] >= 0


def test_assets_exist():
    names = {p.name for p in ASSETS.iterdir()}
    assert {"agp.css", "agp_core.js", "agp_replay.js", "agp_live.js", "agp_pad.js"} <= names


# --------------------------------------------------------------------------- plotly figures
def test_track_figure_layers():
    fig = pf.track_figure("briefing_demo", gate_numbers=True)
    names = [t.name for t in fig.data]
    assert "v_max (centerline)" in names and "start / finish" in names
    assert any(n and "Yellow" in n for n in names)                      # the zone of briefing_demo
    assert any(a.text.startswith("<b>T") for a in fig.layout.annotations)
    assert fig.layout.yaxis.scaleanchor == "x"


def test_overlays_and_telemetry(session):
    rep = session.attempts[0]
    obs = rep.frames[10]["obs"]
    fig = pf.track_figure("oval", vmax=False)
    n0 = len(fig.data)
    pf.add_car(fig, *obs["position"], obs["heading_deg"], name="me")
    pf.add_rays(fig, obs)
    pf.add_target(fig, (0, 0), car=obs["position"])
    pf.add_trajectory(fig, rep)
    pf.add_crashes(fig, rep.crash_points)
    pf.add_turning_circle(fig, 0, 0, 0, 20, 0.5)
    assert len(fig.data) >= n0 + 2 + 7 + 2 + 1
    tf = pf.telemetry_figure(rep, compare=session.attempts[1])
    assert len([t for t in tf.data if t.name == "this lap"]) == 3


def test_learning_heatmap_latency_profile(session):
    sd = session.to_dict()
    lc = pf.learning_curve_figure([sd, {"me": [60.0, None, 55.0]}])
    assert any(t.name == "me" for t in lc.data)
    hm = pf.segment_heatmap(sd)
    assert len(hm.data[0].z) == len(sd["reports"])
    lh = pf.latency_histogram([0.3, 0.8, 1.2, 2.5])
    assert any(s.x0 == 0.5 for s in lh.layout.shapes)                    # the one-tick line
    sp = pf.speed_profile_figure("oval", actual=session.attempts[0])
    assert [t.name for t in sp.data][:2] == ["grip limit  sqrt(a_lat R)", "after the braking pass"]


# --------------------------------------------------------------------------- app/lib + components
DRIVER_SRC = textwrap.dedent('''
    """A test driver."""
    from agentic_gp import llm, driverkit as dk

    class Driver:
        NAME = "Tester"
        AUTHOR = "ada"
        PARAMS = {
            "aggression": 1.05,     # how hard we push
            "laps": 3,
            "use_line": True,
        }
        def plan(self, track):
            self.points = track["centerline"]
            self.note = self._ask(track["briefing"])
        def _ask(self, text):
            try:
                return llm.chat_json(text)
            except llm.LLMError:
                return {}
        def drive(self, obs):
            return 1.0, 0.0
        def reflect(self, report):
            self.best = report["best_lap"]
            if report["next_phase"] in ("qualifying", "race"):
                return
''')


def test_driver_inspection_and_checkpoint(tmp_path):
    from app.lib.checkpoints import where_am_i
    from app.lib.drivers import discover_drivers, inspect_driver

    f = tmp_path / "tester.py"
    f.write_text(DRIVER_SRC, encoding="utf-8")
    info = inspect_driver(f)
    assert (info.name, info.author, info.style) == ("Tester", "ada", "class")
    assert info.params == {"aggression": 1.05, "laps": 3, "use_line": True}
    assert info.param_help["aggression"] == "how hard we push"
    assert info.llm_in == ("plan",)                                       # via the _ask helper
    assert {"map", "briefing", "llm_fallback", "lock_in", "memory"} <= info.features
    w = where_am_i(f)
    assert w.level == "CP8" and w.next_step
    found = discover_drivers([("tmp", tmp_path)], include_template=False)
    assert [d.stem for d in found] == ["tester"]
    bad = tmp_path / "broken.py"
    bad.write_text("def drive(obs:\n", encoding="utf-8")
    assert inspect_driver(bad).syntax_error and where_am_i(bad).level == "-"


def test_example_drivers_are_discovered():
    from app.lib.drivers import discover_drivers

    stems = {d.stem for d in discover_drivers()}
    assert {"l0_dummy", "l2_planner"} <= stems


def test_slider_spec():
    from app.lib.params_ui import params_label, slider_spec

    assert slider_spec(1.0) == (0.0, 2.0, 0.02)
    lo, hi, step = slider_spec(0.92)
    assert lo == 0.0 and hi >= 1.0 and step <= 0.02
    assert slider_spec(3) == (0, 10, 1)
    assert slider_spec(True) is None and slider_spec("x") is None
    assert params_label({"a": 1.0, "b": 2}, {"a": 1.1, "b": 2}) == "a=1.1"


def test_llm_trace_timeline(tmp_path):
    from app.lib.llm_trace import call_table, find_llm_logs, latency_stats, read_llm_log, timeline

    msgs1 = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "plan the lap"}]
    tc = [{"id": "c1", "type": "function", "function": {"name": "corner_speed", "arguments": "{\"r\": 40}"}}]
    rec1 = {"label": "Max", "phases": ["plan"], "seconds": 1.2, "messages": msgs1,
            "reply": {"content": "", "tool_calls": tc, "usage": {"prompt_tokens": 50, "completion_tokens": 9}}}
    msgs2 = msgs1 + [{"role": "assistant", "content": "", "tool_calls": tc},
                     {"role": "tool", "tool_call_id": "c1", "name": "corner_speed", "content": "28.3"}]
    rec2 = {"label": "Max", "phases": ["plan"], "seconds": 0.8, "messages": msgs2,
            "reply": {"content": "{\"cap\": 38}", "usage": {"prompt_tokens": 80, "completion_tokens": 7}}}
    log = tmp_path / "races" / "gp" / "llm_logs" / "Max.jsonl"
    log.parent.mkdir(parents=True)
    log.write_text(json.dumps(rec1) + "\n" + json.dumps(rec2) + "\nnot json\n", encoding="utf-8")
    assert find_llm_logs([tmp_path / "races"]) == [log]
    recs = read_llm_log(log)
    steps = [(s.role, s.kind) for s in timeline(recs)]
    assert steps == [("user", "prompt"), ("assistant", "answer"), ("assistant", "tool_call"),
                     ("tool", "tool_result"), ("assistant", "answer")]
    assert timeline(recs)[2].args == {"r": 40}
    assert call_table(recs)[0]["tool_calls"] == 1
    assert latency_stats([1.2, 0.8])["p50"] in (0.8, 1.2)


def test_live_run_background_thread():
    from app.lib.live_runner import BackgroundJob, LiveRun

    run = LiveRun(_chaser, SessionConfig(track="oval", attempts=2, max_ticks_per_lap=40), pace=None).start()
    assert run.join(60), "the live run did not finish"
    snap = run.snapshot()
    assert snap.done and snap.status == "finished" and snap.seq > 0
    tele = run.telemetry()
    assert len(tele) == 80 and {r["attempt"] for r in tele} == {1, 2}
    assert len(run.reports()) == 2 and run.reports()[0]["summary_text"]
    assert run.replay("all") is not None and len(run.replay("all").cars) == 2
    state = run.canvas_state()
    assert state["car"] and state["track"]["view"]["vmax"] and state["status"] == "finished"

    paused = LiveRun(_chaser, SessionConfig(track="oval", max_ticks_per_lap=40), pace=None)
    paused.pause()
    paused.start()
    time.sleep(1.0)
    t1 = paused.snapshot().tick
    paused.step()
    time.sleep(0.5)
    assert paused.snapshot().tick <= t1 + 1 and not paused.done           # held while paused
    paused.stop()
    assert paused.join(30) and paused.snapshot().status == "stopped"

    job = BackgroundJob(lambda j: [j.post({"driver": "a", "n": k}) for k in range(3)] and "done").start()
    assert job.join(10) and job.state().result == "done" and job.state().latest["a"]["n"] == 2


def test_live_state_from_obs(session):
    from app.components import live_state, pad_controls, rays_from_obs, waypoints_from_obs

    obs = session.attempts[0].frames[5]["obs"]
    rays = rays_from_obs(obs)
    assert [r["angle"] for r in rays] == [-90, -45, -20, 0, 20, 45, 90]
    wp = waypoints_from_obs(obs)
    d = math.dist(obs["position"], wp[0])
    assert d == pytest.approx(obs["distance_to_next_waypoint"], abs=0.2)
    st_ = live_state("oval", obs=obs, action=(0.5, -0.2), debug={"target": [1, 2]}, show_waypoints=True)
    assert st_["car"]["throttle"] == 0.5 and st_["target"] == [1.0, 2.0] and len(st_["waypoints"]) == 2
    json.dumps(st_)                                                        # plain JSON only
    assert pad_controls({"up": True, "left": True}) == (1.0, -0.6)
    assert pad_controls({"down": True}) == (-1.0, 0.0)


def test_components_demo_page_mounts():
    """AppTest: the demo page runs and mounts the keyed canvases with their data."""
    from streamlit.testing.v1 import AppTest

    if not (REPO / "races" / "demo" / "class_demo.json").exists():
        pytest.skip("demo data missing (python races/demo/make_demo.py)")
    at = AppTest.from_file(str(REPO / "app" / "_components_demo.py"), default_timeout=120).run()
    assert not at.exception, at.exception
    comps = {c.key: c for c in at.get("bidi_component")}
    assert {"demo_replay", "demo_live", "hd_pad", "hd_canvas"} <= set(comps)
    data = json.loads(comps["demo_replay"].proto.json)
    assert data["replay"]["cars"] and data["options"]["report"] is True
    at.session_state["demo_replay"] = {"t": 12.0, "follow": None}          # the browser paused at 12 s
    at.run()
    assert any("Paused at t = 12.00 s" in c.value for c in at.caption)
