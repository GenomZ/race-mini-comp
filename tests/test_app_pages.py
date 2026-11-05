"""AppTest smoke + interaction tests for every Race Control page (app/race_control.py -> app/views/*.py).

    .venv/Scripts/python.exe -m pytest -q tests/test_app_pages.py

AGP_FAST=1 shortens every heavy computation (fewer ticks, smaller sweeps); the LLM is the offline mock
(tests/conftest.py). AppTest never runs browser JavaScript and never fires run_every timers, so the
canvases are checked through their data payload and background threads are joined explicitly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = st_testing.AppTest

REPO = Path(__file__).resolve().parents[1]
ENTRY = str(REPO / "app" / "race_control.py")
DEMO = REPO / "races" / "demo"
PAGES = ["home", "live", "replay", "grand_prix", "sensor_xray", "physics_lab", "track_lab", "telemetry",
         "learning_curves", "latency_lab", "pit_wall", "fly_lab", "inbox"]


@pytest.fixture(autouse=True)
def _fast(monkeypatch):
    monkeypatch.setenv("AGP_FAST", "1")


def open_page(name: str, timeout: float = 60) -> AppTest:
    at = AppTest.from_file(ENTRY, default_timeout=timeout).run()
    if name != "home":
        at.switch_page(f"views/{name}.py").run()
    return at


def ok(at: AppTest) -> AppTest:
    assert not at.exception, "\n".join(e.value for e in at.exception)
    return at


def canvases(at: AppTest) -> dict:
    """CCv2 canvases by key -> their JSON payload."""
    return {c.key: json.loads(c.proto.json) for c in at.get("bidi_component")}


def texts(at: AppTest) -> str:
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption] + [m.value for m in at.info]
    parts += [m.value for m in at.success] + [m.value for m in at.error] + [m.value for m in at.warning]
    return "\n".join(str(p) for p in parts)


# ------------------------------------------------------------------------------ every page
@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_with_a_learning_caption(page):
    at = ok(open_page(page))
    assert at.title, f"{page}: no title"
    assert "What you learn here" in texts(at), f"{page}: missing the 'what you learn here' caption"


def test_navigation_lists_every_page_and_the_sidebar_widgets():
    at = ok(open_page("home"))
    views = {p.stem for p in (REPO / "app" / "views").glob("*.py")}
    assert set(PAGES) <= views
    body = at.get("html")[0].proto.body
    for cp in ("CP0", "CP1", "CP2", "CP3", "CP4", "CP5", "CP6", "CP7", "CP8"):
        assert cp in body
    assert any("Where am I" in e.label for e in at.sidebar.expander)


# ------------------------------------------------------------------------------ replay
def test_replay_page_loads_a_replay_and_reports_the_paused_time():
    at = ok(open_page("replay"))
    c = canvases(at)["replay_canvas"]
    assert c["replay"]["cars"], "the replay canvas got no cars"
    assert any(s.value == "Official classification" for s in at.subheader)
    at.session_state["replay_canvas"] = {"t": 12.0, "follow": None}      # what the browser sends on pause
    at.run()
    ok(at)
    assert any("Order at t = 12.00 s" in s.value for s in at.subheader)


def test_pages_survive_missing_data(monkeypatch):
    """No replays, no sessions, no logs, no grand prix, no fly circuit: friendly empty states, no crash."""
    import app.lib.experiments as ex
    import app.lib.llm_trace as lt
    import app.lib.loaders as ld

    monkeypatch.setattr(ld, "list_replays", lambda root=None: [])
    monkeypatch.setattr(ld, "list_json", lambda root=None, kind=None: [])
    monkeypatch.setattr(lt, "find_llm_logs", lambda roots=None: [])
    monkeypatch.setattr(ex, "fly_available", lambda: (False, "circuit file not found (test)"))
    for page in ("replay", "telemetry", "learning_curves", "pit_wall", "grand_prix", "fly_lab"):
        at = ok(open_page(page))
        assert at.info or at.warning, f"{page}: no empty-state message"


# ------------------------------------------------------------------------------ live
def test_live_runs_a_driver_in_a_thread_and_shows_the_session_replay():
    at = ok(open_page("live"))
    assert canvases(at)["live_canvas"]["state"]["message"] == "Press Start"
    at.selectbox(key="live_driver").select_index(
        next(i for i, o in enumerate(at.selectbox(key="live_driver").options) if "l1_reflex" in o)).run()
    at.selectbox(key="live_track").select("oval").run()
    at.number_input(key="live_attempts").set_value(1).run()
    at.select_slider(key="live_pace").set_value(16.0).run()
    at.button(key="live_start").click().run()
    ok(at)
    run = at.session_state["live_run"]
    try:
        assert run.join(60), "the live session did not finish"
        at.run()
        ok(at)
        assert run.result is not None and run.result.attempts[0].result.completed
        assert "live_replay" in canvases(at)
        assert canvases(at)["live_canvas"]["state"]["status"] == "finished"
        assert [b for b in at.button if b.key == "live_save"]
    finally:
        run.stop(kill=True)


# ------------------------------------------------------------------------------ sensor x-ray
def test_sensor_xray_steps_through_ticks():
    if not (DEMO / "l2_ticks.json").exists():
        pytest.skip("demo data missing")
    at = ok(open_page("sensor_xray"))
    t0 = at.session_state["xr_tick"]
    at.button(key="xr_next").click().run()
    ok(at)
    assert at.session_state["xr_tick"] == t0 + 1
    state = canvases(at)["xr_canvas"]["state"]
    assert len(state["rays"]) == 7 and state["waypoints"]
    assert at.get("plotly_chart"), "the driver's-frame figure is missing"


def test_sensor_xray_records_a_driver_now():
    at = ok(open_page("sensor_xray"))
    at.segmented_control(key="xr_mode").set_value("Record a driver now").run()
    at.button(key="xr_record").click().run()
    ok(at)
    assert at.session_state["xr_data"]["ticks"], "nothing recorded"


# ------------------------------------------------------------------------------ physics lab
def test_physics_lab_hand_drive_one_click_one_tick():
    at = ok(open_page("physics_lab"))
    for _ in range(3):
        at.button(key="hd_gas").click().run()
    ok(at)
    obs = at.session_state["hd_obs"]
    assert obs["tick"] == 3 and obs["speed"] > 5
    assert canvases(at)["hd_canvas"]["state"]["car"]["speed"] > 5
    at.toggle(key="hd_blind").set_value(True).run()
    ok(at)
    assert "hd_canvas" not in canvases(at) and any('"distance_to_wall_front"' in c.value for c in at.code)


# ------------------------------------------------------------------------------ track lab
def test_track_lab_races_one_lap_and_validates_a_built_track():
    at = ok(open_page("track_lab"))
    at.selectbox(key="tl_track").select("oval").run()
    at.button(key="tl_explore_go").click().run()
    ok(at)
    assert "tl_explore_replay" in canvases(at)
    assert any("Valid track" in s.value for s in at.success), "the default oval should be valid"


# ------------------------------------------------------------------------------ telemetry + learning
def test_telemetry_and_learning_pages_read_the_demo_sessions():
    if not (DEMO / "l2_session.json").exists():
        pytest.skip("demo data missing")
    at = ok(open_page("telemetry"))
    assert len(at.get("plotly_chart")) >= 2
    at = ok(open_page("learning_curves"))
    assert len(at.get("plotly_chart")) >= 3


# ------------------------------------------------------------------------------ latency lab
def test_latency_lab_simulated_calls_and_the_cliff():
    at = ok(open_page("latency_lab"))
    assert "Real LLM" not in at.radio(key="lat_mode").options, "no key in tests: no real calls offered"
    at.radio(key="lat_mode").set_value("Simulated (free, offline)").run()
    at.button(key="lat_sim").click().run()
    ok(at)
    assert len(at.session_state["lat_rows"]) == 3
    at.button(key="lat_cliff_go").click().run()
    ok(at)
    cliff = at.session_state["lat_cliff"]
    assert cliff[0]["latency_s"] == 0 and cliff[0]["crashes"] <= cliff[-1]["crashes"]


# ------------------------------------------------------------------------------ pit wall
def test_pit_wall_shows_a_short_log(monkeypatch, tmp_path):
    """Regression: a log with fewer than 5 steps used to crash the 'steps to show' input."""
    import app.lib.llm_trace as lt

    log = tmp_path / "llm_logs" / "Tiny.jsonl"
    log.parent.mkdir()
    rec = {"ts": "2026-10-21T10:00:00", "llm": "minimax:MiniMax-M3", "label": "Tiny", "phases": ["plan"],
           "seconds": 1.2, "messages": [{"role": "user", "content": "Corner speeds please"}],
           "reply": {"content": "{\"cap\": 30}", "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                     "tool_calls": [{"function": {"name": "corner_speed", "arguments": "{\"radius\": 20}"}}]}}
    log.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    monkeypatch.setattr(lt, "find_llm_logs", lambda roots=None: [log])
    at = ok(open_page("pit_wall"))
    assert len(at.chat_message) >= 2
    assert any("corner_speed" in m.value for m in at.markdown)


def test_secret_track_files_stay_hidden_until_the_reveal(monkeypatch, tmp_path):
    """Before Show 2 no page may show the mystery track: its LLM logs describe it in the first prompt."""
    import app.lib.llm_trace as lt

    rec = {"ts": "2026-10-21T10:00:00", "llm": "minimax:MiniMax-M3", "label": "Ada", "phases": ["plan"],
           "seconds": 1.0, "messages": [{"role": "user", "content": "Track: Mystery Circuit"}],
           "reply": {"content": "{}", "usage": {}}}
    logs = []
    for race in ("home", "mystery"):
        p = tmp_path / "final" / "llm_logs" / race / "Ada.jsonl"
        p.parent.mkdir(parents=True)
        p.write_text(json.dumps(rec) + "\n", encoding="utf-8")
        logs.append(p)
    monkeypatch.setattr(lt, "find_llm_logs", lambda roots=None: list(logs))
    at = ok(open_page("pit_wall"))
    options = [str(o) for o in at.selectbox(key="pw_log").options]
    assert len(options) == 1 and "mystery" not in options[0].lower()
    assert "hidden until the reveal" in texts(at)
    at.session_state["show_mystery"] = True
    ok(at.run())
    assert len(at.selectbox(key="pw_log").options) == 2


# ------------------------------------------------------------------------------ fly lab
def test_fly_lab_drives_and_runs_the_experiment():
    from app.lib.experiments import fly_available

    if not fly_available()[0]:
        pytest.skip("FlyWire circuit file not present")
    at = ok(open_page("fly_lab", timeout=180))
    at.button(key="fly_go").click().run()
    ok(at)
    label, run = at.session_state["fly_run"]
    assert run["frames"] and len(run["snapshots"]) == len(run["frames"])
    assert "fly_canvas" in canvases(at)
    at.button(key="fly_sweep_go").click().run()
    ok(at)
    assert at.session_state["fly_sweep"]["real"]["steer_raw"]


# ------------------------------------------------------------------------------ grand prix
def test_grand_prix_runs_in_the_background_and_saves_results(monkeypatch, tmp_path):
    import app.lib.paths as paths

    monkeypatch.setattr(paths, "RACES_DIR", tmp_path)            # never write into the repo's races/
    at = ok(open_page("grand_prix"))
    opts = at.multiselect(key="gp_drivers").options
    picks = [o for o in opts if "l1_reflex" in o or "l2_planner.py" in o][:2]
    at.multiselect(key="gp_drivers").set_value(picks).run()
    at.selectbox(key="gp_track").select("oval").run()
    at.text_input(key="gp_stem").input("test_gp").run()
    at.button(key="gp_start").click().run()
    ok(at)
    job = at.session_state["gp_job"]
    assert job.join(120), "grand prix did not finish"
    at.run()
    ok(at)
    assert job.state().status == "done", job.state().error
    assert (tmp_path / "test_gp.json").exists() and (tmp_path / "test_gp_results.json").exists()
    assert "gp_race_canvas" in canvases(at)
    rows = list(job.state().latest.values())
    assert len(rows) == 2 and all(r["status"] == "finished" for r in rows), rows


def test_grand_prix_page_shows_a_saved_event():
    if not (DEMO / "class_demo_results.json").exists():
        pytest.skip("demo data missing")
    at = ok(open_page("grand_prix"))
    assert "gp_race_canvas" in canvases(at)
    assert any(d.key == "gp_dl_results" for d in at.get("download_button")) or at.tabs


# ------------------------------------------------------------------------------ inbox
def _inbox_with(tmp_path, monkeypatch, smoke):
    subs = tmp_path / "submissions"
    monkeypatch.setenv("AGP_SUBMISSIONS_DIR", str(subs))
    staged = tmp_path / "stage" / "ada_lovelace.py"
    staged.parent.mkdir()
    staged.write_text((REPO / "drivers" / "examples" / "l1_reflex.py").read_text(encoding="utf-8"), encoding="utf-8")
    at = AppTest.from_file(ENTRY, default_timeout=60).run()
    at.switch_page("views/inbox.py")
    at.session_state["inbox_staged"] = {"ada_lovelace.py": {"path": str(staged), "orig": "Ada Lovelace.py",
                                                            "smoke": smoke, "digest": "test"}}
    at.run()
    return ok(at), subs


def test_inbox_accepts_a_file_after_a_passed_smoke_test(monkeypatch, tmp_path):
    at, subs = _inbox_with(tmp_path, monkeypatch, (True, "PASSED"))
    assert any("Static check: OK" in s.value for s in at.success)
    assert any("Smoke test: PASSED" in s.value for s in at.success)
    at.button(key="inbox_accept_ada_lovelace.py").click().run()
    ok(at)
    assert (subs / "ada_lovelace.py").exists()
    assert not at.session_state["inbox_staged"]


def test_inbox_needs_a_passed_smoke_test_or_accept_anyway(monkeypatch, tmp_path):
    """Review finding: a file whose smoke test FAILED could be accepted with one click."""
    at, subs = _inbox_with(tmp_path, monkeypatch, (False, "(no output) - the check process exited with code 7"))
    assert at.button(key="inbox_accept_ada_lovelace.py").disabled
    assert any("Smoke test: FAILED" in e.value for e in at.error)
    at.checkbox(key="inbox_force_ada_lovelace.py").check().run()
    ok(at)
    assert not at.button(key="inbox_accept_ada_lovelace.py").disabled
    at.button(key="inbox_accept_ada_lovelace.py").click().run()
    ok(at)
    assert (subs / "ada_lovelace.py").exists()


def test_inbox_not_smoked_yet_cannot_be_accepted(monkeypatch, tmp_path):
    at, _ = _inbox_with(tmp_path, monkeypatch, None)
    assert at.button(key="inbox_accept_ada_lovelace.py").disabled
    assert any("not run yet" in w.value for w in at.warning)


def test_inbox_tools_sanitize_check_and_smoke(tmp_path):
    from app.lib.inbox_tools import sanitize, smoke_check, status_of

    assert sanitize("Ada Lovelace (v3).py") == "ada_lovelace_v3.py"
    assert sanitize("2fast.py") == "d_2fast.py"
    bad = tmp_path / "bad.py"
    bad.write_text("import requests\ndef drive(obs):\n    return 1, 0\n", encoding="utf-8")
    assert status_of(bad)["errors"], "a network import must be a static error"
    passed, out = smoke_check(REPO / "drivers" / "examples" / "l1_reflex.py", timeout_s=120)
    assert passed, out


def test_grand_prix_installs_a_strict_replay_for_unity(monkeypatch, tmp_path):
    import shutil

    import app.lib.paths as paths
    from agentic_gp.viz import unity

    if not (DEMO / "class_demo_results.json").exists():
        pytest.skip("demo data missing")
    for name in ("class_demo.json", "class_demo_results.json", "class_demo_qualifying.json"):
        shutil.copy(DEMO / name, tmp_path / name)
    monkeypatch.setattr(paths, "RACES_DIR", tmp_path)
    installed = []
    monkeypatch.setattr(unity, "project_dir", lambda repo_root=None: tmp_path / "RaceViewer")
    monkeypatch.setattr(unity, "find_viewer", lambda repo_root=None: None)
    monkeypatch.setattr(unity, "install_replay", lambda p, repo_root=None: installed.append(Path(p)) or [tmp_path / "latest.json"])
    at = ok(open_page("grand_prix"))
    at.button(key="gp_unity_install").click().run()
    ok(at)
    assert installed and installed[0].parent == tmp_path / "unity"
    data = json.loads(installed[0].read_text(encoding="utf-8"))
    assert data["format"] == "agentic-gp-replay" and "zones" not in data["track"]      # strict schema v1
    assert any("Installed as" in s.value for s in at.success)


# ------------------------------------------------------------------------------ review round 2 (W2 review)
HANG = "NAME = 'Hanger'\n\ndef drive(obs):\n    while True:\n        pass\n"
EXIT = ("import os\nNAME = 'Exiter'\n\ndef drive(obs):\n    if obs['tick'] >= 3:\n        os._exit(7)\n"
        "    return 1.0, 0.0\n")


def _student(tmp_path, name: str, text: str) -> Path:
    p = tmp_path / "subs" / name
    p.parent.mkdir(exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def test_sandbox_stops_an_endless_loop_and_reports_a_dead_process(tmp_path):
    """A student drive() with `while True` or os._exit() must never run inside the app process."""
    import time

    from app.lib.sandbox import SandboxError, is_trusted, run_driver_job

    hang, dead = _student(tmp_path, "hang.py", HANG), _student(tmp_path, "dead.py", EXIT)
    assert not is_trusted(hang) and is_trusted(REPO / "drivers" / "examples" / "l1_reflex.py")
    t0 = time.monotonic()
    with pytest.raises(SandboxError, match="did not finish"):
        run_driver_job("app.lib.experiments:race_one_lap", hang, {"driver": str(hang), "track": "oval"}, timeout_s=8)
    assert time.monotonic() - t0 < 30
    with pytest.raises(SandboxError, match="code 7"):
        run_driver_job("app.lib.experiments:record_ticks", dead, {"driver": str(dead), "track": "oval", "max_ticks": 40},
                       timeout_s=60)


def test_sandbox_runs_a_student_file_and_returns_json(tmp_path):
    from agentic_gp.race.replay import Replay
    from app.lib.sandbox import run_driver_job

    good = _student(tmp_path, "rookie.py", (REPO / "drivers" / "examples" / "l1_reflex.py").read_text(encoding="utf-8"))
    out = run_driver_job("app.lib.experiments:race_one_lap", good,
                         {"driver": str(good), "track": "oval", "max_ticks": 60}, timeout_s=90)
    assert out["result"]["waypoints_passed"] > 0 and Replay.from_dict(out["replay"]).cars


def test_live_run_of_a_hanging_student_file_stops_on_request(tmp_path):
    """Live with a `while True` drive(): runs in its own process, Stop ends it within seconds."""
    import time

    from agentic_gp.race.session import SessionConfig
    from app.lib.live_runner import LiveRun

    hang = _student(tmp_path, "hang.py", HANG)
    run = LiveRun(str(hang), SessionConfig(track="oval", attempts=1, drive_timeout_s=1.0), isolate=True, pace=None,
                  env={"LLM_PROVIDER": "mock"}).start()
    try:
        deadline = time.monotonic() + 60
        while run.snapshot().status in ("loading", "planning") and time.monotonic() < deadline:
            time.sleep(0.1)
        t0 = time.monotonic()
        run.stop(kill=True)
        assert run.join(20), "Stop did not end the session"
        assert time.monotonic() - t0 < 20
        assert run.snapshot().status in ("stopped", "finished") and run.snapshot().done
    finally:
        run.stop(kill=True)


def test_llm_choice_is_per_run_never_process_wide(monkeypatch):
    """Review finding: one browser's Mock/Real choice changed LLM_PROVIDER for every browser."""
    import os

    from app.lib.ui import llm_run_env, real_llm_possible

    monkeypatch.setenv("MINIMAX_API_KEY", "test-not-a-key")
    assert real_llm_possible()
    assert llm_run_env(True, "x/logs") == {"LLM_PROVIDER": "minimax", "LLM_LOG_DIR": "x/logs"}
    assert llm_run_env(False) == {"LLM_PROVIDER": "mock"}
    assert os.environ["LLM_PROVIDER"] == "mock"                      # untouched
    for page in (REPO / "app" / "views").glob("*.py"):
        assert "set_llm_provider(" not in page.read_text(encoding="utf-8"), page.name


def test_llm_failures_counts_this_runs_failed_calls(tmp_path):
    from app.lib.llm_health import llm_failures

    log = tmp_path / "Strategist.jsonl"
    recs = [{"ts": "2026-10-21T09:00:00Z", "label": "Strategist", "error": "old failure"},
            {"ts": "2026-10-21T10:00:01Z", "label": "Strategist", "error": "MiniMax error 1004: authentication failed"},
            {"ts": "2026-10-21T10:00:05Z", "label": "Strategist", "error": None}]
    log.write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")
    bad = llm_failures(tmp_path, since_utc="2026-10-21T10:00:00Z", label="Strategist")
    assert bad["calls"] == 2 and bad["failed"] == 1 and "1004" in bad["first_error"]


def test_grand_prix_names_are_never_reused(tmp_path):
    from app.lib.gp_worker import unique_stem

    assert unique_stem(tmp_path, "gp_test") == "gp_test"
    (tmp_path / "gp_test_results.json").write_text("{}", encoding="utf-8")
    assert unique_stem(tmp_path, "gp_test") == "gp_test_2"
    (tmp_path / "gp_test_2.json").write_text("{}", encoding="utf-8")
    assert unique_stem(tmp_path, "gp_test") == "gp_test_3"


def test_grand_prix_worker_with_isolated_drivers_and_stop(tmp_path):
    """The race runs in its own process; isolated driver processes must start from it (regression:
    a stdin reader in the worker deadlocked every child on Windows) and Stop must end it."""
    import threading
    import time

    from app.lib.gp_worker import GrandPrixSpec, run_gp_job

    class Job:
        stop_event = threading.Event()

    updates = []
    spec = GrandPrixSpec(entries=[{"source": str(REPO / "drivers" / "examples" / "l1_reflex.py")}],
                         config={"track": "oval", "practice_attempts": 0, "race_laps": 1, "qualifying": False,
                                 "jobs": 1, "ramp_s": 0.5},
                         out_dir=str(tmp_path), stem="iso", inprocess=False, llm_env={"LLM_PROVIDER": "mock"})
    out = run_gp_job(Job(), spec, updates.append)
    assert out and Path(out["paths"]["results"]).exists(), out
    assert out["standings"][0]["race_time"] is not None
    assert any(u.get("phase") == "race" for u in updates)

    job2 = Job()
    job2.stop_event = threading.Event()
    hang = _student(tmp_path, "hang.py", HANG)
    spec2 = GrandPrixSpec(entries=[{"source": str(hang)}], config={"track": "oval", "practice_attempts": 0,
                                                                   "race_laps": 1, "qualifying": False, "jobs": 1},
                          out_dir=str(tmp_path), stem="hang", inprocess=False, llm_env={"LLM_PROVIDER": "mock"})
    threading.Timer(8.0, job2.stop_event.set).start()
    t0 = time.monotonic()
    assert run_gp_job(job2, spec2, lambda u: None) is None          # stopped: nothing saved
    assert time.monotonic() - t0 < 40 and not (tmp_path / "hang_results.json").exists()


def test_replay_upload_and_missing_file_messages():
    from app.lib.loaders import replay_from_upload

    with pytest.raises(ValueError, match="not valid JSON"):
        replay_from_upload(b"not json at all")
    with pytest.raises(ValueError, match="session file"):
        replay_from_upload(json.dumps({"format": "agentic-gp-result", "reports": []}).encode())
    at = AppTest.from_file(ENTRY, default_timeout=60)
    at.query_params["file"] = "races/does_not_exist.json"
    at.run()
    at.switch_page("views/replay.py").run()
    ok(at)
    if (DEMO / "class_demo.json").exists():
        assert any("was not found" in w.value for w in at.warning)


def test_saved_track_lab_tracks_are_offered_everywhere(monkeypatch, tmp_path):
    import app.lib.paths as paths
    from agentic_gp.sim.track import make_track
    from app.lib.loaders import saved_track_specs, track_choices

    monkeypatch.setattr(paths, "RACES_DIR", tmp_path)
    (tmp_path / "tracks").mkdir()
    (tmp_path / "tracks" / "my_oval.py").write_text(
        "NAME = 'My Oval'\nHALF_WIDTH = 9.0\nN_GATES = 12\n"
        "CONTROL_POINTS = [(0, 0), (200, 0), (280, 60), (200, 120), (0, 120), (-80, 60)]\n", encoding="utf-8")
    specs = saved_track_specs()
    assert len(specs) == 1 and specs[0].startswith("file:") and specs[0] in track_choices(0)
    assert make_track(specs[0]).n_gates == 12


def test_dead_heats_are_labelled():
    from app.lib.ui import mark_dead_heats, short_status

    rows = mark_dead_heats([{"time": 96.85, "status": "finished"}, {"time": 96.85, "status": None},
                            {"time": 103.3, "status": "finished"}, {"time": None, "status": "DNF"}])
    assert [("dead heat" in r["status"]) for r in rows] == [True, True, False, False]
    assert short_status("DNF (driver crashed: KeyError 'x' in drive at line 12)") == "DNF (driver crashed)"


def test_latency_lab_counts_only_successful_real_calls(monkeypatch):
    import app.lib.experiments as ex
    import app.lib.ui as ui

    monkeypatch.setattr(ui, "llm_status", lambda: {"provider": "minimax", "model": "M", "effort": "none",
                                                   "available": True, "real": True, "label": "minimax:M"})
    monkeypatch.setattr(ex, "timed_llm_calls", lambda n, prompt, **kw: [
        {"call": i + 1, "seconds": 0.2, "answer": "", "error": "LLMError: authentication failed" if i else None}
        for i in range(n)])
    at = ok(open_page("latency_lab"))
    at.radio(key="lat_mode").set_value("Real LLM").run()
    at.slider(key="lat_n").set_value(3).run()
    used_before = _budget_used(at)
    at.button(key="lat_run").click().run()
    ok(at)
    assert any("2 call(s) failed" in w.value for w in at.warning)
    assert _budget_used(at) == used_before + 1, "failed calls must not use up the class budget"


def _budget_used(at: AppTest) -> int:
    """The 'Budget: n/30 successful real calls' progress text -> n."""
    import re

    text = " ".join(str(p.proto.text) for p in at.get("progress"))
    m = re.search(r"Budget: (\d+)/", text)
    assert m, text
    return int(m.group(1))


def test_latency_cliff_without_the_reference_driver(monkeypatch, tmp_path):
    import app.lib.paths as paths

    monkeypatch.setattr(paths, "DRIVERS_DIR", tmp_path)
    monkeypatch.setattr(paths, "DEMO_DIR", tmp_path)              # no shipped latency_cliff.json either
    at = ok(open_page("latency_lab"))
    assert at.button(key="lat_cliff_go").disabled
    assert any("l2_planner.py" in w.value for w in at.warning)
