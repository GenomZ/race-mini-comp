"""The 2-hour Streamlit tutorial (streamlit_lessons/): conventions + AppTest of every lesson and solution.

    .venv/Scripts/python.exe -m pytest -q tests/test_streamlit_lessons.py

Conventions (CURRICULUM.md section 5): every lesson file <= 40 lines (S10 <= 90) and <= 100
characters per line, a docstring with concept / run command / "you should see" / exercise, exactly
one `# TODO(exercise)`, and a runnable solution in streamlit_lessons/solutions/ for every exercise
whose docstring is the lesson's plus one SOLUTION line. Then each script and solution runs in
AppTest without exceptions, and the key interaction of each lesson works.
"""

from __future__ import annotations

import importlib.util
import json
import math
import re
import sys
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LESSONS_DIR = ROOT / "streamlit_lessons"
SOLUTIONS_DIR = LESSONS_DIR / "solutions"
LESSONS = sorted(LESSONS_DIR.glob("s[0-9][0-9]_*.py"))
SOLUTIONS = sorted(SOLUTIONS_DIR.glob("s[0-9][0-9]_*.py"))
TIMEOUT = 90


def app(path: Path) -> AppTest:
    return AppTest.from_file(str(path), default_timeout=TIMEOUT)


def run(path: Path) -> AppTest:
    at = app(path).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def no_errors(at: AppTest) -> AppTest:
    assert not at.exception, [e.value for e in at.exception]
    return at


def components(at: AppTest) -> list[dict]:
    """The custom canvases (CCv2 bidi components) on the page, as their JSON data."""
    out = []
    for c in at.get("bidi_component"):
        try:
            out.append(json.loads(c.proto.json))
        except (AttributeError, ValueError):
            out.append({})
    return out


def plotly_traces(at: AppTest, k: int = 0) -> list[dict]:
    return json.loads(at.get("plotly_chart")[k].proto.spec)["data"]


def lesson(name: str) -> Path:
    return next(LESSONS_DIR.glob(f"{name}_*.py"))


def solution(name: str) -> Path:
    return next(SOLUTIONS_DIR.glob(f"{name}_*.py"))


def data_module():
    """The lessons' _data module (the same object the AppTest scripts import)."""
    if str(LESSONS_DIR) not in sys.path:
        sys.path.insert(0, str(LESSONS_DIR))
    import _data

    return _data


# ----------------------------------------------------------------------------------------- conventions
def test_ten_lessons_and_ten_solutions():
    assert [p.name[:3] for p in LESSONS] == [f"s{k:02d}" for k in range(1, 11)]
    assert [p.name for p in SOLUTIONS] == [p.name for p in LESSONS]
    assert (LESSONS_DIR / "_data.py").exists() and (LESSONS_DIR / "README.md").exists()
    assert (SOLUTIONS_DIR / "_data.py").exists()


@pytest.mark.parametrize("path", LESSONS, ids=lambda p: p.stem)
def test_lesson_conventions(path: Path):
    text = path.read_text(encoding="utf-8")
    limit = 90 if path.name.startswith("s10") else 40
    assert len(text.splitlines()) <= limit, f"{path.name}: {len(text.splitlines())} lines > {limit}"
    assert text.count("# TODO(exercise)") == 1, f"{path.name}: needs exactly one # TODO(exercise)"
    doc = text.split('"""')[1]
    for word in ("Concept:", "Run:", "You should see:", "Exercise:"):
        assert word in doc, f"{path.name}: docstring lacks {word!r}"
    assert f"python -m streamlit run streamlit_lessons/{path.name}" in doc
    assert "import _data" in text                      # data + sys.path come from _data.py
    long = [n for n, line in enumerate(text.splitlines(), 1) if len(line) > 100]
    assert not long, f"{path.name}: lines over 100 characters (projector!): {long}"


@pytest.mark.parametrize("path", SOLUTIONS, ids=lambda p: p.stem)
def test_solution_conventions(path: Path):
    text = path.read_text(encoding="utf-8")
    assert "TODO(exercise)" not in text
    assert "# SOLUTION" in text
    marker = f"SOLUTION  python -m streamlit run streamlit_lessons/solutions/{path.name}\n"
    assert marker in text
    # the docstring is the lesson's + the marker line, so `git diff --no-index` shows only real changes
    lesson_doc = (LESSONS_DIR / path.name).read_text(encoding="utf-8").split('"""')[1]
    assert text.replace(marker, "", 1).split('"""')[1] == lesson_doc
    long = [n for n, line in enumerate(text.splitlines(), 1) if len(line) > 100]
    assert not long, f"{path.name}: lines over 100 characters: {long}"


def test_files_use_lf_line_endings():
    files = [*LESSONS_DIR.glob("*.py"), *SOLUTIONS_DIR.glob("*.py"), LESSONS_DIR / "README.md"]
    crlf = [f.relative_to(ROOT).as_posix() for f in files if b"\r\n" in f.read_bytes()]
    assert not crlf, f"CRLF line endings (git diff --no-index gets noisy): {crlf}"
    assert "eol=lf" in (LESSONS_DIR / ".gitattributes").read_text(encoding="utf-8")


def test_readme_covers_every_lesson():
    readme = (LESSONS_DIR / "README.md").read_text(encoding="utf-8")
    for p in LESSONS:
        assert f"python -m streamlit run streamlit_lessons/{p.name}" in readme, p.name
    for word in ("You should see", "Common errors", "Windows", "macOS", "13:30", "15:30"):
        assert word in readme
    assert "uv run streamlit" not in readme and "uv run python" not in readme  # no lockfile


def test_old_streamlit_gets_a_clear_message(monkeypatch):
    import streamlit

    monkeypatch.setattr(streamlit, "__version__", "1.50.0")
    spec = importlib.util.spec_from_file_location("_data_version_check", LESSONS_DIR / "_data.py")
    with pytest.raises(RuntimeError, match=r"Streamlit >= 1\.66.*you have 1\.50\.0.*pip install"):
        spec.loader.exec_module(importlib.util.module_from_spec(spec))


# ----------------------------------------------------------------------------------------- smoke
@pytest.mark.parametrize("path", LESSONS + SOLUTIONS, ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_script_runs_without_exceptions(path: Path):
    at = run(path)
    assert at.title, f"{path.name} shows no title"


# ----------------------------------------------------------------------------------------- S01-S05
def test_s01_tiles_and_obs():
    at = run(lesson("s01"))
    assert [m.label for m in at.metric] == ["L0 Dummy", "L1 Reflex", "L2 Planner", "Max (house)"]
    assert at.metric[3].value != "DNF", "the house car Max is in the demo race"
    assert at.json, "the obs dict should be shown with magic st.write"
    sol = run(solution("s01"))
    me = sol.metric[-1]
    assert me.label == "Me" and me.value == "58.0 s"
    sol.number_input[0].set_value(40.0).run()
    assert no_errors(sol).metric[-1].value == "40.0 s"


def test_s02_widgets_rerun_the_script():
    from agentic_gp.sim.config import PhysicsConfig
    from agentic_gp.sim.physics import turn_radius

    at = run(lesson("s02"))
    assert at.metric[0].value == "28.3 u/s"                     # sqrt(20 * 40)
    at.slider[0].set_value(80).run()
    assert no_errors(at).metric[0].value == "40.0 u/s"          # sqrt(20 * 80)
    assert isinstance(at.number_input[0].value, int)            # "30", never "30,00"
    for speed, steer in [(30, 0.5), (10, 0.5), (10, 0.25), (5, -0.25), (40, 1.0)]:
        at.number_input[0].set_value(speed).run()
        at.radio[0].set_value(steer).run()
        sim_r = turn_radius(speed, steer, PhysicsConfig())
        assert plotly_traces(no_errors(at))[0]["name"] == f"your path, R = {sim_r:.0f}", (speed, steer)
    sol = run(solution("s02"))
    sol.number_input[1].set_value(100).run()                     # 30 u/s needs 18 + 15 = 33 units
    assert no_errors(sol).success and not sol.error
    sol.number_input[1].set_value(20).run()
    assert no_errors(sol).error and not sol.success


def test_s03_cache_and_layers():
    at = run(lesson("s03"))
    at.selectbox[0].set_value("sprint").run()
    assert no_errors(at).dataframe, "corner table"
    at.selectbox[0].set_value("grand_prix").run()                # the second visit: from the cache
    assert "from the CACHE" in no_errors(at).caption[0].value
    sol = run(solution("s03"))
    assert len(sol.checkbox) == 4
    sol.checkbox[0].uncheck().run()
    no_errors(sol)


def test_s04_crash_ticks_and_time_lost():
    at = run(lesson("s04"))
    assert len(at.radio[0].options) == 4                          # 4 practice runs
    _data = data_module()
    session = _data.practice()
    rows = [r for r in _data.telemetry_rows(session) if r["attempt"] == 1]
    crash_rows = [r for r in rows if r["crash"]]
    assert len(crash_rows) == session["attempts"][0]["crashes"] == 4
    for c in crash_rows:                       # the chart shows the stop: 0 u/s at the impact time
        k = rows.index(c)                      # ... and flat 0 until the car drives on
        assert c["speed"] == 0 and rows[k - 1]["t"] < c["t"] < rows[k + 1]["t"]
        assert rows[k + 1]["speed"] == rows[k + 1]["throttle"] == 0 and not rows[k + 1]["crash"]
        assert rows[k + 2]["t"] == rows[k + 1]["t"]
    sol = run(solution("s04"))
    sol.radio[0].set_value(1).run()                              # run 1 (sight=0) crashes
    no_errors(sol)
    assert any("time lost" in df.value.columns for df in sol.dataframe)
    assert any("speed_before" in df.value.columns for df in sol.dataframe)
    sol.radio[0].set_value(4).run()                              # the fastest run: no crashes
    assert no_errors(sol).success


def test_s05_layout_and_crashes_tab():
    at = run(lesson("s05"))
    assert len(at.sidebar.selectbox) == 1 and len(at.tabs) == 3
    assert at.metric[2].value.endswith(" u/s")
    sol = run(solution("s05"))
    sol.sidebar.selectbox[0].set_value(1).run()
    no_errors(sol)
    assert len(sol.tabs) == 4 and sol.tabs[3].label.startswith("Crashes (4")


# ----------------------------------------------------------------------------------------- S06-S09
def _select_driver(at: AppTest, stem: str, label: str = "Driver") -> AppTest:
    box = next(s for s in at.selectbox if s.label == label)
    return box.set_value(next(p for p in data_module().driver_files() if p.stem == stem)).run()


def test_s06_race_button_and_compare_mode():
    at = run(lesson("s06"))
    assert at.selectbox[0].value.stem == "l2_planner" or at.selectbox[0].value.parent.name == "submissions"
    assert len(at.slider) >= 1                                   # sliders generated from PARAMS
    at.button[0].click().run()
    no_errors(at)
    assert at.session_state["s06_run"]["best"] is not None
    assert components(at), "replay canvas"
    sol = _select_driver(run(solution("s06")), "l2_planner")
    sol.button[0].click().run()
    sol.slider[0].set_value(sol.slider[0].value * 0.8).run()
    sol.button[0].click().run()
    no_errors(sol)
    assert sol.session_state["s06_prev"] is not None
    assert any(m.label.startswith("Best lap now") for m in sol.metric)
    assert components(sol), "replay canvas with both runs"


def test_s06_int_params_stay_int():
    """An int PARAM (l3_learner racing_line = 0) gets an int slider: the driver gets 0, not 0.0."""
    at = _select_driver(run(lesson("s06")), "l3_learner")
    knobs = {s.label: s.value for s in no_errors(at).slider}
    info = data_module().driver_info(at.selectbox[0].value)
    assert knobs and all(type(knobs[k]) is type(v) for k, v in info.params.items() if k in knobs)
    assert type(knobs["racing_line"]) is int
    at.button[0].click().run()
    assert no_errors(at).session_state["s06_run"]["best"] is not None


def test_s07_session_state_keeps_the_car():
    at = run(lesson("s07"))
    gas = next(b for b in at.button if b.label == "Gas")
    for _ in range(3):
        gas.click().run()
    no_errors(at)
    obs = at.session_state["s07_obs"]
    assert obs["tick"] == 3 and obs["speed"] > 10
    next(b for b in at.button if b.label == "New car").click().run()
    assert no_errors(at).session_state["s07_obs"]["tick"] == 0


def test_s07_solution_undo_and_autopilot():
    at = run(solution("s07"))
    for _ in range(4):
        next(b for b in at.button if b.label == "Gas").click().run()
    speed4 = at.session_state["s07_obs"]["speed"]
    next(b for b in at.button if b.label == "Undo").click().run()
    obs = no_errors(at).session_state["s07_obs"]
    assert obs["tick"] == 3 and obs["speed"] < speed4
    at.toggle[0].set_value(True).run()                           # autopilot on: the driver decides
    for _ in range(6):
        next(b for b in at.button if b.label == "Brake").click().run()
    obs = no_errors(at).session_state["s07_obs"]
    assert obs["tick"] == 9 and obs["crashes"] == 0 and obs["speed"] > 5   # braking buttons, but it drives


def test_s08_frames_start_with_a_car_and_end_finished():
    lap = data_module().frames(str(ROOT / "drivers" / "examples" / "l2_planner.py"), "sprint", 0.0)
    frames = lap["frames"]
    assert all(f["obs"] for f in frames), "every frame has the observation (no empty tick 0)"
    assert frames[0]["tick"] == 0 and frames[0]["t"] == 0.0
    assert [f["tick"] for f in frames] == list(range(len(frames)))
    assert lap["completed"] and frames[-1]["status"] == "finished"
    assert not any("status" in f for f in frames[:-1])


def test_s08_step_and_restart():
    at = run(lesson("s08"))
    at.toggle[0].set_value(False).run()                          # pause
    next(b for b in at.button if b.label == "Restart").click().run()
    assert no_errors(at).session_state["s08_i"] == 0
    state = components(at)[0].get("state", {})
    assert state.get("tick") == 0 and state.get("status") == "paused", "a car on the grid at tick 0"
    next(b for b in at.button if b.label == "Step one tick").click().run()
    assert no_errors(at).session_state["s08_i"] == 1
    assert at.get("progress"), "progress bar"
    sol = run(solution("s08"))
    sol.toggle[1].set_value(True).run()                          # realtime: the command-age gauge appears
    assert len(no_errors(sol).get("progress")) == 2


def test_s08_last_frame_says_finished():
    at = run(lesson("s08"))
    at.toggle[0].set_value(False).run()
    at.session_state["s08_i"] = 10_000                           # clamped to the last frame
    at.run()
    state = components(no_errors(at))[0].get("state", {})
    assert state.get("status") == "finished"
    assert at.get("progress")[0].proto.text.endswith("finished")


def test_s09_track_lab():
    at = run(lesson("s09"))
    assert at.success, "the Grand Prix control points make a valid track"
    next(b for b in at.button if "Race here" in b.label).click().run()
    assert no_errors(at).metric[0].value.endswith(" s")
    sol = run(solution("s09"))
    next(b for b in sol.button if "Race here" in b.label).click().run()
    assert no_errors(sol).session_state["s09_result"]["completed"]


# ----------------------------------------------------------------------------------------- S10
def test_s10_scrub_tower_radio_follow():
    at = run(lesson("s10"))
    assert components(at), "pro player canvas"
    clock = at.slider(key="s10_t")
    clock.set_value(30.0).run()                                  # scrub to t = 30 s
    tower = no_errors(at).dataframe[0].value
    assert list(tower.columns) == ["P", "driver", "lap", "gap"] and len(tower) >= 4
    assert any(re.match(r"\*\*\d+ s ", c.value) for c in at.caption), "radio feed"
    first = json.loads(Path(at.selectbox[0].value).read_text(encoding="utf-8"))["cars"][0]["id"]
    next(s for s in at.selectbox if s.label == "Follow").set_value(first).run()
    layout = json.loads(no_errors(at).get("plotly_chart")[0].proto.spec)["layout"]  # zoomed in
    assert layout["xaxis"]["range"][1] - layout["xaxis"]["range"][0] == pytest.approx(160)
    at.slider(key="s10_t").set_value(clock.max).run()             # the end: everybody finished
    assert any(str(v).startswith("FIN") for v in no_errors(at).dataframe[0].value["gap"])


def test_s10_bunched_cars_get_one_label_each_spot():
    at = run(lesson("s10"))                                      # t = 0: everyone on the grid
    labels = [(d["x"][0], d["y"][0], d["text"][0]) for d in plotly_traces(at)
              if d.get("mode") == "markers+text" and d.get("text") and d["text"][0]]
    assert labels, "at least the leader is labelled"
    for k, (x, y, _) in enumerate(labels):
        assert all(math.dist((x, y), (x2, y2)) > 18 for x2, y2, _ in labels[k + 1:]), labels


def test_s10_navigation_wraps_every_lesson():
    at = run(lesson("s10"))
    for p in LESSONS[:-1]:
        at.switch_page(p.name).run()
        assert not at.exception, (p.name, [e.value for e in at.exception])
        assert at.title, p.name


def test_s10_solution_learning_curve_page(tmp_path, monkeypatch):
    """The page reads results/*.json: current files are drawn, v1-course files are skipped."""
    _data = data_module()
    monkeypatch.setattr(_data, "RESULTS", tmp_path)
    (tmp_path / "reflex.json").write_text(json.dumps(        # a v1 file: no "format", no race_time
        {"name": "reflex", "track": "default", "attempts": [{"attempt": 1, "completed": True}]}))
    result = _data._session(_data.EXAMPLES / "l3_learner.py", "sprint", 3).to_result_dict()
    (tmp_path / "learner_sprint.json").write_text(json.dumps(result), encoding="utf-8")
    assert [p.name for p in _data.result_files()] == ["learner_sprint.json"]

    at = run(solution("s10"))
    # st.navigation function pages cannot be reached with the public AppTest.switch_page (it takes
    # file paths), so this uses AppTest's page registry - and skips if that ever changes.
    pages = getattr(at, "_registered_pages", None) or {}
    page = next((h for h, info in pages.items() if info.get("url_pathname") == "my_learning_curve"), None)
    if page is None:
        pytest.skip("this Streamlit version does not expose st.navigation pages to AppTest")
    at._page_hash = page
    at.run()
    no_errors(at)
    assert at.title[0].value == "My learning curve"
    assert at.metric[0].label == "Learner on sprint" and at.dataframe


def test_cold_start_without_races_demo(tmp_path, monkeypatch):
    """No races/demo: the generated class race still has 8 cars incl. the house car Max."""
    _data = data_module()
    monkeypatch.setattr(_data, "DEMO", tmp_path / "no_demo")
    monkeypatch.setattr(_data, "CACHE", tmp_path / "cache")
    replay = json.loads(_data.class_demo().read_text(encoding="utf-8"))
    names = [c["name"] for c in replay["cars"]]
    assert len(names) == 8 and "Max" in names
    assert {c["id"]: c for c in replay["cars"]}["max"]["best_lap"]   # S01's "Max (house)" tile
