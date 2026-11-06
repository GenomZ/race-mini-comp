"""The race.py command line (run in-process through agentic_gp.cli.main, inside a temp folder)."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from agentic_gp import cli
from agentic_gp.race.replay import Replay
from agentic_gp.viz import unity


@pytest.fixture
def work(tmp_path, monkeypatch):
    """Run every command from an empty temp folder so nothing lands in the repo."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def ex(repo: Path, name: str) -> str:
    return str(repo / "drivers" / "examples" / name)


def run(capsys, *argv: str) -> tuple[int, str]:
    code = cli.main(list(argv))
    out = capsys.readouterr()
    return code, out.out + out.err


def write(folder: Path, name: str, text: str) -> str:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_text(text, encoding="utf-8")
    return str(folder / name)


# ------------------------------------------------------------------ drive
def test_drive_l0_writes_replay_plot_and_result(work, repo, capsys):
    code, out = run(capsys, "drive", ex(repo, "l0_dummy.py"), "--max-ticks", "60", "--replay", "r/l0.json",
                    "--save", "--plot", "l0.png", "--html", "l0.html")
    assert code == 0
    assert "Attempt 1:" in out and "crash" in out
    replay = Replay.from_json(work / "r" / "l0.json")
    assert replay.kind == "session" and replay.cars[0].name == "Dummy" and replay.validate() == []
    result = json.loads((work / "results" / "dummy_grand_prix.json").read_text(encoding="utf-8"))  # per track
    assert result["best_lap"] is None and result["track"] == "grand_prix"
    assert (work / "l0.png").stat().st_size > 1000
    assert "HTML" in out                       # either written or "HTML player not installed yet"


def test_drive_l2_quiet_params_and_states(work, repo, capsys):
    code, out = run(capsys, "drive", ex(repo, "l2_planner.py"), "--track", "oval", "--quiet",
                    "--param", "aggression=0.9", "--dump-states", "states.json", "--dump-n", "5")
    assert code == 0
    assert "FINISHED" in out and "tick" not in out.split("Attempt 1:")[0]   # quiet: no per-tick lines
    states = json.loads((work / "states.json").read_text(encoding="utf-8"))
    assert states["format"] == "agentic-gp-states" and len(states["states"]) == 5
    assert "speed" in states["states"][0]


def test_drive_bad_driver_is_a_friendly_error(work, capsys):
    bad = write(work, "bad.py", "import agentic_gp.sim\ndef drive(obs):\n    return 1, 0\n")
    code, out = run(capsys, "drive", bad)
    assert code == 1 and "ERROR" in out and "agentic_gp.sim" in out


# ------------------------------------------------------------------ check / validate / exam
def test_check_passes_examples(work, repo, capsys):
    code, out = run(capsys, "check", ex(repo, "l2_planner.py"))
    assert code == 0 and "PASSED" in out and "Planner" in out


def test_check_fails_on_errors(work, capsys):
    no_drive = write(work, "nodrive.py", "NAME = 'X'\n")
    assert run(capsys, "check", no_drive)[0] == 1
    crashing = write(work, "boom.py", "NAME='Boom'\nAUTHOR='t'\ndef drive(obs):\n    return 1 / 0\n")
    code, out = run(capsys, "check", crashing)
    assert code == 1 and "ZeroDivisionError" in out


def test_validate_table_and_score(work, repo, capsys):
    code, out = run(capsys, "validate", ex(repo, "l2_planner.py"), "--tracks", "oval", "--random", "1",
                    "--seed", "3", "--quiet")
    assert code == 0
    assert "random:3" in out and "Validation score" in out and "finished 2/2" in out


def test_exam_compares_with_reference(work, repo, capsys):
    run(capsys, "drive", ex(repo, "l2_planner.py"), "--quiet", "--dump-states", "s.json", "--dump-n", "6")
    code, out = run(capsys, "exam", ex(repo, "l0_dummy.py"), "--states", "s.json", "--n", "6",
                    "--reference", ex(repo, "l2_planner.py"))
    assert code == 0
    assert "Steering sign agreement" in out and "p95" in out


# ------------------------------------------------------------------ grand prix / final / watch
def test_grand_prix_inprocess_tiny(work, repo, capsys):
    code, out = run(capsys, "grand-prix", ex(repo, "l0_dummy.py"), ex(repo, "l2_planner.py"), "--track", "oval",
                    "--practice", "1", "--laps", "1", "--inprocess", "--out", "races/gp.json")
    assert code == 0
    assert "Qualifying" in out and "Planner" in out
    replay = Replay.from_json(work / "races" / "gp.json")
    assert [c.name for c in replay.cars] == ["Planner", "Dummy"]
    results = json.loads((work / "races" / "gp_results.json").read_text(encoding="utf-8"))
    assert results["standings"][0]["name"] == "Planner"
    assert (work / "races" / "gp_qualifying.json").is_file() and (work / "races" / "gp_standings.md").is_file()


def test_gp_entries_house_and_scoped_params(repo, work):
    entries = cli._gp_entries([ex(repo, "l2_planner.py"), ex(repo, "l0_dummy.py")], True,
                              ["throttle=0.3", "l2_planner:aggression=0.9"])
    by = {Path(e.source).stem: e for e in entries}
    assert len(entries) == len(by)                       # l2_planner is not raced twice
    assert by["l2_planner"].house and not by["l0_dummy"].house
    assert by["l2_planner"].params == {"throttle": "0.3", "aggression": "0.9"}
    assert by["l0_dummy"].params == {"throttle": "0.3"}


def test_final_two_races_and_championship(work, repo, capsys):
    code, out = run(capsys, "final", ex(repo, "l0_dummy.py"), ex(repo, "l2_planner.py"), "--home", "oval",
                    "--mystery", "random:2", "--practice", "0", "--mystery-practice", "0", "--laps", "1",
                    "--no-qualifying", "--inprocess", "--quiet", "--out", "races/final")
    assert code == 0 and "Championship" in out and "sha256" in out
    champ = json.loads((work / "races" / "final" / "championship.json").read_text(encoding="utf-8"))
    assert champ["standings"][0]["name"] == "Planner" and champ["standings"][0]["points"] == 4
    for name in ("home.json", "mystery.json", "championship.md"):
        assert (work / "races" / "final" / name).is_file()


def test_watch_summary_and_unity_install(work, repo, capsys, monkeypatch):
    run(capsys, "drive", ex(repo, "l2_planner.py"), "--track", "oval", "--quiet", "--replay", "w.json")
    code, out = run(capsys, "watch", "w.json")
    assert code == 0 and "P1" in out and "Planner" in out and "--unity" in out
    calls = []
    monkeypatch.setattr(unity, "show_in_unity", lambda path, **kw: calls.append(Path(path)) or True)
    assert run(capsys, "watch", "w.json", "--unity")[0] == 0
    assert calls == [Path("w.json")]
    assert run(capsys, "watch", "missing.json")[0] == 1


# ------------------------------------------------------------------ tracks / leaderboard / doctor / llm-check
def test_tracks_lists_and_plots(work, capsys, monkeypatch):
    from agentic_gp.viz import mpl                     # real plotting is tested in test_track.py
    monkeypatch.setattr(mpl, "plot_track", lambda track, path, **kw: Path(path).write_bytes(b"png"))
    code, out = run(capsys, "tracks", "--random", "1", "--plot", "plots")
    assert code == 0
    for key in ("grand_prix", "oval", "sprint", "speedway", "briefing_demo", "random:1"):
        assert key in out
    assert (work / "plots" / "random_1.png").is_file()


def test_leaderboard_from_saved_results(work, repo, capsys):
    run(capsys, "drive", ex(repo, "l2_planner.py"), "--track", "oval", "--quiet", "--save")
    code, out = run(capsys, "leaderboard")
    assert code == 0 and "Planner" in out
    assert "Planner" in (work / "LEADERBOARD.md").read_text(encoding="utf-8")


def test_doctor_writes_report(work, capsys):
    code, out = run(capsys, "doctor", "--out", "doctor.txt")
    report = (work / "doctor.txt").read_text(encoding="utf-8")
    assert "Simulator" in report and "verdict:" in report and "Python" in report
    assert code in (0, 1)
    assert "sk-" not in report


def test_llm_check_with_mock_provider(work, capsys):
    code, out = run(capsys, "llm-check", "--load", "2", "--calls", "2", "--ramp", "0")
    assert code == 0, out
    assert "mock" in out and "PASS" in out and "p50" in out


def test_help_lists_every_command(capsys):
    assert cli.main([]) == 0
    out = capsys.readouterr().out
    for cmd in ("drive", "check", "validate", "exam", "grand-prix", "final", "watch", "tracks", "leaderboard",
                "doctor", "llm-check"):
        assert cmd in out


# ------------------------------------------------------------------ viz.unity bridge
def test_unity_bridge_installs_atomically(tmp_path, capsys):
    root = tmp_path / "repo"
    (root / "unity" / "RaceViewer" / "Assets").mkdir(parents=True)
    replay = tmp_path / "r.json"
    replay.write_text('{"format": "agentic-gp-replay"}', encoding="utf-8")
    assert unity.find_repo_root(root / "unity") == root
    assert unity.find_viewer(root) is None
    written = unity.install_replay(replay, root)
    assert written == [root / "unity" / "RaceViewer" / "Assets" / "StreamingAssets" / "replays" / "latest.json"]
    assert written[0].read_text(encoding="utf-8") == replay.read_text(encoding="utf-8")
    lines = []
    assert unity.show_in_unity(replay, repo_root=root, log=lines.append)
    assert any("not built" in line for line in lines)
    with pytest.raises(FileNotFoundError):
        unity.install_replay(tmp_path / "nope.json", root)


# ------------------------------------------------------------------ review fixes
LLM_PLAN_DRIVER = """from agentic_gp import llm
NAME = "LLMPlan"
AUTHOR = "x"
KIND = "llm"

class Driver:
    def plan(self, track):
        data = llm.chat_json("Return {\\"target_speed\\": number}")
        self.v = float(data["target_speed"])

    def drive(self, obs):
        return (1.0 if obs["speed"] < self.v else 0.0), obs["angle_to_next_waypoint"] / 45
"""


def test_check_explains_the_mock_llm_and_shows_the_first_traceback(work, capsys):
    code, out = run(capsys, "check", write(work, "d_llm.py", LLM_PLAN_DRIVER))
    assert code == 1
    assert "LLM: mock" in out and "chat_json() returns {}" in out and "--real-llm" in out
    assert "plan() raised: KeyError: 'target_speed' (d_llm.py:9)" in out
    assert "First error, full traceback" in out and 'self.v = float(data["target_speed"])' in out
    assert "hint: the LLM was mocked" in out


@pytest.mark.parametrize("body, needle", [
    ("return 1.0, obs['angle_to_waypoint'] / 45", "did you mean 'angle_to_next_waypoint'?"),
    ("return {'accel': 1.0, 'steer': 0.3}", "without 'throttle'/'steering'"),
    ("return 'full throttle'", "returned str 'full throttle'"),
])
def test_check_fails_drivers_whose_answers_cannot_be_used(work, capsys, body, needle):
    code, out = run(capsys, "check", write(work, "bad.py", f"NAME='Bad'\nAUTHOR='x'\ndef drive(obs):\n    {body}\n"))
    assert code == 1 and needle in out and "FAILED" in out


def test_check_warns_when_the_car_never_moves(work, capsys):
    code, out = run(capsys, "check", write(work, "coast.py", "NAME='C'\nAUTHOR='x'\ndef drive(obs):\n    return 0, 0\n"))
    assert code == 0 and "WARNING the car never moved" in out


def test_mistakes_are_one_line_errors(work, repo, capsys):
    code, out = run(capsys, "drive", ex(repo, "l2_planner.py"), "--track", "monza", "--quiet")
    assert code == 2 and "ERROR: unknown track 'monza'" in out and "Traceback" not in out
    code, out = run(capsys, "drive", ex(repo, "l2_planner.py"), "--param", "aggression=fast", "--quiet")
    assert code == 2 and "--param aggression expects a number (default 1.0), got 'fast'" in out
    code, out = run(capsys, "exam", ex(repo, "l2_planner.py"), "--states", write(work, "bad.json", "{not json"))
    assert code == 1 and "is not a states file" in out
    code, out = run(capsys, "exam", ex(repo, "l2_planner.py"), "--states", "missing.json")
    assert code == 1 and "states file not found" in out and "--dump-states" in out
    code, out = run(capsys, "grand-prix", "nothere_dir", "--inprocess", "--out", "gp/x.json")
    assert code == 1 and "not found: nothere_dir" in out and not (work / "gp").exists()


def test_final_checks_both_tracks_before_racing(work, repo, capsys):
    code, out = run(capsys, "final", ex(repo, "l2_planner.py"), "--home", "oval", "--mystery", "monza",
                    "--inprocess", "--out", "fin")
    assert code == 2 and "--mystery monza: unknown track" in out
    assert not (work / "fin").exists()                    # the Home GP did not run


def test_unknown_param_keys_are_warned_about(work, repo, capsys):
    code, out = run(capsys, "drive", ex(repo, "l2_planner.py"), "--track", "oval", "--quiet",
                    "--param", "agression=1.5")
    assert code == 0 and "unknown PARAMS key(s) 'agression'" in out and "known: aggression" in out


def test_save_keeps_one_result_per_track_and_the_board_splits_tracks(work, repo, capsys):
    import json as _json

    for track in ("oval", "sprint"):
        assert run(capsys, "drive", ex(repo, "l2_planner.py"), "--track", track, "--quiet", "--save")[0] == 0
    weird = write(work, "weird.py", "NAME='Weird'\nAUTHOR='x'\nPARAMS={'limit': float('inf')}\n"
                                    "def drive(obs):\n    return 0.5, obs['angle_to_next_waypoint'] / 30\n")
    assert run(capsys, "drive", weird, "--track", "oval", "--quiet", "--save", "--max-ticks", "20")[0] == 0
    files = sorted(p.name for p in (work / "results").glob("*.json"))
    assert files == ["planner_oval.json", "planner_sprint.json", "weird_oval.json"]
    assert _json.loads((work / "results" / "weird_oval.json").read_text(encoding="utf-8"))["params"] == \
        {"limit": "inf"}
    code, out = run(capsys, "leaderboard")
    assert code == 0 and "## oval" in out and "## sprint" in out
    oval = out.split("## oval")[1].split("## sprint")[0]
    assert "Planner" in oval and "Weird" in oval
    code, out = run(capsys, "leaderboard", "--track", "sprint", "--out", "sprint.md")
    assert "Weird" not in out and "Planner" in out and "## " not in out


def test_llm_load_test_really_runs_in_parallel(monkeypatch, capsys):
    import types

    from agentic_gp import llm

    monkeypatch.setenv("LLM_PROVIDER", "minimax")
    monkeypatch.setenv("MINIMAX_API_KEY", "fake-key")
    inflight = {"now": 0, "max": 0}
    lock = threading.Lock()

    def create(**kw):
        with lock:
            inflight["now"] += 1
            inflight["max"] = max(inflight["max"], inflight["now"])
        time.sleep(0.3)
        with lock:
            inflight["now"] -= 1
        msg = types.SimpleNamespace(content="OK", tool_calls=None, reasoning_content=None, reasoning_details=None)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg, finish_reason="stop")],
                                     usage=types.SimpleNamespace(prompt_tokens=5, completion_tokens=1),
                                     model_extra={})

    fake = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
    monkeypatch.setattr(llm, "_client", lambda cfg: fake)
    llm.set_llm_semaphore(None)
    assert cli._llm_load_test(workers=4, calls=2, ramp=0.0, effort="none")
    out = capsys.readouterr().out
    assert inflight["max"] == 4 and "up to 4 in flight" in out
    assert llm._semaphore()._value == 1                    # the default (1 per process) is back


def test_ramp_accepts_a_seconds_suffix():
    assert cli._seconds_arg("30s") == 30.0 and cli._seconds_arg("2") == 2.0 and cli._seconds_arg("1m") == 60.0
    args = cli.build_parser().parse_args(["llm-check", "--ramp", "30s"])
    assert args.ramp == 30.0


# ------------------------------------------------------------------ review round 3
def test_ramp_units_ms_and_bad_values():
    assert cli._seconds_arg("30ms") == pytest.approx(0.03) and cli._seconds_arg("500 ms") == pytest.approx(0.5)
    assert cli._seconds_arg("2min") == 120.0 and cli._seconds_arg("1.5 s") == 1.5
    for bad in ("fast", "-3s", "nan", "inf"):
        with pytest.raises(Exception, match="expected a duration"):
            cli._seconds_arg(bad)


def test_track_mistakes_print_no_header_and_no_traceback(work, repo, capsys):
    bad = write(work, "bad_track.py", "CONTROL_POINTS = [(0, 0), (100, 0)\n")
    for argv in (["drive", ex(repo, "l2_planner.py"), "--track", f"file:{bad}"],
                 ["grand-prix", ex(repo, "l2_planner.py"), "--track", f"file:{bad}", "--inprocess"],
                 ["drive", ex(repo, "l2_planner.py"), "--track", "monza"],
                 ["grand-prix", ex(repo, "l2_planner.py"), "--track", "monza", "--inprocess"],
                 ["validate", ex(repo, "l2_planner.py"), "--tracks", "oval", "monza", "--random", "0"]):
        code, out = run(capsys, *argv)
        lines = out.strip().splitlines()
        assert code == 2 and len(lines) == 1 and lines[0].startswith("ERROR:"), (argv, out)
        assert "Traceback" not in out
    assert "syntax error on line 1" in run(capsys, "drive", ex(repo, "l2_planner.py"), "--track", f"file:{bad}")[1]


def test_validate_refuses_import_time_llm_calls(work, capsys):
    src = ("from agentic_gp import llm\ntry:\n    llm.chat('at import')\n    print('IMPORT CALL ALLOWED')\n"
           "except llm.LLMNotAllowed:\n    print('IMPORT CALL REFUSED')\n"
           "NAME = 'Imp'\nAUTHOR = 'x'\ndef drive(obs):\n    return 0.5, 0.0\n")
    code, out = run(capsys, "validate", write(work, "imp.py", src), "--tracks", "oval", "--random", "0")
    assert "IMPORT CALL REFUSED" in out and "ALLOWED" not in out
    code, out = run(capsys, "validate", write(work, "imp.py", src), "--tracks", "oval", "--random", "0",
                    "--llm-policy", "open")
    assert "IMPORT CALL ALLOWED" in out


def test_result_files_of_non_ascii_names_do_not_collide():
    names = [cli._result_file("results", n, "sprint").name for n in ("Żółw", "鬼", "幽霊", "Rocket")]
    assert len(set(names)) == 4 and names[3] == "rocket_sprint.json"
    assert all(n.isascii() and n.endswith("_sprint.json") for n in names)


def test_check_stops_an_endless_drive_loop(work, capsys, monkeypatch):
    """`race.py check` on a drive() that never returns: a clear FAILED instead of hanging forever."""
    monkeypatch.setitem(cli.CHECK_SILENCE_S, "drive", 1.0)
    stuck = write(work, "stuck.py", "import time\nNAME='Stuck'\nAUTHOR='t'\n"
                                    "def drive(obs):\n    if obs['tick'] > 3:\n        time.sleep(3600)\n"
                                    "    return 0.5, 0.0\n")
    t0 = time.perf_counter()
    code, out = run(capsys, "check", stuck)
    assert code == 1 and "did not return within" in out and "endless loop" in out
    assert time.perf_counter() - t0 < 60


def test_check_names_house_cars_and_bad_colours(work, capsys):
    fan = write(work, "fan.py", "NAME='Max'\nAUTHOR='t'\nCOLOR='red'\ndef drive(obs):\n    return 0.5, 0.0\n")
    code, out = run(capsys, "check", fan)
    assert code == 0 and "house car's name" in out and "not a hex colour" in out
