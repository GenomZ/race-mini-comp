"""lesson/instructor/grade.py (autograder + scrutineering) and awards.py, on synthetic class data.

Almost everything runs from cached race.py outputs (no subprocess); one test runs a real
`race.py check` on the template.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FAKE_KEY = "sk-fake-test0123456789abcdefghijklmnop"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"instructor_{name}", REPO / "lesson" / "instructor" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod                     # dataclasses look the module up
    spec.loader.exec_module(mod)
    return mod


grade = _load("grade")
awards = _load("awards")

CARD = """DRIVER CARD (fill in before you submit, one line each):
    Approach:      map + speed plan -> learns per segment
    LLM:           none, because the map is enough
    reflect():     slows down where I crashed
    Anatomy:       perceive, act, map, plan, memory, reflect
    Another day:   <what you would build next>
"""


def student_file(folder: Path, handrail: str, stem: str, name: str, *, card: bool = False, edit: bool = False) -> Path:
    text = (REPO / "lesson" / "checkpoints" / handrail).read_text(encoding="utf-8")
    text = re.sub(r'^NAME = "[^"]*"', f'NAME = "{name}"', text, count=1, flags=re.M)
    text = re.sub(r'^AUTHOR = "[^"]*"', f'AUTHOR = "{stem}"', text, count=1, flags=re.M)
    if card:
        text = text.replace('"""', '"""' + CARD + "\n", 1)
    if edit:
        text += "\n\ndef _my_helper(x):\n    return 2 * x\n"
    p = folder / f"{stem}.py"
    p.write_text(text, encoding="utf-8")
    return p


def seed_run(runs: Path, path: Path, kind: str, output: str, code: int = 0, tag: str = "") -> None:
    """Pretend race.py already ran on this exact file version (grade.py's cache format)."""
    runs.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    (runs / f"{path.stem}.{kind}.json").write_text(json.dumps(
        {"file": str(path), "sha256": sha, "tag": tag, "returncode": code, "output": output}), encoding="utf-8")


def validate_output(rows: dict[str, tuple]) -> str:
    """Text in the exact layout race.py validate prints (rows: track -> (timed, crashes, status))."""
    lines = ["Validating x.py on 5 tracks", "", "track          practice  timed lap  crashes  penalties  status  wall s",
             "-------------  --------  ---------  -------  ---------  ------  ------"]
    for track, (timed, crashes, status) in rows.items():
        lines.append(f"{track:13s}  {timed:>8s}  {timed:>9s}  {crashes:>7}  {'0.0':>9s}  {status}  1.0")
    lines += ["", "Validation score (mean timed lap, DNF = 300 s): 50.00 s   | finished 5/5"]
    return "\n".join(lines)


def car(name, stem, position, race_time, *, house=False, kind="code", crashes=0, best_lap=None, llm_calls=0,
        quali=None, practice=(), status=""):
    """(standings row, qualifying row, drivers row) of a synthetic grand prix result file."""
    s = {"name": name, "author": stem, "kind": kind, "house": house, "position": position, "race_time": race_time,
         "status": status, "best_lap": best_lap, "crashes": crashes, "llm_calls": llm_calls, "llm_tokens": 0,
         "waypoints_passed": 48}
    q = {"name": name, "qualifying_time": quali, "practice_times": list(practice), "house": house}
    atts = [{"phase": "practice", "race_time": t, "completed": t is not None} for t in practice]
    atts.append({"phase": "qualifying", "race_time": quali, "completed": quali is not None})
    d = {"name": name, "source": f"races/grid/{stem}.py", "attempts": atts, "llm": {"calls": llm_calls}}
    return s, q, d


def gp_file(path: Path, rows: list[tuple]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"format": "agentic-gp-grand-prix", "version": 1, "standings": [r[0] for r in rows],
                                "qualifying": [r[1] for r in rows], "drivers": [r[2] for r in rows]}),
                    encoding="utf-8")
    return path


PASSED = "Checking x.py\n  static check: OK\nPASSED - ready to race (python race.py validate x.py)\n"
ALL_CLEAN = {t: ("50.00", 0, "ok") for t in grade.PUBLIC_TRACKS}


# ============================================================================ small pieces
def test_roles_and_handles():
    assert grade.role_of(Path("ada.py")) == "main"
    assert grade.role_of(Path("ada_pilot.py")) == "pilot"
    assert grade.role_of(Path("ada_fly.py")) == "fly"
    assert grade.role_of(Path("_old.py")) == "ignore"
    assert grade.handle_of(Path("Ada_fly.py")) == "ada"


def test_key_leak_never_reveals_the_key(tmp_path):
    assert grade.key_leak(f'KEY = "{FAKE_KEY}"', [FAKE_KEY]).startswith("contains the course API key")
    assert "looks like an API key" in grade.key_leak('k = "sk-abcdefghijklmnopqrstuvwx"', [])
    assert grade.key_leak("steering = 0.5", [FAKE_KEY]) is None
    p = tmp_path / "leaky.py"
    p.write_text(f'NAME = "Leaky"\nAUTHOR = "leaky"\nKIND = "code"\nKEY = "{FAKE_KEY}"\n'
                 "def drive(obs):\n    return 0.5, 0.0\n", encoding="utf-8")
    row = grade.scrutineer_file(p, [FAKE_KEY], tmp_path / "runs")
    assert row["verdict"] == "FAIL" and row["smoke"] == "skipped"       # no smoke test for a leaking file
    assert FAKE_KEY not in json.dumps(row)


def test_verbatim_handrail_is_detected(tmp_path):
    copy = student_file(tmp_path, "ch3_learner.py", "ada", "Ada", card=True)       # metadata + docstring only
    assert grade.verbatim_copy_of(copy) == "lesson/checkpoints/ch3_learner.py"
    own = student_file(tmp_path, "ch3_learner.py", "bob", "Bob", edit=True)
    assert grade.verbatim_copy_of(own) is None


def test_driver_card_ignores_placeholders(tmp_path):
    p = student_file(tmp_path, "ch2_planner.py", "ada", "Ada", card=True)
    card = grade.driver_card(p)
    assert set(card) == {"Approach", "LLM", "reflect()", "Anatomy"}           # "Another day" is still <...>
    assert grade.driver_card(REPO / "drivers" / "template.py") == {}


def test_parse_validate_table():
    out = validate_output({"grand_prix": ("52.50", 0, "ok"), "oval": ("DNF", 3, "DNF (max ticks)"),
                           "sprint": ("41.40", 2, "ok")})
    rows = grade.parse_validate(out)
    assert rows["grand_prix"] == {"finished": True, "time": 52.5, "crashes": 0, "status": "ok"}
    assert rows["oval"]["finished"] is False and rows["oval"]["status"] == "DNF (max ticks)"
    assert rows["sprint"]["crashes"] == 2
    assert grade.parse_validate("nothing here") == {}


# ============================================================================ scrutineer + grid
def test_scrutineer_grid_takes_only_passed_main_files(tmp_path, monkeypatch):
    monkeypatch.setattr(grade, "known_keys", lambda env_file=None: [FAKE_KEY])
    sub = tmp_path / "submissions"
    sub.mkdir()
    runs = tmp_path / "grading" / "runs"
    for p in (student_file(sub, "ch2_planner.py", "ada", "Ada"), student_file(sub, "ch3_learner.py", "bob", "Bob"),
              student_file(sub, "ch7_fly_fixed.py", "ada_fly", "Ada Fly"),
              student_file(sub, "ch4_pilot.py", "bob_pilot", "Bob Pilot")):
        seed_run(runs, p, "check", PASSED)
    (sub / "gus.py").write_text('NAME = "Gus"\nAUTHOR = "gus"\nimport requests\ndef drive(obs):\n    return 1, 0\n',
                                encoding="utf-8")
    (sub / "dup.py").write_text('NAME = "Ada"\nAUTHOR = "dup"\nKIND = "code"\ndef drive(obs):\n    return 0.5, 0\n',
                                encoding="utf-8")
    seed_run(runs, sub / "dup.py", "check", PASSED)
    os.utime(sub / "dup.py", (time.time() + 60, time.time() + 60))          # uploaded after ada.py
    (sub / "_notes.py").write_text("x = 1\n", encoding="utf-8")
    grid = tmp_path / "grid"
    grid.mkdir()
    (grid / "stale.py").write_text("x = 1\n", encoding="utf-8")
    code = grade.main(["scrutineer", "--submissions", str(sub), "--all", "--grid", str(grid),
                       "--out", str(tmp_path / "grading")])
    assert code == 1                                                        # gus and the duplicate NAME fail
    # ada.py was uploaded first: it keeps the NAME and races; the later dup.py fails
    assert sorted(p.name for p in grid.glob("*.py")) == ["ada.py", "bob.py"]
    rows = {r["file"]: r for r in json.loads((tmp_path / "grading" / "scrutineering.json").read_text())}
    assert rows["gus.py"]["verdict"] == "FAIL" and "requests" in rows["gus.py"]["problems"][0]
    assert any("duplicate NAME" in m and "ada.py was uploaded first" in m for m in rows["dup.py"]["problems"])
    assert rows["ada.py"]["verdict"] == "PASS" and any("keeps the name" in w for w in rows["ada.py"]["warnings"])
    assert any("unchanged copy of lesson/checkpoints/ch3_learner.py" in w for w in rows["bob.py"]["warnings"])
    assert not any("plain handle" in w for w in rows["bob_pilot.py"]["warnings"])
    assert rows["ada_fly.py"]["verdict"] == "PASS" and "_notes.py" not in rows

    fly_grid = tmp_path / "flycup"
    grade.main(["scrutineer", "--submissions", str(sub), "--fly", "--grid", str(fly_grid),
                "--out", str(tmp_path / "grading")])
    assert [p.name for p in fly_grid.glob("*.py")] == ["ada_fly.py"]


def test_scrutineer_rejects_house_names_and_cleans_grid_names(tmp_path, monkeypatch):
    monkeypatch.setattr(grade, "known_keys", lambda env_file=None: [FAKE_KEY])
    sub = tmp_path / "submissions"
    sub.mkdir()
    runs = tmp_path / "g" / "runs"
    fan = student_file(sub, "ch2_planner.py", "max_fan", "Max", edit=True)
    odd = student_file(sub, "ch2_planner.py", "Ann Lee (v3)", "Ann", edit=True)
    for p in (fan, odd):
        seed_run(runs, p, "check", PASSED)
    grid = tmp_path / "grid"
    code = grade.main(["scrutineer", "--submissions", str(sub), "--grid", str(grid), "--out", str(tmp_path / "g")])
    rows = {r["file"]: r for r in json.loads((tmp_path / "g" / "scrutineering.json").read_text())}
    assert code == 1 and rows["max_fan.py"]["verdict"] == "FAIL"
    assert any("house car's name" in m for m in rows["max_fan.py"]["problems"])
    assert rows["Ann Lee (v3).py"]["verdict"] == "PASS" and rows["Ann Lee (v3).py"]["handle"] == "ann_lee_v3"
    assert any("ann_lee_v3.py" in w for w in rows["Ann Lee (v3).py"]["warnings"])
    assert [p.name for p in grid.glob("*.py")] == ["ann_lee_v3.py"]


def test_grade_gives_no_working_points_to_a_rejected_file(tmp_path, monkeypatch):
    """A key-like string fails scrutineering: the file never races, so it earns no working points."""
    monkeypatch.setattr(grade, "known_keys", lambda env_file=None: [FAKE_KEY])
    sub = tmp_path / "submissions"
    sub.mkdir()
    p = student_file(sub, "ch2_planner.py", "uma", "Uma", edit=True)
    p.write_text(p.read_text(encoding="utf-8") + f"\nSECRET = '{FAKE_KEY}'\n", encoding="utf-8")
    out = tmp_path / "grading"
    seed_run(out / "runs", p, "check", PASSED)
    none = str(tmp_path / "none")
    grade.main(["grade", "--submissions", str(sub), "--snapshots", none, "--final", none, "--results", none,
                "--evidence", none, "--no-run", "--out", str(out)])
    with open(out / "grades.csv", newline="", encoding="utf-8") as fh:
        row = next(r for r in csv.DictReader(fh) if r["handle"] == "uma")
    assert float(row["working"]) == 0.0 and "failed scrutineering" in row["flags"]


@pytest.mark.slow
def test_scrutineer_runs_the_real_check(tmp_path, monkeypatch):
    monkeypatch.setattr(grade, "known_keys", lambda env_file=None: [FAKE_KEY])
    sub = tmp_path / "submissions"
    sub.mkdir()
    shutil.copy(REPO / "drivers" / "template.py", sub / "tina.py")
    code = grade.main(["scrutineer", "--submissions", str(sub), "--out", str(tmp_path / "grading")])
    rows = json.loads((tmp_path / "grading" / "scrutineering.json").read_text())
    assert code == 0 and rows[0]["smoke"] == "PASSED", rows
    assert (tmp_path / "grading" / "runs" / "tina.check.json").is_file()


# ============================================================================ the rubric
def make_class(tmp_path: Path) -> dict:
    sub = tmp_path / "submissions"
    sub.mkdir()
    out = tmp_path / "grading"
    runs = out / "runs"
    tag = ",".join(grade.PUBLIC_TRACKS)
    ada = student_file(sub, "ch3_learner.py", "ada", "Ada", card=True, edit=True)
    bob = student_file(sub, "ch1_reflex.py", "bob", "Bob")                      # verbatim handrail, no card
    student_file(sub, "ch4_pilot.py", "ada_pilot", "Ada Pilot")
    seed_run(runs, ada, "check", PASSED)
    seed_run(runs, ada, "validate", validate_output(ALL_CLEAN), tag=tag)
    seed_run(runs, bob, "check", PASSED)
    seed_run(runs, bob, "validate", validate_output({**ALL_CLEAN, "sprint": ("61.00", 2, "ok"),
                                                     "oval": ("DNF", 9, "DNF (max ticks)")}), tag=tag)
    snaps = tmp_path / "snapshots"
    gp_file(snaps / "snap1_results.json", [car("Ada", "ada", 1, 55.0, practice=(60.0, 56.0), quali=None),
                                           car("Max", "l6_llm_crew_chief", 2, 44.0, house=True)])
    gp_file(snaps / "snap2_results.json", [car("Ada", "ada", 1, 50.0), car("Bob", "bob", 2, 66.7)])
    final = tmp_path / "final"
    gp_file(final / "home_results.json", [car("Max", "l6_llm_crew_chief", 1, 140.0, house=True, quali=44.0),
                                          car("Ada", "ada", 2, 150.0, quali=48.0, practice=(52.0, 49.0)),
                                          car("Bob", "bob", 3, 200.0, quali=66.7, practice=(66.7, 66.7))])
    gp_file(final / "mystery_results.json", [car("Ada", "ada", 1, 130.0, quali=43.0, practice=(47.0, 45.0, 44.0)),
                                             car("Max", "l6_llm_crew_chief", 2, 131.0, house=True, quali=42.0),
                                             car("Bob", "bob", 3, None, status="DNF (max ticks)")])
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "ada_engineer_report.png").write_bytes(b"\x89PNG")
    return {"sub": sub, "out": out, "snaps": snaps, "final": final, "evidence": evidence}


def grade_args(c: dict, *extra: str) -> list[str]:
    return ["grade", "--submissions", str(c["sub"]), "--snapshots", str(c["snaps"]), "--final", str(c["final"]),
            "--results", str(c["sub"] / "no_results"), "--evidence", str(c["evidence"]), "--out", str(c["out"]),
            "--no-run", *extra]


def test_grade_rubric_from_cached_runs(tmp_path):
    c = make_class(tmp_path)
    assert grade.main(grade_args(c)) == 0
    rows = {r["handle"]: r for r in csv.DictReader(open(c["out"] / "grades.csv", encoding="utf-8"))}
    assert set(rows) == {"ada", "bob"}                                        # the pilot file is not a student
    ada, bob = rows["ada"], rows["bob"]
    assert float(ada["working"]) == 40.0
    assert float(ada["design"]) == pytest.approx(30 * (8 + 5 + 2 + 3 + 6 + 6 * 4 / 5) / 30, abs=0.1)
    assert float(ada["process"]) == 20.0                                      # 2/2 snapshots + screenshot + pilot
    assert float(ada["result"]) == 7.5               # <= median twice, ahead of Max only in the mystery GP
    # bob: check 15 + finished 4/5 * 10 + clean 3/5 * 15; verbatim handrail -> design 0; 1 of 2 snapshots
    assert float(bob["working"]) == pytest.approx(15 + 8 + 9, abs=0.1)
    assert float(bob["design"]) == 0.0 and "verbatim copy of lesson/checkpoints/ch1_reflex.py" in bob["flags"]
    assert float(bob["process"]) == pytest.approx(20 * 6 / 20, abs=0.1)
    assert float(bob["result"]) == 0.0
    md = (c["out"] / "grades.md").read_text(encoding="utf-8")
    assert "## ada" in md and "oval DNF" in md


def test_grade_overrides_and_missing_runs(tmp_path):
    c = make_class(tmp_path)
    ov = tmp_path / "overrides.csv"
    ov.write_text("handle,criterion,points,note\nbob,design,12,pit interview: own racing line\n", encoding="utf-8")
    for f in (c["out"] / "runs").glob("bob.validate.json"):
        f.unlink()                                                             # --no-run: missing stays missing
    grade.main(grade_args(c, "--overrides", str(ov)))
    rows = {r["handle"]: r for r in csv.DictReader(open(c["out"] / "grades.csv", encoding="utf-8"))}
    assert float(rows["bob"]["design"]) == 12.0 and "pit interview" in rows["bob"]["flags"]
    assert float(rows["bob"]["working"]) == 15.0                              # check only: validate not run
    bad = tmp_path / "bad.csv"
    bad.write_text("handle,criterion,points,note\nbob,style,5,x\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        grade.main(grade_args(c, "--overrides", str(bad)))


# ============================================================================ awards
def test_awards_from_final_results(tmp_path):
    final = tmp_path / "final"
    gp_file(final / "home_results.json", [
        car("Max", "l6_llm_crew_chief", 1, 140.0, house=True, kind="llm", llm_calls=6, quali=44.0, best_lap=45.0),
        car("Ada", "ada", 2, 150.0, kind="llm", llm_calls=4, quali=47.0, practice=(50.0, 48.0), best_lap=46.5),
        car("Bob", "bob", 3, 160.0, quali=49.0, practice=(49.0, 49.0), best_lap=52.0),
        car("Cy", "cy", 4, 170.0, crashes=1, quali=52.0, practice=(53.0, 52.0), best_lap=53.0)])
    gp_file(final / "mystery_results.json", [
        car("Bob", "bob", 1, 120.0, quali=40.0, practice=(41.0, 40.5, 40.2), best_lap=39.0),
        car("Max", "l6_llm_crew_chief", 2, 121.0, house=True, kind="llm", llm_calls=6, quali=39.5, best_lap=39.5),
        car("Ada", "ada", 3, 125.0, kind="llm", llm_calls=2, quali=41.0, practice=(46.0, 43.0, 42.0), best_lap=40.0),
        car("Cy", "cy", 4, 140.0, quali=50.0, practice=(60.0, 55.0, 52.0), best_lap=45.0)])
    (final / "mystery.json").write_text(json.dumps({"cars": [
        {"name": "Bob", "events": [{"kind": "zone"}, {"kind": "lap"}], "messages": [{"text": "hi"}] * 3},
        {"name": "Ada", "events": [], "messages": []}]}), encoding="utf-8")
    snap1 = gp_file(tmp_path / "snap1_results.json", [car("Bob", "bob", 1, 66.7), car("Cy", "cy", 2, None, status="DNF")])

    got = {a.title: a for a in awards.compute_awards(final, snap1)}
    assert got["Champion"].winners == ["Bob"]                                 # 2 + 3 points; Max is not scored
    assert got["Home GP winner"].winners == ["Ada"] and "Max (house) finished ahead" in got["Home GP winner"].detail
    assert got["Pole (Home)"].winners == ["Ada"] and "real pole: Max" in got["Pole (Home)"].detail
    assert got["Fastest race lap (Mystery)"].winners == ["Bob"]
    assert got["Beat the House"].winners == ["Bob"]
    assert got["Best Learner"].winners == ["Ada"]                             # 46 -> 41; Cy is > 10 % off pole
    assert got["Cleanest Driver"].winners == ["Ada"]                          # Bob has a zone penalty, Cy a crash
    assert got["LLM Class winner"].winners == ["Ada"]
    assert got["Most Efficient"].winners == ["Ada"]                           # 2 calls on the mystery weekend
    assert got["Biggest Improvement"].winners == ["Cy"]                       # DNF (300 s) -> 52.0 s
    assert got["Best Fly Brain"].winners == []
    assert got["Best Radio"].jury and "Bob: 3 radio messages (Mystery)" in got["Best Radio"].data
    md = awards.to_markdown(list(got.values()), final)
    assert "| Champion | Bob |" in md and "(jury / vote" in md


def test_awards_cli_writes_markdown(tmp_path):
    final = tmp_path / "final"
    gp_file(final / "home_results.json", [car("Ada", "ada", 1, 150.0, quali=47.0, practice=(50.0, 48.0))])
    out = tmp_path / "awards.md"
    assert awards.main(["--final", str(final), "--out", str(out)]) == 0
    assert "| Home GP winner | Ada |" in out.read_text(encoding="utf-8")
    with pytest.raises(SystemExit):
        awards.main(["--final", str(tmp_path / "nothing")])


def test_no_improvement_wins_nothing(tmp_path):
    final = tmp_path / "final"
    gp_file(final / "home_results.json", [car("Ada", "ada", 1, 150.0, quali=50.0, practice=(50.0, 50.0))])
    snap1 = gp_file(tmp_path / "snap1_results.json", [car("Ada", "ada", 1, 50.0)])
    got = {a.title: a for a in awards.compute_awards(final, snap1)}
    assert got["Biggest Improvement"].winners == [] and got["Biggest Improvement"].detail == "nobody got faster"
    assert got["Best Learner"].winners == []
