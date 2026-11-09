import importlib.util
import json
import multiprocessing as mp
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from agentic_gp import llm
from agentic_gp.race import DriverEntry, GrandPrixConfig, Replay, find_driver_files, run_grand_prix
from agentic_gp.sim import PhysicsConfig
from agentic_gp.race.leaderboard import championship, championship_markdown, standings_markdown

ROOT = Path(__file__).resolve().parent.parent


def unity_validator():
    """The Unity team's replay validator (unity/tools/make_sample_replay.py), used read-only."""
    path = ROOT / "unity" / "tools" / "make_sample_replay.py"
    if not path.is_file():
        pytest.skip("unity tools not present")
    spec = importlib.util.spec_from_file_location("agp_unity_sample_replay", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.validate_replay

PLANNER = """
from agentic_gp import driverkit as dk

NAME = "{name}"
AUTHOR = "tests"

class Driver:
    PARAMS = {{"aggression": 1.0}}

    def plan(self, track):
        self.follower = dk.PathFollower.from_track(track, aggression=self.PARAMS["aggression"])
        self.lessons = 0

    def drive(self, obs):
        {extra}
        return self.follower.control(obs)

    def reflect(self, report):
        self.lessons += 1
"""

RAISES = """
NAME = "Raises"
AUTHOR = "tests"

def drive(obs):
    return 1 / 0
"""

LOAD_ERROR = """
NAME = "Broken"
AUTHOR = "tests"
def drive(obs)
    return 1, 0
"""


def planner_source(name, extra="pass", imports=""):
    return imports + textwrap.dedent(PLANNER.format(name=name, extra=extra))


def write(dirpath: Path, name: str, source: str) -> Path:
    dirpath.mkdir(parents=True, exist_ok=True)
    p = dirpath / name
    p.write_text(textwrap.dedent(source), encoding="utf-8")
    return p


@pytest.fixture(autouse=True)
def mock_llm(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("LLM_QUIET", "1")
    llm.set_mock_responder(None)


def test_find_driver_files(tmp_path):
    write(tmp_path, "b.py", RAISES)
    write(tmp_path, "a.py", RAISES)
    write(tmp_path, "_helper.py", RAISES)
    write(tmp_path, "notes.txt", "x")
    assert [p.name for p in find_driver_files(tmp_path)] == ["a.py", "b.py"]
    assert find_driver_files(tmp_path / "a.py") == [tmp_path / "a.py"]


def test_grand_prix_in_process(tmp_path):
    d = tmp_path / "grid"
    fast = write(d, "fast.py", planner_source("Fast"))
    slow = write(d, "slow.py", planner_source("Slow"))
    write(d, "raises.py", RAISES)
    write(d, "broken.py", LOAD_ERROR)
    updates = []
    cfg = GrandPrixConfig(track="oval", practice_attempts=1, race_laps=2, title="Test GP",
                          physics=PhysicsConfig(max_ticks=120))     # a clean oval lap takes ~55 ticks
    sources = [DriverEntry(fast, house=True), DriverEntry(slow, params={"aggression": 0.7}),
               d / "raises.py", d / "broken.py"]
    gp = run_grand_prix(sources, cfg, progress=updates.append, inprocess=True)

    names = [r["name"] for r in gp.standings]
    assert names[:2] == ["Fast", "Slow"]
    by = {r["name"]: r for r in gp.standings}
    assert by["Fast"]["status"] == "" and by["Fast"]["laps_completed"] == 2 and by["Fast"]["gap"] == 0.0
    assert by["Fast"]["house"] is True and by["Slow"]["house"] is False
    assert by["Slow"]["gap"] > 0 and by["Slow"]["gap_text"].startswith("+")
    assert by["Raises"]["status"].startswith("DNF (driver crashed: ZeroDivisionError")
    assert by["broken"]["status"] == "DNF (load error)"
    assert any(e["driver"] == "broken" and "syntax" in e["error"] for e in gp.errors)

    q = gp.qualifying
    assert [r["name"] for r in q[:2]] == ["Fast", "Slow"] and q[0]["qualifying_time"] is not None
    assert by["Fast"]["grid"] == 1 and by["Fast"]["pole"] and by["Slow"]["grid"] == 2
    assert q[0]["attempts"] == 2 and len(q[0]["practice_times"]) == 1

    for rep in (gp.replay, gp.qualifying_replay):
        assert rep.validate() == []
        assert unity_validator()(rep.to_dict(extended=False)) == []
    assert gp.replay.kind == "race" and gp.replay.laps == 2 and gp.qualifying_replay.laps == 1
    assert gp.replay.cars[0].qualifying_time == q[0]["qualifying_time"]

    phases = {u.get("phase") for u in updates if u["type"] == "progress"}
    assert {"load", "plan", "practice", "qualifying", "race"} <= phases
    assert sum(u["type"] == "done" for u in updates) == 3        # the broken file never got that far

    paths = gp.save(tmp_path / "out", "test_gp")
    assert set(paths) == {"replay", "qualifying", "results"}
    assert Replay.from_json(paths["replay"]).validate() == []
    saved = json.loads(paths["results"].read_text(encoding="utf-8"))
    assert saved["standings"][0]["name"] == "Fast" and saved["track"] == "oval"
    md = standings_markdown(gp)
    assert "Fast (house)" in md and "Pole position: **Fast**" in md

    champ = championship([gp, gp])
    assert champ[0]["name"] == "Slow"                  # the house car is not scored
    assert champ[0]["points"] == 2 * 3 and all(r["name"] != "Fast" for r in champ)
    assert "| 1 | Slow |" in championship_markdown(champ, ["Home", "Mystery"])


HANG = """
import time
NAME = "Hang"
AUTHOR = "tests"

def drive(obs):
    if obs["tick"] >= 3:
        while True:
            time.sleep(0.05)
    return 1.0, 0.0
"""

DIES_IN_RACE = "import os\n" + planner_source(
    "Dies", extra='if obs["phase"] == "race" and obs["tick"] == 5:\n            os._exit(3)')

LLM_IN_DRIVE = "from agentic_gp import llm\n" + planner_source(
    "Chatty", extra='if obs["tick"] == 0:\n            llm.chat("which way?")')


def test_grand_prix_with_isolated_processes(tmp_path):
    d = tmp_path / "grid"
    files = [write(d, "fast.py", planner_source("Fast")), write(d, "hang.py", HANG),
             write(d, "dies.py", DIES_IN_RACE), write(d, "chatty.py", LLM_IN_DRIVE)]
    # Chatty also asks the LLM twice in plan() (allowed: <= 2 calls)
    chatty = files[3].read_text(encoding="utf-8").replace(
        "self.lessons = 0", "self.lessons = 0\n        llm.chat('plan?')\n        llm.chat('really?')")
    files[3].write_text(chatty, encoding="utf-8")
    threads = set()

    def progress(update):
        threads.add(threading.get_ident())

    cfg = GrandPrixConfig(track="oval", practice_attempts=1, race_laps=2, drive_timeout_s=0.5, jobs=4, ramp_s=0.1,
                          physics=PhysicsConfig(max_ticks=120),
                          log_dir=str(tmp_path / "logs"), llm_log_dir=str(tmp_path / "llm_logs"))
    t0 = time.monotonic()
    gp = run_grand_prix(files, cfg, progress=progress)
    elapsed = time.monotonic() - t0

    assert mp.active_children() == []                       # no zombies, whatever the drivers did
    assert threads == {threading.get_ident()}               # progress() runs in the calling thread
    by = {r["name"]: r for r in gp.standings}
    assert gp.standings[0]["name"] in ("Fast", "Chatty") and by["Fast"]["status"] == ""
    assert by["Hang"]["status"] == "DNF (stalled)" and by["Hang"]["laps_completed"] == 0
    assert by["Dies"]["status"].startswith("DNF (driver crashed: driver process exited (exit code 3)")
    dies_car = next(c for c in gp.replay.cars if c.name == "Dies")
    assert dies_car.tick[-1] == 5 and len(dies_car.t) > 1      # partial race replay up to the crash
    assert by["Dies"]["qualifying_time"] is not None            # it did qualify before dying

    assert by["Chatty"]["status"] == "" and by["Chatty"]["llm_calls"] == 2
    chatty_errors = [e for e in gp.errors if e["driver"] == "Chatty"]
    assert chatty_errors and "LLMNotAllowed" in chatty_errors[0]["error"]
    logs = list((tmp_path / "llm_logs").glob("*.jsonl"))
    assert [p.stem for p in logs] == ["Chatty"]
    records = [json.loads(line) for line in logs[0].read_text(encoding="utf-8").splitlines()]
    assert sum(r["error"] is None for r in records) == 2 and any("not allowed" in (r["error"] or "") for r in records)

    hang_log = next(x["log"] for x in gp.drivers if x["name"] == "Hang")
    assert Path(hang_log).is_file() and Path(hang_log).parent == tmp_path / "logs"
    for rep in (gp.replay, gp.qualifying_replay):
        assert rep.validate() == []
        assert unity_validator()(rep.to_dict(extended=False)) == []
    assert elapsed < 120


def test_stop_event_before_start(tmp_path):
    stop = threading.Event()
    stop.set()
    gp = run_grand_prix([write(tmp_path, "r.py", RAISES)], GrandPrixConfig(track="oval"), stop_event=stop)
    assert gp.standings[0]["status"] == "DNF (not started)" and gp.replay.validate() == []
    assert mp.active_children() == []


# ------------------------------------------------------------------ review fixes
def test_missing_paths_are_an_error_not_a_dnf(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        find_driver_files([tmp_path / "nothere"])
    with pytest.raises(FileNotFoundError):
        run_grand_prix([str(tmp_path / "typo_dir")], GrandPrixConfig(track="oval"), inprocess=True)
    (tmp_path / "empty").mkdir()
    assert find_driver_files(tmp_path / "empty") == []


def test_two_cars_with_the_same_name_keep_both_sessions(tmp_path):
    d = tmp_path / "twins"
    files = [write(d, "a.py", planner_source("Twin")), write(d, "b.py", planner_source("Twin"))]
    gp = run_grand_prix(files, GrandPrixConfig(track="oval", practice_attempts=0, race_laps=1,
                                               llm_log_dir=str(tmp_path / "llm")), inprocess=True)
    assert sorted(gp.sessions) == ["Twin", "Twin-2"]
    assert sorted(c.id for c in gp.replay.cars) == ["twin", "twin-2"]


SLOW_PLAN = """
import time
NAME = "Slow"
AUTHOR = "tests"

def plan(track):
    time.sleep(60)

def drive(obs):
    return 1.0, 0.0
"""


def test_stop_does_not_wait_for_a_driver_busy_in_plan(tmp_path):
    stop = threading.Event()
    seen = {}

    def progress(u):
        if u.get("phase") == "load" and "t" not in seen:
            seen["t"] = time.monotonic()

    def stopper():
        while "t" not in seen:
            time.sleep(0.05)
        time.sleep(2.0)                                     # the driver is loaded and inside plan() by now
        seen["stop"] = time.monotonic()
        stop.set()

    threading.Thread(target=stopper, daemon=True).start()
    gp = run_grand_prix([write(tmp_path, "slow.py", SLOW_PLAN)], GrandPrixConfig(track="oval", ramp_s=0),
                        progress=progress, stop_event=stop)
    assert time.monotonic() - seen["stop"] < 10             # before the fix: plan's 120 s time limit
    assert gp.standings[0]["status"] == "DNF (stopped)"
    assert mp.active_children() == []


# ------------------------------------------------------------------ review round 3
def test_watchdog_during_loading_is_a_loading_timeout_not_stopped(tmp_path):
    """The watchdog stopping a driver that is still loading used to read "DNF (stopped)", errors: []."""
    from agentic_gp.race import grand_prix as gpmod
    from agentic_gp.sim.track import make_track

    path = write(tmp_path, "slow.py", "import time\ntime.sleep(30)\n" + planner_source("Slow"))
    stop = threading.Event()
    stop.set()                                                    # the watchdog already fired ...
    holder = {"watchdog": True}                                   # ... and said so
    updates: list[dict] = []
    out = gpmod._race_driver(0, DriverEntry(str(path)), GrandPrixConfig(track="oval"), make_track("oval"),
                             updates.append, stop, None, False, tmp_path, holder, {})
    assert out.status == "DNF (timeout: loading)" and "no sign of life" in out.error
    assert any(u["type"] == "error" for u in updates)


def test_non_ascii_names_get_distinct_result_and_log_names():
    from agentic_gp.race.replay import slug

    names = ["Żółw", "Zolw", "鬼", "幽霊", "Rocket 2"]
    slugs = [slug(n) for n in names]
    assert len(set(slugs)) == len(slugs)
    assert slugs[0].startswith("zolw-") and slugs[1] == "zolw" and slugs[2].startswith("car-")
    assert slugs[4] == "rocket-2" and slug("") == "car"
    assert all(s.isascii() and s == s.lower() for s in slugs)


def test_folders_skip_side_files_unless_they_are_a_fly_grid(tmp_path):
    """A class folder races main files only (Ch4 pilots never, Ch7 flies only in a folder of flies)."""
    sub = tmp_path / "sub"
    for name in ("ada.py", "ada_pilot.py", "ada_fly.py", "bob.py"):
        write(sub, name, RAISES)
    skipped: list = []
    assert [p.name for p in find_driver_files(sub, skipped=skipped)] == ["ada.py", "bob.py"]
    assert sorted(p.name for p in skipped) == ["ada_fly.py", "ada_pilot.py"]
    flies = tmp_path / "flycup"
    write(flies, "ada_fly.py", RAISES)
    write(flies, "bob_fly.py", RAISES)
    assert [p.name for p in find_driver_files(flies)] == ["ada_fly.py", "bob_fly.py"]
    assert find_driver_files(sub / "ada_pilot.py") == [sub / "ada_pilot.py"]      # a named file always races


def test_duplicate_names_house_car_keeps_its_name_and_logs_match_ids(tmp_path):
    """A student car called "Max": the house car keeps "Max" (log Max.jsonl, id max) and the student
    gets "Max-2" everywhere (log Max-2.jsonl, id max-2) - whatever order the files were given in."""
    student = write(tmp_path / "s", "fan.py", planner_source("Max"))
    house = write(tmp_path / "h", "house.py", planner_source("Max"))
    gp = run_grand_prix([DriverEntry(student), DriverEntry(house, house=True)],
                        GrandPrixConfig(track="oval", practice_attempts=0, race_laps=1, qualifying=False),
                        inprocess=True)
    by_source = {Path(d["source"]).name: d for d in gp.drivers}
    assert by_source["house.py"]["llm_log"] == "Max.jsonl" and by_source["house.py"]["id"] == "max"
    assert by_source["fan.py"]["llm_log"] == "Max-2.jsonl" and by_source["fan.py"]["id"] == "max-2"
    ids = {c.id: c.house for c in gp.replay.cars}
    assert ids == {"max": True, "max-2": False}
