import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from agentic_gp.race import Replay, ReplayCar, SessionConfig, car_state_at, run_session, standings_at
from agentic_gp.race.replay import (
    ReplayTrack, build_replay, classify_cars, placeholder_car, replay_car_from_trace, validate_replay_dict,
)
from agentic_gp.sim import PhysicsConfig, RaceEnv, make_track

ROOT = Path(__file__).resolve().parent.parent


def unity_validator():
    """unity/tools/make_sample_replay.py: the validator the Unity viewer team ships (read-only use)."""
    path = ROOT / "unity" / "tools" / "make_sample_replay.py"
    if not path.is_file():
        pytest.skip("unity tools not present")
    spec = importlib.util.spec_from_file_location("agp_unity_sample_replay", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.validate_replay


@pytest.fixture(scope="module")
def crashy_replay():
    def wall_lover(obs):
        return 1.0, 0.35

    res = run_session(wall_lover, SessionConfig(track="oval", attempts=2, laps=1, max_ticks_per_lap=80))
    assert res.attempts[0].result.crashes > 2
    return res.to_replay("all", title="Crashes")


def test_round_trip(tmp_path, crashy_replay):
    path = tmp_path / "r.json"
    text = crashy_replay.to_json(path)
    assert json.loads(text) == json.loads(path.read_text(encoding="utf-8"))
    back = Replay.from_json(path)
    assert back.to_dict() == crashy_replay.to_dict()
    assert Replay.from_json(text).to_dict() == crashy_replay.to_dict()
    assert back.duration == pytest.approx(max(c.t[-1] for c in crashy_replay.cars))
    with pytest.raises(ValueError):
        Replay.from_dict({"format": "something-else"})


def test_schema_rules(crashy_replay):
    d = crashy_replay.to_dict()
    assert crashy_replay.validate() == [] and validate_replay_dict(d) == []
    assert d["format"] == "agentic-gp-replay" and d["version"] == 1 and d["kind"] == "session"
    for car in d["cars"]:
        lens = {len(car[k]) for k in ("tick", "t", "x", "y", "heading_deg", "speed", "throttle", "steering", "lap",
                                      "race_progress")}
        assert len(lens) == 1
        assert all(b >= a for a, b in zip(car["t"], car["t"][1:]))
        assert car["race_progress"][0] < 0.05
        assert car["events"][-1]["kind"] == "dnf"
    assert [c["position"] for c in d["cars"]] == [1, 2]


def test_penalty_hold_frame_after_every_crash(crashy_replay):
    for car in crashy_replay.cars:
        crashes = [e for e in car.events if e.kind == "crash"]
        assert len(crashes) == car.crashes > 0
        for e in crashes:
            d = car.to_dict()
            i = next(k for k, t in enumerate(d["t"]) if abs(t - round(e.t, 2)) < 1e-6)
            assert d["t"][i + 1] - d["t"][i] == pytest.approx(5.0, abs=0.011)
            assert (d["x"][i], d["y"][i]) == (d["x"][i + 1], d["y"][i + 1]) and d["speed"][i + 1] == 0
            assert e.value == pytest.approx(5.0) and e.text == "+5s"
            # the wall-contact frame just before: at the crash point, before the crash time
            contact = d["t"].index(d["t"][i]) - 1
            assert d["t"][i] - d["t"][contact] == pytest.approx(0.05, abs=0.011)
            assert (d["x"][contact], d["y"][contact]) == (round(e.x, 2), round(e.y, 2))
        assert len(car.t) == car.tick[-1] + 1 + 2 * car.crashes   # contact + hold frame per crash


def test_unity_validator_accepts_real_replays(crashy_replay, tmp_path):
    validate = unity_validator()
    assert validate(crashy_replay.to_dict(extended=False)) == []
    # A15 additions (motto, anatomy, house, zones) are extra keys for the strict Unity checker
    assert validate(crashy_replay.to_dict(), strict_keys=False) == []
    res = run_session(lambda obs: (1.0, 0.0), SessionConfig(track="briefing_demo", attempts=1, max_ticks_per_lap=60,
                                                            physics=PhysicsConfig(crash_rule="classic")))
    rep = res.to_replay()
    assert validate(rep.to_dict(extended=False)) == []


def test_extended_fields(tmp_path):
    def flat_out(obs):
        return 1.0, 0.0 if obs["distance_to_wall_front"] > 30 else 0.4

    from agentic_gp.race import load_driver
    drv = load_driver(flat_out)
    drv.motto, drv.anatomy = "go", ["perceive", "act"]
    res = run_session(drv, SessionConfig(track="briefing_demo", attempts=1, max_ticks_per_lap=150))
    full, strict = res.to_replay().to_dict(), res.to_replay().to_dict(extended=False)
    assert full["track"]["zones"] == [{"from_index": z["from_index"], "to_index": z["to_index"], "max_speed": 15.0,
                                       "label": z["label"]} for z in full["track"]["zones"]]
    assert len(full["track"]["zones"]) == 1 and "zones" not in strict["track"]
    car, scar = full["cars"][0], strict["cars"][0]
    assert car["motto"] == "go" and car["anatomy"] == ["perceive", "act"] and car["house"] is False
    assert not {"motto", "anatomy", "house"} & set(scar)
    if res.attempts[0].zone_violations:
        assert any(e["kind"] == "zone" for e in car["events"])
        assert not any(e["kind"] == "zone" for e in scar["events"])


def _car(cid, t, rp, finished, race_time=None, crashes=0, heading=None):
    n = len(t)
    return ReplayCar(id=cid, name=cid.upper(), finished=finished, race_time=race_time, crashes=crashes,
                     tick=list(range(n)), t=list(t), x=[float(i) for i in range(n)], y=[0.0] * n,
                     heading_deg=heading or [0.0] * n, speed=[10.0] * n, throttle=[1.0] * n, steering=[0.0] * n,
                     lap=[1] * n, race_progress=list(rp))


def test_standings_and_classification():
    a = _car("a", [0, 10, 20], [0, 0.5, 1.0], True, 20.0)
    b = _car("b", [0, 10, 30], [0, 0.6, 1.0], True, 30.0)
    c = _car("c", [0, 10, 20, 40], [0, 0.2, 0.4, 0.45], False)
    d = _car("d", [0, 10], [0, 0.3], False)
    rep = build_replay(make_track("oval"), [c, d, b, a], title="t", laps=1)
    assert [x.id for x in rep.cars] == ["a", "b", "c", "d"] and [x.position for x in rep.cars] == [1, 2, 3, 4]
    assert [r[0] for r in standings_at(rep, 5.0)] == ["b", "a", "d", "c"]
    late = standings_at(rep, 25.0)
    assert late[0] == ("a", 1.0, True, 20.0)
    assert late[1][0] == "b" and late[1][2] is False and late[1][1] == pytest.approx(0.6 + 0.4 * 15 / 20)
    assert standings_at(rep.to_dict(), 100.0)[1] == ("b", 1.0, True, 30.0)
    tie = classify_cars([_car("x", [0, 1], [0, 1], True, 50.0, crashes=2), _car("y", [0, 1], [0, 1], True, 50.0)])
    assert [c.id for c in tie] == ["y", "x"]                   # same time: fewer crashes wins


def test_car_state_interpolation():
    car = _car("a", [0.0, 10.0, 20.0], [0.0, 0.5, 1.0], True, 20.0, heading=[170.0, -170.0, -90.0])
    s = car_state_at(car, 5.0)
    assert s["x"] == pytest.approx(0.5) and s["race_progress"] == pytest.approx(0.25)
    assert abs(abs(s["heading_deg"]) - 180.0) < 1e-6           # 170 -> -170 goes through 180, not 0
    assert car_state_at(car, 15.0)["heading_deg"] == pytest.approx(-130.0)
    assert car_state_at(car, -1)["x"] == 0.0 and car_state_at(car, 99)["x"] == 2.0
    assert car_state_at(car, 20.0)["finished"] and not car_state_at(car, 19.0)["finished"]
    assert car_state_at(car.to_dict(), 5.0)["x"] == pytest.approx(0.5)


def test_validate_catches_broken_files(crashy_replay):
    good = crashy_replay.to_dict()

    def broken(fn):
        d = copy.deepcopy(good)
        fn(d)
        return validate_replay_dict(d)

    assert any("missing" in e for e in broken(lambda d: d["cars"][0].pop("lap_times")))
    assert any("non-decreasing" in e for e in broken(lambda d: d["cars"][0]["t"].__setitem__(3, -1.0)))
    assert any("color" in e for e in broken(lambda d: d["cars"][0].__setitem__("color", "red")))
    assert any("differ in length" in e for e in broken(lambda d: d["cars"][0]["x"].pop()))
    assert any("sorted" in e or "1..N" in e for e in broken(lambda d: d["cars"].reverse()))
    assert any("hold frame" in e for e in broken(lambda d: d["cars"][0]["speed"].__setitem__(
        d["cars"][0]["t"].index(next(ev["t"] for ev in d["cars"][0]["events"] if ev["kind"] == "crash")) + 1, 3.0)))
    assert any("unknown event kind" in e for e in broken(lambda d: d["cars"][0]["events"].append(
        {"t": 0, "tick": 0, "kind": "teleport", "lap": 1, "x": 0, "y": 0, "value": 0, "text": ""})))
    assert validate_replay_dict({"format": "x"}) != []


def test_builders_from_env_traces_and_placeholders():
    track = make_track("oval")
    env = RaceEnv(track, laps=1)
    obs = env.reset()
    while not obs["attempt_over"]:
        obs = env.step(0.0, 0.0) if obs["tick"] < 5 else env.stop()
    car = replay_car_from_trace({"name": "Env Car", "color": "#123456"}, env.attempt_traces[-1])
    assert car.id == "env-car" and car.n_frames == 6 and car.status == "DNF (stopped)"
    ghost = placeholder_car({"name": "Ghost"}, track, "DNF (load error)")
    rep = build_replay(track, [car, ghost, placeholder_car({"name": "Ghost"}, track, "DNF (load error)")], title="x")
    assert rep.validate() == [] and sorted(c.id for c in rep.cars) == ["env-car", "ghost", "ghost-2"]
    assert ghost.events[-1].kind == "dnf" and ghost.events[-1].text == "DNF (load error)"
    rt = ReplayTrack.from_track(track)
    assert len(rt.center_x) == track.n_points and len(rt.gate_x1) == track.n_gates and rt.key == "oval"


def test_zero_crash_penalty_keeps_time_moving_forward():
    res = run_session(lambda obs: (0.5, 0.0), SessionConfig(track="oval", max_ticks_per_lap=80,
                                                            physics=PhysicsConfig(crash_penalty_seconds=0.0)))
    rep = res.to_replay()
    car = rep.cars[0]
    assert car.crashes > 0 and rep.validate() == []
    assert all(e.value == 0.0 and e.text == "+0s" for e in car.events if e.kind == "crash")
    frames = res.attempts[0].frames
    crash_ticks = [f for f in frames if f.get("crashed")]
    assert [c["t"] for c in res.attempts[0].crash_log] == [round(f["t"], 2) for f in crash_ticks]
    validate = unity_validator()
    assert validate(rep.to_dict(extended=False)) == []
