"""Ch7 fly drivers: the 57-neuron toy, the broken-fly exercise, the FlyWire connectome driver, the
Fly-class lint, and the library fixes (nose-to-wall reflex, graded looming, branch-free control path).
Connectome tests skip when the circuit file is missing."""

from __future__ import annotations

import ast
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from agentic_gp.race import SessionConfig, check_driver_source, load_driver, run_session
from agentic_gp.sim.config import PhysicsConfig
from agentic_gp.sim.env import RaceEnv

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "drivers" / "examples"
TOY, BROKEN, CONNECTOME = EXAMPLES / "fly_toy.py", EXAMPLES / "fly_broken.py", EXAMPLES / "fly_connectome.py"


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _lap(path, track="grand_prix", params=None, physics=None, attempts=1):
    result = run_session(load_driver(path, params=params),
                         SessionConfig(track=track, laps=1, physics=physics, attempts=attempts))
    return result.attempts[0] if attempts == 1 else result


def _circuit_available() -> bool:
    try:
        from agentic_gp.fly.connectome import circuit_path
        return circuit_path().is_file()
    except Exception:
        return False


needs_circuit = pytest.mark.skipif(not _circuit_available(),
                                   reason="FlyWire circuit file missing (python -m agentic_gp.fly.extract)")


def _corner_of(track: str, point) -> str:
    """Name of the corner (T1, T2, ...) whose apex is closest along the centerline to `point`."""
    m = RaceEnv(track).track_map()
    c = np.asarray(m["centerline"])
    i = int(np.argmin(((c - np.asarray(point)) ** 2).sum(1)))
    n = len(c)
    return min(m["corners"], key=lambda k: min(abs(i - k["apex_index"]), n - abs(i - k["apex_index"])))["id"]


# ----------------------------------------------------------------------------------------- all files
@pytest.mark.parametrize("path", [TOY, BROKEN, CONNECTOME], ids=lambda p: p.stem)
def test_static_check_is_clean(path):
    issues = check_driver_source(path)
    assert not [i for i in issues if i.level == "error"], issues
    assert not [i for i in issues if "NAME" in i.message or "KIND" in i.message], issues
    drv = load_driver(path)
    assert drv.kind == "fly" and len(drv.name) <= 16


@pytest.mark.parametrize("path", [TOY, BROKEN, CONNECTOME], ids=lambda p: p.stem)
def test_example_flies_pass_the_fly_class_lint(path):
    from agentic_gp.fly.rules import check_fly_rules
    assert check_fly_rules(path) == []


# ----------------------------------------------------------------------------------------- Fly-class lint
BAD_FLY = '''
import numpy as np
class Driver:
    NAME = "Bad Fly"
    KIND = "fly"
    PARAMS = {"gain": 1.0}
    def encode(self, obs):
        front = obs["distance_to_wall_front"]
        if front < 5:                       # a branch on a sensor
            return np.ones(3)
        return np.zeros(3)
    def drive(self, obs):
        rates = self.encode(obs)            # the brain's output: neurons, not sensors
        steer = 1.0 if rates[0] > 10 else 0.0
        return {"throttle": 0.5, "steering": steer}
'''
BAD_SPINAL = '''
SPINAL_CORD = "throttle from speed"
class Driver:
    KIND = "fly"
    SPINAL_CORD = SPINAL_CORD
    PARAMS = {"SCRAMBLE": None}
    def drive(self, obs):
        return {"throttle": self.spinal(obs), "steering": 0.0}
    # ==== SPINAL_CORD
    def spinal(self, obs):
        return 1.0 if obs["distance_to_wall_front"] > 9 else 0.0
    # ==== end SPINAL_CORD
'''


def test_fly_class_lint_flags_sensor_branches_and_missing_declarations(tmp_path):
    from agentic_gp.fly.rules import check_fly_rules
    bad = tmp_path / "bad_fly.py"
    bad.write_text(BAD_FLY, encoding="utf-8")
    msgs = [i.message for i in check_fly_rules(bad)]
    assert any("rule 2" in m and "front < 5" in m for m in msgs), msgs
    assert not any("rates[0]" in m for m in msgs), msgs           # a branch on neuron output is fine
    assert any("rule 3" in m and "no SPINAL_CORD" in m for m in msgs)
    assert any("rule 5" in m and "SCRAMBLE" in m for m in msgs)
    spinal = tmp_path / "spinal_fly.py"
    spinal.write_text(BAD_SPINAL, encoding="utf-8")
    msgs = [i.message for i in check_fly_rules(spinal)]
    assert any("SPINAL_CORD section reads 'distance_to_wall_front'" in m for m in msgs), msgs
    assert not any("rule 2" in m for m in msgs)                   # branches inside SPINAL_CORD: rule 3's job
    assert check_fly_rules(EXAMPLES / "l1_reflex.py") == []       # not a fly: the rule does not apply
    assert check_fly_rules(EXAMPLES / "l1_reflex.py", only_if_fly=False)   # ...but it would fail it


def test_flybrain_control_path_has_no_sensor_branches():
    from agentic_gp.fly.brain import CONTROL_PATH
    from agentic_gp.fly.rules import sensor_branches
    src = (ROOT / "agentic_gp" / "fly" / "brain.py").read_text(encoding="utf-8")
    found = sensor_branches(src, CONTROL_PATH, sensors=("obs", "bearing_deg", "strength", "speed", "gr"),
                            class_name="FlyBrain")
    assert found == []
    tree = ast.parse(src)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FlyBrain")
    assert set(CONTROL_PATH) <= {f.name for f in cls.body if isinstance(f, ast.FunctionDef)}


def test_looming_is_graded_not_a_threshold():
    from agentic_gp.fly.brain import EncoderParams, FlyBrain
    me = SimpleNamespace(encoder=EncoderParams())
    loom = lambda front, speed: FlyBrain.looming(me, {"distance_to_wall_front": front, "speed": speed})  # noqa: E731
    ttcs = np.linspace(1.2, 0.5, 30)
    hz = [loom(10.0 * t, 10.0) for t in ttcs]                     # speed 10, wall getting closer
    assert hz[0] == 0 and hz[-1] == pytest.approx(100.0)
    assert np.all(np.diff(hz) >= 0) and sum(0 < h < 100 for h in hz) >= 5   # a ramp, not a step
    assert loom(7.9, 10.0) == pytest.approx(loom(8.0, 10.0), abs=5.0)     # no jump at the old 0.8 s
    assert loom(0.5, 0.5) == 0 and loom(0.5, 1.5) < loom(0.5, 2.5)        # a parked car does not loom


# ----------------------------------------------------------------------------------------- the toy
@pytest.fixture(scope="module")
def toy():
    return _module(TOY, "fly_toy_under_test")


@pytest.fixture(scope="module")
def broken():
    return _module(BROKEN, "fly_broken_under_test")


def test_toy_has_the_named_cell_types(toy):
    assert 50 <= toy.N_NEURONS <= 57
    for t in ("LPLC2", "LC10a_L", "LC10a_R", "AOTU019_L", "AOTU019_R", "P1", "DNa02_L", "DNa02_R", "P9", "GF",
              "MDN", "AND_STUCK", "WALL_L20", "WALL_R20", "WALL_L45", "WALL_R45", "WALL_L90", "WALL_R90"):
        assert t in toy.NEURONS, t
    for pre, post, w, why in toy.WIRING:
        assert pre in toy.NEURONS and post in toy.NEURONS and isinstance(why, str) and why
    assert any(w < 0 for *_, w, _ in toy.WIRING)              # there are inhibitory synapses
    assert len(toy.NEURON_NAMES) == toy.N_NEURONS
    assert toy.SPINAL_CORD == "" and toy.Driver.SPINAL_CORD == ""
    assert "look-ahead" in toy.NEURONS["LC10a_L"][2]          # the encoder's look-ahead is disclosed


def test_toy_wall_wiring_is_crossed(toy):
    walls = [(pre, post) for pre, post, *_ in toy.WIRING if pre.startswith("WALL_")]
    assert len(walls) == 6
    for pre, post in walls:
        assert post == ("DNa02_R" if pre.startswith("WALL_L") else "DNa02_L"), (pre, post)


def _rates_after(toy, obs: dict, windows: int = 3) -> dict:
    d = toy.Driver()
    for _ in range(windows):
        d.drive(obs)
    return d.rates


def test_coincidence_neuron_needs_both_inputs(toy):
    open_road = {"speed": 15.0, "distance_to_wall_front": 200.0, "angle_to_next_waypoint": 0.0,
                 "distance_to_next_waypoint": 80.0}
    wall_ahead_moving = dict(open_road, speed=8.0, distance_to_wall_front=4.0)
    standing_open = dict(open_road, speed=0.0)
    stuck = dict(open_road, speed=0.0, distance_to_wall_front=3.0)
    assert _rates_after(toy, wall_ahead_moving)["AND_STUCK"] == 0
    assert _rates_after(toy, standing_open)["AND_STUCK"] == 0
    r = _rates_after(toy, stuck)
    assert r["AND_STUCK"] > 20 and r["MDN"] > 20
    assert toy.Driver().decode(r)[0] < 0                      # throttle < 0: the moonwalker backs up


def test_toy_steers_toward_the_target_and_away_from_walls(toy):
    base = {"speed": 15.0, "distance_to_wall_front": 200.0, "distance_to_next_waypoint": 80.0}
    for bearing in (-40.0, 40.0):
        _, steer = toy.Driver().decode(_rates_after(toy, dict(base, angle_to_next_waypoint=bearing)))
        assert np.sign(steer) == np.sign(bearing) and abs(steer) > 0.2
    _, steer = toy.Driver().decode(_rates_after(toy, dict(base, angle_to_next_waypoint=0.0,
                                                          distance_to_wall_left_20=4.0, distance_to_wall_left_45=4.0)))
    assert steer > 0.2                                        # left wall -> turn right


def test_toy_has_no_sensor_branches_in_the_control_path():
    tree = ast.parse(TOY.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Driver")
    for fn in cls.body:
        if isinstance(fn, ast.FunctionDef) and fn.name in ("encode", "simulate", "decode", "drive"):
            branches = [n for n in ast.walk(fn) if isinstance(n, (ast.If, ast.IfExp, ast.Match))]
            assert not branches, f"{fn.name}() branches on line {branches[0].lineno}"


def test_toy_finishes_grand_prix_and_oval_cleanly_and_deterministically():
    gp = _lap(TOY, "grand_prix").result
    assert gp.completed and gp.crashes == 0 and 55 < gp.race_time < 75, (gp.race_time, gp.crashes)
    assert _lap(TOY, "grand_prix").result.race_time == gp.race_time        # no noise: same lap
    oval = _lap(TOY, "oval").result
    assert oval.completed and oval.crashes == 0 and oval.race_time < 40


def test_toy_uncrossed_wall_wiring_is_much_worse():
    r = _lap(TOY, "grand_prix", params={"crossed": False}).result
    assert not r.completed or r.crashes >= 3


def test_toy_without_mdn_stays_pinned_under_the_classic_rule():
    classic = PhysicsConfig(crash_rule="classic")
    with_mdn = _lap(TOY, "oval", params={"crossed": False}, physics=classic).result
    without = _lap(TOY, "oval", params={"crossed": False, "mdn": False}, physics=classic).result
    assert with_mdn.completed and not without.completed


def test_toy_scramble_control_shuffles_targets_and_loses(toy):
    real = toy.Driver()
    d = load_driver(TOY, params={"SCRAMBLE": "0"})
    inner = d._drive.__self__
    assert toy._seed("none") is None and toy._seed("3") == 3
    assert not np.array_equal(inner.W, real.W)
    r = _lap(TOY, "oval", params={"SCRAMBLE": 0}).result
    assert not r.completed or r.race_time > 2 * 29.0           # the real toy: 29.0 s


def test_toy_lookahead_param_is_the_disclosed_blend():
    slower = _lap(TOY, "grand_prix", params={"lookahead": 0.0}).result
    assert slower.completed and slower.crashes == 0 and slower.race_time > 63.3 + 2


def test_toy_radio_starts_fresh_every_attempt(toy):
    obs = {"speed": 15.0, "distance_to_wall_front": 200.0, "angle_to_next_waypoint": 40.0,
           "distance_to_next_waypoint": 80.0}
    d = toy.Driver()
    first = d.drive(obs)["say"]
    d.reflect({})
    again = d.drive(obs)["say"]
    assert first and again == first


def test_toy_mirror_self_check_passes(toy, capsys):
    assert toy.mirror_check(verbose=True)
    assert "NOT" not in capsys.readouterr().out


# ----------------------------------------------------------------------------------------- broken fly
def _the_diff(toy, broken) -> int:
    rows = [i for i, (a, b) in enumerate(zip(toy.WIRING, broken.WIRING)) if a[:3] != b[:3]]
    assert len(toy.WIRING) == len(broken.WIRING) and len(rows) == 1, rows
    return rows[0]


def test_broken_fly_differs_from_the_toy_in_exactly_one_wiring_row(toy, broken):
    assert broken.NEURONS == toy.NEURONS
    assert (broken.STEER_HZ, broken.GO_HZ, broken.BRAKE_HZ, broken.BACK_HZ) == (toy.STEER_HZ, toy.GO_HZ,
                                                                                toy.BRAKE_HZ, toy.BACK_HZ)
    assert broken.WALL_RANGE == toy.WALL_RANGE and broken.Driver.PARAMS == toy.Driver.PARAMS
    i = _the_diff(toy, broken)
    assert broken.WIRING[i][0].startswith("WALL_")              # a wall row ...
    # ... whose comment does not give it away: wall rows name the stimulus, not the turn it should cause
    assert all("turn" not in row[3] for row in broken.WIRING if row[0].startswith("WALL_"))
    assert "Instructor" not in (broken.__doc__ or "")
    readme = (ROOT / "agentic_gp" / "fly" / "README.md").read_text(encoding="utf-8")
    assert f"`{broken.WIRING[i][0]} → {broken.WIRING[i][1]}`" not in readme   # not published


def _fixed_source(toy, broken) -> str:
    """fly_broken.py with its one differing row replaced by the toy's row (no literal answer here)."""
    i = _the_diff(toy, broken)
    lines = BROKEN.read_text(encoding="utf-8").splitlines(keepends=True)
    pre, bad_post, w = broken.WIRING[i][:3]
    good_post = toy.WIRING[i][1]
    hits = [k for k, ln in enumerate(lines) if ln.lstrip().startswith(f'("{pre}",') and f'"{bad_post}"' in ln]
    assert len(hits) == 1
    lines[hits[0]] = lines[hits[0]].replace(f'"{bad_post}"', f'"{good_post}"')
    return "".join(lines)


def test_broken_fly_hits_the_hairpin_and_the_one_row_fix_gives_the_toys_exact_lap(tmp_path, toy, broken):
    a = _lap(BROKEN, "grand_prix")
    assert a.result.completed and a.result.crashes >= 1
    assert {_corner_of("grand_prix", p) for p in a.crash_points} == {"T5"}
    assert not broken.mirror_check()
    fixed = tmp_path / "fly_fixed.py"
    fixed.write_text(_fixed_source(toy, broken), encoding="utf-8")
    assert _module(fixed, "fly_fixed_under_test").mirror_check()
    r = _lap(fixed, "grand_prix").result
    toy_lap = _lap(TOY, "grand_prix").result
    assert r.completed and r.crashes == 0 and r.race_time == toy_lap.race_time == 63.3


def test_broken_fly_success_test_rejects_lucky_non_fixes(toy, broken):
    """One-row changes that dodge the hairpin but leave the bug (review finding): the mirror self-check
    fails, and the lap is not the toy's exact 63.3 s."""
    i = _the_diff(toy, broken)
    pre, post, w, why = broken.WIRING[i]

    def variant(change: dict):
        wiring = [change.get(k, row) for k, row in enumerate(broken.WIRING)]
        return type("Variant", (broken.Driver,), {"WIRING": wiring})

    row_of = {r[0]: k for k, r in enumerate(broken.WIRING) if r[0].startswith("WALL_L")}
    others = [p for p in ("WALL_L20", "WALL_L45", "WALL_L90") if p != pre][:2]
    variants = [variant({i: (pre, post, 0.0, why)}), variant({i: (pre, post, -w, why)})]
    for p in others:
        k = row_of[p]
        variants.append(variant({k: broken.WIRING[k][:2] + (broken.WIRING[k][2] * 2.4,) + broken.WIRING[k][3:]}))
    for cls in variants:
        assert not broken.mirror_check(cls), cls.WIRING
    lucky = run_session(load_driver(variants[0]), SessionConfig(track="grand_prix", laps=1)).attempts[0].result
    assert lucky.race_time != 63.3                              # 0 crashes, but not the healthy fly's lap


def test_broken_fly_radio_names_the_left_wall_neurons_before_the_first_crash():
    a = _lap(BROKEN, "grand_prix")
    first = a.crash_log[0]["tick"]
    before = [m["text"] for m in a.messages if first - 5 <= m["tick"] < first]
    assert any("turning left" in t and "WALL_L" in t for t in before), before


# ----------------------------------------------------------------------------------------- connectome
def test_connectome_without_circuit_file_is_graceful(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("AGENTIC_GP_FLY_CIRCUIT", str(tmp_path / "missing.npz"))
    drv = load_driver(CONNECTOME)
    drv.plan(RaceEnv("oval").track_map())
    assert "agentic_gp.fly.extract" in capsys.readouterr().err
    obs = RaceEnv("oval").reset()
    first, second = drv.drive(obs), drv.drive(obs)
    assert (first.throttle, first.steering) == (0.0, 0.0) and "extract" in (first.say or "")
    assert second.say is None


def test_connectome_param_parsing_and_declarations():
    mod = _module(CONNECTOME, "fly_connectome_params")
    assert [mod._seed_or_none(v) for v in (None, False, "none", "", True, "true", 3, "4")] == [None, None, None, None,
                                                                                              0, 0, 3, 4]
    assert mod._groups("DNa02, AOTU019") == ["DNa02", "AOTU019"] and mod._groups(["LC10a"]) == ["LC10a"]
    assert mod._groups("") == []
    assert [mod._mirror(v) for v in ("off", False, "all", True, "true", "eyes")] == [False, False, True, True,
                                                                                    True, "eyes"]
    assert isinstance(mod.Driver.PARAMS["MIRROR"], str)        # so --param MIRROR=eyes is not cast to bool
    assert "MDN" in mod.REFLEX and mod.Driver.REFLEX == mod.REFLEX
    assert "zero lines" not in mod.Driver.MOTTO
    doc = mod.__doc__
    assert "AOTU019" in doc.split("steering =")[0] and "SHALLOW" in doc


@needs_circuit
def test_connectome_builds_the_requested_brain():
    mod = _module(CONNECTOME, "fly_connectome_build")
    marshal = RaceEnv("oval").track_map()
    classic = RaceEnv("oval", config=PhysicsConfig(crash_rule="classic")).track_map()
    d = mod.Driver()
    d.PARAMS = dict(d.PARAMS, SCRAMBLE=2, LESION="DNa02", MIRROR=True)
    d.plan(marshal)
    label = d.brain.circuit.label
    assert "scrambled(seed=2)" in label and "lesion(DNa02)" in label and "mirrored" in label
    assert d.brain.encoder.touch_hz == 0.0 and d.brain.readout.steer_gain == pytest.approx(0.4)
    d2 = mod.Driver()
    d2.plan(classic)                                          # TOUCH="auto": on under the classic rule only
    assert d2.brain.circuit.label == "real" and d2.brain.encoder.touch_hz > 0
    d3 = mod.Driver()
    d3.PARAMS = dict(d3.PARAMS, SHALLOW=True, READOUT="DN", MIRROR="eyes")
    d3.plan(marshal)
    assert "shallow" in d3.brain.circuit.label and "mirrored(LC10a+LC16+LPLC2+LC4)" in d3.brain.circuit.label
    assert d3.brain.readout.aotu019_weight == 0.0
    assert d3.brain.circuit.n_edges < 0.01 * d2.brain.circuit.n_edges


@needs_circuit
def test_shallow_keeps_only_direct_input_to_readout_connections():
    from agentic_gp.fly import load_circuit
    from agentic_gp.fly.brain import INPUT_GROUPS, READOUT_GROUPS
    from agentic_gp.fly.connectome import shallow
    c = load_circuit()
    s = shallow(c, INPUT_GROUPS, READOUT_GROUPS)
    inputs = np.concatenate([c.group(g) for g in INPUT_GROUPS])
    outputs = np.concatenate([c.group(g) for g in READOUT_GROUPS])
    assert 0 < s.n_edges < 0.005 * c.n_edges and s.n == c.n
    assert np.isin(s.pre, inputs).all() and np.isin(s.post, outputs).all()


@needs_circuit
def test_mirror_all_still_chases_the_target_and_mirror_eyes_turns_away():
    from agentic_gp.fly import FlyBrain
    full, eyes = FlyBrain.from_default(mirror=True), FlyBrain.from_default(mirror="eyes")
    for bearing in (-45.0, 45.0):
        assert np.sign(full.bearing_response(bearing, ms=300)["steer"]) == np.sign(bearing)
        assert np.sign(eyes.bearing_response(bearing, ms=300)["steer"]) == -np.sign(bearing)


@needs_circuit
def test_connectome_drives_speaks_and_declares_its_spinal_cord():
    mod = _module(CONNECTOME, "fly_connectome_drive")
    assert "speed" in mod.SPINAL_CORD and mod.Driver.SPINAL_CORD == mod.SPINAL_CORD
    src = CONNECTOME.read_text(encoding="utf-8")
    assert src.count("# ====") >= 2 and "SPINAL_CORD" in src.split("def spinal_cord")[0]
    env = RaceEnv("oval")
    d = mod.Driver()
    d.plan(env.track_map())
    obs, says = env.reset(), []
    for _ in range(12):
        a = d.drive(obs)
        assert -1 <= a["throttle"] <= 1 and -1 <= a["steering"] <= 1
        motor = d.snapshots[-1]["motor"]                      # the library's own decode gives the same throttle
        assert a["throttle"] == pytest.approx(motor["throttle"])
        says.append(a["say"])
        obs = env.step(a["throttle"], a["steering"])
    assert any(says) and set(a["debug"]) >= {"DNa02_L", "DNa02_R", "MDN_L", "DNp01_L", "brake", "steer_AOTU019"}


@needs_circuit
def test_connectome_snapshots_hold_one_attempt_on_one_clock():
    from agentic_gp.fly import viz
    drv = load_driver(CONNECTOME)
    res = run_session(drv, SessionConfig(track="oval", laps=1, attempts=2, max_ticks_per_lap=30))
    inner = drv._drive.__self__
    snaps = inner.snapshots
    assert len(res.attempts) == 2 and 0 < len(snaps) <= 31
    assert {s["attempt"] for s in snaps} == {1}
    t = np.array([s["t_ms"] for s in snaps])
    assert np.all(np.diff(t) > 0)
    assert np.all(np.diff(viz.raster_from_snapshots(snaps)["ticks"]) > 0)


@needs_circuit
def test_connectome_real_wiring_beats_its_scrambled_control_on_oval():
    real = _lap(CONNECTOME, "oval").result
    scrambled = _lap(CONNECTOME, "oval", params={"SCRAMBLE": 0}).result
    assert real.completed and real.crashes <= 2
    assert not scrambled.completed or scrambled.race_time > 1.8 * real.race_time
    assert scrambled.crashes > real.crashes + 4


# ----------------------------------------------------------------------------------------- library fix
def _nose_to_wall(front: float, bearing: float) -> dict:
    """A flat wall straight ahead at `front` units, car stopped, waypoint 20 units away."""
    obs = {"speed": 0.0, "distance_to_next_waypoint": 20.0, "angle_to_next_waypoint": bearing}
    rays = ((0, "distance_to_wall_front"), (-20, "distance_to_wall_left_20"), (20, "distance_to_wall_right_20"),
            (-45, "distance_to_wall_left_45"), (45, "distance_to_wall_right_45"))
    for ang, key in rays:
        obs[key] = front / math.cos(math.radians(ang))
    obs["distance_to_wall_left_90"] = obs["distance_to_wall_right_90"] = 9.0
    return obs


@needs_circuit
def test_parked_nose_to_wall_backs_up_with_touch_and_creeps_without():
    from agentic_gp.fly import EncoderParams, FlyBrain, load_circuit
    c = load_circuit()

    def mean_throttle(enc, bearing):
        out = []
        for seed in range(3):
            b = FlyBrain(c, encoder_params=enc, seed=seed)
            for _ in range(4):
                ctl, snap = b.step(_nose_to_wall(1.5, bearing))
            out.append(ctl["throttle"])
        return float(np.mean(out)), snap

    for bearing in (0.0, 30.0, 60.0):
        backs, snap = mean_throttle(EncoderParams(), bearing)
        assert backs < -0.5 and snap["inputs"]["touch_MDN"] > 0, (bearing, backs)
        creeps, _ = mean_throttle(EncoderParams(touch_hz=0.0), bearing)
        assert creeps > 0.2, (bearing, creeps)                # the old behaviour, now opt-out


@needs_circuit
def test_encoder_drives_only_input_neurons_and_the_reflex_is_separate():
    from agentic_gp.fly import FlyBrain, load_circuit
    from agentic_gp.fly.brain import INPUT_GROUPS
    b = FlyBrain(load_circuit())
    obs = dict(_nose_to_wall(1.5, 30.0), speed=8.0)
    rates, info = b.encode(obs)
    inputs = np.concatenate([b.circuit.group(g) for g in INPUT_GROUPS])
    others = np.setdiff1d(np.flatnonzero(rates), inputs)
    assert others.size == 0                                    # rule 2: no encoder input into read neurons
    b.reflex(obs, rates, info)
    assert info["touch_MDN"] > 0 and rates[b.circuit.group("MDN")].min() > 0   # the declared REFLEX


@needs_circuit
def test_steer_gain_scales_the_decoded_steering():
    from agentic_gp.fly import FlyBrain, ReadoutParams, load_circuit
    c = load_circuit()
    gr = {g: 0.0 for g in ("DNa02_L", "DNa02_R", "AOTU019_L", "AOTU019_R", "MDN_L", "MDN_R", "DNp01_L", "DNp01_R")}
    gr["DNa02_R"] = 65.0                                       # raw steer = 65 / 130 = 0.5
    full = FlyBrain(c).decode(gr, speed=10.0)["steering"]
    calm = FlyBrain(c, readout_params=ReadoutParams(steer_gain=0.4)).decode(gr, speed=10.0)["steering"]
    assert full == pytest.approx(0.5) and calm == pytest.approx(0.2)
