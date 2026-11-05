"""Tests for agentic_gp.fly: the LIF simulator always; the FlyWire circuit only if extracted.

The circuit tests skip when data/fly/steering_circuit_783.npz is missing
(build it with `python -m agentic_gp.fly.extract`).
"""
import json

import numpy as np
import pytest

from agentic_gp.fly.lif import LIFNetwork, LIFParams


# ---------------------------------------------------------------- LIF unit tests
def test_no_input_is_silent():
    rng = np.random.default_rng(0)
    pre, post = rng.integers(0, 50, 400), rng.integers(0, 50, 400)
    net = LIFNetwork(50, pre, post, rng.normal(0, 5, 400), backend="numpy")
    rates, spikes = net.run(200.0)
    assert rates.sum() == 0 and spikes.idx.size == 0
    assert net.t_ms == pytest.approx(200.0)


def test_poisson_drive_makes_spikes_at_the_input_rate():
    net = LIFNetwork(3, [0], [1], [0.0], seed=1, backend="numpy")
    drive = np.array([100.0, 0.0, 0.0])
    rates, spikes = net.run(2000.0, drive)
    # every Poisson kick (68.75 mV) fires a driven neuron, which has no refractory period
    assert 80 < rates[0] < 120
    assert rates[1] == 0 and rates[2] == 0
    assert set(spikes.idx.tolist()) == {0}


def test_spikes_propagate_with_delay_and_sign():
    # 0 -> 1 strongly excitatory, 0 -> 2 strongly inhibitory
    net = LIFNetwork(3, [0, 0], [1, 2], [2000.0, -2000.0], seed=2, backend="numpy")
    rates, spikes = net.run(500.0, np.array([50.0, 0.0, 0.0]))
    assert rates[1] > 10 and rates[2] == 0
    steps0 = np.round(spikes.t_ms[spikes.idx == 0] / net.dt).astype(int)
    steps1 = np.round(spikes.t_ms[spikes.idx == 1] / net.dt).astype(int)
    # Brian2 schedule: a spike at step t lands in g at step t + D (after that step's threshold),
    # is integrated at t + D + 1, and a big enough input fires the target right then.
    assert net.D == 7
    assert steps1[0] - steps0[0] == net.D + 1
    assert set((steps1 - net.D - 1).tolist()) <= set(steps0.tolist())


def test_refractory_period_drops_input():
    p = LIFParams()                                   # dt 0.25: delay 7 steps, input dropped 7 steps after a spike

    def fires_again(arrival_step):
        net = LIFNetwork(1, [], [], [], params=p, backend="numpy")
        net.u[0] = 50.0                               # far above threshold: spikes at step 0
        net.buf[arrival_step % net.D, 0] += 2000.0    # a huge input arriving at `arrival_step`
        _, spikes = net.run(30.0)
        return spikes.idx.size - 1

    assert LIFNetwork(1, [], [], [], params=p).R == 7   # Brian2: int(2.2 / 0.25 + 1e-3) - 1
    assert fires_again(4) == 0                        # inside the refractory period: dropped
    assert fires_again(6) == 0
    assert fires_again(1) == 0
    net = LIFNetwork(1, [], [], [], params=p, backend="numpy")
    net.u[0] = 50.0
    net.run(2.0)                                      # steps 0..7: spike at 0, refractory 1..7
    net.buf[net.step_count % net.D, 0] += 2000.0      # arrives at step 8: the first open step
    _, spikes = net.run(10.0)
    assert spikes.idx.size == 1


def _reference(n, pre, post, w, p, steps, kick_ptr, kick_idx, driven):
    """Plain step-by-step LIF on Brian2's default schedule (slow, written independently):
    groups (integrate unless refractory) -> thresholds -> synapses (g += w and Poisson v += kick,
    unless refractory) -> resets.  not_refractory = timestep(t - lastspike) >= timestep(t_ref)."""
    D = int(round(p.t_delay / p.dt))
    Rb = int(p.t_refractory / p.dt + 1e-3)
    em, es = np.exp(-p.dt / p.tau_m), np.exp(-p.dt / p.tau_syn)
    cvg = p.tau_syn / (p.tau_syn - p.tau_m) * (es - em)
    theta = p.v_threshold - p.v_rest
    u, g = np.zeros(n), np.zeros(n)
    last = np.full(n, -10**9)
    arriving = np.zeros((steps + D + 1, n))
    out = []
    for t in range(steps):
        notref = driven | (t - last >= Rb)
        u = np.where(notref, u * em + g * cvg, u)                       # groups
        g = np.where(notref, g * es, g)
        spk = np.flatnonzero((u > theta) & notref)                      # thresholds
        last[spk] = t
        notref[spk] = driven[spk]
        g += np.where(notref, arriving[t], 0.0)                         # synapses
        for k in range(kick_ptr[t], kick_ptr[t + 1]):
            if notref[kick_idx[k]]:
                u[kick_idx[k]] += p.poisson_kick
        for i in spk:
            out.append((t, i))
            m = pre == i
            np.add.at(arriving[t + D], post[m], w[m])
        u[spk] = 0.0                                                    # resets
        g[spk] = 0.0
    return out


@pytest.mark.parametrize("dt", [0.25, 0.1])
def test_matches_step_by_step_brian2_schedule_reference(dt):
    rng = np.random.default_rng(3)
    n, m = 60, 900
    pre, post = rng.integers(0, n, m), rng.integers(0, n, m)
    w = rng.integers(-40, 120, m) * 0.25               # exactly representable: sums are order-independent
    p = LIFParams(dt=dt)
    drive = np.zeros(n)
    drive[:8] = 120.0
    steps = int(round(150.0 / dt))
    net = LIFNetwork(n, pre, post, w, params=p, seed=5, backend="numpy")
    kp, ki = LIFNetwork(n, pre, post, w, params=p, seed=5)._poisson_schedule(drive, drive > 0, steps)
    ref = _reference(n, pre, post, w, p, steps, kp, ki, drive > 0)
    _, spikes = net.run(150.0, drive)
    got = list(zip(np.round(spikes.t_ms / dt).astype(int).tolist(), spikes.idx.tolist()))
    assert len(ref) > 100
    assert sorted(got) == sorted(ref)


def test_state_carries_across_calls():
    rng = np.random.default_rng(4)
    pre, post = rng.integers(0, 40, 500), rng.integers(0, 40, 500)
    w = rng.normal(20, 10, 500)
    drive = np.zeros(40)
    drive[:5] = 150.0
    a = LIFNetwork(40, pre, post, w, seed=9, backend="numpy")
    a.run(37.5, drive)
    assert a.step_count == 150 and np.abs(a.buf).sum() + np.abs(a.g).sum() > 0   # spikes in flight survive
    a.reset()
    assert a.step_count == 0 and not a.buf.any() and not a.u.any()


def test_numba_backend_gives_identical_spikes():
    pytest.importorskip("numba")
    rng = np.random.default_rng(6)
    n, m = 80, 1500
    pre, post, w = rng.integers(0, n, m), rng.integers(0, n, m), rng.normal(15, 20, m)
    drive = np.zeros(n)
    drive[:10] = 150.0
    a = LIFNetwork(n, pre, post, w, seed=2, backend="numpy")
    b = LIFNetwork(n, pre, post, w, seed=2, backend="numba")
    for _ in range(3):
        sa, sb = a.run(41.0, drive)[1], b.run(41.0, drive)[1]
        oa, ob = np.lexsort((sa.idx, sa.t_ms)), np.lexsort((sb.idx, sb.t_ms))
        assert np.array_equal(sa.idx[oa], sb.idx[ob]) and np.allclose(sa.t_ms[oa], sb.t_ms[ob])


def test_drive_summary_explains_a_race():
    from agentic_gp.fly.viz import drive_summary

    def snap(bearing, steering, emergency=False, brake=0.0):
        return {"inputs": {"bearing_deg": bearing}, "motor": {"steering": steering, "throttle": 0.2,
                                                             "emergency": emergency, "brake": brake}}
    s = drive_summary([snap(30, 0.5), snap(-30, -0.4), snap(30, -0.6), snap(-20, 0.0, True, 0.9)])
    assert s["ticks"] == 4 and s["emergency"] == 0.25 and s["braking"] == 0.25
    assert s["toward"] == 0.5 and s["away"] == 0.25 and s["weak_steer"] == 0.25


# ------------------------------------------------------------- circuit-dependent tests
@pytest.fixture(scope="module")
def circuit():
    from agentic_gp.fly.connectome import CircuitNotFoundError, load_circuit

    try:
        return load_circuit()
    except CircuitNotFoundError:
        pytest.skip("FlyWire circuit not extracted (python -m agentic_gp.fly.extract)")


def test_circuit_contents(circuit):
    assert circuit.n == 4494 and circuit.n_edges == 407139
    for g, size in {"LC10a_L": 115, "LC10a_R": 119, "DNa02_L": 1, "DNa02_R": 1, "AOTU019_L": 1, "MDN_L": 2,
                    "DNp01_R": 1, "LC16_L": 77, "LPLC2_R": 102, "LC4_L": 54}.items():
        assert circuit.group(g).size == size, g
    assert circuit.group("DNa02").size == 2
    pref = circuit.pref_ecc[circuit.group("LC10a")]
    assert np.all((pref >= 0) & (pref <= 1))


def test_scrambled_preserves_degrees_and_signs(circuit):
    from agentic_gp.fly.connectome import scrambled

    s = scrambled(circuit, 0)
    n = circuit.n
    assert np.array_equal(np.bincount(s.pre, minlength=n), np.bincount(circuit.pre, minlength=n))
    assert np.array_equal(np.bincount(s.post, minlength=n), np.bincount(circuit.post, minlength=n))
    assert np.array_equal(s.syn, circuit.syn) and np.array_equal(s.pre, circuit.pre)
    assert np.mean(s.post == circuit.post) < 0.01
    assert np.array_equal(scrambled(circuit, 0).post, s.post)          # seeded
    assert not np.array_equal(scrambled(circuit, 1).post, s.post)


def test_strength_scramble_keeps_input_synapse_totals(circuit):
    from agentic_gp.fly.connectome import scrambled

    def inputs(c):
        return (np.bincount(c.post, weights=np.clip(c.syn, 0, None), minlength=c.n),
                np.bincount(c.post, weights=np.clip(-c.syn, 0, None), minlength=c.n))
    exc, inh = inputs(circuit)
    big = exc + inh >= 200
    for method, lo, hi in (("degree", 0.3, 9.9), ("strength", 0.0, 0.12)):
        s = scrambled(circuit, 0, method=method)
        assert np.array_equal(np.bincount(s.post, minlength=circuit.n), np.bincount(circuit.post, minlength=circuit.n))
        e, i = inputs(s)
        err = np.median(np.abs(e[big] - exc[big]) / exc[big].clip(1))
        assert lo <= err <= hi, (method, err)
        assert np.mean(s.post == circuit.post) < 0.05


def test_lesion_silences_the_group_and_mirror_swaps_sides(circuit):
    from agentic_gp.fly.brain import FlyBrain
    from agentic_gp.fly.connectome import lesioned, mirrored

    les = lesioned(circuit, ["DNa02"])
    dn = circuit.group("DNa02")
    assert not (np.isin(les.pre, dn) | np.isin(les.post, dn)).any()
    assert set(les.silenced_index().tolist()) == set(dn.tolist())
    b = FlyBrain(les)
    for a in (-60, 60):
        r = b.bearing_response(a, ms=200, warmup_ms=50)
        assert r["DNa02_L"] == 0 and r["DNa02_R"] == 0          # silenced: no DNa02 term at all
    assert b.bearing_response(-15, ms=200, warmup_ms=50)["steer"] < -0.2   # AOTU019 still sees the side
    blind = FlyBrain(lesioned(circuit, "LC10a"))                  # input neurons silenced too
    assert blind.bearing_response(-60, ms=200, warmup_ms=50)["steer"] == 0
    mir = mirrored(circuit)
    assert np.array_equal(mir.group("LC10a_L"), circuit.group("LC10a_R"))


def test_save_is_deterministic_and_round_trips(circuit, tmp_path):
    from agentic_gp.fly.connectome import content_sha256, load_circuit, save_circuit

    save_circuit(circuit, tmp_path / "a.npz")
    save_circuit(circuit, tmp_path / "b.npz")
    assert (tmp_path / "a.npz").read_bytes() == (tmp_path / "b.npz").read_bytes()
    assert content_sha256(load_circuit(tmp_path / "a.npz")) == content_sha256(circuit)


def test_real_wiring_steers_toward_the_target_scrambled_does_not(circuit):
    from agentic_gp.fly.viz import bearing_sweep, lateralization

    sw = bearing_sweep(circuit, angles=[-60, -30, 30, 60], scramble_seeds=[0, 1, 2], ms=300, warmup_ms=50)
    real = sw["real"]
    assert real["DNa02_L"][0] > 60 and real["DNa02_R"][0] < 10      # target left -> left DNa02
    assert real["DNa02_R"][3] > 40 and real["DNa02_L"][3] < 10      # target right -> right DNa02
    assert real["AOTU019_R"][2] > 100 > real["AOTU019_L"][2]        # central-right target -> right AOTU019
    li_real = lateralization(real, sw["angles"])
    li_scr = [lateralization(c, sw["angles"]) for c in sw["scrambled"].values()]
    assert li_real > 0.7
    assert max(abs(x) for x in li_scr) < 0.4


def test_lc16_recruits_mdn_and_looming_drives_giant_fiber(circuit):
    from agentic_gp.fly.viz import stimulate

    wall = stimulate(circuit, {"LC16": 150}, ms=400, groups=["MDN_L", "MDN_R"])
    assert (wall["MDN_L"] + wall["MDN_R"]) / 2 > 10
    loom = stimulate(circuit, {"LPLC2": 100, "LC4": 100}, ms=300, groups=["DNp01_L", "DNp01_R"])
    assert loom["DNp01_L"] > 50 and loom["DNp01_R"] > 50
    quiet = stimulate(circuit, {}, ms=200)
    assert max(quiet.values()) == 0


FAR = {"distance_to_wall_front": 150.0, "distance_to_wall_left_20": 60.0, "distance_to_wall_right_20": 60.0,
       "distance_to_wall_left_45": 30.0, "distance_to_wall_right_45": 30.0, "distance_to_wall_left_90": 20.0,
       "distance_to_wall_right_90": 20.0, "distance_to_next_waypoint": 25.0, "angle_to_next_waypoint": 0.0}


def _settle(brain, obs, n=3):
    brain.reset()
    for _ in range(n):
        controls, snap = brain.step(obs)
    return controls, snap


def test_flybrain_step_api(circuit):
    from agentic_gp.fly.brain import FlyBrain

    brain = FlyBrain(circuit)
    obs = dict(FAR, speed=10.0, angle_to_next_waypoint=-50.0, distance_to_wall_front=80.0)
    controls, snap = _settle(brain, obs)
    assert set(controls) == {"throttle", "steering"}
    assert -1 <= controls["throttle"] <= 1 and controls["steering"] < -0.3     # turns left toward the target
    json.dumps(snap)
    assert snap["rates"]["DNa02_L"] > snap["rates"]["DNa02_R"]
    assert snap["t_ms"] == pytest.approx(300.0) and brain.last_window_start_ms == pytest.approx(200.0)
    controls, snap = brain.step({})                                            # missing sensors tolerated
    assert -1 <= controls["steering"] <= 1
    r = brain.bearing_rates(3.0)                                               # near the midline both eyes
    near, far = r[circuit.group("LC10a_R")].max(), r[circuit.group("LC10a_L")].max()   # see it, far eye less
    assert near > far > 0


def test_brake_emergency_and_no_false_stops(circuit):
    from agentic_gp.fly.brain import FlyBrain
    from agentic_gp.fly.connectome import scrambled

    brain = FlyBrain(circuit)
    # wall 3 ahead, waypoint ahead, rolling slowly (no looming).  Visible (in front of the wall):
    # LC10a pursuit suppresses the LC16 -> MDN brake.  Hidden behind it: LC10a is dimmed, MDN brakes.
    def motor(wp):
        brain.reset()
        obs = dict(FAR, speed=2.0, distance_to_wall_front=3.0, distance_to_next_waypoint=wp)
        snaps = [brain.step(obs)[1] for _ in range(5)][1:]
        return (snaps[-1]["inputs"]["visibility"], np.mean([x["motor"]["brake"] for x in snaps]),
                np.mean([x["motor"]["target_speed"] for x in snaps]))
    vis, brake, target = motor(2.0)
    assert vis == 1 and brake < 0.05
    vis_h, brake_h, target_h = motor(20.0)
    assert vis_h == 0 and brake_h > 0.25 and target_h < 0.7 * target
    # looming wall at speed: LPLC2/LC4 -> giant fiber -> emergency stop
    controls, snap = _settle(brain, dict(FAR, speed=12.0, distance_to_wall_front=3.0, distance_to_next_waypoint=20.0))
    assert snap["motor"]["emergency"] and controls["throttle"] < 0
    # driving along a side wall is not a collision course
    controls, snap = _settle(brain, dict(FAR, speed=15.0, distance_to_wall_left_20=8.0, distance_to_wall_left_45=4.0))
    assert not snap["motor"]["emergency"] and controls["throttle"] > 0
    # scrambled wiring: giant-fiber noise stays below the emergency threshold on an open road
    for seed in (0, 1):
        b = FlyBrain(scrambled(circuit, seed))
        hz = [b.step(dict(FAR, speed=15.0, angle_to_next_waypoint=a))[1]["motor"]["dnp01_hz"] for a in (0, 20, -20, 45)]
        assert max(hz) < b.readout.dnp01_min_hz


def test_calibrate_refuses_to_amplify_a_scrambled_brain(circuit):
    from agentic_gp.fly.brain import FlyBrain, ReadoutParams
    from agentic_gp.fly.connectome import scrambled

    b = FlyBrain(scrambled(circuit, 0))
    with pytest.warns(RuntimeWarning, match="does not steer"):
        p = b.calibrate(ms=200, warmup_ms=50)
    d = ReadoutParams()
    assert (p.dna02_gain_left, p.dna02_gain_right) == (d.dna02_gain_left, d.dna02_gain_right)
    assert abs(p.steer_offset) <= 0.3
