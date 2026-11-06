import math

import numpy as np
import pytest

from agentic_gp import RaceEnv, make_track
from agentic_gp.sim.track import RANDOM_LENGTH, RANDOM_MIN_RADIUS, make_random_track
from test_env import follow_centerline, run   # the tiny test planner (tests/ is on sys.path)

SEEDS = range(20)


def test_same_seed_same_track():
    a, b = make_track("random:42"), make_random_track(42)
    assert np.array_equal(a.centerline, b.centerline)
    assert a.key == "random:42" and a.name == "Random #42"
    a.centerline[0] += 100.0                       # tracks never share arrays (the generator is cached)
    assert not np.array_equal(make_track("random:42").centerline, a.centerline)
    assert not np.array_equal(make_track("random:43").centerline, b.centerline)


def test_string_seeds():
    t = make_track("random:mystery-monday")
    assert t.key == "random:mystery-monday"
    assert np.array_equal(t.centerline, make_track("random:mystery-monday").centerline)
    assert t.validate(min_radius=RANDOM_MIN_RADIUS) == []


@pytest.mark.parametrize("seed", SEEDS)
def test_random_tracks_stay_in_the_envelope(seed):
    tr = make_track(f"random:{seed}")
    assert RANDOM_LENGTH[0] <= tr.length <= RANDOM_LENGTH[1]
    assert tr.half_width == 9.0 and tr.n_gates == 16
    assert tr.min_turn_radius() >= RANDOM_MIN_RADIUS
    assert tr.validate(min_radius=RANDOM_MIN_RADIUS) == []        # incl. no wall overlap, start on a straight
    # normalised pose: the start line at the origin, the start straight heading +x
    assert np.allclose(tr.centerline[0], 0.0) and abs(math.degrees(tr.heading_at(0))) < 1.0
    m = tr.to_map()
    assert m["key"] == f"random:{seed}" and len(m["centerline"]) == tr.n_points and m["corners"]


def test_random_tracks_differ():
    lengths = {round(make_track(f"random:{s}").length) for s in SEEDS}
    assert len(lengths) > len(SEEDS) // 2


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
def test_tiny_planner_laps_random_tracks_cleanly(seed):
    env = RaceEnv(track=f"random:{seed}", laps=2)
    run(env, follow_centerline(env.track, env.cfg))
    res = env.attempts[-1]
    assert res.completed and res.crashes == 0 and res.laps_completed == 2
    assert sum(res.lap_times) == pytest.approx(res.race_time, abs=2e-3)


def test_negative_and_odd_seeds_make_tracks():
    for spec in ("random:-5", "random:1.5", "random:" + str(2 ** 40)):
        t = make_track(spec)
        assert t.key == spec and t.validate(min_radius=RANDOM_MIN_RADIUS) == []
    assert np.array_equal(make_track("random:-5").centerline, make_track("random:-5").centerline)
    assert not np.array_equal(make_track("random:-5").centerline, make_track("random:5").centerline)
