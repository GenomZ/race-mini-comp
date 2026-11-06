import math

import numpy as np

from agentic_gp.sim import geometry as geo


def test_raycast_hits_segment_ahead_and_misses_behind():
    seg = np.array([[10.0, -5.0, 10.0, 5.0]])
    assert math.isclose(geo.raycast(np.zeros(2), 0.0, seg, 100.0), 10.0, abs_tol=1e-9)
    assert geo.raycast(np.zeros(2), math.pi, seg, 100.0) == 100.0


def test_raycast_respects_max_distance_and_left_convention():
    seg = np.array([[0.0, 8.0, 20.0, 8.0]])  # wall 8 units to the left (positive y) of a car heading +x
    assert math.isclose(geo.raycast(np.zeros(2), math.pi / 2, seg, 100.0), 8.0, abs_tol=1e-9)
    assert geo.raycast(np.zeros(2), math.pi / 2, seg, 5.0) == 5.0


def test_raycast_many_matches_single_rays():
    rng = np.random.default_rng(0)
    segs = rng.uniform(-50, 50, size=(40, 4))
    origin = np.array([1.5, -2.0])
    angles = np.linspace(-math.pi, math.pi, 25)
    many = geo.raycast_many(origin, angles, segs, 80.0)
    single = [geo.raycast(origin, a, segs, 80.0) for a in angles]
    assert np.allclose(many, single)
    # no segments at all -> every ray sees max distance
    assert geo.raycast_many(origin, angles, np.zeros((0, 4)), 80.0).tolist() == [80.0] * len(angles)


def test_segment_hits_detects_crossing_only():
    walls = np.array([[5.0, -1.0, 5.0, 1.0], [50.0, -1.0, 50.0, 1.0]])
    hits = geo.segment_hits(np.array([0.0, 0.0]), np.array([10.0, 0.0]), walls)
    assert hits.tolist() == [True, False]
    assert not geo.segment_hits(np.array([0.0, 0.0]), np.array([4.0, 0.0]), walls).any()
    assert geo.crosses(np.array([4.0, 0.5]), np.array([6.0, -0.5]), walls[0])
    assert geo.crosses_xy(4.0, 0.5, 6.0, -0.5, walls[0])
    assert not geo.crosses_xy(4.0, 0.5, 4.5, -0.5, walls[0])


def test_first_hit_returns_the_earliest_crossing():
    walls = np.array([[25.0, -1.0, 25.0, 1.0], [12.0, -1.0, 12.0, 1.0]])
    path = np.column_stack([np.arange(0.0, 40.0, 5.0), np.zeros(8)])   # pieces of 5 units along +x
    k, t, p = geo.first_hit(path, walls)
    assert k == 2 and math.isclose(t, 0.4) and np.allclose(p, [12.0, 0.0])
    assert geo.first_hit(path[:2], walls) is None
    assert geo.first_hit(path, np.zeros((0, 4))) is None


def test_segments_intersect_matrix():
    a = np.array([[0.0, 0.0, 10.0, 0.0]])
    b = np.array([[5.0, -1.0, 5.0, 1.0], [20.0, -1.0, 20.0, 1.0], [0.0, 1.0, 10.0, 1.0]])
    assert geo.segments_intersect(a, b).tolist() == [[True, False, False]]


def test_catmull_rom_passes_through_control_points():
    pts = np.array([(0, 0), (10, 0), (10, 10), (0, 10)], dtype=float)
    curve = geo.catmull_rom_closed(pts, samples_per_segment=8)
    assert curve.shape == (32, 2)
    for i, p in enumerate(pts):
        assert np.allclose(curve[i * 8], p)


def test_resample_closed_is_uniform():
    square = np.array([(0, 0), (10, 0), (10, 10), (0, 10)], dtype=float)
    pts = geo.resample_closed(square, 40)
    assert pts.shape == (40, 2)
    step = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
    assert np.allclose(step, 1.0)
    assert math.isclose(geo.polyline_length(pts), 40.0)


def test_curvature_of_circle_radius_and_sign():
    t = np.linspace(0, 2 * math.pi, 200, endpoint=False)
    circle = np.column_stack([25 * np.cos(t), 25 * np.sin(t)])          # counter-clockwise
    assert np.allclose(geo.curvature_radius_closed(circle), 25.0, rtol=1e-3)
    assert np.allclose(geo.signed_curvature_closed(circle), 1 / 25.0, rtol=1e-3)      # + = turning left
    assert np.allclose(geo.signed_curvature_closed(circle[::-1]), -1 / 25.0, rtol=1e-3)
    tan = geo.tangents_closed(circle)
    assert np.allclose(np.linalg.norm(tan, axis=1), 1.0)
    assert np.allclose(tan[0], [0.0, 1.0], atol=1e-3)                 # at (25, 0) a CCW circle heads +y


def test_signed_angle_and_wrap():
    assert math.isclose(abs(geo.wrap_angle(3 * math.pi)), math.pi)
    # a target straight to the left of a car heading +x is +90 degrees (CCW)
    assert math.isclose(geo.signed_angle_to(0.0, np.array([0.0, 1.0])), math.pi / 2)
    assert math.isclose(geo.signed_angle_to(0.0, np.array([0.0, -1.0])), -math.pi / 2)
