import math

import numpy as np

from agentic_gp import geometry as geo


def test_raycast_hits_segment_ahead_and_misses_behind():
    seg = np.array([[10.0, -5.0, 10.0, 5.0]])
    assert math.isclose(geo.raycast(np.zeros(2), 0.0, seg, 100.0), 10.0, abs_tol=1e-9)
    assert geo.raycast(np.zeros(2), math.pi, seg, 100.0) == 100.0


def test_raycast_respects_max_distance_and_left_convention():
    seg = np.array([[0.0, 8.0, 20.0, 8.0]])  # wall 8 units to the left (positive y) of a car heading +x
    assert math.isclose(geo.raycast(np.zeros(2), math.pi / 2, seg, 100.0), 8.0, abs_tol=1e-9)
    assert geo.raycast(np.zeros(2), math.pi / 2, seg, 5.0) == 5.0


def test_segment_hits_detects_crossing_only():
    walls = np.array([[5.0, -1.0, 5.0, 1.0], [50.0, -1.0, 50.0, 1.0]])
    hits = geo.segment_hits(np.array([0.0, 0.0]), np.array([10.0, 0.0]), walls)
    assert hits.tolist() == [True, False]
    assert not geo.segment_hits(np.array([0.0, 0.0]), np.array([4.0, 0.0]), walls).any()
    assert geo.crosses(np.array([4.0, 0.5]), np.array([6.0, -0.5]), walls[0])


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


def test_curvature_radius_of_circle():
    t = np.linspace(0, 2 * math.pi, 200, endpoint=False)
    circle = np.column_stack([25 * np.cos(t), 25 * np.sin(t)])
    r = geo.curvature_radius_closed(circle)
    assert np.allclose(r, 25.0, rtol=1e-3)


def test_signed_angle_and_wrap():
    assert math.isclose(geo.wrap_angle(3 * math.pi), math.pi) or math.isclose(geo.wrap_angle(3 * math.pi), -math.pi)
    # a target straight to the left of a car heading +x is +90 degrees (CCW)
    assert math.isclose(geo.signed_angle_to(0.0, np.array([0.0, 1.0])), math.pi / 2)
    assert math.isclose(geo.signed_angle_to(0.0, np.array([0.0, -1.0])), -math.pi / 2)
