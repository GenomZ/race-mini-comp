import json

import numpy as np

from agentic_gp import geometry as geo
from agentic_gp.track import make_default_track, make_oval_track, TRACKS


def test_default_track_geometry_is_sane():
    tr = make_default_track()
    assert tr.n_gates == 16
    assert tr.wall_segments.shape == (2 * tr.n_points, 4)
    # offsetting a centerline by half_width only stays clean if every corner is wider than that
    assert tr.min_turn_radius() > tr.half_width * 1.2
    # walls stay half_width away from the centerline
    assert np.allclose(np.linalg.norm(tr.left_wall - tr.centerline, axis=1), tr.half_width)
    assert np.allclose(np.linalg.norm(tr.right_wall - tr.centerline, axis=1), tr.half_width)
    assert 1000 < tr.length < 2500


def test_gates_span_the_track():
    tr = make_default_track()
    for g in range(tr.n_gates):
        x1, y1, x2, y2 = tr.gates[g]
        assert np.isclose(np.hypot(x2 - x1, y2 - y1), 2 * tr.half_width)
        assert np.allclose(tr.gate_centers[g], [(x1 + x2) / 2, (y1 + y2) / 2])


def test_start_is_just_before_the_finish_line_on_the_straight():
    tr = make_default_track()
    d = np.linalg.norm(tr.start_position - tr.gate_centers[0])
    assert 0 < d < 3 * tr.length / tr.n_points
    # the start straight runs along +x
    assert abs(tr.start_heading) < 0.1


def test_track_map_is_json_serialisable():
    tr = make_default_track()
    m = tr.to_map()
    s = json.dumps(m)
    assert "waypoints" in m and len(m["waypoints"]) == tr.n_gates
    assert m["waypoints"][0]["index"] == 0
    assert all(w["min_turn_radius_until_next"] is None or w["min_turn_radius_until_next"] > 0 for w in m["waypoints"])
    assert len(s) > 100


def test_all_registered_tracks_build():
    for name, factory in TRACKS.items():
        tr = factory()
        assert tr.n_points > 0, name
        assert geo.polyline_length(tr.centerline) > 0


def test_oval_is_simpler_than_the_default_track():
    oval, default = make_oval_track(), make_default_track()
    assert oval.min_turn_radius() > default.min_turn_radius()
    assert oval.length < default.length
