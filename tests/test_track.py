import json
import math

import numpy as np
import pytest

from agentic_gp import TRACKS, Track, make_track
from agentic_gp.sim.track import (
    PUBLIC_TRACKS, Zone, load_track_file, make_grand_prix_track, track_from_control_points,
)

MAP_KEYS = {
    "key", "name", "track_width", "half_width", "lap_length", "n_waypoints", "start_position",
    "start_heading_deg", "waypoints", "centerline", "left_wall", "right_wall", "corners", "briefing", "notes",
}
CORNER_KEYS = {"id", "start_index", "apex_index", "end_index", "min_radius", "direction", "v_max", "from_wp", "to_wp"}


@pytest.mark.parametrize("key", PUBLIC_TRACKS)
def test_public_tracks_have_valid_geometry(key):
    tr = make_track(key)
    assert tr.key == key and tr.name
    assert tr.validate() == [], tr.validate()
    assert tr.min_turn_radius() > 1.2 * tr.half_width
    assert tr.wall_segments.shape == (2 * tr.n_points, 4)
    # walls stay half_width away from the centerline
    assert np.allclose(np.linalg.norm(tr.left_wall - tr.centerline, axis=1), tr.half_width)
    assert np.allclose(np.linalg.norm(tr.right_wall - tr.centerline, axis=1), tr.half_width)
    # the car starts one sample before the start/finish gate, on a straight
    d = np.linalg.norm(tr.start_position - tr.gate_centers[0])
    assert 0 < d < 3 * tr.length / tr.n_points
    assert tr.start_straight_radius() > 100


def test_registry_and_aliases():
    assert set(PUBLIC_TRACKS) <= set(TRACKS)
    gp = make_track("grand_prix")
    assert gp.name == "Grand Prix Circuit" and gp.n_gates == 16
    assert np.array_equal(make_track("default").centerline, gp.centerline)
    assert np.array_equal(make_track(None).centerline, gp.centerline)
    assert make_track(gp) is gp
    assert 1500 < gp.length < 1600 and abs(gp.start_heading) < 0.1
    with pytest.raises(KeyError):
        make_track("nope")


def test_new_tracks_have_their_character():
    sprint, speedway, gp = make_track("sprint"), make_track("speedway"), make_track("grand_prix")
    assert sprint.min_turn_radius() >= 12 and len(sprint.corners()) >= 6         # short and technical
    assert sprint.length < gp.length
    assert speedway.min_turn_radius() >= 30 and speedway.length > gp.length       # fast and flowing
    tight = lambda tr: sum(c["v_max"] < 30 for c in tr.corners())
    assert tight(sprint) > tight(speedway)


def test_track_map_contents():
    tr = make_track("grand_prix")
    m = tr.to_map()
    assert set(m) == MAP_KEYS
    json.dumps(m)
    assert m["key"] == "grand_prix" and m["half_width"] == 9.0 and m["n_waypoints"] == 16
    assert len(m["centerline"]) == len(m["left_wall"]) == len(m["right_wall"]) == tr.n_points
    assert all(round(v, 2) == v for p in m["centerline"][:20] for v in p)
    assert m["centerline"][-1] == m["start_position"]
    for w in m["waypoints"]:
        assert {"index", "center", "track_heading_deg", "min_turn_radius_until_next", "centerline_index"} <= set(w)
        assert m["centerline"][w["centerline_index"]] == w["center"]
    assert m["corners"] and all(set(c) == CORNER_KEYS for c in m["corners"])
    assert "No speed restrictions" in m["briefing"]
    assert "zones" not in m        # structured zones are NOT given to drivers: they must read the briefing


def test_corners_are_detected_in_lap_order():
    tr = make_track("sprint")
    cs = tr.corners()
    assert [c["id"] for c in cs] == [f"T{k + 1}" for k in range(len(cs))]
    assert [c["start_index"] for c in cs] == sorted(c["start_index"] for c in cs)
    for c in cs:
        assert c["direction"] in ("left", "right")
        assert c["start_index"] <= c["apex_index"] <= c["end_index"]
        assert math.isclose(c["v_max"], min(40.0, math.sqrt(20.0 * c["min_radius"])), abs_tol=0.11)
        assert 0 <= c["from_wp"] < tr.n_gates and 0 <= c["to_wp"] < tr.n_gates
    # the sprint's first corner is a left-hander, the chicane gives a right + left pair
    assert cs[0]["direction"] == "left"
    dirs = [c["direction"] for c in cs]
    assert any(a == "right" and b == "left" for a, b in zip(dirs, dirs[1:]))
    assert make_track("oval").corners() and all(c["direction"] == "left" for c in make_track("oval").corners())


def test_project_sign_convention_and_arc_length():
    tr = make_track("grand_prix")             # start straight runs along +x, so "left" is +y
    p0 = tr.centerline[0]
    s, off, i = tr.project(p0 + [0.0, 3.0])
    assert off == pytest.approx(-3.0, abs=0.05) and i == 0 and min(s, tr.length - s) < 0.5
    s, off, i = tr.project(p0 + [2.0, -4.0])
    assert off == pytest.approx(4.0, abs=0.05) and s == pytest.approx(2.0, abs=0.05)
    # along the centerline: offset ~0 and s grows with the sample index
    ss = []
    for k in range(0, tr.n_points, 7):
        s, off, i = tr.project(tr.centerline[k])
        assert abs(off) < 1e-6 and i == k
        ss.append(s)
    assert np.all(np.diff(ss) > 0)
    assert tr.project(tr.start_position)[0] == pytest.approx(tr.length - tr.arc[-1] + tr.arc[-2], abs=1e-6)


def test_project_offset_matches_walls():
    tr = make_track("sprint")
    for k in range(0, tr.n_points, 11):
        assert tr.project(tr.left_wall[k] * 0.999 + tr.centerline[k] * 0.001)[1] < -8.5
        assert tr.project(tr.right_wall[k] * 0.999 + tr.centerline[k] * 0.001)[1] > 8.5


def test_respawn_pose_stays_inside_the_segment():
    tr = make_track("grand_prix")
    for seg in range(tr.n_gates):
        idx = tr.segment_indices(seg)
        far_point = tr.centerline[idx[0]] + [500.0, 500.0]
        pos, heading = tr.respawn_pose(far_point, seg)
        k = int(np.argmin(np.linalg.norm(tr.centerline - pos, axis=1)))
        assert k in idx and tr.segment_of_index(k) == seg
        assert heading == pytest.approx(tr.heading_at(k))


def test_zones_and_briefing():
    z = Zone(15, 2, 12.0, "Safety car")
    assert z.segments(16) == [15, 0, 1]
    demo = make_track("briefing_demo")
    assert demo.key == "briefing_demo" and demo.validate() == []
    assert [zz.from_wp for zz in demo.zones] == [9]
    assert demo.zone_for_segment(9).max_speed == 15.0 and demo.zone_for_segment(11) is None
    b = demo.to_map()["briefing"]
    assert "between waypoints 9 and 11: max speed 15" in b and "Marshals" in b
    r = demo.zone_ranges()[0]
    assert (r["from_index"], r["to_index"]) == (int(demo.gate_indices[9]), int(demo.gate_indices[11]))
    for key in ("grand_prix", "oval", "sprint", "speedway"):
        assert make_track(key).zones == []
    # the demo is the grand prix circuit, same geometry
    assert np.array_equal(demo.centerline, make_grand_prix_track().centerline)


def test_validate_reports_bad_geometry():
    tiny = track_from_control_points([(0, 0), (12, 0), (12, 12), (0, 12)], name="tiny")
    assert any("radius" in p for p in tiny.validate())
    t = np.linspace(0, 2 * math.pi, 300, endpoint=False)
    eight = np.column_stack([200 * np.sin(t), 100 * np.sin(2 * t)])     # figure eight crosses itself
    assert "walls intersect" in Track(name="8", centerline=eight, half_width=9.0, n_gates=16).validate()


def test_track_files(tmp_path):
    f = tmp_path / "mystery.py"
    f.write_text(
        "NAME = 'Mystery'\n"
        "CONTROL_POINTS = [(150, 0), (300, 0), (380, 100), (300, 200), (0, 200), (-80, 100), (0, 0)]\n"
        "ZONES = [(3, 5, 18.0, 'Debris')]\n"
        "BRIEFING = 'Oil on the track in turn 2.'\n"
    )
    tr = make_track(f"file:{f}")
    assert tr.name == "Mystery" and tr.key == "file:mystery" and tr.validate() == []
    assert tr.zones == [Zone(3, 5, 18.0, "Debris")]
    assert "Debris between waypoints 3 and 5" in tr.briefing() and "Oil" in tr.briefing()
    assert make_track(str(f)).name == "Mystery"
    g = tmp_path / "custom.py"
    g.write_text("from agentic_gp.sim.track import make_oval_track\n"
                 "def make_track():\n    t = make_oval_track()\n    t.name = 'Custom'\n    t.key = 'custom'\n    return t\n")
    tr2 = load_track_file(g)
    assert tr2.name == "Custom" and tr2.key == "file:custom"
    with pytest.raises(FileNotFoundError):
        make_track(f"file:{tmp_path / 'missing.py'}")


def test_plot_track_accepts_track_map_and_replay_track(tmp_path):
    from agentic_gp.viz.mpl import plot_track

    tr = make_track("briefing_demo")
    m = tr.to_map()
    replay_track = {
        "name": tr.name, "half_width": tr.half_width, "length": tr.length,
        "center_x": tr.centerline[:, 0].tolist(), "center_y": tr.centerline[:, 1].tolist(),
        "left_x": tr.left_wall[:, 0].tolist(), "left_y": tr.left_wall[:, 1].tolist(),
        "right_x": tr.right_wall[:, 0].tolist(), "right_y": tr.right_wall[:, 1].tolist(),
        "gate_x1": tr.gates[:, 0].tolist(), "gate_y1": tr.gates[:, 1].tolist(),
        "gate_x2": tr.gates[:, 2].tolist(), "gate_y2": tr.gates[:, 3].tolist(), "zones": tr.zone_ranges(),
    }
    for k, src in enumerate((tr, m, {"track": replay_track})):
        out = plot_track(src, tmp_path / f"t{k}.png")
        assert out.exists() and out.stat().st_size > 5000
