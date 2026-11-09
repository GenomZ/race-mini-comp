"""Tests for agentic_gp.driverkit on synthetic geometry only (no simulator import).

A tiny kinematic car below mirrors the sim's bicycle model so PathFollower can be checked end to end
(sign conventions: world heading CCW+, driver frame "+ = right").
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pytest

from agentic_gp import driverkit as dk


# ------------------------------------------------------------------ synthetic tracks
def stadium(r: float = 30.0, straight: float = 150.0, step: float = 2.0) -> np.ndarray:
    """Counter-clockwise stadium, evenly resampled every ~`step` units (like a real track centerline).

    Bottom straight left->right from (0, -r), right semicircle, top straight, left semicircle.
    """
    a = np.linspace(-math.pi / 2, math.pi / 2, 400)
    fine = np.vstack([
        np.c_[np.linspace(0, straight, 400, endpoint=False), np.full(400, -r)],
        np.c_[straight + r * np.cos(a), r * np.sin(a)][:-1],
        np.c_[np.linspace(straight, 0, 400, endpoint=False), np.full(400, r)],
        np.c_[r * np.cos(a + math.pi), r * np.sin(a + math.pi)][:-1],
    ])
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(fine, axis=0), axis=1))])
    total = s[-1] + np.linalg.norm(fine[0] - fine[-1])
    n = int(round(total / step))
    t = np.linspace(0, total, n, endpoint=False)
    closed = np.vstack([fine, fine[:1]])
    sc = np.append(s, total)
    return np.c_[np.interp(t, sc, closed[:, 0]), np.interp(t, sc, closed[:, 1])]


def track_dict(points: np.ndarray, half_width: float = 9.0, n_wp: int = 8, briefing: str = "") -> dict:
    """The parts of a track map that driverkit reads."""
    idx = np.linspace(0, len(points), n_wp, endpoint=False).astype(int)
    return {
        "key": "test", "name": "Test Stadium", "half_width": half_width, "laps": 1,
        "centerline": points.round(2).tolist(),
        "waypoints": [{"index": i, "center": points[k].tolist(), "centerline_index": int(k)} for i, k in enumerate(idx)],
        "physics": {"max_speed": 40.0, "accel": 12.0, "brake": 25.0, "drag": 0.15,
                    "max_lateral_accel": 20.0, "max_yaw_rate": 1.5, "tick_seconds": 0.5},
        "briefing": briefing,
    }


# ------------------------------------------------------------------ a mini sim (same model as the real one)
def integrate(state: dict, throttle: float, steering: float, p: dict, dt: float = 0.05) -> None:
    v = state["v"]
    a = throttle * (p["accel"] if throttle >= 0 else p["brake"]) - p["drag"] * v
    v_new = max(0.0, min(p["max_speed"], v + a * dt))
    v_mid = 0.5 * (v + v_new)
    yaw = steering * min(p["max_yaw_rate"], p["max_lateral_accel"] / max(abs(v_mid), 1e-6))
    h_new = state["h"] - yaw * dt                       # + steering = clockwise = right
    h_mid = 0.5 * (state["h"] + h_new)
    state["x"] += v_mid * math.cos(h_mid) * dt
    state["y"] += v_mid * math.sin(h_mid) * dt
    state["h"], state["v"] = h_new, v_new


def open_obs(state: dict, **extra) -> dict:
    """Observation with no walls in sight (rays long), enough for the follower's maths."""
    obs = {"speed": state["v"], "position": [state["x"], state["y"]], "heading_deg": math.degrees(state["h"]),
           "zone_speed_limit": None}
    for ray in ("front", "left_20", "right_20", "left_45", "right_45", "left_90", "right_90"):
        obs[f"distance_to_wall_{ray}"] = 100.0
    obs.update(extra)
    return obs


def drive_laps(follower: dk.PathFollower, points: np.ndarray, ticks: int, p: dict) -> dict:
    state = {"x": points[-1][0], "y": points[-1][1], "h": 0.0, "v": 0.0}
    max_off, speeds, travelled = 0.0, [], 0.0
    for _ in range(ticks):
        throttle, steering = follower.control(open_obs(state))
        x0, y0 = state["x"], state["y"]
        for _ in range(10):
            integrate(state, throttle, steering, p)
        travelled += math.hypot(state["x"] - x0, state["y"] - y0)
        max_off = max(max_off, float(np.min(np.linalg.norm(points - [state["x"], state["y"]], axis=1))))
        speeds.append(state["v"])
    return {"max_offset": max_off, "speeds": speeds, "distance": travelled}


# ------------------------------------------------------------------ small maths
def test_clamp_and_wrap():
    assert dk.clamp(2.0) == 1.0 and dk.clamp(-3.0) == -1.0 and dk.clamp(0.3) == 0.3
    assert dk.clamp(float("nan")) == 0.0 and dk.clamp(50, 0, 40) == 40
    assert dk.wrap_deg(190) == -170 and dk.wrap_deg(-190) == 170 and dk.wrap_deg(360) == 0


def test_speed_for_radius_and_braking_distance():
    assert dk.speed_for_radius(20.0) == pytest.approx(20.0)            # sqrt(20 * 20)
    assert dk.speed_for_radius(1e6) == 40.0 and dk.speed_for_radius(math.inf) == 40.0
    assert dk.braking_distance(40, 15) == pytest.approx((1600 - 225) / 50)
    assert dk.braking_distance(10, 20) == 0.0
    assert dk.stopping_speed(dk.braking_distance(30, 10), v_end=10) == pytest.approx(30)


def test_steer_towards_sign_and_speed_dependence():
    assert dk.steer_towards(20, 10) > 0 and dk.steer_towards(-20, 10) < 0       # + = right
    assert dk.steer_towards(0, 30) == 0.0
    assert dk.steer_towards(120, 5) == 1.0 and dk.steer_towards(-95, 30) == -1.0
    # Same aim, faster car: needs MORE steering (steering is a fraction of grip).
    slow, fast = dk.steer_towards(10, 10, 30.0), dk.steer_towards(10, 30, 30.0)
    assert 0 < slow < fast <= 1.0


# ------------------------------------------------------------------ geometry
def test_curvature_radii_circle_and_straight():
    a = np.linspace(0, 2 * math.pi, 200, endpoint=False)
    circle = np.c_[25 * np.cos(a), 25 * np.sin(a)]
    assert np.allclose(dk.curvature_radii(circle), 25.0, rtol=1e-3)
    assert np.all(dk.turn_directions(circle) == -1)                       # CCW = turning left
    pts = stadium()
    r = dk.curvature_radii(pts)
    assert np.isinf(r[5]) and abs(np.median(r[np.isfinite(r)]) - 30.0) < 1.0


def test_speed_profile_braking_pass():
    pts = stadium()
    p = track_dict(pts)["physics"]
    v = dk.speed_profile(pts, p["max_lateral_accel"], p["brake"], p["max_speed"])
    grip = np.minimum(40.0, np.sqrt(20.0 * np.minimum(dk.curvature_radii(pts), 1e9)))
    assert np.all(v <= grip + 1e-9)                                      # never above the grip limit
    ds = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
    nxt = np.roll(v, -1)
    assert np.all(v ** 2 <= nxt ** 2 + 2 * 25.0 * ds + 1e-6)              # can always brake in time
    assert v.max() == pytest.approx(40.0) and v.min() == pytest.approx(math.sqrt(20 * 30), rel=0.02)
    # Speed falls monotonically on the approach to the first bend (end of the bottom straight).
    bend = int(150.0 / np.mean(np.linalg.norm(np.diff(pts, axis=0), axis=1)))   # end of the bottom straight
    approach = v[bend - 15:bend + 1]
    assert np.all(np.diff(approach) <= 1e-9) and approach[0] > approach[-1]
    slower = dk.speed_profile(pts, aggression=0.8)
    assert np.all(slower <= v + 1e-9) and slower.sum() < v.sum()
    capped = dk.speed_profile(pts, caps=np.where(np.arange(len(pts)) == 10, 12.0, np.inf))
    assert capped[10] == pytest.approx(12.0) and capped[5] < v[5]       # brakes before the cap


def test_find_corners_on_stadium():
    pts = stadium()
    tr = track_dict(pts)
    corners = dk.track_corners(tr)
    assert [c["id"] for c in corners] == ["T1", "T2"]
    for c in corners:
        assert c["direction"] == "left" and c["min_radius"] == pytest.approx(30, abs=1.5)
        assert c["v_max"] == pytest.approx(math.sqrt(600), abs=0.6)
        assert {"start_index", "apex_index", "end_index", "from_wp", "to_wp"} <= set(c)


def test_racing_line_stays_inside_and_is_straighter():
    pts = stadium(r=25.0, straight=80.0)
    tr = track_dict(pts, half_width=9.0)
    line = dk.racing_line(tr, margin=2.0)
    assert line.shape == pts.shape
    offsets = np.linalg.norm(line - np.asarray(tr["centerline"]), axis=1)
    assert offsets.max() <= 7.0 + 1e-6 and offsets.max() > 3.0           # uses the width, within limits
    assert lap_time(line) < 0.95 * lap_time(pts)                          # faster on the same physics


def lap_time(points: np.ndarray) -> float:
    """Ideal lap time of a speed profile: sum of segment length / planned speed."""
    ds = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    return float(np.sum(ds / dk.speed_profile(points)))


# ------------------------------------------------------------------ control
def test_recover_if_stuck():
    stuck = open_obs({"x": 0, "y": 0, "h": 0, "v": 0.2}, distance_to_wall_front=2.0,
                     distance_to_wall_left_45=10.0, distance_to_wall_right_45=3.0)
    state: dict = {}
    assert dk.recover_if_stuck(stuck, state, ticks=2) == (-1.0, -1.0)    # reverse, nose to the open left
    moving = dict(stuck, speed=20.0, distance_to_wall_front=50.0)
    assert dk.recover_if_stuck(moving, state) == (-1.0, -1.0)             # still finishing the manoeuvre
    assert dk.recover_if_stuck(moving, state) is None


def test_path_follower_drives_laps_on_the_centerline():
    pts = stadium()
    tr = track_dict(pts)
    follower = dk.PathFollower.from_track(tr)
    lap = float(np.sum(np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)))
    result = drive_laps(follower, pts, ticks=60, p=tr["physics"])
    assert result["max_offset"] < 4.0                                     # stays well inside +-9
    assert result["distance"] > 1.5 * lap                                  # keeps lapping
    assert max(result["speeds"]) > 35.0                                   # flat out on the straights
    assert set(follower.debug) >= {"target", "v_plan", "v_target"}


def test_path_follower_localize_and_segments():
    pts = stadium()
    tr = track_dict(pts)
    f = dk.PathFollower.from_track(tr)
    assert f.localize(pts[40] + [0.5, 0.5]) == 40
    assert f.localize(pts[45]) == 45                                      # windowed search from last index
    assert f.segments[tr["waypoints"][3]["centerline_index"]] == 3
    assert f.segments[tr["waypoints"][3]["centerline_index"] - 1] == 2


def test_target_speed_respects_walls_zones_and_caps():
    pts = stadium()
    tr = track_dict(pts)
    f = dk.PathFollower.from_track(tr)
    state = {"x": pts[10][0], "y": pts[10][1], "h": 0.0, "v": 30.0}
    free = f.target_speed(open_obs(state))
    assert free == pytest.approx(40.0)
    assert f.target_speed(open_obs(state, distance_to_wall_front=20.0)) < 20.0   # must be able to stop
    assert f.target_speed(open_obs(state, zone_speed_limit=15.0)) == 15.0
    f.set_speed_limit(1, 2, 15.0)                                        # zone on segment 1 only
    assert set(f.segment_speed_caps) == {1}
    seg1 = tr["waypoints"][1]["centerline_index"]
    assert f.target_speed(open_obs({"x": pts[seg1 + 3][0], "y": pts[seg1 + 3][1], "h": 0, "v": 14})) <= 14.5
    # The braking pass makes it slow down BEFORE the zone starts.
    f.localize(pts[seg1 - 12])
    before = f.target_speed(open_obs({"x": pts[seg1 - 12][0], "y": pts[seg1 - 12][1], "h": 0, "v": 30}))
    assert before < 40.0
    throttle, _ = f.control(open_obs({"x": pts[seg1 - 12][0], "y": pts[seg1 - 12][1], "h": 0, "v": 30}))
    assert throttle < 0                                                   # braking for the zone


def test_set_speed_limit_wraps_past_waypoint_zero():
    f = dk.PathFollower.from_track(track_dict(stadium(), n_wp=8))
    f.set_speed_limit(7, 1, 20.0, margin=0.0)
    assert f.segment_speed_caps == {7: 20.0, 0: 20.0}


# ------------------------------------------------------------------ text helpers
def test_track_brief_mentions_corners_physics_and_briefing():
    tr = track_dict(stadium(), briefing="Yellow flag between waypoints 2 and 3: max speed 15.")
    text = dk.track_brief(tr)
    assert "T1" in text and "T2" in text and "+ = right" in text and "sqrt(20 * R)" in text
    assert "Yellow flag between waypoints 2 and 3" in text
    assert len(text) < 2000


def make_report(tr: dict) -> dict:
    pts = np.asarray(tr["centerline"])
    telemetry = [{"tick": k, "t": 0.5 * k, "lap": 1, "segment": 0, "progress": k / len(pts),
                  "x": float(pts[k][0]), "y": float(pts[k][1]), "speed": 25.0 if k % 7 else 18.0,
                  "throttle": 1.0, "steering": 0.0, "offset_from_center": 0.0} for k in range(0, len(pts), 3)]
    apex = dk.track_corners(tr)[1]["apex_index"]
    return {
        "attempt": 2, "completed": False, "race_time": None, "lap_times": [], "best_lap": None, "sim_time": 250.0,
        "penalty_time": 10.0, "crashes": 2, "waypoints_passed": 5, "laps": 1, "laps_completed": 0, "ticks": 500,
        "ended_by": "max_ticks", "next_phase": "qualifying", "best_lap_so_far": 61.2,
        "crash_log": [{"tick": 40, "t": 25.0, "lap": 1, "segment": 4, "progress": 0.6,
                       "position": pts[apex].tolist(), "speed_before": 31.0, "throttle": 1.0, "steering": -0.4,
                       "rays": {"distance_to_wall_front": 2.5, "distance_to_wall_left_45": 9.0}}] * 2,
        "segments": [{"segment": s, "time": 5.0, "entry_speed": 30.0, "min_speed": 20.0, "max_speed": 40.0,
                      "avg_speed": 31.0, "crashes": int(s == 4), "ticks": 10} for s in range(8)],
        "telemetry": telemetry, "zone_violations": 3, "stalls": 1, "driver_errors": [],
        "llm_usage": {"calls": 2, "prompt_tokens": 800, "completion_tokens": 200},
    }


def test_corner_stats_from_telemetry():
    tr = track_dict(stadium())
    rows = dk.corner_stats(make_report(tr), tr)
    assert [r["id"] for r in rows] == ["T1", "T2"]
    assert rows[1]["crashes"] == 2 and rows[0]["crashes"] == 0
    assert rows[0]["min_speed"] == 18.0 and rows[0]["passes"] == 1 and rows[0]["time"] > 0


def test_summarize_report_is_compact_and_complete():
    tr = track_dict(stadium())
    text = dk.summarize_report(make_report(tr), tr)
    for needle in ("Attempt 2", "next phase: qualifying", "DNF (max_ticks)", "Segments", "Corners",
                   "Crashes (2)", "(T2)", "rays F 2.5", "Zone violations: 3", "stalls 1", "llm calls 2"):
        assert needle in text, needle
    assert len(text) < 6000                                                # ~1.5k tokens
    finished = dict(make_report(tr), completed=True, race_time=62.5, lap_times=[62.5], best_lap=62.5,
                    laps_completed=1, crash_log=[], crashes=0)
    assert "FINISHED 1/1" in dk.summarize_report(finished) and "62.50" in dk.summarize_report(finished)


def test_driverkit_never_imports_the_sim():
    tree = ast.parse(Path(dk.__file__).read_text(encoding="utf-8"))
    modules = {getattr(n, "module", None) or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    modules |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not any(m.startswith(("agentic_gp", ".")) or m.startswith("sim") for m in modules), modules
