import json
import math
import time

import numpy as np
import pytest

from agentic_gp import PhysicsConfig, RaceEnv, make_track
from agentic_gp.sim import OBS_KEYS
from agentic_gp.sim.track import PUBLIC_TRACKS

FRAME_KEYS = {
    "tick", "t", "sim_time", "x", "y", "heading_deg", "speed", "throttle", "steering", "lap", "segment",
    "progress", "race_progress", "offset_from_center", "events", "crashed",
}


def follow_centerline(track, cfg, aggression=0.85, respect_zones=True):
    """A tiny planner for tests: speed profile from curvature (+ zone limits, as if it had read
    the briefing) with a backward braking pass, curvature feed-forward steering plus
    heading/offset feedback via Track.project, and proportional throttle/brake."""
    n, ds = track.n_points, track.length / track.n_points
    v = np.minimum(cfg.max_speed, np.sqrt(cfg.max_lateral_accel * aggression * track.radii))
    if respect_zones:
        for z in track.zones:
            for seg in z.segments(track.n_gates):
                idx = track.segment_indices(seg)
                v[idx] = np.minimum(v[idx], z.max_speed - 1.0)
                v[idx[-1] - 3:idx[-1] + 1] = np.minimum(v[idx[-1] - 3:idx[-1] + 1], z.max_speed - 1.0)
    for i in range(2 * n - 1, -1, -1):                       # brake early enough for what comes next
        v[i % n] = min(v[i % n], math.sqrt(v[(i + 1) % n] ** 2 + 2 * 0.7 * cfg.brake * ds))

    def drive(obs):
        spd = max(obs["speed"], 0.5)
        _, off, i = track.project(obs["position"])
        k = track.curvature[(i + max(1, int(spd * 0.25 / ds))) % n]          # + = left
        yaw_per_steer = min(cfg.max_yaw_rate, cfg.max_lateral_accel / spd)
        tx, ty = track.tangents[i]
        heading_err = (math.radians(obs["heading_deg"]) - math.atan2(ty, tx) + math.pi) % (2 * math.pi) - math.pi
        steering = -spd * k / yaw_per_steer + 1.2 * heading_err - 0.08 * off  # + = right everywhere
        target = min(v[(i + q) % n] for q in range(int(spd * 0.5 / ds) + 2))
        if spd > target:
            throttle = -(spd - target) / (cfg.brake * cfg.tick_seconds)
        else:
            throttle = ((target - spd) / cfg.tick_seconds + cfg.drag * spd) / cfg.accel
        return max(-1.0, min(1.0, throttle)), max(-1.0, min(1.0, steering))

    return drive


def run(env, drive, laps=None, until=None):
    obs = env.reset(laps)
    while not obs["attempt_over"] and not (until and until(obs)):
        obs = env.step(*drive(obs))
    return obs


# ------------------------------------------------------------------ observations
def test_reset_observation():
    env = RaceEnv(laps=2)
    obs = env.reset()
    assert set(obs) == set(OBS_KEYS) | {"events"}
    assert obs["speed"] == 0 and obs["tick"] == 0 and obs["next_waypoint_index"] == 1 and obs["segment"] == 0
    assert obs["progress"] == 0 and obs["race_progress"] == 0 and obs["lap"] == 1 and obs["laps_total"] == 2
    assert obs["finished"] is False and obs["attempt_over"] is False and obs["events"] == []
    assert obs["zone_speed_limit"] is None and obs["phase"] == "session" and obs["dt"] == 0.5
    assert obs["ticks_remaining"] == 2 * env.cfg.max_ticks
    assert obs["distance_to_wall_left_90"] == pytest.approx(9.0, abs=0.2)
    assert obs["distance_to_wall_right_90"] == pytest.approx(9.0, abs=0.2)
    assert abs(obs["offset_from_center"]) < 0.01
    json.dumps(obs)


def test_step_advances_one_tick_and_is_deterministic():
    a, b = RaceEnv(), RaceEnv()
    oa, ob = a.step(1.0, 0.0), b.step(1.0, 0.0)
    assert oa == ob and oa["tick"] == 1 and oa["speed"] > 0 and oa["sim_time"] == a.cfg.tick_seconds
    drive = follow_centerline(a.track, a.cfg)
    run(a, drive, laps=2)
    run(b, drive, laps=2)
    fa, fb = a.attempt_traces[-1]["frames"], b.attempt_traces[-1]["frames"]
    assert fa == fb and a.attempts[-1] == b.attempts[-1]


def test_controls_are_clipped_and_sanitised():
    a, b = RaceEnv(), RaceEnv()
    assert a.step(5.0, -9.0) == b.step(1.0, -1.0)
    c, d = RaceEnv(), RaceEnv()
    assert c.step(float("nan"), "left") == d.step(0.0, 0.0)


def test_left_is_negative():
    env = RaceEnv()
    env.step(1.0, 0.0)
    for _ in range(3):
        obs = env.step(0.3, -0.4)              # steer left on the start straight (+x), drift to +y
    assert obs["position"][1] > 0 and obs["offset_from_center"] < 0
    assert obs["distance_to_wall_left_90"] < obs["distance_to_wall_right_90"]
    assert obs["angle_to_next_waypoint"] > 0   # the track is now to the right of the nose


# ------------------------------------------------------------------ crashes
def _crash_into_first_corner(env):
    obs = env.reset()
    for _ in range(80):
        before = obs
        obs = env.step(1.0, 0.0)               # the oval's first corner is a wall straight ahead
        if obs["crashes"]:
            return before, obs
    raise AssertionError("no crash")


def test_marshal_rule_puts_the_car_back_on_track():
    env = RaceEnv(track="oval")
    before, obs = _crash_into_first_corner(env)
    tr = env.track
    assert obs["crashes"] == 1 and obs["penalty_time"] == 5.0 and obs["speed"] == 0
    assert any(e.startswith("crash: hit a wall, +5s penalty") for e in obs["events"])
    assert abs(obs["offset_from_center"]) < 0.01
    k = int(np.argmin(np.linalg.norm(tr.centerline - obs["position"], axis=1)))
    assert tr.segment_of_index(k) == obs["segment"] == before["segment"]
    assert math.radians(obs["heading_deg"]) == pytest.approx(math.atan2(*tr.tangents[k][::-1]), abs=1e-3)
    obs = env.step(1.0, 0.0)                   # facing along the track again: no crash loop
    assert obs["crashes"] == 1 and obs["speed"] > 0
    frame = env.frames[-2]
    assert frame["crashed"] and env.crash_points


def test_classic_rule_stops_the_car_where_it_hit():
    env = RaceEnv(track="oval", config=PhysicsConfig(crash_rule="classic"))
    before, obs = _crash_into_first_corner(env)
    assert obs["crashes"] == 1 and obs["speed"] == 0 and obs["penalty_time"] == 5.0
    assert obs["heading_deg"] == before["heading_deg"]          # not turned back along the track
    assert abs(obs["offset_from_center"]) > 3                   # still out by the wall
    with pytest.raises(ValueError):
        PhysicsConfig(crash_rule="bumper cars")


def test_marshal_respawn_never_skips_a_waypoint():
    """Chaotic driving with many crashes: gates are still passed strictly in order."""
    env = RaceEnv(track="grand_prix", config=PhysicsConfig(max_ticks=400))
    rng = np.random.default_rng(3)
    obs = env.reset()
    while not obs["attempt_over"]:
        obs = env.step(rng.uniform(0.3, 1.0), rng.uniform(-1, 1))
    frames = env.attempt_traces[-1]["frames"]
    passed = [int(e.split(": ")[1]) for f in frames for e in f["events"] if e.startswith("waypoint_passed")]
    assert env.attempts[-1].crashes >= 5
    assert passed == [(1 + k) % 16 for k in range(len(passed))]
    for f in frames:
        if f["crashed"]:
            k = int(np.argmin(np.linalg.norm(env.track.centerline - [f["x"], f["y"]], axis=1)))
            assert env.track.segment_of_index(k) == f["segment"]


# ------------------------------------------------------------------ laps & timing
def test_dnf_after_max_ticks_per_lap():
    env = RaceEnv(config=PhysicsConfig(max_ticks=5), laps=2)
    obs = env.reset()
    for _ in range(9):
        obs = env.step(0.0, 0.0)
    assert not obs["attempt_over"] and obs["ticks_remaining"] == 1
    obs = env.step(0.0, 0.0)
    assert obs["attempt_over"] and env.done and "attempt_over: max ticks reached (DNF)" in obs["events"]
    res = env.attempts[0]
    assert res.ended_by == "max_ticks" and not res.completed and res.race_time is None and res.ticks == 10
    after = env.step(1.0, 0.0)                 # stepping a finished attempt is a no-op with a hint
    assert after["tick"] == 10 and after["events"] == ["attempt_over: call reset to start a new attempt"]


@pytest.mark.parametrize("key", PUBLIC_TRACKS)
def test_tiny_planner_laps_every_public_track_cleanly(key):
    env = RaceEnv(track=key)
    obs = run(env, follow_centerline(env.track, env.cfg))
    res = env.attempts[-1]
    assert obs["finished"] and res.completed and res.ended_by == "finished"
    assert res.crashes == 0 and res.zone_violations == 0
    assert res.waypoints_passed == env.track.n_gates
    assert 20 < res.race_time < 90


def test_multi_lap_timing():
    env = RaceEnv(track="sprint", laps=3)
    obs = run(env, follow_centerline(env.track, env.cfg))
    res = env.attempts[-1]
    assert res.completed and res.laps == res.laps_completed == 3 and len(res.lap_times) == 3
    assert sum(res.lap_times) == pytest.approx(res.race_time, abs=2e-3)
    assert res.lap_times[0] > max(res.lap_times[1:])            # standing start
    assert res.lap_times[1] == pytest.approx(res.lap_times[2], abs=1.0)
    assert res.best_lap == min(res.lap_times) == env.best_lap
    assert res.waypoints_passed == 3 * env.track.n_gates
    assert obs["lap"] == 3 and obs["race_progress"] == 3.0 and obs["progress"] == 1.0
    frames = env.attempt_traces[-1]["frames"]
    events = [e for f in frames for e in f["events"]]
    laps = [e for e in events if e.startswith("lap_complete")]
    assert laps == [f"lap_complete: lap {k + 1} in {t:.2f}s" for k, t in enumerate(res.lap_times)]
    assert events[-1] == f"finished: 3 laps in {res.race_time:.2f}s (incl. 0s penalties)"
    # every lap starts with a frame on lap k + 1 just after crossing the line
    assert [f["lap"] for f in frames].count(2) > 50


def test_penalties_belong_to_the_lap_of_the_crash():
    track = make_track("oval")
    clean = RaceEnv(track=track, laps=2)
    drive = follow_centerline(track, clean.cfg)
    run(clean, drive)
    env = RaceEnv(track=track, laps=2)
    state = {"crash_tick": None}

    def crash_in_lap_two(obs):
        if obs["lap"] == 2 and obs["segment"] == 2 and not obs["crashes"]:
            return 1.0, 1.0                     # swerve right into the wall
        return drive(obs)

    obs = run(env, crash_in_lap_two)
    res, ref = env.attempts[-1], clean.attempts[-1]
    assert res.completed and res.crashes == 1 and res.penalty_time == 5.0
    assert res.lap_times[0] == ref.lap_times[0]                    # lap 1 untouched
    assert res.lap_times[1] > ref.lap_times[1] + 5.0               # penalty + lost time in lap 2
    assert sum(res.lap_times) == pytest.approx(res.race_time, abs=2e-3)
    crash_frames = [f for f in env.attempt_traces[-1]["frames"] if f["crashed"]]
    assert len(crash_frames) == 1 and crash_frames[0]["lap"] == 2
    assert "(incl. 5s penalties)" in env.attempt_traces[-1]["frames"][-1]["events"][-1]


def test_progress_starts_at_zero_and_grows():
    env = RaceEnv(track="grand_prix", laps=2)
    run(env, follow_centerline(env.track, env.cfg))
    frames = env.attempt_traces[-1]["frames"]
    assert frames[0]["progress"] == 0 and frames[0]["race_progress"] == 0
    assert all(f["progress"] < 0.05 for f in frames[:5])          # no jump to ~1 behind the start line
    rp = [f["race_progress"] for f in frames]
    assert np.all(np.diff(rp) >= 0) and rp[-1] == 2.0
    lap1 = [f["progress"] for f in frames if f["lap"] == 1]
    assert np.all(np.diff(lap1) >= 0) and lap1[-1] > 0.95
    # race_progress = laps completed + progress, and progress stays inside the current segment
    tr = env.track
    for f in frames[1:-1]:
        assert f["race_progress"] == pytest.approx(f["lap"] - 1 + f["progress"])
        lo = tr.gate_arc(f["segment"]) / tr.length
        hi = 1.0 if f["segment"] == tr.n_gates - 1 else tr.gate_arc(f["segment"] + 1) / tr.length
        assert lo - 1e-9 <= f["progress"] <= hi + 1e-9


def test_frames_are_complete():
    env = RaceEnv(track="oval")
    run(env, follow_centerline(env.track, env.cfg))
    trace = env.attempt_traces[-1]
    frames, res = trace["frames"], trace["result"]
    assert trace["attempt"] == 1 and trace["crash_points"] == []
    assert len(frames) == res.ticks + 1
    assert all(FRAME_KEYS <= set(f) for f in frames)
    f0 = frames[0]
    assert f0["tick"] == 0 and f0["t"] == 0 and f0["throttle"] == 0 and f0["events"] == []
    assert [f["tick"] for f in frames] == list(range(len(frames)))
    assert np.all(np.diff([f["t"] for f in frames]) >= 0)
    assert frames[-1]["t"] == pytest.approx(res.race_time, abs=1e-3)
    json.dumps(frames)


# ------------------------------------------------------------------ zones
def test_zone_penalties():
    track = make_track("briefing_demo")
    env = RaceEnv(track=track)
    obs = run(env, follow_centerline(track, env.cfg, respect_zones=False))
    res = env.attempts[-1]
    assert res.completed and res.crashes == 0 and res.zone_violations > 3
    assert res.penalty_time == pytest.approx(res.zone_violations * env.cfg.zone_penalty_seconds)
    assert res.race_time == pytest.approx(res.sim_time + res.penalty_time, abs=2e-3)
    assert sum(res.lap_times) == pytest.approx(res.race_time, abs=2e-3)
    frames = env.attempt_traces[-1]["frames"]
    hits = [f for f in frames if f.get("zone_violation")]
    assert len(hits) == res.zone_violations and all(f["segment"] in (9, 10) for f in hits)
    assert all(f["speed"] > 15.5 for f in hits)
    msg = next(e for f in hits for e in f["events"] if e.startswith("zone_violation"))
    assert "limit 15" in msg and "+1s penalty" in msg

    careful = RaceEnv(track=track)
    seen = []
    drive = follow_centerline(track, careful.cfg)

    def spy(obs):
        seen.append((obs["segment"], obs["zone_speed_limit"]))
        return drive(obs)

    run(careful, spy)
    assert careful.attempts[-1].completed and careful.attempts[-1].zone_violations == 0
    assert {lim for seg, lim in seen if seg in (9, 10)} == {15.0}
    assert {lim for seg, lim in seen if seg not in (9, 10)} == {None}


# ------------------------------------------------------------------ lifecycle
def test_stop_ends_the_attempt():
    env = RaceEnv()
    env.step(1.0, 0.0)
    obs = env.stop()
    assert obs["attempt_over"] and env.attempts[-1].ended_by == "stopped" and obs["events"] == ["attempt_over: stopped"]
    assert env.stop()["events"] == []               # idempotent
    assert env.step(1.0, 0.0)["events"][0].startswith("attempt_over")
    obs = env.reset()
    assert env.attempt_no == 2 and obs["tick"] == 0 and not obs["attempt_over"]


def test_reset_records_only_driven_attempts():
    env = RaceEnv()
    env.reset()
    env.reset()                       # not driven -> not recorded
    assert env.attempts == [] and env.attempt_no == 1
    env.step(0.5, 0.0)
    obs = env.reset(laps=3)           # driven -> recorded as "reset"; the new attempt has 3 laps
    assert len(env.attempts) == 1 and env.attempts[0].ended_by == "reset" and env.attempt_no == 2
    assert obs["laps_total"] == 3 and obs["ticks_remaining"] == 3 * env.cfg.max_ticks


def test_best_lap_over_attempts():
    env = RaceEnv(track="oval")
    assert env.best_lap is None
    run(env, follow_centerline(env.track, env.cfg, aggression=0.6))
    slow = env.attempts[-1].race_time
    run(env, follow_centerline(env.track, env.cfg, aggression=0.9), laps=2)
    assert env.best_lap == min(slow, *env.attempts[-1].lap_times) < slow
    assert [a.attempt for a in env.attempts] == [1, 2]
    d = env.attempts[-1].to_dict()
    assert {"attempt", "completed", "race_time", "lap_times", "best_lap", "sim_time", "penalty_time", "crashes",
            "waypoints_passed", "laps", "laps_completed", "ticks", "ended_by"} <= set(d)


def test_track_map_has_physics_and_laps():
    env = RaceEnv(track="sprint", laps=3)
    m = env.track_map()
    assert set(env.track.to_map()) | {"physics", "laps"} == set(m)
    assert m["laps"] == 3 and m["key"] == "sprint"
    p = m["physics"]
    assert p["max_speed"] == 40.0 and p["crash_rule"] == "marshal" and p["max_ticks_per_lap"] == 500
    assert p["tick_seconds"] == 0.5 and p["max_lateral_accel"] == 20.0
    json.dumps(m)


def test_jsonl_logging(tmp_path):
    env = RaceEnv(log_dir=tmp_path, run_name="t")
    env.step(1.0, 0.0)
    env.close()
    lines = [json.loads(l) for l in (tmp_path / "t.jsonl").read_text().splitlines()]
    assert [l["event"] for l in lines] == ["reset", "step", "attempt_end"]
    assert lines[1]["throttle"] == 1.0 and lines[2]["ended_by"] == "stopped"


def test_step_is_fast():
    env = RaceEnv(track="grand_prix", laps=3)
    drive = follow_centerline(env.track, env.cfg)
    obs = env.reset()
    t0, n = time.perf_counter(), 0
    while not obs["attempt_over"]:
        obs = env.step(*drive(obs))
        n += 1
    per_step = (time.perf_counter() - t0) / n
    assert per_step < 0.01, f"{per_step * 1000:.2f} ms per step"     # typically < 1 ms


def test_plots_from_an_env_trace(tmp_path):
    from agentic_gp.viz.mpl import plot_attempt, plot_telemetry

    env = RaceEnv(track="oval")
    run(env, follow_centerline(env.track, env.cfg))
    trace = env.attempt_traces[-1]
    assert plot_attempt(env.track, trace, tmp_path / "a.png").exists()
    assert plot_telemetry(trace, tmp_path / "t.png").exists()


# ------------------------------------------------------------------ review fixes
def test_zone_at_the_start_line_is_not_judged_after_the_finish():
    """A zone on segment 0 (e.g. a pit exit) judges the tick that crosses gate 0 in the NEW lap; the
    finish line has no new lap, so the final crossing is never penalised."""
    from agentic_gp.sim.track import Zone, make_oval_track

    track = make_oval_track()
    track.zones = [Zone(0, 1, 5.0, "Pit exit")]
    env = RaceEnv(track, laps=2)
    drive = follow_centerline(track, env.cfg, respect_zones=False)
    run(env, drive)
    frames = env.attempt_traces[-1]["frames"]
    finish = frames[-1]
    assert any(e.startswith("finished") for e in finish["events"])
    assert not finish.get("zone_violation") and not any("zone_violation" in e for e in finish["events"])
    lap_change = next(f for f in frames if any(e.startswith("lap_complete: lap 1") for e in f["events"]))
    assert lap_change.get("zone_violation")                  # judged in (and booked to) lap 2
    res = env.attempt_traces[-1]["result"]
    assert res.race_time == pytest.approx(sum(res.lap_times))
