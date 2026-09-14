import json

from agentic_gp import RaceEnv, PhysicsConfig, make_tools
from agentic_gp.agents.reflex import run as run_reflex
from agentic_gp.track import make_oval_track

SENSOR_KEYS = {
    "speed", "distance_to_wall_front", "distance_to_wall_left_20", "distance_to_wall_right_20",
    "distance_to_wall_left_45", "distance_to_wall_right_45", "distance_to_wall_left_90",
    "distance_to_wall_right_90", "distance_to_next_waypoint", "angle_to_next_waypoint",
    "distance_to_waypoint_after_next", "angle_to_waypoint_after_next", "next_waypoint_index",
    "waypoints_passed", "waypoints_total", "position", "heading_deg", "sim_time", "penalty_time",
    "crashes", "tick", "ticks_remaining", "lap_complete", "attempt_over",
}


def test_reset_observation():
    env = RaceEnv()
    obs = env.reset()
    assert set(obs) == SENSOR_KEYS
    assert obs["speed"] == 0 and obs["tick"] == 0 and obs["next_waypoint_index"] == 1
    assert obs["distance_to_wall_left_90"] > 0 and obs["distance_to_wall_right_90"] > 0
    json.dumps(obs)  # serialisable


def test_step_advances_one_tick_and_is_deterministic():
    a, b = RaceEnv(), RaceEnv()
    oa = a.step(1.0, 0.0)
    ob = b.step(1.0, 0.0)
    assert oa == ob
    assert oa["tick"] == 1
    assert abs(oa["sim_time"] - a.cfg.tick_seconds) < 1e-9
    assert oa["speed"] > 0
    assert "events" in oa


def test_controls_are_clipped():
    a, b = RaceEnv(), RaceEnv()
    assert a.step(5.0, -9.0) == b.step(1.0, -1.0)


def test_crash_penalty_and_stop():
    env = RaceEnv(track=make_oval_track())
    obs = env.reset()
    for _ in range(60):
        obs = env.step(1.0, 0.0)          # drive straight into the first corner
        if obs["crashes"]:
            break
    assert obs["crashes"] == 1
    assert obs["penalty_time"] == env.cfg.crash_penalty_seconds
    assert obs["speed"] == 0
    assert any(e.startswith("crash") for e in obs["events"])


def test_max_ticks_ends_attempt_as_dnf():
    env = RaceEnv(config=PhysicsConfig(max_ticks=5))
    obs = env.reset()
    for _ in range(5):
        obs = env.step(0.0, 0.0)
    assert obs["attempt_over"] and env.done
    assert len(env.attempts) == 1
    assert env.attempts[0].ended_by == "max_ticks" and not env.attempts[0].completed
    # stepping a finished attempt is a no-op with a hint
    after = env.step(1.0, 0.0)
    assert after["tick"] == 5 and after["events"][0].startswith("attempt_over")


def test_reset_records_only_driven_attempts():
    env = RaceEnv()
    env.reset()
    env.reset()                       # not driven -> not recorded
    assert env.attempts == [] and env.attempt_no == 1
    env.step(0.5, 0.0)
    env.reset()                       # driven -> recorded as "reset"
    assert len(env.attempts) == 1 and env.attempts[0].ended_by == "reset"
    assert env.attempt_no == 2


def test_reference_driver_completes_a_clean_lap():
    """Regression guard: the competition track must be drivable through the tools alone."""
    env = RaceEnv()
    run_reflex(env, make_tools(env), attempts=1, verbose=False)
    res = env.attempts[-1]
    assert res.completed and res.ended_by == "lap_complete"
    assert res.crashes == 0
    assert res.waypoints_passed == env.track.n_gates
    assert 40 < res.lap_time < 120
    assert env.best_lap == res.lap_time
    assert len(env.attempt_traces) == 1 and len(env.attempt_traces[0]["trajectory"]) == res.ticks + 1


def test_jsonl_logging(tmp_path):
    env = RaceEnv(log_dir=tmp_path, run_name="t")
    env.step(1.0, 0.0)
    env.close()
    lines = [json.loads(l) for l in (tmp_path / "t.jsonl").read_text().splitlines()]
    kinds = [l["event"] for l in lines]
    assert kinds == ["reset", "step", "attempt_end"]
    assert lines[1]["throttle"] == 1.0
