"""Tests for the reference agentic workflow agent."""

import sys
from pathlib import Path

from agentic_gp import RaceEnv, make_tools
from agentic_gp.agents import agentic

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_baseline_sector_speeds_computation():
    env = RaceEnv()
    t_map = env.track_map()
    speeds = agentic.compute_baseline_sector_speeds(t_map, env.cfg)
    assert len(speeds) == t_map["n_waypoints"]
    for idx, v in speeds.items():
        assert 5.0 <= v <= env.cfg.max_speed


def test_reflection_and_adaptation():
    class DummyResult:
        crashes = 1
        completed = False
        lap_time = 0.0

    initial_speeds = {0: 25.0, 1: 30.0, 2: 12.0}
    # With telemetry dict
    memory = [{"segment": 1, "events": ["crash: wall"]}]
    adapted = agentic.reflect_and_adapt(DummyResult(), memory, initial_speeds, verbose=False)
    # Crashed sector 1 should be reduced
    assert adapted[1] < initial_speeds[1]
    # Sector 0 was untouched
    assert adapted[0] == initial_speeds[0]


def test_agentic_workflow_finishes_clean_lap():
    env = RaceEnv()
    tools = make_tools(env)
    agentic.run(env, tools, attempts=1, verbose=False)
    res = env.attempts[-1]
    assert res.completed, res
    assert res.crashes == 0
    assert res.ticks < env.cfg.max_ticks
    # Ensure it's faster than the starter agent (~207s)
    assert res.lap_time < 120.0
