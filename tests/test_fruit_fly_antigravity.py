"""Tests for the Fruit Fly Connectome Agentic Driver (fruit_fly_antigravity)."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure repository root is on path
repo_root = str(Path(__file__).resolve().parents[1])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from agentic_gp import RaceEnv, PhysicsConfig, make_tools
import fruit_fly_antigravity as ffa


def test_connectome_topology_and_synapses():
    """Verify that the fruit fly connectome has the expected neuron count and signed synapse properties."""
    assert ffa.N_NEURONS == 48
    assert len(ffa.NEURON_NAMES) == 48
    assert len(ffa.N2I) == 48

    W = ffa.W_CONNECTOME
    assert W.shape == (48, 48)

    # Synapses should have both excitatory (+1) and inhibitory (-1) connections
    assert np.any(W > 0.0), "Connectome missing excitatory connections"
    assert np.any(W < 0.0), "Connectome missing inhibitory connections"

    # Row normalization check: each postsynaptic neuron's incoming absolute weights sum to <= 1.0 (with float tolerance)
    row_abs_sums = np.sum(np.abs(W), axis=1)
    for dst_idx, total in enumerate(row_abs_sums):
        assert total <= 1.0001, f"Neuron {ffa.NEURON_NAMES[dst_idx]} has unnormalized incoming weights: {total}"


def test_brain_dynamics_stability():
    """Verify that leaky-tanh recurrent dynamics remain bounded in [-1, 1] under constant stimulation."""
    brain = ffa.FruitFlyBrain(alpha=0.7, beta=1.4)
    u = np.zeros(ffa.N_NEURONS)
    u[ffa.N2I["LC4_F"]] = 1.0  # High frontal looming stimulation
    u[ffa.N2I["HS_R"]] = 0.8   # Rightward yaw optic flow

    for _ in range(50):
        h = brain.step(u)
        assert np.all(h >= -1.0) and np.all(h <= 1.0), "Activations exceeded [-1, 1] bounds"

    # Giant Fiber should be excited by LC4 looming
    assert brain.h[ffa.N2I["GF_L"]] > 0.1
    assert brain.h[ffa.N2I["GF_R"]] > 0.1

    brain.reset()
    assert np.all(brain.h == 0.0)


def test_sensory_encoding():
    """Verify that observation dictionary correctly translates to biological neuron inputs u."""
    cfg = PhysicsConfig()
    obs = {
        "speed": 25.0,
        "distance_to_wall_front": 12.0,
        "distance_to_wall_left_20": 4.0,
        "distance_to_wall_right_20": 18.0,
        "distance_to_wall_left_45": 5.0,
        "distance_to_wall_right_45": 22.0,
        "distance_to_wall_left_90": 3.0,
        "distance_to_wall_right_90": 20.0,
        "angle_to_next_waypoint": 20.0,
        "distance_to_next_waypoint": 30.0,
    }
    plan = {"aim_deg": 20.0, "target_speed": 25.0}

    u = ffa.encode_sensory_drives(obs, plan, cfg)
    assert len(u) == ffa.N_NEURONS

    # Left wall is close -> LC11_L should have strong activation
    assert u[ffa.N2I["LC11_L"]] > u[ffa.N2I["LC11_R"]]
    # Rightward goal angle -> FB_GOAL_R should be active
    assert u[ffa.N2I["FB_GOAL_R"]] > 0.0
    # Looming wall at front (12 units) -> LC4_F active
    assert u[ffa.N2I["LC4_F"]] > 0.3


def test_reflection_and_adaptation():
    """Test Mushroom Body dopaminergic adaptation on simulated crashes vs clean laps."""
    cfg = PhysicsConfig()
    initial_speeds = {0: 30.0, 1: 35.0, 2: 20.0}

    # Crash scenario
    class CrashAttempt:
        crashes = 2
        completed = False
        lap_time = 0.0

    memory = [{"segment": 1, "events": ["crash: left wall"]}]
    punished = ffa.reflect_and_adapt(CrashAttempt(), memory, initial_speeds, cfg, verbose=False)
    assert punished[1] < initial_speeds[1], "Crashed sector 1 was not slowed down"
    assert punished[0] == initial_speeds[0], "Uncrashed sector 0 was modified unexpectedly"

    # Clean lap scenario
    class CleanAttempt:
        crashes = 0
        completed = True
        lap_time = 85.0

    rewarded = ffa.reflect_and_adapt(CleanAttempt(), [], initial_speeds, cfg, verbose=False)
    assert rewarded[1] >= initial_speeds[1], "Clean fast sector was not reinforced"


def test_fruit_fly_antigravity_completes_clean_lap():
    """Integration test: fruit_fly_antigravity finishes a complete lap with 0 crashes under 100s."""
    env = RaceEnv()
    tools = make_tools(env)
    ffa.run(env, tools, attempts=1, verbose=False, save_trace=False)

    res = env.attempts[-1]
    assert res.completed, f"Agent did not complete lap: {res.ended_by}"
    assert res.crashes == 0, f"Agent suffered {res.crashes} crash(es)"
    assert res.ticks < env.cfg.max_ticks, f"Exceeded tick limit: {res.ticks}"
    assert res.lap_time < 95.0, f"Lap time {res.lap_time:.2f}s is slower than target"
