import math

import numpy as np
import pytest

from agentic_gp.driver import (
    COAST, DEBUG_MAX_BYTES, OBS_FIELDS, REPORT_FIELDS, TRACK_FIELDS, Action, Driver, action_problem,
    normalize_action,
)
from agentic_gp.sim import OBS_KEYS, RaceEnv


@pytest.mark.parametrize("raw, expected", [
    ((0.8, -0.2), Action(0.8, -0.2)),
    ([1, 0], Action(1.0, 0.0)),
    ((1, 0, "flat out"), Action(1.0, 0.0, "flat out")),
    ((2.5, -7), Action(1.0, -1.0)),
    ((0.5,), Action(0.5, 0.0)),
    ((), COAST),
    (None, COAST),
    ("go fast", COAST),
    (42, COAST),
    ({"throttle": 0.3, "steering": 0.1}, Action(0.3, 0.1)),
    ({"throttle": "1", "steering": "left"}, Action(1.0, 0.0)),
    ({"throttle": 1, "message": "hi"}, Action(1.0, 0.0, "hi")),
    ({"throttle": 1, "thought": "  braking\n soon "}, Action(1.0, 0.0, "braking soon")),
    ({"say": ""}, COAST),
    ((float("nan"), float("inf")), COAST),
    ((True, False), Action(1.0, 0.0)),
    (np.array([0.25, -0.5]), Action(0.25, -0.5)),
    (np.float64(0.7), COAST),                         # a 0-d numpy value: no TypeError
    (np.array(0.7), COAST),
])
def test_normalize_action(raw, expected):
    assert normalize_action(raw) == expected


def test_say_is_capped_and_debug_is_kept_small():
    a = normalize_action({"throttle": 1, "say": "x" * 500, "debug": {"target": np.array([1.5, 2.0])}})
    assert len(a.say) == 160
    assert a.debug == {"target": [1.5, 2.0]}
    big = normalize_action({"debug": {"blob": "y" * (DEBUG_MAX_BYTES + 10)}})
    assert "too large" in big.debug["error"]
    bad = normalize_action({"debug": {"f": object()}})
    assert "error" in bad.debug
    assert normalize_action({"debug": {"x": math.nan}}).debug == {"error": "debug dict is not JSON-serialisable"}
    assert normalize_action(Action(3, -3, "a")) == Action(1.0, -1.0, "a")
    assert normalize_action((0.5, 0.5)).as_tuple() == (0.5, 0.5)


def test_base_class_is_a_coasting_noop():
    d = Driver()
    assert d.plan({}) is None and d.reflect({}) is None
    assert normalize_action(d.drive({})) == COAST


def test_obs_fields_document_exactly_the_observation():
    assert set(OBS_FIELDS) == set(OBS_KEYS) | {"events"}          # events: documented too (lesson docs)
    obs = RaceEnv(track="oval").reset()
    assert set(obs) == set(OBS_FIELDS)
    assert all(isinstance(v, str) and v for v in OBS_FIELDS.values())


def test_track_fields_document_the_track_map():
    assert set(RaceEnv(track="sprint").track_map()) == set(TRACK_FIELDS)


def test_report_fields_are_documented():
    # the exact match with AttemptReport.to_reflect_dict() is checked in test_session.py
    for key in ("next_phase", "summary_text", "stalls", "driver_errors", "llm_usage", "telemetry", "crash_log",
                "segments", "best_lap_so_far"):
        assert key in REPORT_FIELDS


@pytest.mark.parametrize("raw, needle", [
    ((1, 0), None), (None, None), (Action(1, 0), None), ({"throttle": 1}, None), ({"say": "hi"}, None),
    (np.array([0.5, 0.1]), None), ((True, False), None), ((2.5, -7), None),       # clipped, not an error
    ((float("nan"), 0), "throttle = nan - not a finite number"),
    ((float("inf"), float("-inf")), "throttle = inf - not a finite number"),
    ((0.5, float("nan")), "steering = nan"),
    ({"throttle": float("nan")}, "throttle = nan"),
    (Action(float("inf"), 0), "throttle = inf"),
    (np.array([np.nan, 0.0]), "throttle = nan"),
    (np.float64(0.7), "one number (0.7)"),
    ("full throttle", "returned str 'full throttle'"),
    ({"accel": 1.0, "steer": 0.3}, "without 'throttle'/'steering' (its keys: accel, steer)"),
    ({"throttle": "fast"}, "throttle = 'fast'"),
    ((1, "left"), "steering = 'left'"),
    ((), "empty tuple"),
    (0.7, "one number (0.7)"),
    (object(), "returned object"),
])
def test_action_problem_explains_unusable_answers(raw, needle):
    problem = action_problem(raw)
    assert (problem is None) if needle is None else (needle in problem)
