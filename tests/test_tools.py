import json

from agentic_gp import RaceEnv, make_tools
from agentic_gp.tools import RaceTools, tools_by_name


def test_tool_names_and_json_outputs():
    env = RaceEnv()
    tools = make_tools(env)
    by = tools_by_name(tools)
    assert set(by) == {"read_sensors", "drive", "reset_environment", "get_track_map"}

    obs = json.loads(by["read_sensors"].invoke({}))
    assert obs["tick"] == 0

    out = json.loads(by["drive"].invoke({"throttle": 0.8, "steering": 0.0}))
    assert out["tick"] == 1 and "events" in out and out["lap_complete"] is False

    m = json.loads(by["get_track_map"].invoke({}))
    assert m["n_waypoints"] == env.track.n_gates and "physics" in m

    again = json.loads(by["reset_environment"].invoke({}))
    assert again["tick"] == 0 and again["speed"] == 0


def test_out_of_range_controls_are_clipped_not_rejected():
    a, b = RaceEnv(), RaceEnv()
    ta, tb = tools_by_name(make_tools(a)), tools_by_name(make_tools(b))
    oa = json.loads(ta["drive"].invoke({"throttle": 3.0, "steering": -2.0}))
    ob = json.loads(tb["drive"].invoke({"throttle": 1.0, "steering": -1.0}))
    assert oa == ob


def test_raw_tools_record_calls():
    env = RaceEnv()
    raw = RaceTools(env)
    tools = make_tools(env, raw)
    by = tools_by_name(tools)
    by["drive"].invoke({"throttle": 1.0, "steering": 0.0})
    by["read_sensors"].invoke({})
    assert [c["tool"] for c in raw.calls] == ["drive", "read_sensors"]
    assert raw.calls[0]["args"] == {"throttle": 1.0, "steering": 0.0}


def test_tool_schemas_are_llm_friendly():
    env = RaceEnv()
    by = tools_by_name(make_tools(env))
    schema = by["drive"].args_schema.model_json_schema()
    assert set(schema["required"]) == {"throttle", "steering"}
    assert "left" in by["drive"].description.lower()
    assert by["read_sensors"].args_schema.model_json_schema()["properties"] == {}
