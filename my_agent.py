"""YOUR AGENT - the file you edit for the Agentic Grand Prix.

Entry point (do not rename - evaluate.py calls it):

    def run(env, tools, attempts=1, **kwargs) -> None

`tools` is the list of LangChain tools: read_sensors, drive, reset_environment,
get_track_map. `env` is the RaceEnv; you may read env.attempts for your own
reflection loop, but move the car only through the `drive` tool.

This starter is a tiny LangGraph with two nodes:

    planner  -> decides a target speed for the next stretch   (TODO: make it an LLM)
    executor -> converts the plan + sensors into one `drive` call (fast, no LLM)

and loops until the attempt is over. Out of the box it is *slow but safe*: it
finishes a lap, far behind the reference time. Ideas to beat Hermes: call
`get_track_map` once and plan speeds per segment, add a reflection node that
reads the crash events after each attempt and adjusts the plan, keep memory
across attempts, or split planning/execution between a strong model and a fast
one (see agentic_gp/llm.py).
"""

from __future__ import annotations

import json
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from agentic_gp.physics import clamp
from agentic_gp.tools import tools_by_name

# from agentic_gp.llm import get_chat_model   # uncomment when you add an LLM planner


class RaceState(TypedDict):
    obs: dict           # latest sensor reading (see README for the fields)
    plan: dict          # whatever your planner decides (target_speed, notes, ...)
    memory: list[str]   # events collected during this attempt (crashes, waypoints)


def build_graph(tools):
    by = tools_by_name(tools)

    # ---------------------------------------------------------------- planner
    def planner(state: RaceState) -> dict:
        """Decide how to drive the next stretch.

        TODO: replace with an LLM call. Example skeleton:

            llm = get_chat_model()          # provider/model from environment variables
            msg = llm.invoke([("system", "You are a race strategist..."),
                              ("human", json.dumps(state["obs"]))])
            plan = json.loads(msg.content)  # e.g. {"target_speed": 22}
        """
        obs = state["obs"]
        upcoming_turn = abs(obs["angle_to_next_waypoint"]) + 0.5 * abs(obs["angle_to_waypoint_after_next"])
        target_speed = 16.0 if upcoming_turn < 12 else 10.0
        return {"plan": {"target_speed": target_speed}}

    # --------------------------------------------------------------- executor
    def executor(state: RaceState) -> dict:
        obs, plan = state["obs"], state["plan"]
        speed = obs["speed"]
        front = obs["distance_to_wall_front"]
        l20, r20 = obs["distance_to_wall_left_20"], obs["distance_to_wall_right_20"]
        l45, r45 = obs["distance_to_wall_left_45"], obs["distance_to_wall_right_45"]

        if speed < 1.5 and min(front, l20, r20) < 4.0:
            # Stopped against a wall: back off towards the open side.
            throttle, steering = -1.0, (-1.0 if l45 > r45 else 1.0)
        else:
            # Aim at the next waypoint, looking through to the one after when close.
            w = clamp((25.0 - obs["distance_to_next_waypoint"]) / 20.0, 0.0, 0.5)
            aim = (1.0 - w) * obs["angle_to_next_waypoint"] + w * obs["angle_to_waypoint_after_next"]
            steering = aim / 30.0
            steering += 0.6 * max(0.0, 1.0 - l20 / 10.0) + 0.4 * max(0.0, 1.0 - l45 / 8.0)   # off the left wall
            steering -= 0.6 * max(0.0, 1.0 - r20 / 10.0) + 0.4 * max(0.0, 1.0 - r45 / 8.0)   # off the right wall
            steering = clamp(steering, -1.0, 1.0)
            # Never drive faster than the space in front of us allows (~2 ticks of travel).
            room = min(front, 1.3 * l20, 1.3 * r20)
            target = clamp(min(plan["target_speed"], 0.8 * room), 4.0, 40.0)
            throttle = clamp((target - speed) / 6.0, -1.0, 1.0)

        new_obs = json.loads(by["drive"].invoke({"throttle": throttle, "steering": steering}))
        return {"obs": new_obs, "memory": state["memory"] + list(new_obs.get("events", []))}

    def keep_driving(state: RaceState) -> str:
        return END if state["obs"]["attempt_over"] else "planner"

    g = StateGraph(RaceState)
    g.add_node("planner", planner)
    g.add_node("executor", executor)
    g.add_edge(START, "planner")
    g.add_edge("planner", "executor")
    g.add_conditional_edges("executor", keep_driving, {"planner": "planner", END: END})
    return g.compile()


def run(env, tools, attempts: int = 1, verbose: bool = True, **kwargs) -> None:
    by = tools_by_name(tools)
    graph = build_graph(tools)
    for i in range(attempts):
        obs = json.loads(by["reset_environment"].invoke({}))
        final = graph.invoke(
            {"obs": obs, "plan": {}, "memory": []},
            config={"recursion_limit": 10_000},   # 2 nodes per tick, up to 500 ticks
        )
        result = env.attempts[-1]
        if verbose:
            status = f"lap {result.lap_time:.2f}s" if result.completed else "DNF"
            print(f"[my_agent] attempt {i + 1}: {status}, crashes={result.crashes}, events={len(final['memory'])}")
        # TODO reflection: look at final["memory"] / result and change the plan for the next attempt.
