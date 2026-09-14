"""The reference Agentic Workflow: hierarchical planning, fast execution, memory, and reflection.

This agent demonstrates the cognitive architecture students are tasked with building:
1. Global Track Strategist: Queries `get_track_map`, analyzes corner radii and track sectors,
   derives physics-safe speed targets (v = sqrt(20 * R)), and optionally consults an LLM strategist.
2. Tactical Executor: High-frequency reactive driver in LangGraph that converts the plan, waypoint
   geometry, and 7-ray LiDAR sensors into smooth throttle and steering commands with stuck recovery.
3. Episodic Memory: Records telemetry, split times, and incident logs across ticks.
4. Reflection Loop: After each attempt, analyzes crashes and sector performance from env.attempts
   and memory, adapting sector speed limits for subsequent attempts.
"""

from __future__ import annotations

import json
import math
from typing import Any, TypedDict

import numpy as np
from langgraph.graph import END, START, StateGraph

from ..config import PhysicsConfig
from ..llm import get_chat_model
from ..physics import available_yaw_rate, clamp
from ..tools import tools_by_name


class RaceState(TypedDict):
    obs: dict                       # Latest sensor reading
    plan: dict                      # Tactical plan for current tick
    sector_speeds: dict[int, float] # Per-waypoint target speed profile (indexed by segment)
    memory: list[dict[str, Any]]    # Telemetry and events for this attempt
    recovery_ticks: int             # Multi-tick stuck recovery counter
    crashes_this_lap: int           # Crash counter for reactive adjustments


def compute_baseline_sector_speeds(
    track_map: dict,
    cfg: PhysicsConfig | None = None,
    aggression: float = 0.85,
) -> dict[int, float]:
    """Compute physics-informed speed limits per waypoint segment from corner radii.

    Segment `i` is the track stretch between waypoint `i` and waypoint `(i + 1) % n`.
    Physics formula: turn radius R at maximum lateral acceleration (20.0 units/s²) gives
    v_max = sqrt(20 * R). A backward braking pass ensures the car can slow down in time.
    """
    cfg = cfg or PhysicsConfig()
    waypoints = track_map.get("waypoints", [])
    n = len(waypoints)
    if n == 0:
        return {}

    centers = np.array([w["center"] for w in waypoints], dtype=float)
    diffs = np.roll(centers, -1, axis=0) - centers
    distances = np.linalg.norm(diffs, axis=1)

    # 1. Corner radius speed cap for segment i
    v_caps = []
    for w in waypoints:
        r_min = w.get("min_turn_radius_until_next", 50.0)
        # Safe corner speed with grip margin
        v_safe = math.sqrt(cfg.max_lateral_accel * max(r_min, 4.0)) * aggression
        v_caps.append(float(clamp(v_safe, 8.0, cfg.max_speed)))

    # 2. Backward braking pass (twice to wrap around closed circuit)
    for _ in range(2):
        for i in reversed(range(n)):
            nxt = (i + 1) % n
            dist = float(distances[i])
            max_entry_v = math.sqrt(v_caps[nxt] ** 2 + 2.0 * cfg.brake * 0.70 * dist)
            v_caps[i] = min(v_caps[i], max_entry_v)

    return {int(w["index"]): round(float(v_caps[i]), 1) for i, w in enumerate(waypoints)}


def consult_llm_strategist(track_map: dict, sector_speeds: dict[int, float]) -> dict[str, Any]:
    """Optionally consult an LLM to provide strategic notes on key track sectors."""
    try:
        llm = get_chat_model()
        summary = [
            {
                "segment": w["index"],
                "radius": w.get("min_turn_radius_until_next"),
                "speed_limit": sector_speeds.get(w["index"]),
            }
            for w in track_map.get("waypoints", [])
        ]
        prompt = (
            f"You are the race strategist for the Agentic Grand Prix. Here is the circuit sector analysis:\n"
            f"{json.dumps(summary)}\n"
            f"Identify the 2 tightest corners and return a short JSON: "
            f'{{"tightest_gates": [list of gate indices], "strategy_notes": "advice"}}'
        )
        msg = llm.invoke([
            ("system", "You are an expert racing strategist. Respond ONLY with valid JSON."),
            ("human", prompt),
        ])
        content = msg.content if isinstance(msg.content, str) else json.dumps(msg.content)
        start = content.find("{")
        end = content.rfind("}") + 1
        if start != -1 and end != 0:
            return json.loads(content[start:end])
    except Exception:
        pass

    tightest = sorted(track_map.get("waypoints", []), key=lambda w: w.get("min_turn_radius_until_next", 100))[:2]
    return {
        "tightest_gates": [w["index"] for w in tightest],
        "strategy_notes": "Hairpin and chicane sectors require disciplined entry speeds.",
    }


def reflect_and_adapt(
    attempt_res,
    memory: list[dict[str, Any]],
    sector_speeds: dict[int, float],
    verbose: bool = True,
) -> dict[int, float]:
    """Episodic reflection node: inspects attempt outcome and tunes sector speeds."""
    new_speeds = dict(sector_speeds)
    crashes = attempt_res.crashes
    completed = attempt_res.completed

    # Find sectors where crashes occurred from telemetry memory
    crashed_sectors = set()
    for entry in memory:
        if isinstance(entry, dict):
            events = entry.get("events", [])
            if any("crash" in str(e).lower() for e in events):
                if entry.get("segment") is not None:
                    crashed_sectors.add(entry.get("segment"))
        elif isinstance(entry, str) and "crash" in entry.lower():
            # Fallback if raw string events are provided
            pass

    if crashes > 0:
        if verbose:
            print(f"[reflection] Attempt had {crashes} crash(es) in segments {sorted(crashed_sectors) or 'unknown'}. Refining...")
        # Reduce speed specifically on crashed sectors, or all high-speed sectors
        if crashed_sectors:
            for seg in crashed_sectors:
                if seg in new_speeds and new_speeds[seg] > 8.0:
                    new_speeds[seg] = round(new_speeds[seg] * 0.85, 1)
        else:
            for idx in new_speeds:
                if new_speeds[idx] > 12.0:
                    new_speeds[idx] = round(new_speeds[idx] * 0.90, 1)
    elif completed:
        if verbose:
            print(f"[reflection] Clean lap finished in {attempt_res.lap_time:.2f}s! Tuning up fast sectors...")
        for idx in new_speeds:
            if new_speeds[idx] >= 16.0:
                new_speeds[idx] = round(min(40.0, new_speeds[idx] * 1.03), 1)

    return new_speeds


def build_agent_graph(tools, initial_sector_speeds: dict[int, float], cfg: PhysicsConfig | None = None):
    """Build the LangGraph cognitive loop (tactical_planner -> executor -> keep_driving)."""
    by = tools_by_name(tools)
    cfg = cfg or PhysicsConfig()
    n_sectors = len(initial_sector_speeds)

    def tactical_planner(state: RaceState) -> dict:
        obs = state["obs"]
        gate_idx = int(obs.get("next_waypoint_index", 1))
        # The car is currently driving on segment (gate_idx - 1) % n_sectors heading towards gate_idx
        curr_segment = (gate_idx - 1) % n_sectors
        curr_speed_limit = state["sector_speeds"].get(curr_segment, 18.0)
        next_speed_limit = state["sector_speeds"].get(gate_idx, 18.0)

        target_speed = curr_speed_limit

        # Lookahead braking as car approaches the next gate
        dist_to_next = obs.get("distance_to_next_waypoint", 50.0)
        if dist_to_next < 35.0 and next_speed_limit < target_speed:
            blend = (35.0 - dist_to_next) / 35.0
            target_speed = (1.0 - blend) * target_speed + blend * next_speed_limit

        # Sharp turn anticipation
        turn_angle = abs(obs.get("angle_to_next_waypoint", 0.0))
        if turn_angle > 32.0:
            target_speed = min(target_speed, 12.0)
        elif turn_angle > 18.0:
            target_speed = min(target_speed, 18.0)

        return {"plan": {"target_speed": target_speed, "current_segment": curr_segment}}

    def executor(state: RaceState) -> dict:
        obs, plan = state["obs"], state["plan"]
        speed = obs["speed"]
        front = obs["distance_to_wall_front"]
        l20, r20 = obs["distance_to_wall_left_20"], obs["distance_to_wall_right_20"]
        l45, r45 = obs["distance_to_wall_left_45"], obs["distance_to_wall_right_45"]
        recovery = state.get("recovery_ticks", 0)

        # 1. Multi-tick stuck recovery when stopped or pinned against a wall
        if recovery > 0:
            recovery -= 1
            throttle = -1.0
            steering = -1.0 if l45 > r45 else 1.0
        elif speed < 1.5 and min(front, l20, r20) < 4.0:
            recovery = 2  # reverse for 2 ticks
            throttle = -1.0
            steering = -1.0 if l45 > r45 else 1.0
        else:
            # 2. Waypoint curvature and grip-fraction steering
            dist_next = max(obs["distance_to_next_waypoint"], 1.0)
            angle_next = math.radians(obs["angle_to_next_waypoint"])
            angle_after = math.radians(obs.get("angle_to_waypoint_after_next", obs["angle_to_next_waypoint"]))

            w = clamp((28.0 - dist_next) / 20.0, 0.0, 0.40)
            aim_angle = (1.0 - w) * angle_next + w * angle_after

            # Pure pursuit curvature with lookahead floor
            lookahead_dist = clamp(dist_next, 8.0, 30.0)
            curvature = 2.0 * math.sin(aim_angle) / lookahead_dist
            eff_speed = max(abs(speed), 5.0)
            yaw_needed = eff_speed * curvature
            avail_yaw = available_yaw_rate(eff_speed, cfg)
            steering = yaw_needed / avail_yaw

            # 3. Dynamic LiDAR wall repulsion
            steering += 0.55 * max(0.0, 1.0 - l20 / 7.5) + 0.30 * max(0.0, 1.0 - l45 / 5.5)
            steering -= 0.55 * max(0.0, 1.0 - r20 / 7.5) + 0.30 * max(0.0, 1.0 - r45 / 5.5)
            steering = clamp(steering, -1.0, 1.0)

            # 4. Kinematic safe braking distance to wall
            margin = 3.5 + 1.3 * abs(speed) * cfg.tick_seconds
            v_front = math.sqrt(2.0 * cfg.brake * max(front - margin, 0.0))
            v_side = cfg.max_speed
            if min(l20, r20) < 4.5:
                v_side = 10.0
            elif min(l45, r45) < 4.5:
                v_side = 14.0

            target = clamp(min(plan["target_speed"], v_front, v_side), 5.0, cfg.max_speed)
            throttle = clamp((target - speed) / (cfg.accel * cfg.tick_seconds), -1.0, 1.0)

        new_obs = json.loads(by["drive"].invoke({"throttle": throttle, "steering": steering}))
        events = list(new_obs.get("events", []))
        crashes = state["crashes_this_lap"] + sum(1 for e in events if "crash" in e.lower())

        telemetry = {
            "tick": new_obs.get("tick"),
            "segment": plan.get("current_segment"),
            "speed": new_obs.get("speed"),
            "events": events,
        }

        return {
            "obs": new_obs,
            "memory": state["memory"] + [telemetry],
            "recovery_ticks": recovery,
            "crashes_this_lap": crashes,
        }

    def keep_driving(state: RaceState) -> str:
        return END if state["obs"]["attempt_over"] else "tactical_planner"

    graph = StateGraph(RaceState)
    graph.add_node("tactical_planner", tactical_planner)
    graph.add_node("executor", executor)
    graph.add_edge(START, "tactical_planner")
    graph.add_edge("tactical_planner", "executor")
    graph.add_conditional_edges("executor", keep_driving, {"tactical_planner": "tactical_planner", END: END})

    return graph.compile()


def run(env, tools, attempts: int = 1, verbose: bool = True, aggression: float = 0.85, **kwargs) -> None:
    """Entry point for evaluate.py: orchestrates planning, execution, and reflection."""
    by = tools_by_name(tools)
    cfg = getattr(env, "cfg", PhysicsConfig())

    # Phase 1: Strategic Planning from Track Map
    track_map = json.loads(by["get_track_map"].invoke({}))
    sector_speeds = compute_baseline_sector_speeds(track_map, cfg, aggression=aggression)
    strategy_notes = consult_llm_strategist(track_map, sector_speeds)

    if verbose:
        print(f"[agentic] Global Track Planning complete ({len(sector_speeds)} segments mapped).")
        print(f"[agentic] Sector speed profile: {sector_speeds}")
        print(f"[agentic] Strategy: {strategy_notes.get('strategy_notes')}")

    # Phase 2: Multi-Attempt Driving Loop with Reflection
    for attempt_idx in range(attempts):
        graph = build_agent_graph(tools, sector_speeds, cfg)
        obs = json.loads(by["reset_environment"].invoke({}))

        initial_state: RaceState = {
            "obs": obs,
            "plan": {},
            "sector_speeds": sector_speeds,
            "memory": [],
            "recovery_ticks": 0,
            "crashes_this_lap": 0,
        }

        final_state = graph.invoke(
            initial_state,
            config={"recursion_limit": 10_000},
        )

        res = env.attempts[-1]
        status = f"lap {res.lap_time:.2f}s" if res.completed else f"DNF ({res.ended_by})"
        if verbose:
            print(f"[agentic] Attempt {attempt_idx + 1}/{attempts}: {status}, crashes={res.crashes}, ticks={res.ticks}")

        # Phase 3: Cognitive Reflection across attempts
        if attempt_idx < attempts - 1:
            sector_speeds = reflect_and_adapt(
                res,
                final_state["memory"],
                sector_speeds,
                verbose=verbose,
            )
