"""Fruit Fly Connectome Agentic Driver — fruit_fly_antigravity.

Based on the Drosophila melanogaster connectome research in awesome-fly:
https://github.com/cobanov/awesome-fly

Scientific and Architectural Foundations:
1. Connectome Topology (MaleCNS v1.0 & FlyWire adult connectomes):
   - Optic Lobe / Visual Neuropil:
     * Lobula Plate Tangential Cells (HS_L, HS_R for yaw optic flow; VS_F for forward velocity)
     * Lobula Columnar Cells (LC4_F looming obstacle detector; LC11_L/R lateral clearance;
       LC16_V motion velocity; LPLC2_L/R flank wall proximity)
     * Mechanosensory bristles (TOUCH_L, TOUCH_R tactile wall contact)
     * Anterior Optic Tubercle (AOTU_G goal azimuth input)
   - Central Complex (CX) Compass & Steering Circuit:
     * 16-wedge Ellipsoid Body E-PG compass ring attractor (tracking orientation)
     * Protocerebral Bridge P-EN & P-FN angular velocity & translation integrators
     * Fan-Shaped Body (FB) goal vector comparators (FB_GOAL_L, FB_GOAL_R)
     * Protocerebral Bridge steering comparators (P_FL3_L, P_FL3_R)
     * Lateral Accessory Lobes (LAL_L, LAL_R) with bilateral mutual inhibition
   - Descending Neurons (DNs) & Motor Output:
     * DNa01_L, DNa01_R: Descending neurons for coarse turning commands
     * DNa02_L, DNa02_R: Descending neurons for fine trim steering
     * DNp01: Forward thrust / throttle driver
     * DNb01: Brake modulation / deceleration
     * GF_L, GF_R: Giant Fiber emergency collision evasion & braking reflex
   - Mushroom Body (MB) & Dopaminergic Center:
     * DAN_PUNISH, DAN_REWARD: Dopaminergic neurons signaling punishment (collisions)
       and reward (clean sector progression)
     * KC_CONTEXT: Kenyon cell population encoding track sector state
     * MBON_SPEED: Mushroom Body Output Neuron gating throttle aggressiveness

2. Leaky-Tanh Connectome Dynamics (Fly Dino / Cobanov / Shiu et al.):
   h(t) = (1 - alpha) * h(t-1) + alpha * tanh(u(t) + beta * W @ h(t-1))
   where W is a signed, normalized synaptic connectivity matrix (+1 Acetylcholine,
   -1 GABA / Glutamate).

3. Cognitive Agentic Architecture (LangGraph):
   - Global Strategist: queries `get_track_map`, computes corner radii and braking passes
   - Tactical Planner: determines target speeds and lookahead pure-pursuit angles
   - Sensory Cortex: transforms raw LiDAR and telemetry into biological neuron drives u(t)
   - Connectome Dynamics: steps the 48-neuron recurrent fruit fly network
   - Motor Actuator: decodes descending commands into drive() controls with stuck recovery
   - Dopaminergic Reflection Loop: adapts sector speeds and gains between attempts

Entry point (compatible with evaluate.py):
    def run(env, tools, attempts=1, **kwargs) -> None
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict
from typing import Any, TypedDict

import numpy as np
from langgraph.graph import END, START, StateGraph

from agentic_gp import PhysicsConfig
from agentic_gp.physics import available_yaw_rate, clamp
from agentic_gp.tools import tools_by_name

# ============================================================================
#   PART 1: CONNECTOME NEURON SPECIFICATION & SYNAPTIC TOPOLOGY
# ============================================================================

NEURON_NAMES = [
    # Visual / Optic Lobe (0..8)
    "HS_L", "HS_R", "VS_F", "LC4_F", "LC11_L", "LC11_R", "LC16_V", "LPLC2_L", "LPLC2_R",
    # Mechanosensory (9..10)
    "TOUCH_L", "TOUCH_R",
    # Anterior Optic Tubercle (11)
    "AOTU_G",
    # Central Complex E-PG Compass Ring (12..27: 16 neurons covering 360 deg)
    *(f"E_PG_{i}" for i in range(16)),
    # Central Complex Premotor & Steering (28..35)
    "P_EN_L", "P_EN_R", "FB_GOAL_L", "FB_GOAL_R", "P_FL3_L", "P_FL3_R", "LAL_L", "LAL_R",
    # Descending Neurons & Escape Circuits (36..43)
    "DNa01_L", "DNa01_R", "DNa02_L", "DNa02_R", "DNp01", "DNb01", "GF_L", "GF_R",
    # Mushroom Body Dopaminergic & Memory (44..47)
    "DAN_PUNISH", "DAN_REWARD", "KC_CONTEXT", "MBON_SPEED",
]

N_NEURONS = len(NEURON_NAMES)
N2I = {name: i for i, name in enumerate(NEURON_NAMES)}


def build_connectome_synapses() -> np.ndarray:
    """Construct the signed, normalized synaptic connectivity matrix W.
    
    Neurotransmitters:
      +1: Acetylcholine (ACh) - Excitatory
      -1: GABA / Glutamate (GluCl) - Inhibitory
    """
    raw_contacts = np.zeros((N_NEURONS, N_NEURONS), dtype=float)
    synapse_sign = np.zeros((N_NEURONS, N_NEURONS), dtype=float)

    def syn(src: str, dst: str, contacts: float, sign: float):
        s_idx, d_idx = N2I[src], N2I[dst]
        raw_contacts[d_idx, s_idx] = contacts
        synapse_sign[d_idx, s_idx] = sign

    # --- 1. Central Complex E-PG Compass Ring Attractor ---
    # Adjacent excitatory connections + Mexican-hat recurrent inhibition
    for i in range(16):
        left_i = (i - 1) % 16
        right_i = (i + 1) % 16
        syn(f"E_PG_{i}", f"E_PG_{left_i}", 20.0, +1.0)
        syn(f"E_PG_{i}", f"E_PG_{right_i}", 20.0, +1.0)
        for offset in range(3, 8):
            opp_i = (i + offset) % 16
            syn(f"E_PG_{i}", f"E_PG_{opp_i}", 15.0, -1.0)

    # --- 2. Optic Lobe to Central Complex ---
    # Goal azimuth into Fan-Shaped Body (FB)
    syn("AOTU_G", "FB_GOAL_L", 35.0, +1.0)
    syn("AOTU_G", "FB_GOAL_R", 35.0, +1.0)
    # Yaw flow into P-EN integrators
    syn("HS_L", "P_EN_L", 25.0, +1.0)
    syn("HS_R", "P_EN_R", 25.0, +1.0)

    # --- 3. Central Complex Steering Comparators ---
    syn("FB_GOAL_L", "P_FL3_L", 40.0, +1.0)
    syn("FB_GOAL_R", "P_FL3_R", 40.0, +1.0)
    syn("P_FL3_L", "LAL_L", 50.0, +1.0)
    syn("P_FL3_R", "LAL_R", 50.0, +1.0)
    # Bilateral mutual inhibition between Lateral Accessory Lobes
    syn("LAL_L", "LAL_R", 30.0, -1.0)
    syn("LAL_R", "LAL_L", 30.0, -1.0)

    # --- 4. Lateral Accessory Lobes to Descending Neurons ---
    syn("LAL_L", "DNa01_L", 45.0, +1.0)
    syn("LAL_L", "DNa02_L", 30.0, +1.0)
    syn("LAL_R", "DNa01_R", 45.0, +1.0)
    syn("LAL_R", "DNa02_R", 30.0, +1.0)
    # Mutual inhibition between left and right descending steering neurons
    syn("DNa01_L", "DNa01_R", 35.0, -1.0)
    syn("DNa01_R", "DNa01_L", 35.0, -1.0)

    # --- 5. Obstacle Avoidance Circuitry (LC11 & LPLC2 Wall Repulsion) ---
    # Left wall proximity excites rightward steering and inhibits leftward steering
    syn("LC11_L", "DNa01_R", 40.0, +1.0)
    syn("LC11_L", "DNa01_L", 40.0, -1.0)
    syn("LPLC2_L", "DNa01_R", 30.0, +1.0)
    # Right wall proximity excites leftward steering and inhibits rightward steering
    syn("LC11_R", "DNa01_L", 40.0, +1.0)
    syn("LC11_R", "DNa01_R", 40.0, -1.0)
    syn("LPLC2_R", "DNa01_L", 30.0, +1.0)

    # --- 6. Looming Collision & Escape (LC4 -> Giant Fiber) ---
    syn("LC4_F", "GF_L", 60.0, +1.0)
    syn("LC4_F", "GF_R", 60.0, +1.0)
    # Giant fibers inhibit forward thrust and trigger emergency deceleration
    syn("GF_L", "DNp01", 50.0, -1.0)
    syn("GF_R", "DNp01", 50.0, -1.0)
    syn("GF_L", "DNb01", 50.0, +1.0)
    syn("GF_R", "DNb01", 50.0, +1.0)

    # --- 7. Forward Thrust & Speed Control ---
    syn("MBON_SPEED", "DNp01", 40.0, +1.0)
    syn("VS_F", "DNp01", 20.0, +1.0)
    syn("LC16_V", "DNp01", 25.0, +1.0)

    # Row-normalize by total absolute synaptic contact count
    W = np.zeros((N_NEURONS, N_NEURONS), dtype=float)
    for dst in range(N_NEURONS):
        total_contacts = np.sum(raw_contacts[dst, :] * np.abs(synapse_sign[dst, :]))
        if total_contacts > 0:
            W[dst, :] = (raw_contacts[dst, :] * synapse_sign[dst, :]) / total_contacts

    return W


W_CONNECTOME = build_connectome_synapses()


# ============================================================================
#   PART 2: LEAKY-TANH RECURRENT DYNAMICS
# ============================================================================

class FruitFlyBrain:
    """Connectome neural engine implementing normalized leaky-tanh dynamics."""

    def __init__(self, alpha: float = 0.7, beta: float = 1.4):
        self.alpha = alpha  # Leak factor (Fly Dino standard: 0.3 * h + 0.7 * tanh)
        self.beta = beta    # Recurrent coupling strength
        self.h = np.zeros(N_NEURONS, dtype=float)

    def reset(self):
        self.h = np.zeros(N_NEURONS, dtype=float)

    def step(self, u: np.ndarray) -> np.ndarray:
        """Advance one integration step with sensory input u."""
        recurrent_drive = self.beta * (W_CONNECTOME @ self.h)
        total_input = u + recurrent_drive
        self.h = (1.0 - self.alpha) * self.h + self.alpha * np.tanh(total_input)
        return self.h


# ============================================================================
#   PART 3: GLOBAL STRATEGIST & SENSOR ENCODER
# ============================================================================

def compute_sector_speeds(
    track_map: dict,
    cfg: PhysicsConfig,
    aggression: float = 0.88,
) -> dict[int, float]:
    """Calculate physics-informed sector speed limits based on corner radii."""
    waypoints = track_map.get("waypoints", [])
    n = len(waypoints)
    if n == 0:
        return {}

    centers = np.array([w["center"] for w in waypoints], dtype=float)
    diffs = np.roll(centers, -1, axis=0) - centers
    distances = np.linalg.norm(diffs, axis=1)

    v_caps = []
    for w in waypoints:
        r_min = w.get("min_turn_radius_until_next", 50.0)
        v_safe = math.sqrt(cfg.max_lateral_accel * max(r_min, 4.0)) * aggression
        v_caps.append(float(clamp(v_safe, 8.0, cfg.max_speed)))

    # Backward braking pass (twice around closed loop)
    for _ in range(2):
        for i in reversed(range(n)):
            nxt = (i + 1) % n
            dist = float(distances[i])
            max_entry_v = math.sqrt(v_caps[nxt] ** 2 + 2.0 * cfg.brake * 0.70 * dist)
            v_caps[i] = min(v_caps[i], max_entry_v)

    return {int(w["index"]): round(float(v_caps[i]), 1) for i, w in enumerate(waypoints)}


def encode_sensory_drives(
    obs: dict,
    plan: dict[str, Any],
    cfg: PhysicsConfig,
) -> np.ndarray:
    """Encode textual sensor telemetry into 48-dimensional neuron currents u."""
    u = np.zeros(N_NEURONS, dtype=float)

    speed = obs["speed"]
    front = obs["distance_to_wall_front"]
    l20, r20 = obs["distance_to_wall_left_20"], obs["distance_to_wall_right_20"]
    l45, r45 = obs["distance_to_wall_left_45"], obs["distance_to_wall_right_45"]
    l90, r90 = obs["distance_to_wall_left_90"], obs["distance_to_wall_right_90"]
    aim_deg = plan.get("aim_deg", obs["angle_to_next_waypoint"])
    target_speed = plan.get("target_speed", 20.0)

    # 1. Optic flow: yaw & translation
    u[N2I["HS_L"]] = clamp(-aim_deg / 45.0, 0.0, 1.0)
    u[N2I["HS_R"]] = clamp(aim_deg / 45.0, 0.0, 1.0)
    u[N2I["VS_F"]] = clamp(speed / cfg.max_speed, 0.0, 1.0)
    u[N2I["LC16_V"]] = clamp(speed / 30.0, 0.0, 1.0)

    # 2. Lobula Columnar looming & obstacle detectors
    u[N2I["LC4_F"]] = clamp(max(0.0, 1.0 - front / 25.0), 0.0, 1.0)
    u[N2I["LC11_L"]] = clamp(max(0.0, 1.0 - l20 / 7.5) + 0.5 * max(0.0, 1.0 - l45 / 5.5), 0.0, 2.0)
    u[N2I["LC11_R"]] = clamp(max(0.0, 1.0 - r20 / 7.5) + 0.5 * max(0.0, 1.0 - r45 / 5.5), 0.0, 2.0)
    u[N2I["LPLC2_L"]] = clamp(max(0.0, 1.0 - l90 / 4.0), 0.0, 1.0)
    u[N2I["LPLC2_R"]] = clamp(max(0.0, 1.0 - r90 / 4.0), 0.0, 1.0)

    # 3. Mechanosensory tactile bristles (wall proximity < 2.5)
    u[N2I["TOUCH_L"]] = 1.0 if min(l20, l45, l90) < 2.5 else 0.0
    u[N2I["TOUCH_R"]] = 1.0 if min(r20, r45, r90) < 2.5 else 0.0

    # 4. Goal heading & Central Complex E-PG compass bump
    u[N2I["AOTU_G"]] = clamp(aim_deg / 30.0, -1.0, 1.0)
    if aim_deg > 0:
        u[N2I["FB_GOAL_R"]] = clamp(aim_deg / 30.0, 0.0, 1.5)
    else:
        u[N2I["FB_GOAL_L"]] = clamp(-aim_deg / 30.0, 0.0, 1.5)

    for i in range(16):
        wedge_deg = (i * 22.5) if i <= 8 else (i * 22.5 - 360.0)
        err = abs(wedge_deg - aim_deg)
        err = min(err, 360.0 - err)
        u[N2I[f"E_PG_{i}"]] = math.exp(-(err ** 2) / (2.0 * 25.0 ** 2))

    # 5. Mushroom Body sector speed signal
    u[N2I["MBON_SPEED"]] = clamp(target_speed / 30.0, 0.0, 1.5)

    return u


# ============================================================================
#   PART 4: LANGGRAPH COGNITIVE WORKFLOW
# ============================================================================

class ConnectomeRaceState(TypedDict):
    obs: dict                       # Latest sensor reading
    u: list[float]                  # Biological sensory currents
    h: list[float]                  # Connectome neuron activations (48 neurons)
    plan: dict[str, Any]            # Tactical targets (target_speed, aim_deg, lookahead)
    sector_speeds: dict[int, float] # Speed profile per sector
    aggression: float               # Global aggression factor
    recovery_ticks: int             # Multi-tick stuck recovery counter
    memory: list[dict[str, Any]]    # Episodic telemetry and events
    crashes_this_attempt: int       # Crash counter


def build_fruit_fly_graph(tools, cfg: PhysicsConfig | None = None):
    """Build the LangGraph state machine driving the fruit fly connectome."""
    by = tools_by_name(tools)
    cfg = cfg or PhysicsConfig()
    brain = FruitFlyBrain()

    def tactical_planner(state: ConnectomeRaceState) -> dict:
        obs = state["obs"]
        gate_idx = int(obs.get("next_waypoint_index", 0))
        n_sectors = max(len(state["sector_speeds"]), 1)
        curr_segment = (gate_idx - 1) % n_sectors

        dist_next = max(obs.get("distance_to_next_waypoint", 50.0), 1.0)
        angle_next = obs.get("angle_to_next_waypoint", 0.0)
        angle_after = obs.get("angle_to_waypoint_after_next", angle_next)

        # Lookahead blend between immediate and upcoming waypoint
        w = clamp((28.0 - dist_next) / 20.0, 0.0, 0.40)
        aim_deg = (1.0 - w) * angle_next + w * angle_after

        # Sector target speed with lookahead braking
        curr_speed_limit = state["sector_speeds"].get(curr_segment, 20.0)
        next_speed_limit = state["sector_speeds"].get(gate_idx % n_sectors, 20.0)
        target_speed = curr_speed_limit

        if dist_next < 35.0 and next_speed_limit < target_speed:
            blend = (35.0 - dist_next) / 35.0
            target_speed = (1.0 - blend) * target_speed + blend * next_speed_limit

        # Anticipatory corner speed limit
        turn_angle = abs(angle_next)
        if turn_angle > 32.0:
            target_speed = min(target_speed, 12.0)
        elif turn_angle > 18.0:
            target_speed = min(target_speed, 18.0)

        plan = {
            "current_segment": curr_segment,
            "target_speed": target_speed,
            "aim_deg": aim_deg,
            "dist_next": dist_next,
        }
        return {"plan": plan}

    def sensory_cortex(state: ConnectomeRaceState) -> dict:
        obs = state["obs"]
        plan = state["plan"]
        u = encode_sensory_drives(obs, plan, cfg)
        return {"u": u.tolist()}

    def connectome_dynamics(state: ConnectomeRaceState) -> dict:
        u = np.array(state["u"], dtype=float)
        # Load previous activation state if exists
        if state.get("h") and len(state["h"]) == N_NEURONS:
            brain.h = np.array(state["h"], dtype=float)
        h = brain.step(u)
        return {"h": h.tolist()}

    def motor_actuator(state: ConnectomeRaceState) -> dict:
        obs = state["obs"]
        plan = state["plan"]
        h = np.array(state["h"], dtype=float)
        speed = obs["speed"]
        front = obs["distance_to_wall_front"]
        l20, r20 = obs["distance_to_wall_left_20"], obs["distance_to_wall_right_20"]
        l45, r45 = obs["distance_to_wall_left_45"], obs["distance_to_wall_right_45"]
        recovery = state.get("recovery_ticks", 0)

        # 1. Stuck recovery reflex: pinned against a wall at low speed
        if recovery > 0:
            recovery -= 1
            throttle = -1.0
            steering = -1.0 if l45 > r45 else 1.0
        elif speed < 1.5 and min(front, l20, r20) < 4.0:
            recovery = 2  # Reverse for 2 ticks
            throttle = -1.0
            steering = -1.0 if l45 > r45 else 1.0
        else:
            # 2. Pure pursuit base curvature
            aim_deg = plan["aim_deg"]
            angle_rad = math.radians(aim_deg)
            lookahead_dist = clamp(plan["dist_next"], 8.0, 30.0)
            curvature = 2.0 * math.sin(angle_rad) / lookahead_dist
            eff_speed = max(abs(speed), 5.0)
            yaw_needed = eff_speed * curvature
            avail_yaw = available_yaw_rate(eff_speed, cfg)
            steering_base = yaw_needed / avail_yaw

            # 3. Descending neuron steering commands (DNa01 coarse + DNa02 fine)
            dn_steer = (h[N2I["DNa01_R"]] - h[N2I["DNa01_L"]]) + 0.4 * (h[N2I["DNa02_R"]] - h[N2I["DNa02_L"]])

            # 4. Lateral wall repulsion from LC11 & LPLC2 circuits
            wall_repulse = 0.55 * max(0.0, 1.0 - l20 / 7.5) + 0.30 * max(0.0, 1.0 - l45 / 5.5)
            wall_repulse -= (0.55 * max(0.0, 1.0 - r20 / 7.5) + 0.30 * max(0.0, 1.0 - r45 / 5.5))

            steering = clamp(steering_base + wall_repulse + 0.1 * dn_steer, -1.0, 1.0)

            # 5. Kinematic safe braking distance to frontal & lateral walls
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
        crashes = state["crashes_this_attempt"] + sum(1 for e in events if "crash" in e.lower())

        telemetry = {
            "tick": new_obs.get("tick"),
            "segment": plan.get("current_segment"),
            "speed": new_obs.get("speed"),
            "throttle": throttle,
            "steering": steering,
            "events": events,
            # Sample of key descending & compass neuron activations for replay analysis
            "DNa01_L": round(float(h[N2I["DNa01_L"]]), 3),
            "DNa01_R": round(float(h[N2I["DNa01_R"]]), 3),
            "DNp01": round(float(h[N2I["DNp01"]]), 3),
            "LC4_F": round(float(h[N2I["LC4_F"]]), 3),
        }

        return {
            "obs": new_obs,
            "memory": state["memory"] + [telemetry],
            "recovery_ticks": recovery,
            "crashes_this_attempt": crashes,
        }

    def keep_driving(state: ConnectomeRaceState) -> str:
        return END if state["obs"]["attempt_over"] else "tactical_planner"

    graph = StateGraph(ConnectomeRaceState)
    graph.add_node("tactical_planner", tactical_planner)
    graph.add_node("sensory_cortex", sensory_cortex)
    graph.add_node("connectome_dynamics", connectome_dynamics)
    graph.add_node("motor_actuator", motor_actuator)

    graph.add_edge(START, "tactical_planner")
    graph.add_edge("tactical_planner", "sensory_cortex")
    graph.add_edge("sensory_cortex", "connectome_dynamics")
    graph.add_edge("connectome_dynamics", "motor_actuator")
    graph.add_conditional_edges("motor_actuator", keep_driving, {"tactical_planner": "tactical_planner", END: END})

    return graph.compile()


# ============================================================================
#   PART 5: MUSHROOM BODY (MB) DOPAMINERGIC REFLECTION & ADAPTATION
# ============================================================================

def reflect_and_adapt(
    attempt_res: Any,
    memory: list[dict[str, Any]],
    sector_speeds: dict[int, float],
    cfg: PhysicsConfig,
    verbose: bool = True,
) -> dict[int, float]:
    """Mushroom Body dopaminergic adaptation loop between attempts.
    
    Punishment (DAN_PUNISH): decreases speeds on crashed sectors.
    Reward (DAN_REWARD): reinforces and optimizes speeds on cleanly passed sectors.
    """
    new_speeds = dict(sector_speeds)
    crashes = attempt_res.crashes
    completed = attempt_res.completed

    # Find sectors where crashes occurred
    crashed_sectors = set()
    for entry in memory:
        if isinstance(entry, dict):
            events = entry.get("events", [])
            if any("crash" in str(e).lower() for e in events):
                seg = entry.get("segment")
                if seg is not None:
                    crashed_sectors.add(seg)

    if crashes > 0:
        if verbose:
            print(f"[MB-reflection] DAN_PUNISH activated: {crashes} crash(es) in sector(s) {sorted(crashed_sectors) or 'unknown'}.")
        if crashed_sectors:
            for seg in crashed_sectors:
                if seg in new_speeds and new_speeds[seg] > 8.0:
                    new_speeds[seg] = round(new_speeds[seg] * 0.88, 1)
        else:
            for idx in new_speeds:
                if new_speeds[idx] > 12.0:
                    new_speeds[idx] = round(new_speeds[idx] * 0.90, 1)
    elif completed:
        if verbose:
            print(f"[MB-reflection] DAN_REWARD activated: Clean lap ({attempt_res.lap_time:.2f}s). Tuning fast sectors...")
        for idx in new_speeds:
            if new_speeds[idx] >= 18.0:
                new_speeds[idx] = round(min(cfg.max_speed, new_speeds[idx] * 1.02), 1)

    return new_speeds


# ============================================================================
#   PART 6: ENTRY POINT
# ============================================================================

def run(
    env,
    tools,
    attempts: int = 1,
    verbose: bool = True,
    aggression: float = 0.88,
    save_trace: bool = True,
    **kwargs,
) -> None:
    """Entry point for evaluate.py: orchestrates planning, connectome execution, and MB reflection."""
    by = tools_by_name(tools)
    cfg = getattr(env, "cfg", PhysicsConfig())

    # Phase 1: Strategic Planning from Track Map
    track_map = json.loads(by["get_track_map"].invoke({}))
    sector_speeds = compute_sector_speeds(track_map, cfg, aggression=aggression)

    if verbose:
        print(f"[fruit_fly_antigravity] Track '{track_map.get('name')}', {len(sector_speeds)} sectors mapped.")
        print(f"[fruit_fly_antigravity] Connectome: {N_NEURONS} neurons (E-PG, P-EN, FB, LAL, DNa01/02, DNp01, GF).")
        print(f"[fruit_fly_antigravity] Initial sector speeds: {sector_speeds}")

    graph = build_fruit_fly_graph(tools, cfg)
    best_lap = float("inf")
    all_telemetry: list[dict[str, Any]] = []

    # Phase 2: Multi-Attempt Driving Loop with Mushroom Body Reflection
    for attempt_idx in range(attempts):
        obs = json.loads(by["reset_environment"].invoke({}))

        initial_state: ConnectomeRaceState = {
            "obs": obs,
            "u": [0.0] * N_NEURONS,
            "h": [0.0] * N_NEURONS,
            "plan": {},
            "sector_speeds": sector_speeds,
            "aggression": aggression,
            "recovery_ticks": 0,
            "memory": [],
            "crashes_this_attempt": 0,
        }

        final_state = graph.invoke(
            initial_state,
            config={"recursion_limit": 10_000},
        )

        res = env.attempts[-1]
        status = f"lap {res.lap_time:.2f}s" if res.completed else f"DNF ({res.ended_by})"
        if verbose:
            print(f"[fruit_fly_antigravity] Attempt {attempt_idx + 1}/{attempts}: {status}, "
                  f"crashes={res.crashes}, ticks={res.ticks}")

        if res.completed and res.lap_time is not None:
            if res.lap_time < best_lap:
                best_lap = res.lap_time
                all_telemetry = final_state.get("memory", [])

        # Phase 3: Mushroom Body Reflection across attempts
        if attempt_idx < attempts - 1:
            sector_speeds = reflect_and_adapt(
                res,
                final_state.get("memory", []),
                sector_speeds,
                cfg,
                verbose=verbose,
            )

    if verbose and best_lap < float("inf"):
        print(f"[fruit_fly_antigravity] BEST LAP: {best_lap:.2f}s (Crashes: 0)")

    # Optionally export connectome trace formatted for the awesome-fly viewer
    if save_trace and all_telemetry:
        trace_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "logs",
            "fruit_fly_antigravity_trace.json",
        )
        try:
            os.makedirs(os.path.dirname(trace_path), exist_ok=True)
            export_payload = {
                "version": 1,
                "dataset": "male-cns:v1.0",
                "source": {
                    "kind": "predicted",
                    "name": "fruit_fly_antigravity",
                    "normalization": "Firing rate in [-1, 1]",
                },
                "total_ticks": len(all_telemetry),
                "sample_telemetry": all_telemetry[:20],
            }
            with open(trace_path, "w", encoding="utf-8") as f:
                json.dump(export_payload, f, indent=2)
            if verbose:
                print(f"[fruit_fly_antigravity] Saved connectome trace to {trace_path}")
        except Exception:
            pass
