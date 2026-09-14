# Implementation Plan: Agentic Car Racing Competition

This document outlines the architecture, implementation steps, and student materials for a data
science bootcamp competition where students build agentic workflows to race a virtual car, competing
against an autonomous "Hermes" baseline agent.

> Implementation status against this plan is tracked at the [bottom of this document](#implementation-status).
> Design details of what was actually built are in [ARCHITECTURE.md](ARCHITECTURE.md).

## Part 1: Project Architecture & System Design

The system requires three main components: a lightweight simulation environment (the "Gym"), an API
to interact with it using tools, and the evaluation framework for the competition.

### 1.1 The Simulation Environment (The "Gym")

We need a simple, deterministic 2D physics environment. Using a full 3D simulator is overkill and too
slow for agentic workflows, which rely on LLM API calls.

* Technology Choice: Python with `pygame` (for visualization if needed, though headless is preferred
  for speed) and `pymunk` (for simple 2D rigid body physics) or a custom grid/waypoint-based track logic.
* Track Representation:
  * A set of track boundaries (inner and outer walls).
  * A sequence of waypoints (checkpoints) the car must pass through in order to complete a lap.
  * A start/finish line.
* Car State:
  * Position (X, Y).
  * Velocity (speed and direction).
  * Orientation (angle).
* Action Space (Discrete or Continuous):
  * Acceleration (throttle/brake): -1.0 to 1.0.
  * Steering: -1.0 (left) to 1.0 (right).
* Sensors (What the Agent "Sees"):
  * Raycasts (LiDAR-like): Distances to track walls at various angles (e.g., front, 45 deg
    left/right, 90 deg left/right).
  * Current speed.
  * Vector to the next waypoint (distance and relative angle).

### 1.2 The Tool API Layer

The LLM agents will not interact directly with Python arrays; they need discrete tools they can call
to perceive the world and take action.

* Technology Choice: FastAPI (for a local server approach) or direct Python function bindings if the
  agent runs in the same process as the gym. Direct Python functions are easier for a bootcamp setup.
* Required Tools (LangChain `Tool` objects):
  1. `get_sensor_data()`: Returns JSON containing speed, distance to walls, and vector to the next waypoint.
  2. `apply_controls(throttle, steering)`: Takes values between -1 and 1. Advances the simulation by
     $n$ steps (the "tick rate") and returns the new sensor data and lap status.
  3. `reset_environment()`: Places the car at the starting line and resets the timer.

### 1.3 The Agentic Framework (LangGraph/LangChain)

Students will build their agents here. The flow generally looks like this:

1. Agent calls `get_sensor_data()`.
2. Agent analyzes data (using an LLM).
3. Agent decides on controls and calls `apply_controls(throttle, steering)`.
4. Loop repeats until the lap is complete or the car crashes.

### 1.4 The "Hermes" Baseline Agent

* The baseline agent should use a powerful open-source model (like Hermes) or an API model if budget permits.
* It operates zero-shot or few-shot using a generic prompt like: "You are driving a virtual car.
  Analyze the sensor data and output the optimal throttle and steering values to complete the track
  quickly without hitting walls."
* It will likely be slow and prone to errors because it re-evaluates every frame without a persistent
  state or specialized workflow (which is exactly what the students need to beat).

## Part 2: Implementation Roadmap

### Phase 1: Environment Setup (Week 1)

1. Build the 2D Engine: Implement the simple physics and collision detection (walls vs. car).
2. Design the Track: Create a standard track layout (e.g., an oval or a figure-8) defined by
   coordinates and waypoints.
3. Implement the API Wrapper: Write the `get_sensor_data`, `apply_controls`, and `reset` functions.
   Ensure they return clean, JSON-serializable outputs.

### Phase 2: Agent Tooling & Baseline (Week 2)

1. Create LangChain Tool Bindings: Wrap the API functions into `StructuredTool` objects that LLMs can understand.
2. Develop the Hermes Baseline: Build a simple LangChain loop that feeds sensor data to the Hermes
   model and executes its control outputs. Record its lap time (or failure rate).
3. Implement Logging: Ensure every action, state change, and API call is logged so students can debug their agents.

### Phase 3: Competition Infrastructure (Week 3)

1. Package the Environment: Create a pip-installable package or a clean Docker container for students to run locally.
2. Create Starter Code: Provide a basic repository with the environment running, the tools defined,
   and a dummy agent that just drives straight.
3. Establish Leaderboard: Set up a simple way to submit agents (e.g., a GitHub classroom repo where
   you run their code and post times, or a simple web form).

## Part 3: Student Materials

The student-facing competition README lives at the repository root: [README.md](../README.md).
It contains the problem definition, environment rules, tool reference, strategy hints and the
getting-started steps described in this plan.

---

## Implementation status

| Plan item | Status | Where |
|-----------|--------|-------|
| 2D engine: physics + wall collision | done, custom numpy model (no pymunk; see ARCHITECTURE.md for why) | `agentic_gp/physics.py`, `agentic_gp/env.py` |
| Track: walls, ordered waypoints, start/finish | done, spline-based "Grand Prix Circuit" + a test oval | `agentic_gp/track.py` |
| Sensors: rays (front, ±20, ±45, ±90), speed, vector to next waypoint | done, plus waypoint-after-next, position, heading | `RaceEnv.sensors()` |
| Tool API: sensors / controls / reset as direct Python bindings | done as `read_sensors`, `drive`, `reset_environment` (+ `get_track_map`) | `agentic_gp/tools.py` |
| LangChain `StructuredTool` bindings | done | `agentic_gp.make_tools(env)` |
| Hermes baseline (zero-shot, generic prompt, per-frame) | done; Hermes 3 via Ollama by default, any provider via env vars | `agentic_gp/agents/hermes.py`, `run_hermes.py` |
| Logging of every action/state | done, JSONL per run + tool-call ledger | `RaceEnv._log`, `RaceTools.calls` |
| Tick rate 0.5 s, crash = stop + 5 s penalty, ordered waypoints | done | `agentic_gp/config.py` |
| pip-installable package | done (`pyproject.toml`); Docker not provided | repo root |
| Starter repo: env running, tools defined, dummy straight driver | done | `run_dummy.py`, `my_agent.py` (LangGraph starter) |
| Leaderboard | done: `evaluate.py` writes `results/*.json`, `python -m agentic_gp.leaderboard` renders `LEADERBOARD.md` | `agentic_gp/leaderboard.py` |
| Visualisation | headless matplotlib plots of track + trajectory (pygame live view not built) | `agentic_gp/render.py` |
| Tests + CI | pytest suite, GitHub Actions | `tests/`, `.github/workflows/tests.yml` |
