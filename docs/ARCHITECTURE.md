# Architecture notes

What was built, how it works, and where it deviates from the [implementation plan](IMPLEMENTATION_PLAN.md).

## Module map

```
agentic_gp/
  config.py      PhysicsConfig: every tunable constant in one frozen dataclass
  geometry.py    numpy helpers: raycast, segment intersection, Catmull-Rom spline, arc-length
                 resampling, curvature radius
  track.py       Track dataclass (centerline -> left/right walls -> wall segments -> gates),
                 make_default_track(), make_oval_track(), Track.to_map()
  physics.py     CarState + integrate(): longitudinal accel/brake/drag, grip-fraction steering
  env.py         RaceEnv: reset/step/sensors, collision + penalty, ordered gates, lap timing,
                 AttemptResult records, per-attempt traces, JSONL logging
  tools.py       RaceTools (plain Python) + make_tools() -> 4 LangChain StructuredTools
  llm.py         get_chat_model(): provider switch (ollama | anthropic | openai) via env vars
  render.py      matplotlib (Agg) plots: track, trajectory coloured by speed, crash markers
  leaderboard.py results/*.json -> LEADERBOARD.md
  agents/        dummy (straight), reflex (rule-based reference), hermes (zero-shot LLM baseline)
evaluate.py      loads an agent module, runs run(env, tools, attempts), writes results + plot
my_agent.py      student starter: LangGraph planner -> executor loop
```

Data flow for one tick:

```
agent -> drive(throttle, steering) -> RaceEnv.step()
           10 x integrate(dt=0.05)   -> wall check (segment vs 720 wall segments, vectorised)
                                     -> gate check (movement segment vs next gate)
           sensors()                 -> 7 raycasts + waypoint geometry -> dict
        <- JSON string (+ events)
```

## Simulation design

### Track
A closed Catmull-Rom spline through ~16 control points (scaled 0.8x) is resampled to 360 points at
equal arc length. Walls are the centerline offset by ±9 units along the normal. Gates are the wall-
to-wall segments at 16 evenly spaced indices; gate 0 is start/finish. The car starts one sample
(~4 units) before gate 0 on the main straight, heading along +x.

Default circuit: lap 1554 units, width 18, tightest corner radius ~14 (hairpin), plus 30-100 unit
sweepers and a 200-unit straight. `Track.to_map()` exposes waypoint centers, local track heading and
the tightest radius per segment so planners can derive safe speeds.

### Car model (`physics.integrate`)
* Longitudinal: `a = throttle*12` (throttle >= 0), `throttle*25` when braking while moving, half
  acceleration in reverse when stopped; `a -= 0.15 * v`; speed clipped to [-6, 40]. Braking cannot
  flip direction inside a step.
* Lateral: **steering commands a fraction of the available lateral grip**
  `yaw = steering * min(1.5, 20/|v|)` rad/s, heading decreases for positive (right) steering.
  Hence `R = v² / (20*|s|)`, `v_max(R) = sqrt(20*R)`.

  *Why not a bicycle model?* The first version used a kinematic bicycle model (wheelbase 2.6,
  30 deg max steer) with the same grip cap. At speed 30 a steering input of just 0.11 already
  saturated the grip limit and turned the car 19 deg per tick, so the effective control range was
  a sliver around zero and any driver oscillated. Mapping steering to grip fraction makes the
  response linear and predictable at every speed, which is what an LLM (or a student's executor)
  can actually reason about.
* Integration: 10 substeps of 0.05 s per 0.5 s tick, midpoint heading for position update.

### Collision and penalties
Each substep the movement segment `p0 -> p1` is tested against all wall segments (vectorised cross
products). On a hit: crashes += 1, penalty += 5 s, speed := 0, and the car is placed at `p0` nudged
2 units backwards along its heading (only if that nudge does not itself cross a wall). Throttle is
ignored for the rest of that tick so a single tick can only register one crash. Because the
movement segment is tested (not the end point), tunnelling through walls is impossible at any speed.

### Waypoints and laps
`next_gate` starts at 1. A gate counts only when the movement segment crosses *that* gate, so gates
cannot be taken out of order or backwards. Crossing gate 0 with all others done sets `lap_complete`
and ends the attempt. `lap_time = sim_time + penalty_time`. Attempts also end after `max_ticks`
(500) or when `reset_environment` is called mid-attempt (recorded as `ended_by="reset"`); a reset
before any tick is not recorded.

### Sensors
Seven rays (0, ±20, ±45, ±90 deg; negative = left), capped at 200. Waypoint geometry for the next
and the one after (distance + signed angle, degrees, negative = left). Plus position, heading,
timers and counters. Everything is rounded for compact JSON.

## Tool layer
`make_tools(env)` returns `StructuredTool`s named `read_sensors`, `drive`, `reset_environment`,
`get_track_map` with pydantic arg schemas and LLM-oriented descriptions. Returns are compact JSON
strings. Bounds on `drive` are documented rather than enforced: the env clips, so an LLM emitting
`1.2` does not abort the race with a validation error. `RaceTools` keeps a ledger of every call
(`tool_calls` on the leaderboard).

## LLM access
`agentic_gp.llm.get_chat_model()` picks the provider from `RACE_LLM_PROVIDER` and the model from
`RACE_LLM_MODEL`. Defaults: Ollama + `hermes3` (OpenAI-compatible endpoint), `claude-opus-5` for
Anthropic (no sampling params are passed: Claude 5-family models reason adaptively and reject
`temperature`), `gpt-4o-mini` for OpenAI. The Hermes baseline and the starter share this factory so
switching models is a config change.

## Hermes baseline
Per tick: `[system generic prompt] + [human: sensor JSON]` -> `llm.bind_tools([drive])` ->
execute the tool call (fallback: parse `{throttle, steering}` JSON or `throttle: x` text; if nothing
parses, coast). No memory by default (`--agent-arg history=N` adds a sliding window). Prints LLM
latency, parse failures and provider errors per attempt.

## Reference and starter agents
* `agents/reflex.py`: one-off plan from `get_track_map` (spline through waypoint centers ->
  curvature -> speed profile with a backward braking pass), then pure-pursuit steering, planned
  speed, wall-distance braking and a stuck-recovery reverse. Clean lap **69.45 s** at aggression 1.0
  (probe: 1.1 -> 70.45 s clean, 1.2 -> 2 crashes). Uses only the four tools.
* `my_agent.py`: LangGraph `planner -> executor -> (loop | END)`; rule-based placeholder planner
  (target speed 16/10), executor with wall repulsion, room-ahead speed cap and stuck recovery.
  Clean lap **207.3 s** in 415 ticks: finishes, but leaves everything on the table.
* `agents/dummy.py`: throttle 0.5, steering 0. DNF.

## Evaluation and leaderboard
`evaluate.py <module|path> --name X --attempts N [--plot]` builds an env with JSONL logging into
`logs/`, calls the agent's `run`, then writes `results/X.json` (attempts, best lap, crashes, tool
calls, wall clock) and `results/X.png` (best attempt trajectory). `python -m agentic_gp.leaderboard`
sorts finishers by best lap (then crashes), DNFs by waypoints reached, and writes `LEADERBOARD.md`
with deltas against the `hermes` entry.

## Deviations from the plan
| Plan | Built | Reason |
|------|-------|--------|
| pymunk physics | ~80-line custom model | deterministic, trivially inspectable, no native deps, fast enough for 10 substeps/tick |
| pygame visualisation | matplotlib PNGs | headless, works in CI and notebooks; a live viewer can be added on top of the JSONL logs |
| FastAPI option | not built | in-process Python bindings are simpler for a bootcamp, as the plan anticipated |
| 3 tools | 4 tools | `get_track_map` gives planners something to plan with; without it the "planner" hint is hollow |
| 5 rays | 7 rays | ±20 deg rays see the corner entry that ±45 misses at 15 units/tick |
| 12 waypoints | 16 | denser targets make waypoint-chasing agents (including Hermes) viable |
| Docker image | not built | `pip install -r requirements.txt` on Python 3.10+ suffices |
