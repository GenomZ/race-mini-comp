# Drivers

One file = one driver. `template.py` is where you start; `examples/` is the ladder of reference drivers,
one per chapter, each a little more "agentic" than the one before. Every example is self-contained and
readable top to bottom; its docstring explains the agent type, shows the loop as a diagram, tells you how
to run it and ends with "Try this" exercises. `submissions/` is where your file goes (see its README).

```bash
python race.py drive drivers/examples/l2_planner.py --attempts 3            # one driver, one track
python race.py drive drivers/examples/l3_learner.py --track random:7 --attempts 6
python race.py check drivers/template.py                                    # the contract check
python race.py grand-prix drivers/examples/l3_learner.py --house --track oval   # + house cars (see below)
```

`--house` adds the house team (Max, Planner, Fly Toy) and marks them as house cars: they race but score
no championship points. Pass Max with `--house`, never as a plain path, or it is scored like a student.

## The ladder

| File | Ch | Agent type (Russell & Norvig) | Agentic concept | In one line |
|---|---|---|---|---|
| `template.py` | 0-1 | simple reflex (starter) | perceive -> act | slow; clean on the public tracks, finished all 300 random tracks tried; TODOs show the upgrades |
| `examples/l0_dummy.py` | 0 | none (open loop) | no perception | half throttle, straight ahead; the marshals carry it round |
| `examples/l1_reflex.py` | 1 | simple reflex | condition -> action rules | aim, look-through, line of sight, stop/turn-away speed, recover |
| `examples/l2_planner.py` | 2 | model-based, goal-based | world model + planning | v = sqrt(a R) on the dense centerline + braking pass + pure pursuit |
| `examples/l2b_racing_line.py` | 2 | model-based, goal-based | better model / path | the L2 planner on `driverkit.racing_line()` (Ch2 Gold) |
| `examples/l3_learner.py` | 3 | learning agent | memory + reflection | per-segment factors + safety margin; keep-best, revert, lock-in |
| `examples/l4_llm_rookie.py` | 4 | LLM as the reflex policy | the latency wall | MiniMax every tick (thinking off), 20 LLM ticks, then code fallback |
| `examples/l5_llm_strategist.py` | 5 | goal-based, LLM planner | structured output, validation, briefing | LLM writes a corner plan JSON once; validated, clamped, fallback |
| `examples/l6_llm_crew_chief.py` | 6 | learning + roles (**Max**, the house racer) | tools, reflection, memory, guardrails, budgets | Strategist + Race Engineer (MiniMax, tools) + Steward + Driver (code) |
| `examples/l6b_pit_crew_langgraph.py` | 6+ | learning, multi-agent | orchestration (LangGraph) | analyst -> strategist -> safety officer (retry) -> commit |
| `examples/fly_toy.py`, `fly_connectome.py` | 7 | spiking "brains" | neurons as controllers | owned by the fly team, see their docstrings |

All examples set `NAME`, `AUTHOR`, `COLOR` (distinct), `KIND`, `MOTTO`, `ANATOMY` and numeric `PARAMS`
(`--param key=value`, Streamlit sliders). LLM drivers also set `MODEL` and never call the LLM inside
`drive()` except L4 (a lesson tool that is refused at official events and then drives on code reflexes).
No example contains a per-track number: everything is computed from the track map it is given.

## Measured lap times (code drivers, and LLM drivers with `LLM_PROVIDER=mock`)

Default crash rule (marshal), one-lap attempts, measured 2026-10-21 with `run_session` (the same thing
`python race.py drive ... --quiet` runs). Times are race times = sim time + 5 s per crash + 1 s per tick
too fast in a zone; `(Nz)` = zone violations on `briefing_demo`. Learners: attempt 1 -> best of 4.
Code drivers are deterministic: you get these exact numbers.

| Driver | Attempts | grand_prix | oval | sprint | speedway | briefing_demo | random:1-10 (mean) |
|---|---|---|---|---|---|---|---|
| l0_dummy | 1 | 307.40 (31 crashes) | 151.60 (15) | 207.30 (21) | 284.35 (26) | 316.40 (31, 9z) | 244.1 (233 crashes) |
| template | 1 | 85.85 | 44.40 | 64.80 | 106.30 | 91.85 (1z) | 80.1, 0 crashes |
| l1_reflex | 1 | 66.70 | 31.65 | 50.00 | 74.30 | 71.70 (1z) | 59.4, 0 crashes |
| l2_planner | 1 | 52.50 | 25.95 | 41.40 | 60.60 | 59.75 (2z) | 46.4, 0 crashes |
| l2b_racing_line | 1 | 49.95 | 25.00 | 38.40 | 58.20 | 57.65 (2z) | 44.6, 0 crashes |
| l3_learner | 4 | 52.50 -> 48.30 | 25.95 -> 24.70 | 41.40 -> 39.50 | 60.60 -> 57.00 | 59.75 -> 57.20 (2z/lap) | 43.4, 0 crashes |
| l3_learner `racing_line=1` (instructor's best) | 4 | 49.95 -> 44.95 | 25.00 -> 24.45 | 38.40 -> 35.85 | 58.20 -> 54.30 | 57.65 -> 53.50 (2z/lap) | 43.1, 0 crashes |
| l4_llm_rookie (mock = fallback) | 1 | 71.80 | 34.50 | 49.35 | 82.35 | 76.65 (1z) | 65.8 (1 crash) |
| l5_llm_strategist (mock = physics plan) | 1 | 49.95 | 25.00 | 38.40 | 58.20 | 57.65 (2z) | 44.6, 0 crashes |
| l6_llm_crew_chief / Max (mock = code plan) | 4 | 49.95 | 25.00 | 38.40 | 58.20 | 57.30 (0z) | 44.6, 0 crashes |
| l6b_pit_crew_langgraph (mock) | 4 | 49.95 | 25.00 | 38.40 | 58.20 | 57.30 (0z) | 44.6, 0 crashes |

The ladder test (`tests/test_drivers_ladder.py`) checks the ordering on grand_prix: L2 clean,
L2b < L2, L3 best-of-4 < L2 with no DNF and >= 3 % gain, L1 and the template clean on every public track,
and `python race.py check` PASSED for every example and the template.

Wider runs (one lap each, `random:1..N`): the template and L1 finished every track (template 300 of 300,
crashes on 7; L1 200 of 200, one crash on 9). L2b beats L2 by a median 1.4 s on 200 random tracks but is
slower on 12 of them.

## With MiniMax (real LLM, `MiniMax-M3`)

Grand-prix weekend format, official LLM rules: plan -> 3 practice laps (reflect) -> qualifying.
Qualifying time = the best plan, locked in. The code drivers and two FIXED scripts run the same format.
A fixed script is Max with a mock "LLM" that always answers the same two tool calls (no thinking at all):
`1.3 / 0.8` = aggression 1.3 + safety margin 0.8 (the best single setting of a 25-track sweep),
`1.6 / 0.8` = every knob at the Steward's limit.

| Track | L2b | L3 | L3 racing line | fixed 1.3 / 0.8 | fixed 1.6 / 0.8 (max) | Max, mock | **Max, MiniMax** practice 1 -> 2 -> 3 = qualifying |
|---|---|---|---|---|---|---|---|
| grand_prix | 49.95 | 49.30 | 45.45 | 43.00 | 44.65 | 49.95 | **46.05 -> 46.05 -> 44.65 = 44.65** |
| briefing_demo | 57.65 (2z) | 57.25 (2z) | 53.90 (2z) | 52.45 | 53.10 | 57.30 | **54.30 -> 54.30 -> 53.45 = 53.45 (0z)** |
| random:7 | 44.15 | 41.50 | 44.00 | 41.80 | 42.95 | 44.15 | **43.00 -> 42.00 -> 43.35 = 42.00** |
| random:12 | 39.15 | 39.75 | 37.20 | 36.85 | 37.90 | 39.15 | **37.60 -> 37.20 -> 37.45 = 37.20** |
| sprint | 38.40 | 39.50 | 36.85 | 35.05 | 41.40 (1 crash) | 38.40 | **36.35 -> 35.80 -> 35.55 = 35.55** |
| random:21 | 42.85 | 42.35 | 42.20 | 42.20 | 41.15 | 42.85 | **41.10 -> 42.15 -> 41.20 = 41.10** |

`(2z)` = 2 zone violations (+1 s each, included in the time). L1 and L2 are slower everywhere (see above).

- Max was clean (0 crashes, 0 zone violations) in every MiniMax weekend. It beats the best published
  code example (L3, with or without the racing line) on grand_prix, briefing_demo (it reads the yellow
  flag), sprint and random:21, ties on random:12 and loses to L3 on random:7.
- **Max is beatable by tuning.** The fixed 1.3 / 0.8 script is faster than MiniMax on 5 of the 6 tracks
  (all but random:21): the LLM is a decent tuner, not an optimal one, and a sweep over 25 tracks is
  information it does not have. "Everything at the limit" (1.6 / 0.8) is not a free win: it crashed on
  sprint, is slower than MiniMax on 4 of 6 tracks and ties on grand_prix, because more speed or less
  margin can make the car brake harder for the walls. That is why the LLM's choices matter.
- LLM runs vary: three grand_prix weekends with the current prompts qualified at 44.65, 44.95 and
  45.75 s. Run it twice before you draw conclusions.
- LLM usage per weekend: 6 calls (2 plan + 2 + 2 reflect; the 3rd practice reflect is the lock-in and
  needs no LLM), 12.6-21.6k prompt + 3.6-8.8k completion tokens, 31-152 s of LLM time in total (one
  slow-server weekend hit the 45 s request timeout once; Max kept its plan and stayed clean).
- Oval grand prix (command below, real MiniMax): Max (house) pole 23.90 s, won the 3-lap race in 67.40 s,
  4 LLM calls; Learner 72.70, Planner (house) 73.50, Fly Toy (house) 83.70, Rookie 100.05; all clean.
- L5 strategist (one plan, one lap): grand_prix 49.4 s, briefing_demo 58.2 s with 0 zone violations; 1-2 calls.
- L6b pit crew (grand_prix, 3 practice): 49.95 -> 46.80 -> 45.90, qualifying 45.90 s, 2 LLM calls.
- L4 rookie (grand_prix, 20 LLM ticks, turn-based): 78.15 s, 0 crashes, p50 1.16 s / p95 1.95 s per call.
- Grand prix demo with real MiniMax:
  `python race.py grand-prix drivers/examples/l3_learner.py drivers/examples/l4_llm_rookie.py --house --track oval --practice 2 --laps 3 --jobs 4 --out races/demo/llm_gp_oval.json`
  (Max, Planner and Fly Toy are house cars; the Learner and the Rookie are scored).

## Max in five sentences

Max is the house racer (#0, black and gold): same rules and budget as every student. The **Strategist**
(MiniMax) reads the corner table and the briefing and sets the plan through tools; the **Steward** (code)
checks every value (factors 0.6-1.6, at most 10 % change per debrief, safety margin 0.8-4, every briefing
zone respected, no zone that is not in the briefing, a first plan not estimated > 5 % slower than the code
plan) and returns its verdict to the model. The **Driver** (code) follows the plan on the racing
line and records, per corner, whether the plan or the safety net limited the speed. After each practice lap
the **Race Engineer** (MiniMax) reads `report["summary_text"]`, that telemetry and the lessons notebook,
calls tools and proposes changes; code guardrails revert a clearly slower plan, slow down crash corners and
lock in the best plan for qualifying and the race. Without an LLM it races its code plan (the racing line).
Read `self.crew_log` after a session to see every call, tool call and verdict.
