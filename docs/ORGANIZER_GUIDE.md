# Organiser guide

## Setup

```bash
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest                                   # ~5 s, includes a clean-lap regression test
```

## Establish the Hermes baseline

The baseline model is Hermes 3 served by Ollama:

```bash
ollama pull hermes3
python run_hermes.py --attempts 3
```

This writes `results/hermes.json` and `results/hermes.png`. Each tick is one LLM call, so an attempt
costs up to 500 calls; with a local model at ~1-3 s per call budget 5-25 minutes per attempt. A
hosted model is faster: `RACE_LLM_PROVIDER=anthropic RACE_LLM_MODEL=claude-haiku-4-5 python run_hermes.py`.

Options: `--agent-arg history=2` gives Hermes a 2-turn memory (still no plan). Use the same
configuration for the published baseline as students see, and publish the model name (it is stored
in the results JSON and shown on the leaderboard).

## Evaluating submissions

```bash
python evaluate.py submissions/team_a/my_agent.py --name team-a --attempts 3 --plot
python evaluate.py submissions/team_b/my_agent.py --name team-b --attempts 3 --plot
python -m agentic_gp.leaderboard          # regenerates LEADERBOARD.md from results/*.json
```

* Best completed lap counts; ties break on fewer crashes; DNFs rank by waypoints reached.
* Every evaluation logs each tick to `logs/<name>_<timestamp>.jsonl` (controls, sensors, events),
  and `results/<name>.json` records the number of tool calls and wall-clock time.
* Run submissions in a fresh virtualenv or container. `evaluate.py` imports and executes student
  code, so treat it as untrusted.

### Fair-play checks
Students receive the `RaceEnv` object as well as the tools (so they can read `env.attempts` for
reflection). The rules forbid moving the car through anything but `drive`. Quick audit:

```bash
grep -nE "env\.(car|track|sim_time|penalty|crashes|next_gate|_)" submissions/team_a/*.py
```

Also compare `tool_calls` in the results JSON with `ticks`: a lap with fewer `drive` calls than
ticks is impossible.

## Tuning the competition

All constants live in `agentic_gp/config.py` (`PhysicsConfig`). Useful knobs:

| Knob | Effect |
|------|--------|
| `tick_seconds` | reaction time per decision; 0.5 s is the spec, 0.25 s makes Hermes much stronger |
| `max_lateral_accel` | corner speeds (`v = sqrt(a*R)`); lower = more braking needed |
| `crash_penalty_seconds`, `max_ticks` | severity of mistakes, attempt length |
| `ray_angles_deg` | what the agent sees |

Tracks live in `agentic_gp/track.py` (`TRACKS` registry; `--track oval` in `evaluate.py`). To add a
circuit, list control points and register a factory. The tests assert the reference driver laps the
default track clean, so re-run `pytest` after changing physics or track and re-tune
`agents/reflex.py` if needed.

Plot any track without driving it:

```bash
python -c "from agentic_gp.track import make_default_track; from agentic_gp.render import plot_track; plot_track(make_default_track(), 'results/track.png')"
```

## Suggested schedule

| Week | Students | Organisers |
|------|----------|------------|
| 1 | clone, run dummy + starter, read sensors, build a code-only executor | publish Hermes baseline time and plot |
| 2 | add LLM planner (`get_track_map`), memory across attempts | office hours on LangGraph state / tool calling |
| 3 | reflection on crash logs, model split (strong planner / fast executor), tuning | collect submissions, run `evaluate.py`, publish `LEADERBOARD.md` |
