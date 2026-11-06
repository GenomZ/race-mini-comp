# Legacy v1 agents (archived)

These are the agents, scripts and results added to `main` on 2026-09-15, before the 2.0 restructure:

- `my_agent.py` - the winning LangGraph agent (68.50 s) and `my_agent_simple.py`, the teaching version
- `fruit_fly_antigravity.py`, `agentic_gp/agents/fruit_fly_minimax.py` - fruit-fly connectome agents,
  with `train_fruit_fly_minimax.py`
- `agentic_gp/agents/hermes.py` - the Hermes baseline with per-tick logging
- `load_env.py`, `run_*.py`, `results/`, `LEADERBOARD.md` - runners, traces and the v1 leaderboard

The files are kept unchanged at the paths they had in v1. They use the v1 API (`agentic_gp.tools`,
`agentic_gp.physics`, `evaluate.py`), which 2.0 removed, so they do **not** run on this branch and
are not part of the test suite. To run them, check out the last v1 commit:

```bash
git switch --detach 9c873e7
```

To bring one into 2.0, port it to a driver in `drivers/` (see `drivers/template.py`).
