"""The shipped starter must finish a clean (if slow) lap, so students start from a working agent."""

import sys
from pathlib import Path

from agentic_gp import RaceEnv, make_tools

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import my_agent  # noqa: E402


def test_starter_agent_finishes_a_clean_lap():
    env = RaceEnv()
    my_agent.run(env, make_tools(env), attempts=1, verbose=False)
    res = env.attempts[-1]
    assert res.completed, res
    assert res.crashes == 0
    assert res.ticks < env.cfg.max_ticks
