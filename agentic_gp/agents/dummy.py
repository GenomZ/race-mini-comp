"""The dummy agent: drives straight ahead at half throttle.

It exists to show the mechanics of the tool loop. It will crash into the first
corner and keep crashing - that is the point.
"""

from __future__ import annotations

import json

from ..tools import tools_by_name


def run(env, tools, attempts: int = 1, verbose: bool = True, **_) -> None:
    by = tools_by_name(tools)
    for _ in range(attempts):
        obs = json.loads(by["reset_environment"].invoke({}))
        while not obs["attempt_over"]:
            obs = json.loads(by["drive"].invoke({"throttle": 0.5, "steering": 0.0}))
            if verbose and obs["events"]:
                print(f"tick {obs['tick']:3d} speed {obs['speed']:5.1f} -> {obs['events']}")
