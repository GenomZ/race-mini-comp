"""L0 Dummy - the agent with no perception at all (open loop).

Agent type (Russell & Norvig): not even a simple reflex agent - it never looks at `obs`, so it
cannot react to anything (open loop). Agentic concept: an agent without PERCEPTION. It is the
"before" picture of the whole course.

    obs ──(ignored)──>  [ always: half throttle, straight ahead ]  ──> (0.5, 0.0)

How it works
- drive() returns the same command every tick: throttle 0.5, steering 0.0.
- The car drives straight into the first corner's wall. Under the default "marshal" crash rule it
  is put back on the centerline, pointing along the track (+5 s), drives a bit further and hits
  the next wall. The marshals carry it round the lap, one crash at a time.
- With the "classic" crash rule (--crash-rule classic) nobody helps: it stays stuck at the first
  wall until the tick limit ends the attempt (DNF).

Run it:
    python race.py drive drivers/examples/l0_dummy.py --attempts 3
    python race.py drive drivers/examples/l0_dummy.py --replay races/me_l0.json   # then watch the replay
    python race.py drive drivers/examples/l0_dummy.py --crash-rule classic

Expected on grand_prix: ~307 s with ~31 crashes (every one of them +5 s), the same every attempt;
DNF with the classic crash rule.

Try this
1. Print `obs` once and read every key (see agentic_gp/driver.py OBS_FIELDS).
2. Steer towards the next waypoint: steering = obs["angle_to_next_waypoint"] / 30 ("+ = right").
3. Brake when distance_to_wall_front gets small. Congratulations: that is a reflex agent (L1).
"""

NAME = "Dummy"
AUTHOR = "Agentic GP"
COLOR = "#8d99ae"
KIND = "code"
MOTTO = "Eyes closed, foot down."
ANATOMY = ["act"]
PARAMS = {"throttle": 0.5}


def drive(obs: dict) -> tuple[float, float]:
    """Ignore the world: same command every tick."""
    return PARAMS["throttle"], 0.0
