"""L1 Reflex - a simple reflex agent: condition -> action rules on the sensors, nothing else.

Agent type (Russell & Norvig): SIMPLE REFLEX AGENT. It sees only the current `obs` (7 wall rays,
the angles to the next two waypoints, its speed) and maps it straight to (throttle, steering) with
if-then rules. No map, no plan, no memory of earlier ticks (except "am I reversing out of a wall?").
Agentic concept: PERCEIVE -> ACT. Every rule here is a reflex you could explain in one sentence.

    obs (sensors, this tick only)
      │
      ├─ angle to next waypoint ──┐ look-through: blend in the waypoint after next
      ├─ wall rays (7) ───────────┼─> AIM: is the aim blocked by a wall? -> aim at the open side
      │                           └─> STEER: steer_towards(aim) + push away from close walls
      ├─ front ray, speed ────────> SPEED: slow enough to stop / to turn away before the wall
      └─ zone_speed_limit ────────> SPEED: obey a yellow flag (only seen once you are inside it)
                                     │
                                     v
                          (throttle, steering)  every 0.5 s

How it works
- AIM: steer at the next waypoint gate; when it is close, blend in the waypoint after it, so the car
  starts turning before it reaches the gate ("look through the corner").
- LINE OF SIGHT: if the wall in the aim direction is closer than the gate, the gate is hidden behind
  a wall (a blind corner) - aim halfway towards the most open ray instead.
- STEER: driverkit.steer_towards turns "the angle I want" into a steering command (steering is a
  fraction of grip, so the same command turns less at high speed), plus a push away from close walls.
- SPEED: the slowest of four limits: top speed cap, "can I still STOP before the wall in front?"
  (v = sqrt(2 * brake * distance)), "can I still TURN AWAY from it?" (v = sqrt(grip * distance)), and
  "how sharp is the turn ahead?" (cosine of the aim angle).
- RECOVER: stopped against a wall? Reverse towards the open side for a moment (recover_if_stuck).

Run it:
    python race.py drive drivers/examples/l1_reflex.py --attempts 3
    python race.py drive drivers/examples/l1_reflex.py --track sprint --param cap=30
    python race.py check drivers/examples/l1_reflex.py

Expected: clean (0 crashes) on every public track: grand_prix ~67 s, oval ~32 s, sprint ~50 s,
speedway ~74 s. Random tracks ~45-75 s: it finished all 200 we tried, with one crash on 9 of them
(reflexes cannot see a tricky corner coming). Same result every attempt: a reflex agent cannot learn.

Try this
1. --param sight=0 switches the line-of-sight rule off. Where does the car crash on sprint, and why
   is the gate "straight ahead behind a wall"?
2. --param cap=40 and --param turn_k=1.0: faster on grand_prix? Now run sprint and speedway too.
   Tuning on one track and failing on another is your first taste of overfitting.
3. Delete the recover_if_stuck line, use the "classic" crash rule (--crash-rule classic) and watch
   a crash loop: a stopped car cannot turn.
4. The yellow flag on briefing_demo is only seen once you are inside it (zone_speed_limit), so the
   first tick is always too fast. What would you need to brake BEFORE it? (A map: that is L2.)
"""

from __future__ import annotations

import math

from agentic_gp import driverkit as dk

NAME = "Reflex"
AUTHOR = "Agentic GP"
COLOR = "#e76f51"
KIND = "code"
MOTTO = "See wall, avoid wall."
ANATOMY = ["perceive", "act"]
PARAMS = {
    "cap": 34.0,           # top speed we allow ourselves (u/s; the car can do 40)
    "blend_near": 10.0,    # look-through: fully blended when the next gate is this close (u)
    "blend_far": 40.0,     # ... and starts blending at this distance (u)
    "blend_max": 0.6,      # at most this much weight on the waypoint after next
    "sight": 40.0,         # line-of-sight check distance (u); 0 = rule off
    "wall_gain": 1.5,      # how hard we push away from close side walls
    "margin": 4.0,         # free space we keep in front of the car (u)
    "turn_k": 0.6,         # "turn away" speed limit: v = sqrt(grip * turn_k * front distance)
    "ahead_k": 0.7,        # how much the turn at the waypoint after next slows us down
}

# The seven rays, as (angle, obs key). Driver frame: + = right, - = left.
RAYS = [(-90, "left_90"), (-45, "left_45"), (-20, "left_20"), (0, "front"),
        (20, "right_20"), (45, "right_45"), (90, "right_90")]

_recovery: dict = {}        # the only memory: "am I reversing out of a wall right now?"
_radio = {"last": None}     # (and the last radio message, so we only speak when it changes)


def free_distance(obs: dict, angle: float) -> float:
    """How far we can see in direction `angle` (interpolated between the two nearest rays)."""
    angle = max(-90.0, min(90.0, angle))
    for (a0, k0), (a1, k1) in zip(RAYS, RAYS[1:]):
        if a0 <= angle <= a1:
            d0, d1 = obs["distance_to_wall_" + k0], obs["distance_to_wall_" + k1]
            return d0 + (angle - a0) / (a1 - a0) * (d1 - d0)
    return obs["distance_to_wall_front"]


def drive(obs: dict) -> dict:
    P = PARAMS
    # Rule 0: stuck against a wall -> reverse towards the open side.
    recover = dk.recover_if_stuck(obs, _recovery)
    if recover is not None:
        return {"throttle": recover[0], "steering": recover[1], "say": radio("Reversing out of the wall")}

    speed = obs["speed"]
    a_next, d_next = obs["angle_to_next_waypoint"], obs["distance_to_next_waypoint"]
    a_after = obs["angle_to_waypoint_after_next"]

    # Rule 1: AIM at the next gate, looking through to the one after it as we get close.
    w = dk.clamp(1.0 - (d_next - P["blend_near"]) / (P["blend_far"] - P["blend_near"]), 0.0, P["blend_max"])
    aim = (1.0 - w) * a_next + w * a_after

    # Rule 2: LINE OF SIGHT. A wall nearer than the gate in that direction? Aim towards the open side.
    say = None
    if free_distance(obs, aim) < min(d_next, P["sight"]):
        open_angle = max((a for a, _ in RAYS if abs(a) <= 45), key=lambda a: free_distance(obs, a))
        aim = 0.5 * aim + 0.5 * open_angle
        say = "Blind corner - aiming for the gap"

    # Rule 3: STEER towards the aim, pushed away from walls that are close beside us.
    steering = dk.steer_towards(aim, speed) + dk.wall_repulsion(obs, P["wall_gain"])

    # Rule 4: SPEED = the slowest of the limits.
    front = obs["distance_to_wall_front"]
    v_stop = math.sqrt(2 * 25.0 * max(front - P["margin"] - 0.5 * speed, 0.0))   # brake 25 u/s^2, 0.5 s blind
    v_turn = math.sqrt(20.0 * P["turn_k"] * max(front, 0.0))                     # grip 20 u/s^2
    turn_ahead = max(abs(aim), P["ahead_k"] * abs(a_after) if d_next < P["blend_far"] else 0.0)
    v_bend = P["cap"] * max(0.35, math.cos(math.radians(min(turn_ahead, 80.0))))
    target = max(5.0, min(P["cap"], v_stop, v_turn, v_bend))
    if obs.get("zone_speed_limit") is not None:                                  # yellow flag (seen inside only)
        target = min(target, obs["zone_speed_limit"] - 0.5)
        say = say or f"Yellow flag: max {obs['zone_speed_limit']:g}"

    # Rule 5: THROTTLE towards the target speed (accel 12, brake 25 u/s^2, one tick = 0.5 s).
    dv = target - speed
    throttle = (dv + 0.075 * speed) / 6.0 if dv >= 0 else dv / 12.5
    return {"throttle": dk.clamp(throttle), "steering": dk.clamp(steering), "say": radio(say)}


def radio(text: str | None) -> str | None:
    """Send a radio message only when it is new (viewers show every message)."""
    if text is None or text == _radio["last"]:
        return None
    _radio["last"] = text
    return text
