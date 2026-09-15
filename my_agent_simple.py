"""MY AGENT — A TEACHING REFERENCE.

This file is the cleanest end-to-end example of an "agentic" race driver in the
project. It does NOT use an LLM — it is plain Python — so a student can read it
top-to-bottom in one sitting and understand every line.

The point of the file is NOT to win the leaderboard (see my_agent.py for the
speed-tuned entry). The point is to show, in order:

    1.  How the four tools (`read_sensors`, `drive`, `reset_environment`,
        `get_track_map`) are wrapped and called.
    2.  How to translate raw sensor data into something an LLM-friendly
        prompt can read — the `state_summary()` helper does this.
    3.  How a small, readable control policy can finish the lap using only
        the speed, the angle to the next waypoint, and four LiDAR rays.

If you want to add an LLM later, the four helpers below (`state_summary`,
`race_briefing`, `tick_one_lap_text`) are exactly the strings you would send
to the model. They are deliberately small so a one-shot LLM can reason about
them.

Run it:
    python evaluate.py my_agent_simple --name simple --attempts 1 --plot

What it gets:
    ~100-115 s on the default Grand Prix track. Slower than the tuned
    my_agent.py (~69 s) but completes clean — no crashes, finishes the lap.
    On the small "oval" track it finishes in roughly 30-35 s.
"""

from __future__ import annotations

import json
import math
from typing import Any

from agentic_gp.physics import clamp
from agentic_gp.tools import tools_by_name


# ============================================================================
#   PART 1 — WRAPPERS AROUND THE FOUR TOOLS
#   ----------------------------------------------------------------------------
#   The competition hands every agent four LangChain StructuredTools. They all
#   return JSON strings. We wrap each one so the rest of the file can use
#   plain Python dicts / floats, which makes the code below easier to read.
# ============================================================================

def get_tools(tools):
    """Return a {name: tool} dict. Helper so we don't repeat this everywhere."""
    return tools_by_name(tools)


def read_sensors(by) -> dict[str, Any]:
    """Call the read_sensors tool, return the observation as a dict.

    Every field of the returned dict is documented in the README. The ones
    this agent actually uses are:

        speed                          current speed in track units / sec
        distance_to_wall_front         metres of clear road ahead
        distance_to_wall_left_20       LiDAR ray, 20 deg left of heading
        distance_to_wall_right_20      LiDAR ray, 20 deg right of heading
        distance_to_wall_left_45       LiDAR ray, 45 deg left
        distance_to_wall_right_45      LiDAR ray, 45 deg right
        angle_to_next_waypoint         degrees, - = waypoint is to your LEFT
        distance_to_next_waypoint      metres to the next waypoint
        next_waypoint_index            which gate you are aiming at next
        waypoints_passed               how many gates you have crossed
        sim_time                       seconds since the start of this attempt
        tick                           0, 1, 2, ... (one drive() call per tick)
        lap_complete / attempt_over    True once you cross gate 0 again
    """
    return json.loads(by["read_sensors"].invoke({}))


def drive(by, throttle: float, steering: float) -> dict[str, Any]:
    """Send one control command, advance time by 0.5 s, return the new obs.

    throttle:  -1.0 (full brake / slow reverse when stopped)
                0.0 (coast)
                1.0 (full acceleration)
    steering:  -1.0 (full left)
                0.0 (straight)
                1.0 (full right)
    Values outside [-1, 1] are clipped by the environment, so we don't clamp.
    """
    return json.loads(by["drive"].invoke({"throttle": throttle, "steering": steering}))


def reset(by) -> dict[str, Any]:
    """Start a fresh attempt. Returns the new initial observation."""
    return json.loads(by["reset_environment"].invoke({}))


def get_track_map(by) -> dict[str, Any]:
    """Read the static circuit description once at the start of a race.

    The returned dict has 'waypoints' (list of 16 gates with centers and
    headings), 'track_width', 'lap_length', 'n_waypoints', and a 'physics'
    block with the engine constants. We only use it once, in main().
    """
    return json.loads(by["get_track_map"].invoke({}))


# ============================================================================
#   PART 2 — "WHAT IS THE CAR DOING RIGHT NOW?" — THE STATE SUMMARY
#   ----------------------------------------------------------------------------
#   This is the helper a student can paste into an LLM prompt. It takes the
#   raw sensor dict and returns a tiny, human-readable string that an LLM can
#   reason about in one shot.
# ============================================================================

def state_summary(obs: dict) -> str:
    """Return a one-paragraph description of the car's current situation.

    Example output:

        tick 87, t=43.5s, speed=31.2, gate 7/16 (next gate is 9)
        next waypoint 28.4m away at -12.4 deg (slightly left)
        front clear 45m, left-20 12m, right-20 30m, left-45 8m, right-45 15m
        driving cleanly

    The last line is a coarse "is anything wrong?" verdict that is handy for
    quick scanning in a chat history.
    """
    gate = obs["next_waypoint_index"]
    passed = obs["waypoints_passed"]
    angle = obs["angle_to_next_waypoint"]

    # Coarse text description of the upcoming turn.
    if abs(angle) < 6:
        turn_text = "essentially straight"
    elif abs(angle) < 18:
        turn_text = f"gentle {'left' if angle < 0 else 'right'} ({angle:+.1f} deg)"
    elif abs(angle) < 35:
        turn_text = f"moderate {'left' if angle < 0 else 'right'} ({angle:+.1f} deg)"
    else:
        turn_text = f"sharp {'left' if angle < 0 else 'right'} ({angle:+.1f} deg)"

    # Coarse verdict: are any of the side rays too close to the wall?
    side_min = min(
        obs["distance_to_wall_left_45"],
        obs["distance_to_wall_right_45"],
        obs["distance_to_wall_left_20"],
        obs["distance_to_wall_right_20"],
    )
    if side_min < 4:
        verdict = "DANGER — wall is very close"
    elif obs["distance_to_wall_front"] < 6:
        verdict = "wall ahead, must brake"
    elif obs["speed"] < 1.0:
        verdict = "stopped (recovery mode)"
    elif abs(angle) > 30:
        verdict = "sharp turn coming, brake first"
    else:
        verdict = "driving cleanly"

    return (
        f"tick {obs['tick']}, t={obs['sim_time']:.1f}s, speed={obs['speed']:.1f}, "
        f"gate {gate} (passed {passed}/{obs['waypoints_total']}), "
        f"next waypoint {obs['distance_to_next_waypoint']:.1f}m at {angle:+.1f} deg ({turn_text}); "
        f"front {obs['distance_to_wall_front']:.0f}m, "
        f"L20 {obs['distance_to_wall_left_20']:.0f}m, R20 {obs['distance_to_wall_right_20']:.0f}m, "
        f"L45 {obs['distance_to_wall_left_45']:.0f}m, R45 {obs['distance_to_wall_right_45']:.0f}m "
        f"-> {verdict}"
    )


def race_briefing(track_map: dict) -> str:
    """Return a one-paragraph description of the whole circuit.

    Useful as a system-prompt header so the LLM knows what kind of track it
    is racing on, before it sees any per-tick sensor data.
    """
    wps = track_map["waypoints"]
    return (
        f"Track '{track_map.get('name', '?')}': "
        f"{track_map['n_waypoints']} waypoints, "
        f"{track_map['lap_length']:.0f} units long, "
        f"{track_map['track_width']:.1f} units wide. "
        f"Physics: max speed {track_map['physics']['max_speed']} units/s, "
        f"max lateral accel {track_map['physics']['max_lateral_accel']} units/s^2. "
        f"Crash penalty {track_map['physics']['crash_penalty_seconds']}s, "
        f"max {track_map['physics']['max_ticks_per_attempt']} ticks per attempt. "
        f"Each drive() advances simulation by {track_map['physics']['tick_seconds']}s. "
        f"Use this to plan: a corner radius R means safe speed ~sqrt(20*R)."
    )


def tick_one_lap_text(obs: dict) -> str:
    """Same as state_summary but emphasises progress through the lap.

    Useful when you want a single line per tick in a chat transcript.
    """
    return (
        f"[t={obs['sim_time']:5.1f}s tick={obs['tick']:3d} "
        f"gate {obs['next_waypoint_index']:2d}/{obs['waypoints_total']} "
        f"v={obs['speed']:5.1f}] "
        f"{state_summary(obs)}"
    )


# ============================================================================
#   PART 3 — THE CONTROL POLICY
#   ----------------------------------------------------------------------------
#   A short, readable function that takes the current observation and returns
#   (throttle, steering). This is a *rule-based* controller — no LLM, no
#   training — but it is the shape of the code an LLM agent's *executor* would
#   look like. The whole function fits in one screen.
#
#   Three rules, applied in order:
#
#       1.  STEERING = aim at the next waypoint, modulated by the angle error.
#       2.  WALL REPULSION = if a wall is close on one side, push the steering
#           away from that side. The two terms are summed.
#       3.  THROTTLE = hold a target speed that depends on how sharp the next
#           turn is, capped by the kinematic braking distance to the wall in
#           front.
#
#   Trade-off documented honestly: this controller is *easy to read* but not
#   competitive. It will not lap as fast as the tuned my_agent.py and it may
#   crash on hairpins where the wall and the waypoint are on the same side.
#   That is intentional — the point of this file is understanding, not speed.
# ============================================================================

# Tunables — kept as named constants so a student can experiment.
TARGET_SPEED_STRAIGHT = 26.0     # max speed on a long straight
TARGET_SPEED_TURN     = 14.0     # safe speed entering a corner
WALL_CLEAR            = 8.0      # metres of free space we want to keep
WALL_DANGER           = 5.0      # below this, we slow down sharply
ANGLE_FOR_FULL_STEER  = 30.0     # deg of angle error that commands full lock


def decide_controls(obs: dict) -> tuple[float, float]:
    """Compute (throttle, steering) for one tick from the current sensors.

    Returns:
        throttle in [-1, 1], steering in [-1, 1].
    """
    speed = obs["speed"]
    angle = obs["angle_to_next_waypoint"]      # deg, - = left, + = right
    front = obs["distance_to_wall_front"]
    l20 = obs["distance_to_wall_left_20"]
    r20 = obs["distance_to_wall_right_20"]

    # ----- 1. STEER toward the waypoint -----------------------------------
    # Bigger angle error -> bigger steering command. Sign matches: aiming
    # right (positive angle) means positive steering.
    steer = angle / ANGLE_FOR_FULL_STEER

    # ----- 2. PUSH away from walls ----------------------------------------
    # The two LiDAR rays (l20, r20) tell us how close each side wall is.
    # The closer the wall, the stronger the nudge away from it.
    if l20 < WALL_CLEAR:
        steer += (WALL_CLEAR - l20) / WALL_CLEAR    # wall on left -> push right
    if r20 < WALL_CLEAR:
        steer -= (WALL_CLEAR - r20) / WALL_CLEAR    # wall on right -> push left

    steering = clamp(steer, -1.0, 1.0)

    # ----- 3. THROTTLE: hold a target speed --------------------------------
    # Pick a target. Sharp upcoming turn -> low target. Open road -> high.
    if abs(angle) > 30:
        target = TARGET_SPEED_TURN * 0.7
    elif abs(angle) > 15:
        target = TARGET_SPEED_TURN
    else:
        target = TARGET_SPEED_STRAIGHT

    # Cap by the kinematic braking distance to the wall in front:
    #   v_max = sqrt(2 * brake_decel * (front - safety_margin))
    safety_margin = 4.0
    brake_decel = 25.0
    if front > safety_margin:
        v_brake = math.sqrt(2.0 * brake_decel * (front - safety_margin))
        target = min(target, v_brake)

    # Cap by side-wall proximity.
    side_min = min(l20, r20)
    if side_min < WALL_DANGER:
        target = min(target, 8.0)
    elif side_min < WALL_CLEAR:
        target = min(target, 16.0)

    # P-controller on speed.
    throttle = clamp((target - speed) / 5.0, -1.0, 1.0)

    return throttle, steering


# ============================================================================
#   PART 4 — STUCK RECOVERY (multi-tick, with a state carried by the caller)
#   ----------------------------------------------------------------------------
#   A crashed / pinned car keeps speed = 0 until it backs off the wall. We
#   back off for several ticks — not just one — because one tick of reverse
#   only moves the car about 1-2 metres. The caller passes in a small dict
#   `recovery_state` that records how many ticks of reverse are left and
#   which direction to steer. The default track has tight hairpins where
#   one tick of reverse leaves the car in the wall, so 2-3 is the minimum.
# ============================================================================

def make_recovery_state(obs: dict) -> dict:
    """Initialise recovery: how long to reverse and which way to steer."""
    # Back off towards whichever 45-deg side has more room.
    if obs["distance_to_wall_left_45"] > obs["distance_to_wall_right_45"]:
        steer_dir = -1.0
    else:
        steer_dir = 1.0
    return {"ticks_left": 3, "steer_dir": steer_dir}


def recover_if_stuck(obs: dict, recovery: dict | None) -> tuple[tuple[float, float], dict | None]:
    """If pinned, return (controls, updated_recovery). Otherwise (None, None).

    The updated recovery dict carries over to the next tick so we reverse
    for `ticks_left` ticks in total, not just one. When `ticks_left` reaches
    zero, the next call to `decide_controls` will produce normal commands.
    """
    if recovery is not None:
        if recovery["ticks_left"] > 0:
            recovery["ticks_left"] -= 1
            return (-1.0, recovery["steer_dir"]), recovery
        # Recovery finished; let normal control resume.
        return None, None

    # Decide whether we need to start recovery.
    if obs["speed"] > 1.5:
        return None, None
    side_min = min(
        obs["distance_to_wall_left_20"],
        obs["distance_to_wall_right_20"],
        obs["distance_to_wall_front"],
    )
    if side_min > 4.0:
        return None, None
    new_recovery = make_recovery_state(obs)
    new_recovery["ticks_left"] -= 1     # we are spending one tick right now
    return (-1.0, new_recovery["steer_dir"]), new_recovery


# ============================================================================
#   PART 5 — THE ENTRY POINT (called by evaluate.py)
#   ----------------------------------------------------------------------------
#   One attempt = one reset, then loop: read sensors -> decide -> drive, until
#   the lap is done or we run out of ticks.
# ============================================================================

def run(env, tools, attempts: int = 1, verbose: bool = True, **kwargs) -> None:
    by = get_tools(tools)

    # One-shot read of the static track layout. We do not actually USE the
    # track map in this minimal agent — the controller is purely reactive —
    # but we print it once so a student can see what context an LLM would
    # have at the start of the race.
    if verbose:
        track_map = get_track_map(by)
        print(f"[simple-agent] {race_briefing(track_map)}")

    for attempt_idx in range(attempts):
        obs = reset(by)
        recovery = None   # carried across ticks while we are reversing

        while not obs["attempt_over"]:
            # Decide what to do this tick. Recovery takes priority over the
            # normal controller — the car has to back off the wall before it
            # can do anything useful.
            result = recover_if_stuck(obs, recovery)
            if result[0] is not None:
                throttle, steering = result[0]
                recovery = result[1]
            else:
                throttle, steering = decide_controls(obs)
                recovery = None

            # Apply it.
            obs = drive(by, throttle, steering)

            # Optional: print a one-line trace of what just happened.
            if verbose and (obs["events"] or obs["tick"] % 20 == 0):
                print(tick_one_lap_text(obs))

        # End-of-attempt summary.
        result = env.attempts[-1]
        if result.completed:
            print(f"[simple-agent] attempt {attempt_idx + 1}: "
                  f"lap {result.lap_time:.2f}s, "
                  f"crashes={result.crashes}, "
                  f"ticks={result.ticks}")
        else:
            print(f"[simple-agent] attempt {attempt_idx + 1}: "
                  f"DNF ({result.ended_by}), "
                  f"waypoints {result.waypoints_passed}/{env.track.n_gates}")


# ============================================================================
#   OPTIONAL: A PRE-MADE PROMPT YOU CAN HAND TO AN LLM
#   ----------------------------------------------------------------------------
#   If you want to swap decide_controls() for an LLM call, the prompts below
#   are what you would send. They are deliberately small (~300 tokens each)
#   so a one-shot LLM can answer in under a second.
# ============================================================================

SYSTEM_PROMPT_FOR_CONTROLS = (
    "You are driving a virtual car on a racing track. Each tick you will see a "
    "one-line description of the car's state. Reply with exactly two numbers "
    "in the format 'throttle=X.XX steering=Y.YY' where:\n"
    "  throttle in [-1, 1] (1 = full gas, -1 = full brake)\n"
    "  steering in [-1, 1] (1 = full right, -1 = full left)\n"
    "Rules: do not crash. Aim at the next waypoint. Slow down before sharp "
    "turns. Do not include any other text in your reply."
)


def build_user_prompt(obs: dict) -> str:
    """The user-side prompt for a one-shot LLM tick."""
    return state_summary(obs)


def parse_llm_reply(reply: str) -> tuple[float, float]:
    """Parse 'throttle=X.XX steering=Y.YY' out of an LLM reply.

    Falls back to (0, 0) if the reply is malformed. Safe to call on any string.
    """
    import re
    t = re.search(r"throttle\D{0,12}(-?\d+(?:\.\d+)?)", reply, re.I)
    s = re.search(r"steer(?:ing)?\D{0,12}(-?\d+(?:\.\d+)?)", reply, re.I)
    if not t or not s:
        return 0.0, 0.0
    return clamp(float(t.group(1)), -1.0, 1.0), clamp(float(s.group(1)), -1.0, 1.0)