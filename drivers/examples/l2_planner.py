"""L2 Planner - a model-based, goal-based agent: plan the lap from the map, then execute it.

Agent type (Russell & Norvig): model-based + goal-based. plan() reads the track map once and turns
physics (v_max = sqrt(a_lat * R)) into a speed for every metre of the lap; drive() is a fast
executor that follows the plan with pure pursuit. No LLM, no memory between attempts.
Agentic concepts: a WORLD MODEL (the map sees beyond the sensor horizon) and PLANNING before acting.

    plan(track)                         drive(obs)  (every 0.5 s)
    ┌─────────────────────────┐         ┌──────────────────────────────┐
    │ dense centerline        │         │ where am I on the path?      │
    │  -> curvature radius R  │  plan   │ steer: pure pursuit + walls  │
    │  -> v = sqrt(a_lat * R) │ ──────> │ speed: planned v ahead,      │
    │  -> backward braking    │         │        stop-before-wall net  │
    └─────────────────────────┘         └──────────────────────────────┘

How it works
- Uses the DENSE centerline from track["centerline"] (not a spline through 16 gates).
- Corner speed from curvature: v = sqrt(a_lat * R) x AGGRESSION, capped at max speed.
- A backward pass makes sure every speed can be reached by braking from the one before it,
  so the car brakes BEFORE the corner instead of in it.
- drive() hands the plan to driverkit.PathFollower: pure pursuit steering, wall repulsion,
  and a safety net that never drives faster than it could stop before the wall in front.
- Radio: announces the plan and calls out each corner (T1, T2, ...) as it brakes for it.

Run it:
    python race.py drive drivers/examples/l2_planner.py --attempts 3
    python race.py drive drivers/examples/l2_planner.py --track random:7 --param aggression=1.1

Expected (clean, 0 crashes): grand_prix 52.5 s, sprint 41.4 s, speedway 60.6 s, oval 25.95 s;
clean on random tracks too (mean 46.4 s on random:1-10), with no per-track tuning. Same time every
attempt (no learning). On briefing_demo it collects zone penalties: it never reads the briefing.

Try this
1. --param aggression=1.15: faster or crash? Find the edge on grand_prix, then try sprint.
2. --param brake_margin=0.8: brake earlier. Slower? Safer on random tracks?
3. Replace the centerline with dk.racing_line(track, margin=3.0) (that is l2b_racing_line).
4. Read track["briefing"] and call self.follower.set_speed_limit(...) for a yellow flag.
"""

from __future__ import annotations

import numpy as np

from agentic_gp import driverkit as dk


class Driver:
    NAME = "Planner"
    AUTHOR = "Agentic GP"
    COLOR = "#2a9d8f"
    KIND = "code"
    MOTTO = "Plan the lap, drive the plan."
    ANATOMY = ["perceive", "act", "map", "plan"]
    PARAMS = {
        "aggression": 1.0,        # x physics corner speed (1.0 = exactly the grip limit)
        "brake_margin": 1.0,      # x brake strength used by the planner (< 1 = brake earlier)
    }

    def __init__(self) -> None:
        self.follower: dk.PathFollower | None = None
        self.corners: list[dict] = []
        self.last_say: str | None = None

    # ------------------------------------------------------------------ plan: once per session
    def plan(self, track: dict) -> None:
        p = dk.physics_of(track)                              # grip, brakes, top speed of this sim
        points = np.asarray(track["centerline"], dtype=float)

        # 1. how tight is the track at every point?
        radii = dk.curvature_radii(points)
        # 2. the fastest speed each point allows (grip limit), times our aggression
        v = np.array([dk.speed_for_radius(r, p["max_lateral_accel"], p["max_speed"]) for r in radii])
        v = np.clip(v * self.PARAMS["aggression"], 5.0, p["max_speed"])
        # 3. backward pass: brake in time for every slow point ahead
        speeds = dk.braking_pass(v, points, p["brake"] * self.PARAMS["brake_margin"])

        # 4. the executor: pure pursuit on the same points, with our speeds
        self.follower = dk.PathFollower.from_track(track, speeds=speeds)
        self.corners = dk.track_corners(track)
        self.last_say = None
        slowest = min(self.corners, key=lambda c: c["v_max"]) if self.corners else None
        self.plan_note = (f"Plan: {len(self.corners)} corners, slowest {slowest['id']} at "
                          f"{slowest['v_max']:.0f}" if slowest else "Plan: flat out, no corners")

    # ------------------------------------------------------------------ drive: every tick
    def drive(self, obs: dict) -> dict:
        if self.follower is None:                             # plan() failed: crawl along the waypoints
            return {"throttle": 0.3, "steering": dk.clamp(obs["angle_to_next_waypoint"] / 30.0)}
        throttle, steering = self.follower.control(obs)
        return {"throttle": throttle, "steering": steering, "say": self._radio(obs, throttle),
                "debug": dict(self.follower.debug)}

    def _radio(self, obs: dict, throttle: float) -> str | None:
        """Only speak when the message changes (viewers show every new line)."""
        if obs["tick"] == 0:
            text = self.plan_note
        elif throttle < -0.2:
            idx = self.follower.idx or 0
            ahead = [c for c in self.corners if 0 <= c["apex_index"] - idx < 60]
            text = f"Braking for {ahead[0]['id']}" if ahead else None
        else:
            text = None
        if text is None or text == self.last_say:
            return None
        self.last_say = text
        return text
