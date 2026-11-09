"""L2b Racing Line - the L2 planner on a smarter path: outside, apex, outside.

Agent type (Russell & Norvig): model-based + goal-based, like L2. Same map, same physics, same
executor - only the PATH is better. Agentic concept: the quality of your world model and of your
plan matters as much as the controller (this is the "Gold" step of Ch2).

    plan(track)                                     drive(obs)  (every 0.5 s)
    ┌──────────────────────────────────┐            ┌──────────────────────────────┐
    │ centerline                       │            │ where am I on the line?      │
    │  -> racing_line(track, margin)   │   plan     │ steer: pure pursuit + walls  │
    │     (least curvature inside the  │ ─────────> │ speed: planned v ahead,      │
    │      walls: wider corners)       │            │        stop-before-wall net  │
    │  -> v = sqrt(a_lat * R_line)     │            └──────────────────────────────┘
    │  -> backward braking pass        │
    └──────────────────────────────────┘

How it works
- driverkit.racing_line() slides every centerline point sideways (at most half_width - margin) so the
  whole loop bends as little as possible: it swings wide before a corner, clips the inside at the
  apex and runs wide again on exit.
- Wider corners = bigger radius R = higher corner speed v = sqrt(a_lat * R) for the same grip.
- Point i of the racing line is the sideways shift of centerline point i, so waypoint and corner
  indices from the track map still match.
- The rest is exactly L2: backward braking pass, PathFollower (pure pursuit + safety net).

Run it:
    python race.py drive drivers/examples/l2b_racing_line.py --attempts 3
    python race.py drive drivers/examples/l2b_racing_line.py --track sprint --param line_margin=2

Expected (clean, 0 crashes): grand_prix ~50 s (L2: 52.5 s), oval ~25 s, sprint ~38.4 s,
speedway ~58 s. Random tracks (200 measured): a median 1.4 s faster than L2 (most gain 0.1-2.8 s),
but slower on 12 of 200 - a "better" line is not better everywhere. Same time every attempt.

Try this
1. --param line_margin=1 (closer to the walls, wider line): faster? Still clean on every track?
2. --param line_margin=9 makes the "racing line" the centerline again: you should get L2's time.
3. Plot the line: matplotlib, plt.plot(*np.array(track["centerline"]).T) and the same for
   self.line. Where does the line touch the walls?
4. Combine it with the learner of L3 (per-segment speed factors): that is the instructor's best.
"""

from __future__ import annotations

import numpy as np

from agentic_gp import driverkit as dk


class Driver:
    NAME = "Racing Line"
    AUTHOR = "Agentic GP"
    COLOR = "#264653"
    KIND = "code"
    MOTTO = "Outside, apex, outside."
    ANATOMY = ["perceive", "act", "map", "plan"]
    PARAMS = {
        "line_margin": 3.0,       # units kept free between the racing line and the walls
        "aggression": 1.0,        # x physics corner speed on the line
    }

    def __init__(self) -> None:
        self.follower: dk.PathFollower | None = None
        self.corners: list[dict] = []
        self.last_say: str | None = None
        self.plan_note = ""

    # ------------------------------------------------------------------ plan: once per session
    def plan(self, track: dict) -> None:
        p = dk.physics_of(track)
        # 1. the path: least-curvature line inside the walls (same indexing as the centerline)
        self.line = dk.racing_line(track, margin=self.PARAMS["line_margin"])
        # 2. speeds along THIS line (its radii are larger than the centerline's)
        speeds = dk.speed_profile(self.line, p["max_lateral_accel"], p["brake"], p["max_speed"],
                                  aggression=self.PARAMS["aggression"])
        # 3. the same executor as L2
        self.follower = dk.PathFollower.from_track(track, line=self.line, speeds=speeds)
        self.corners = dk.track_corners(track)
        self.last_say = None
        gain = self._radius_gain(track)
        self.plan_note = f"Racing line: corners {gain:.0f}% wider on average"

    def _radius_gain(self, track: dict) -> float:
        """How much larger the apex radii are on our line than on the centerline (for the radio)."""
        r_center = dk.curvature_radii(track["centerline"], span=2)
        r_line = dk.curvature_radii(self.line, span=2)
        apexes = [c["apex_index"] for c in self.corners] or [int(np.argmin(r_center))]
        ratio = [r_line[i] / r_center[i] for i in apexes if np.isfinite(r_line[i]) and r_center[i] > 0]
        return 100.0 * (float(np.mean(ratio)) - 1.0) if ratio else 0.0

    # ------------------------------------------------------------------ drive: every tick
    def drive(self, obs: dict) -> dict:
        if self.follower is None:                             # plan() failed: crawl along the waypoints
            return {"throttle": 0.3, "steering": dk.clamp(obs["angle_to_next_waypoint"] / 30.0)}
        throttle, steering = self.follower.control(obs)
        return {"throttle": throttle, "steering": steering, "say": self._radio(obs, throttle),
                "debug": dict(self.follower.debug)}

    def _radio(self, obs: dict, throttle: float) -> str | None:
        """Speak only when the message changes."""
        if obs["tick"] == 0:
            text = self.plan_note
        elif throttle < -0.2:
            idx = self.follower.idx or 0
            ahead = [c for c in self.corners if 0 <= c["apex_index"] - idx < 60]
            text = f"Turn-in for {ahead[0]['id']}" if ahead else None
        else:
            text = None
        if text is None or text == self.last_say:
            return None
        self.last_say = text
        return text
