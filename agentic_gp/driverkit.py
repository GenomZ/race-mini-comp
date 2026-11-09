"""driverkit - optional helpers a driver file may import (pure functions on obs/track dicts + numpy).

Nothing here touches the simulator: every function works on the plain dicts your driver receives
(`obs` every tick, `track` in plan(), `report` in reflect()), so you can read, copy or replace any
of it. Driver-frame convention everywhere: "+ = right" for angles, offsets and steering.

Physics cheat sheet (defaults; the real numbers are in track["physics"]):
    corner speed      v_max = sqrt(a_lat * R)            a_lat = 20 u/s^2
    turn radius       R = v^2 / (a_lat * |steering|)     steering is a fraction of grip
    stopping distance d = (v^2 - v_end^2) / (2 * brake)  brake = 25 u/s^2

Usage:
    from agentic_gp import driverkit as dk

    class Driver:
        def plan(self, track):
            self.follower = dk.PathFollower.from_track(track)              # centerline + speed profile
            # or: dk.PathFollower.from_track(track, line=dk.racing_line(track, margin=3.0))
            self.follower.set_speed_limit(9, 11, 15.0)                      # a yellow flag from the briefing
            self.track = track
            print(dk.track_brief(track))                                    # corner table for prompts

        def drive(self, obs):
            return self.follower.control(obs)                               # (throttle, steering)

        def reflect(self, report):
            print(dk.summarize_report(report, self.track))                  # lap summary for an LLM

Contents: clamp, wrap_deg, speed_for_radius, braking_distance, stopping_speed, steer_towards,
curvature_radii, speed_profile (+ braking_pass), find_corners / track_corners, racing_line,
recover_if_stuck, wall_repulsion, PathFollower, track_brief, corner_stats, corner_at, summarize_report.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

# Same values as agentic_gp.sim.config.PhysicsConfig (copied: driver files never import the sim).
DEFAULT_PHYSICS = {
    "max_speed": 40.0,
    "accel": 12.0,
    "brake": 25.0,
    "drag": 0.15,
    "max_lateral_accel": 20.0,
    "max_yaw_rate": 1.5,
    "tick_seconds": 0.5,
}


def physics_of(track: dict | None) -> dict:
    """Physics constants from a track map (track["physics"]), with defaults for missing keys."""
    return {**DEFAULT_PHYSICS, **((track or {}).get("physics") or {})}


# ------------------------------------------------------------------ small maths
def clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    """Limit x to [lo, hi]; NaN becomes the value closest to 0 inside the range."""
    x = float(x)
    if x != x:                              # NaN
        x = 0.0
    return lo if x < lo else hi if x > hi else x


def wrap_deg(angle: float) -> float:
    """Wrap an angle in degrees to [-180, 180)."""
    return (float(angle) + 180.0) % 360.0 - 180.0


def speed_for_radius(radius: float, a_lat: float = 20.0, max_speed: float = 40.0) -> float:
    """Fastest speed the grip allows on a circle of `radius`: v = sqrt(a_lat * R), capped."""
    if radius is None or not math.isfinite(radius):
        return float(max_speed)
    return float(min(max_speed, math.sqrt(a_lat * max(radius, 0.0))))


def braking_distance(v_from: float, v_to: float = 0.0, brake: float = 25.0) -> float:
    """Distance needed to slow from v_from to v_to at full brake (0 if no braking is needed)."""
    return max(0.0, (v_from ** 2 - v_to ** 2) / (2.0 * brake))


def stopping_speed(distance: float, brake: float = 25.0, v_end: float = 0.0) -> float:
    """Fastest speed from which you can still slow to v_end within `distance`: sqrt(v_end^2 + 2*b*d)."""
    return math.sqrt(v_end ** 2 + 2.0 * brake * max(distance, 0.0))


def steer_towards(angle_deg: float, speed: float, distance: float | None = None, *, horizon: float = 1.0,
                  a_lat: float = 20.0, max_yaw_rate: float = 1.5, min_speed: float = 4.0) -> float:
    """Steering (-1..1, + = right) that turns the car towards a target `angle_deg` away (+ = right).

    Steering is a fraction of the available grip, so the same command turns much less at high speed
    (R = v^2 / (a_lat * |s|)). This converts the turn you WANT into the command you NEED:
      * with `distance`: pure pursuit - the arc through the target has curvature 2*sin(angle)/distance;
      * without: remove the heading error in `horizon` seconds.
    """
    if abs(angle_deg) >= 90.0:                       # target beside/behind us: full lock towards it
        return 1.0 if angle_deg > 0 else -1.0
    v = max(abs(speed), min_speed)
    if distance is not None and distance > 1e-6:
        yaw_needed = v * 2.0 * math.sin(math.radians(angle_deg)) / distance     # rad/s
    else:
        yaw_needed = math.radians(angle_deg) / horizon
    yaw_available = min(max_yaw_rate, a_lat / v)    # what full steering gives at this speed
    return clamp(yaw_needed / yaw_available)


# ------------------------------------------------------------------ geometry of a closed path
def _as_points(points: Any) -> np.ndarray:
    return np.asarray(points, dtype=float).reshape(-1, 2)


def curvature_radii(points: Any, span: int = 1) -> np.ndarray:
    """Turning radius at every point of a CLOSED path (circle through points i-span, i, i+span).

    Straight sections give very large radii (inf when exactly straight). span > 1 smooths noise.
    """
    p = _as_points(points)
    a, b, c = np.roll(p, span, axis=0), p, np.roll(p, -span, axis=0)
    ab, bc, ca = (np.linalg.norm(b - a, axis=1), np.linalg.norm(c - b, axis=1), np.linalg.norm(a - c, axis=1))
    cross = np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = ab * bc * ca / (2.0 * cross)
    return np.where(cross < 1e-9, np.inf, r)


def turn_directions(points: Any, span: int = 1) -> np.ndarray:
    """+1 where a closed path bends right (clockwise), -1 where it bends left, 0 when straight."""
    p = _as_points(points)
    a, b, c = np.roll(p, span, axis=0), p, np.roll(p, -span, axis=0)
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - b[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - b[:, 0])
    return -np.sign(np.where(np.abs(cross) < 1e-9, 0.0, cross))       # CCW (left) has cross > 0


def _spacing(points: np.ndarray) -> np.ndarray:
    """Distance from point i to point i+1 (closed)."""
    return np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)


def braking_pass(speeds: Any, points: Any, brake: float = 25.0) -> np.ndarray:
    """Backward pass: lower each speed so the car can brake in time for every slower point ahead.

    Walking backwards, v[i] may be at most sqrt(v[i+1]^2 + 2 * brake * ds). Twice round the loop
    because the path is closed (the last corner limits the end of the previous lap).
    """
    v = np.array(speeds, dtype=float)
    ds = _spacing(_as_points(points))
    n = len(v)
    for _ in range(2):
        for i in reversed(range(n)):
            j = (i + 1) % n
            v[i] = min(v[i], math.sqrt(v[j] ** 2 + 2.0 * brake * ds[i]))
    return v


def speed_profile(points: Any, a_lat: float = 20.0, brake: float = 25.0, max_speed: float = 40.0,
                  aggression: float = 1.0, min_speed: float = 5.0, caps: Any = None) -> np.ndarray:
    """Planned speed at every point of a closed path: grip limit per corner, then the braking pass.

    aggression scales the grip limit (1.0 = physics limit; > 1 is optimistic and crashes).
    caps (optional, one value per point) are extra limits, e.g. a yellow-flag zone.
    """
    radii = curvature_radii(points)
    v = np.minimum(max_speed, np.sqrt(a_lat * np.minimum(radii, 1e9))) * aggression
    v = np.clip(v, min_speed, max_speed)
    if caps is not None:
        v = np.minimum(v, np.asarray(caps, dtype=float))
    return braking_pass(v, points, brake)


# ------------------------------------------------------------------ track helpers
def waypoint_indices(track: dict, points: Any = None) -> list[int]:
    """Index of each waypoint (gate) on `points` (default: the track centerline)."""
    center = track.get("centerline")
    pts = _as_points(points if points is not None else center)
    same_line = center is not None and len(pts) == len(center)
    out = []
    for wp in track["waypoints"]:
        if same_line and "centerline_index" in wp:
            out.append(int(wp["centerline_index"]))
        else:
            out.append(int(np.argmin(np.linalg.norm(pts - np.asarray(wp["center"], dtype=float), axis=1))))
    return out


def segment_ids(n_points: int, gate_indices: list[int]) -> np.ndarray:
    """Waypoint segment of every path point: segment g runs from gate g to gate g+1 (like obs["segment"])."""
    order = np.argsort(gate_indices)
    starts = np.asarray(gate_indices)[order]
    pos = np.searchsorted(starts, np.arange(n_points), side="right") - 1     # -1 = before the first gate
    return order[pos % len(starts)].astype(int)


def find_corners(points: Any, a_lat: float = 20.0, max_speed: float = 40.0,
                 gate_indices: list[int] | None = None, min_points: int = 3) -> list[dict]:
    """Corners of a closed path: runs tighter than the flat-out radius, split where the direction flips.

    Returns dicts like track["corners"]: id, start_index, apex_index, end_index, min_radius,
    direction ("left"/"right"), v_max, from_wp, to_wp (from_wp/to_wp only with gate_indices).
    """
    p = _as_points(points)
    n = len(p)
    ds = float(np.mean(_spacing(p)))
    span = max(1, int(round(4.0 / max(ds, 1e-6))))          # look ~4 units each way: less noise
    radii = curvature_radii(p, span)
    turn = turn_directions(p, span)
    tight = radii < (max_speed ** 2 / a_lat)                # slower than flat-out here
    if tight.all() or not tight.any():
        start = 0
    else:
        start = int(np.argmin(tight))                      # begin the scan on a straight
    runs, current = [], []
    for k in range(n):
        i = (start + k) % n
        if tight[i] and (not current or turn[i] == turn[current[-1]]):
            current.append(i)
            continue
        if len(current) >= min_points:
            runs.append(current)
        current = [i] if tight[i] else []
    if len(current) >= min_points:
        runs.append(current)
    runs.sort(key=lambda r: r[0])                          # number corners from the start line
    seg = segment_ids(n, gate_indices) if gate_indices else None
    corners = []
    for k, run in enumerate(runs, start=1):
        apex = run[int(np.argmin(radii[run]))]
        r_min = float(radii[apex])
        corner = {"id": f"T{k}", "start_index": run[0], "apex_index": apex, "end_index": run[-1],
                  "min_radius": round(r_min, 1), "direction": "right" if turn[apex] > 0 else "left",
                  "v_max": round(speed_for_radius(r_min, a_lat, max_speed), 1)}
        if seg is not None:
            corner["from_wp"], corner["to_wp"] = int(seg[run[0]]), int(seg[run[-1]])
        corners.append(corner)
    return corners


def track_corners(track: dict) -> list[dict]:
    """track["corners"] if the runner provided them, else detected from the centerline."""
    if track.get("corners"):
        return list(track["corners"])
    phys = physics_of(track)
    return find_corners(track["centerline"], phys["max_lateral_accel"], phys["max_speed"],
                        waypoint_indices(track) if track.get("waypoints") else None)


def _curvature_hessian(normal: np.ndarray, ds: float, center_weight: float) -> Any:
    """H for  sum(curvature^2) = offset.H.offset - 2 g.offset + const  (banded: only neighbours i-2..i+2).

    The squared second difference has the stencil [1, -4, 6, -4, 1] / ds^4; sliding point j along
    its normal n_j moves the curvature at i by (n_i . n_j) times that. Sparse when scipy is there.
    """
    n = len(normal)
    rows, cols, vals = [], [], []
    for shift, weight in ((-2, 1.0), (-1, -4.0), (0, 6.0), (1, -4.0), (2, 1.0)):
        i = np.arange(n)
        j = (i + shift) % n
        rows.append(i)
        cols.append(j)
        vals.append(weight / ds ** 4 * np.sum(normal[i] * normal[j], axis=1) + (center_weight if shift == 0 else 0.0))
    rows, cols, vals = np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
    try:
        from scipy.sparse import csc_matrix               # fast + single-threaded; optional
        return csc_matrix((vals, (rows, cols)), shape=(n, n))
    except ImportError:
        H = np.zeros((n, n))
        np.add.at(H, (rows, cols), vals)
        return H


def _solve(H: Any, rhs: np.ndarray) -> np.ndarray:
    if isinstance(H, np.ndarray):
        return np.linalg.solve(H, rhs)
    from scipy.sparse.linalg import spsolve

    return np.atleast_1d(spsolve(H, rhs))


def racing_line(track: dict, margin: float = 3.0, *, rounds: int = 60, center_weight: float = 1e-6) -> np.ndarray:
    """A smooth, low-curvature line inside the walls: outside-apex-outside through the corners.

    Each centerline point may slide sideways (along its normal) by at most half_width - margin.
    We pick the offsets that minimise the squared curvature of the whole loop (least squares with
    box limits): solve, pin points that would leave the corridor to the wall, solve again, and let
    pinned points go when the solution pulls them back inside ("active set").
    Returns an (n, 2) array with the SAME number of points as track["centerline"] (point i is the
    sideways shift of centerline[i], so waypoint indices still match).
    """
    c = _as_points(track["centerline"])
    n = len(c)
    bound = max(float(track.get("half_width", 9.0)) - margin, 0.0)
    tangent = np.roll(c, -1, axis=0) - np.roll(c, 1, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])           # left normal
    ds = float(np.mean(_spacing(c)))
    H = _curvature_hessian(normal, ds, center_weight)
    # Linear term: the centerline's own curvature, seen along each normal.
    dd = (np.roll(c, 2, axis=0) - 4 * np.roll(c, 1, axis=0) + 6 * c - 4 * np.roll(c, -1, axis=0)
          + np.roll(c, -2, axis=0)) / ds ** 4
    g = -np.sum(normal * dd, axis=1)

    offset, fixed = np.zeros(n), np.zeros(n, dtype=bool)
    for _ in range(rounds):
        free = np.flatnonzero(~fixed)
        rhs = g - H @ np.where(fixed, offset, 0.0)                     # fixed points act as known values
        sub = H[free][:, free] if not isinstance(H, np.ndarray) else H[np.ix_(free, free)]
        offset[free] = _solve(sub, rhs[free])
        outside = ~fixed & (np.abs(offset) > bound)
        if outside.any():                                               # pin to the wall and re-solve
            offset[outside] = np.clip(offset[outside], -bound, bound)
            fixed |= outside
            continue
        pull_in = fixed & (np.sign(offset) * (H @ offset - g) > 1e-12)  # the wall is not needed here
        if not pull_in.any():
            break
        fixed &= ~pull_in
    offset = np.clip(offset, -bound, bound)
    return c + offset[:, None] * normal


# ------------------------------------------------------------------ control
def recover_if_stuck(obs: dict, state: dict | None = None, *, wall: float = 4.0, min_speed: float = 1.5,
                     ticks: int = 2) -> tuple[float, float] | None:
    """If the car is (nearly) stopped with a wall just ahead, reverse towards the open side.

    Returns (throttle, steering) while recovering, else None. Pass the same `state` dict every
    tick to keep reversing for `ticks` ticks. Why steer to the open side while reversing? In this
    sim steering turns the nose the same way whatever the direction of travel.
    """
    state = {} if state is None else state
    if state.get("ticks", 0) > 0:
        state["ticks"] -= 1
        return state["action"]
    front = obs["distance_to_wall_front"]
    l20, r20 = obs["distance_to_wall_left_20"], obs["distance_to_wall_right_20"]
    l45, r45 = obs["distance_to_wall_left_45"], obs["distance_to_wall_right_45"]
    if abs(obs["speed"]) < min_speed and min(front, l20, r20) < wall:
        action = (-1.0, -1.0 if l45 > r45 else 1.0)
        state["ticks"], state["action"] = ticks - 1, action
        return action
    return None


def wall_repulsion(obs: dict, gain: float = 1.0) -> float:
    """Steering nudge away from close side walls (+ = right, i.e. away from a close LEFT wall)."""
    push_right = 0.5 * max(0.0, 1.0 - obs["distance_to_wall_left_20"] / 7.0) \
        + 0.3 * max(0.0, 1.0 - obs["distance_to_wall_left_45"] / 5.0)
    push_left = 0.5 * max(0.0, 1.0 - obs["distance_to_wall_right_20"] / 7.0) \
        + 0.3 * max(0.0, 1.0 - obs["distance_to_wall_right_45"] / 5.0)
    return gain * (push_right - push_left)


class PathFollower:
    """Follow a planned path: pure pursuit + wall repulsion, planned speed + stopping-distance safety.

        follower = PathFollower(points, speeds)       # speeds default: speed_profile(points)
        throttle, steering = follower.control(obs)
        follower.segment_speed_caps[9] = 15.0         # e.g. a yellow flag in segment 9 (brakes in time)

    `follower.debug` holds the last target point / speeds (handy as drive()'s "debug" dict).
    """

    def __init__(self, points: Any, speeds: Any = None, *, lookahead: float | None = None,
                 physics: dict | None = None, segments: Any = None, aggression: float = 1.0,
                 wall_gain: float = 1.0, safety_margin: float = 3.0, blind_factor: float = 1.0,
                 recover: bool = True, respect_zone_limit: bool = True) -> None:
        self.points = _as_points(points)
        self.physics = {**DEFAULT_PHYSICS, **(physics or {})}
        p = self.physics
        if speeds is None:
            speeds = speed_profile(self.points, p["max_lateral_accel"], p["brake"], p["max_speed"], aggression)
        self.base_speeds = np.asarray(speeds, dtype=float)
        self.ds = float(np.mean(_spacing(self.points)))
        self.segments = None if segments is None else np.asarray(segments, dtype=int)  # segment id per point
        self.segment_speed_caps: dict[int, float] = {}
        self.lookahead = lookahead              # None = speed-dependent (6 + 0.45 v, within 8..22)
        self.wall_gain = wall_gain
        self.safety_margin = safety_margin      # units kept free in front of the car
        self.blind_factor = blind_factor        # x distance covered in one tick (we only look every 0.5 s)
        self.recover = recover
        self.respect_zone_limit = respect_zone_limit
        self.idx: int | None = None
        self.recovery_state: dict = {}
        self.debug: dict = {}
        self._caps_seen: tuple | None = None
        self.speeds = self.base_speeds.copy()

    @classmethod
    def from_track(cls, track: dict, *, line: Any = None, speeds: Any = None, aggression: float = 1.0,
                   **kwargs: Any) -> "PathFollower":
        """Follower on the track centerline (or on `line`, e.g. racing_line(track)) with track physics."""
        points = _as_points(line if line is not None else track["centerline"])
        segments = segment_ids(len(points), waypoint_indices(track, points)) if track.get("waypoints") else None
        return cls(points, speeds, physics=physics_of(track), segments=segments, aggression=aggression, **kwargs)

    def set_speed_limit(self, from_wp: int, to_wp: int, max_speed: float, margin: float = 0.5) -> None:
        """Respect a speed-limit zone "between waypoints from_wp and to_wp" (e.g. read from the briefing).

        The car is inside after crossing from_wp and before crossing to_wp, i.e. on segments
        from_wp .. to_wp-1 (wrapping past waypoint 0). The braking pass slows down BEFORE the zone.
        """
        n = int(self.segments.max()) + 1 if self.segments is not None else max(from_wp, to_wp) + 1
        for k in range((to_wp - from_wp) % n or n):
            seg = (from_wp + k) % n
            self.segment_speed_caps[seg] = min(self.segment_speed_caps.get(seg, math.inf), max_speed - margin)

    def set_speeds(self, speeds: Any) -> None:
        """Replace the planned speeds (e.g. a new plan from reflect()). Segment caps still apply."""
        self.base_speeds = np.asarray(speeds, dtype=float)
        self._caps_seen = None

    # ---------------------------------------------------------- where am I?
    def localize(self, position: Any) -> int:
        """Index of the path point nearest to `position` (searches near the last index first)."""
        pos = np.asarray(position, dtype=float)
        n = len(self.points)
        if self.idx is not None:
            ahead = int(45.0 / max(self.ds, 1e-6)) + 5          # ~2 ticks at top speed
            window = np.arange(self.idx - 5, self.idx + ahead) % n
            d = np.linalg.norm(self.points[window] - pos, axis=1)
            if d.min() < 30.0:
                self.idx = int(window[int(np.argmin(d))])
                return self.idx
        self.idx = int(np.argmin(np.linalg.norm(self.points - pos, axis=1)))     # lost: full search
        return self.idx

    def lookahead_distance(self, speed: float) -> float:
        if self.lookahead is not None:
            return self.lookahead
        return clamp(6.0 + 0.45 * abs(speed), 8.0, 22.0)

    # ---------------------------------------------------------- steering
    def steering(self, obs: dict) -> float:
        """Pure pursuit towards a point `lookahead` ahead on the path, plus wall repulsion."""
        p = self.physics
        speed = obs["speed"]
        pos = np.asarray(obs["position"], dtype=float)
        i = self.localize(pos)
        n = len(self.points)
        target = self.points[(i + int(round(self.lookahead_distance(speed) / self.ds))) % n]
        vec = target - pos
        bearing = math.degrees(math.atan2(vec[1], vec[0]))          # world frame, CCW +
        angle = -wrap_deg(bearing - obs["heading_deg"])              # driver frame, + = right
        steer = steer_towards(angle, speed, float(np.linalg.norm(vec)),
                              a_lat=p["max_lateral_accel"], max_yaw_rate=p["max_yaw_rate"])
        self.debug.update(target=[round(float(target[0]), 2), round(float(target[1]), 2)],
                          path_index=i, target_angle=round(angle, 1))
        return clamp(steer + wall_repulsion(obs, self.wall_gain))

    # ---------------------------------------------------------- speed
    def _refresh_caps(self) -> None:
        """Bake segment caps into the profile (with a braking pass, so we slow down BEFORE the segment)."""
        key = tuple(sorted(self.segment_speed_caps.items()))
        if key == self._caps_seen:
            return
        self._caps_seen = key
        v = self.base_speeds.copy()
        if self.segments is not None and key:
            caps = np.array([self.segment_speed_caps.get(int(s), np.inf) for s in self.segments])
            v = braking_pass(np.minimum(v, caps), self.points, self.physics["brake"])
        self.speeds = v

    def target_speed(self, obs: dict) -> float:
        """Slowest planned speed in the next tick of travel, limited by walls ahead/beside and zones."""
        p = self.physics
        self._refresh_caps()
        speed = abs(obs["speed"])
        i = self.localize(obs["position"])
        n = len(self.points)
        ahead = int(round(max(speed, 5.0) * p["tick_seconds"] / self.ds))
        v_plan = float(np.min(self.speeds[(i + np.arange(0, ahead + 2)) % n]))
        if self.segments is None and obs.get("segment") in self.segment_speed_caps:
            v_plan = min(v_plan, self.segment_speed_caps[obs["segment"]])      # no map: cap where we are
        # Safety net: we must be able to stop before the wall in front. We only look every tick, so
        # also keep the distance we cover before the next look ("blind" distance) free.
        margin = self.safety_margin + self.blind_factor * speed * p["tick_seconds"]
        v_front = stopping_speed(obs["distance_to_wall_front"] - margin, p["brake"])
        v_side = p["max_speed"]
        if min(obs["distance_to_wall_left_45"], obs["distance_to_wall_right_45"]) < 5.0:
            v_side = 14.0
        if min(obs["distance_to_wall_left_20"], obs["distance_to_wall_right_20"]) < 5.0:
            v_side = 10.0
        target = min(v_plan, v_front, v_side)
        zone = obs.get("zone_speed_limit")
        if self.respect_zone_limit and zone is not None:
            target = min(target, float(zone))
        self.debug.update(v_plan=round(v_plan, 1), v_target=round(target, 1))
        floor = 5.0 if zone is None else min(5.0, float(zone))
        return clamp(target, floor, p["max_speed"])

    def throttle_for(self, target: float, speed: float) -> float:
        """Throttle that reaches `target` speed in about one tick (brake harder than we accelerate)."""
        p = self.physics
        dv = target - speed
        if dv >= 0:
            return clamp((dv + p["drag"] * speed * p["tick_seconds"]) / (p["accel"] * p["tick_seconds"]))
        return clamp(dv / (p["brake"] * p["tick_seconds"]))

    def control(self, obs: dict) -> tuple[float, float]:
        """(throttle, steering) for this tick."""
        if self.recover:
            action = recover_if_stuck(obs, self.recovery_state)
            if action is not None:
                self.debug["recovering"] = True
                return action
        self.debug.pop("recovering", None)
        steering = self.steering(obs)
        throttle = self.throttle_for(self.target_speed(obs), obs["speed"])
        return throttle, steering


# ------------------------------------------------------------------ text for LLM prompts
def _fmt(x: Any, digits: int = 1) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def _table(rows: list[list[Any]], header: list[str]) -> list[str]:
    cells = [header] + [[_fmt(c) for c in r] for r in rows]
    widths = [max(len(r[k]) for r in cells) for k in range(len(header))]
    return ["  ".join(c.rjust(w) for c, w in zip(r, widths)) for r in cells]


def track_brief(track: dict) -> str:
    """Compact text description of a track for an LLM prompt: physics, corner table, straights, briefing."""
    phys = physics_of(track)
    corners = track_corners(track)
    center = _as_points(track["centerline"])
    ds = _spacing(center)
    length = track.get("lap_length") or float(ds.sum())
    lines = [
        f"Track: {track.get('name', '?')} ({track.get('key', '?')}): lap {length:.0f} u, "
        f"half-width {track.get('half_width', '?')}, {len(track.get('waypoints', []))} waypoints, "
        f"{track.get('laps', 1)} lap(s).",
        f"Physics: top speed {phys['max_speed']:g}, accel {phys['accel']:g}, brake {phys['brake']:g} u/s^2, "
        f"grip {phys['max_lateral_accel']:g} u/s^2 -> corner speed v_max = sqrt({phys['max_lateral_accel']:g} * R). "
        f"One tick = {phys['tick_seconds']:g} s. Sign convention: + = right.",
    ]
    if corners:
        rows = [[c["id"], f"{c.get('from_wp', '?')}->{c.get('to_wp', '?')}", c.get("direction", "?"),
                 float(c["min_radius"]), float(c["v_max"])] for c in corners]
        lines.append(f"Corners ({len(corners)}):")
        lines += ["  " + r for r in _table(rows, ["id", "wp", "dir", "min_R", "v_max"])]
        # Straights = gaps between consecutive corners, measured along the centerline.
        cum = np.concatenate([[0.0], np.cumsum(ds)])
        gaps = []
        for k, c in enumerate(corners):
            nxt = corners[(k + 1) % len(corners)]
            a, b = c["end_index"], nxt["start_index"]
            gap = cum[b] - cum[a] if b > a else cum[-1] - cum[a] + cum[b]
            gaps.append((gap, c["id"], nxt["id"]))
        gap, a, b = max(gaps)
        lines.append(f"Longest straight: {gap:.0f} u from {a} to {b} "
                     f"(braking from {phys['max_speed']:g} to 15 needs {braking_distance(phys['max_speed'], 15, phys['brake']):.0f} u).")
    else:
        lines.append("Corners: none tighter than flat-out.")
    if track.get("briefing"):
        lines.append(f"Briefing: {track['briefing']}")
    return "\n".join(lines)


def _corner_names(track: dict | None) -> dict[int, str]:
    """{segment: "T3/T4"} for the corners that touch each waypoint segment."""
    if not track:
        return {}
    corners = track_corners(track)
    n = len(track.get("waypoints", [])) or 1
    names: dict[int, list[str]] = {}
    for c in corners:
        if "from_wp" not in c:
            continue
        s = c["from_wp"]
        for _ in range(n):
            names.setdefault(s, []).append(c["id"])
            if s == c["to_wp"]:
                break
            s = (s + 1) % n
    return {s: "/".join(ids) for s, ids in names.items()}


def nearest_index(points: Any, xy: Any) -> np.ndarray:
    """Index of the nearest path point for each (x, y) in `xy` (array of shape (k, 2) -> (k,))."""
    p, q = _as_points(points), _as_points(xy)
    d2 = ((q[:, None, :] - p[None, :, :]) ** 2).sum(axis=2)
    return np.argmin(d2, axis=1)


def _corner_mask(n: int, corner: dict) -> np.ndarray:
    """Boolean mask over centerline indices covered by a corner (wraps past index 0)."""
    a, b = int(corner["start_index"]), int(corner["end_index"])
    idx = np.arange(n)
    return (idx >= a) & (idx <= b) if a <= b else (idx >= a) | (idx <= b)


def corner_at(track: dict, position: Any, lookahead: float = 40.0) -> str | None:
    """Corner id at a position ("T3"), "before T3" if a corner starts within `lookahead` units, else None."""
    center = _as_points(track["centerline"])
    corners = track_corners(track)
    i = int(nearest_index(center, [position])[0])
    for c in corners:
        if _corner_mask(len(center), c)[i]:
            return c["id"]
    ds = float(np.mean(_spacing(center)))
    for k in range(1, int(lookahead / max(ds, 1e-6)) + 1):
        j = (i + k) % len(center)
        for c in corners:
            if int(c["start_index"]) == j:
                return f"before {c['id']}"
    return None


def corner_stats(report: dict, track: dict) -> list[dict]:
    """Per-corner numbers from a report's telemetry: how fast you entered, the slowest point, the exit.

    One row per corner: id, direction, v_max (grip limit at the apex), passes, time (s per pass),
    entry/min/exit speed (averaged over passes; min is the overall minimum) and crashes there.
    """
    telemetry = report.get("telemetry") or []
    corners = track_corners(track)
    if not telemetry or not corners:
        return []
    center = _as_points(track["centerline"])
    dt = physics_of(track)["tick_seconds"]
    idx = nearest_index(center, [[f["x"], f["y"]] for f in telemetry])
    speed = np.array([f["speed"] for f in telemetry], dtype=float)
    crash_ids = [corner_at(track, c["position"]) for c in report.get("crash_log") or [] if c.get("position")]
    rows = []
    for c in corners:
        inside = _corner_mask(len(center), c)[idx]
        passes = []                                   # runs of consecutive ticks inside the corner
        for k in np.flatnonzero(inside):
            if passes and k == passes[-1][-1] + 1:
                passes[-1].append(int(k))
            else:
                passes.append([int(k)])
        if not passes:
            continue
        rows.append({
            "id": c["id"], "direction": c.get("direction"), "v_max": float(c["v_max"]), "passes": len(passes),
            "time": round(float(np.mean([len(p) * dt for p in passes])), 2),
            "entry_speed": round(float(np.mean([speed[p[0]] for p in passes])), 1),
            "min_speed": round(float(min(speed[p].min() for p in passes)), 1),
            "exit_speed": round(float(np.mean([speed[p[-1]] for p in passes])), 1),
            "crashes": sum(1 for cid in crash_ids if cid in (c["id"], f"before {c['id']}")),
        })
    return rows


def _rays_text(entry: dict) -> str:
    rays = entry.get("rays") or {k: v for k, v in entry.items() if k.startswith("distance_to_wall_")}
    short = []
    for k, v in rays.items():
        name = k.replace("distance_to_wall_", "").replace("left_", "L").replace("right_", "R").replace("front", "F")
        short.append(f"{name} {_fmt(v)}")
    return " rays " + " ".join(short) if short else ""


def summarize_report(report: dict, track: dict | None = None, max_crashes: int = 8) -> str:
    """Compact, LLM-friendly text for a reflect() report: result, per-segment table, crashes, zones.

    Pass `track` (the plan() map) to label segments and crashes with corner ids (T1, T2, ...).
    """
    corner_of = _corner_names(track)
    corners = report.get("corners") or (corner_stats(report, track) if track else [])
    done = report.get("completed")
    head = (f"Attempt {report.get('attempt', '?')}"
            + (f" ({report['phase']})" if report.get("phase") else "")
            + (f", next phase: {report['next_phase']}" if report.get("next_phase") else ""))
    lines = [head + "."]
    if done:
        lines.append(f"Result: FINISHED {report.get('laps_completed', '?')}/{report.get('laps', '?')} lap(s), "
                     f"race time {_fmt(report.get('race_time'), 2)} s incl. {_fmt(report.get('penalty_time'))} s "
                     f"penalties, {report.get('crashes', 0)} crash(es).")
    else:
        lines.append(f"Result: DNF ({report.get('ended_by', '?')}) after {report.get('waypoints_passed', '?')} "
                     f"waypoints, {report.get('crashes', 0)} crash(es).")
    if report.get("lap_times"):
        lines.append("Lap times: " + ", ".join(f"{t:.2f}" for t in report["lap_times"])
                     + f" | best lap {_fmt(report.get('best_lap'), 2)}"
                     + (f" | session best {_fmt(report.get('best_lap_so_far'), 2)}"
                        if report.get("best_lap_so_far") is not None else ""))
    if report.get("segments"):
        rows = [[s.get("segment"), corner_of.get(s.get("segment"), ""), s.get("time"), s.get("entry_speed"),
                 s.get("min_speed"), s.get("max_speed"), s.get("crashes", 0)] for s in report["segments"]]
        lines.append("Segments (time s, speeds u/s):")
        lines += ["  " + r for r in _table(rows, ["seg", "corner", "time", "entry", "min", "max", "crashes"])]
    if corners:                          # per-corner table: compare min speed with the grip limit v_max
        keys = [k for k in ("id", "direction", "v_max", "time", "entry_speed", "min_speed", "exit_speed", "crashes")
                if any(k in c for c in corners)]
        lines.append("Corners (time s per pass, speeds u/s; v_max = grip limit at the apex):")
        lines += ["  " + r for r in _table([[c.get(k) for k in keys] for c in corners],
                                            [k.replace("_speed", "").replace("direction", "dir") for k in keys])]
    crashes = report.get("crash_log") or []
    if crashes:
        lines.append(f"Crashes ({len(crashes)}):")
        for c in crashes[:max_crashes]:
            seg = c.get("segment")
            where = c.get("corner") or (corner_at(track, c["position"]) if track and c.get("position") else None)                 or corner_of.get(seg, "")
            lines.append(f"  tick {c.get('tick')} t={_fmt(c.get('t'))}s lap {c.get('lap', '?')} seg {seg}"
                         + (f" ({where})" if where else "")
                         + f": speed {_fmt(c.get('speed_before'))} throttle {_fmt(c.get('throttle'), 2)} "
                         f"steering {_fmt(c.get('steering'), 2)}" + _rays_text(c))
        if len(crashes) > max_crashes:
            lines.append(f"  ... and {len(crashes) - max_crashes} more")
    zones = report.get("zone_violations")
    if zones:
        if isinstance(zones, list):
            lines.append(f"Zone violations ({len(zones)}):")
            for z in zones[:max_crashes]:
                lines.append("  " + ", ".join(f"{k} {_fmt(v)}" for k, v in z.items()))
        else:
            lines.append(f"Zone violations: {zones}")
    extras = []
    for key in ("stalls", "driver_errors"):
        val = report.get(key)
        if val:
            extras.append(f"{key} {len(val) if isinstance(val, list) else val}")
    usage = report.get("llm_usage") or {}
    if usage.get("calls"):
        extras.append(f"llm calls {usage['calls']}, tokens "
                      f"{usage.get('prompt_tokens', 0) + usage.get('completion_tokens', 0)}")
    if extras:
        lines.append("Other: " + "; ".join(extras) + ".")
    return "\n".join(lines)
