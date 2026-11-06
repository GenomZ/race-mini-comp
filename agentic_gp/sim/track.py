"""Track geometry: centerline, walls, waypoint gates, corners and speed-limit zones.

    from agentic_gp.sim.track import make_track, TRACKS
    track = make_track("grand_prix")          # or "oval", "sprint", "speedway", "briefing_demo",
                                              #    "random:42", "file:lesson_private/mystery.py"
    s, offset, i = track.project((120.0, 3.0))  # arc length, signed offset (+ = right), nearest index
    track.to_map()                            # the dict a driver's plan(track) receives

A track is defined by a handful of control points (a closed Catmull-Rom spline is
fitted through them) or by "turtle" commands (straights + arcs with exact radii).
The centerline is resampled to equal arc-length spacing (~4.3 units) and offset
left/right by `half_width` to produce the two walls. Waypoint "gates" are segments
spanning the track from wall to wall at evenly spaced centerline indices. Gate 0 is
the start/finish line; the car starts one centerline sample before it.
"""

from __future__ import annotations

import importlib.util
import math
import zlib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from . import geometry as geo
from .config import PhysicsConfig, DEFAULT_CONFIG

SAMPLE_SPACING = 4.3        # target distance between centerline samples (units)


@dataclass(frozen=True)
class Zone:
    """Speed-limit zone (e.g. a yellow flag) from gate `from_wp` to gate `to_wp`.

    The car is "inside" after crossing waypoint `from_wp` and before crossing `to_wp`,
    i.e. on segments from_wp .. to_wp - 1 (wrapping around gate 0 if needed).
    """

    from_wp: int
    to_wp: int
    max_speed: float
    label: str = "Yellow flag"

    def segments(self, n_gates: int) -> list[int]:
        n = (self.to_wp - self.from_wp) % n_gates
        return [(self.from_wp + k) % n_gates for k in range(n)]

    def describe(self) -> str:
        return f"{self.label} between waypoints {self.from_wp} and {self.to_wp}: max speed {self.max_speed:g}."

    def to_dict(self) -> dict:
        return {"from_wp": self.from_wp, "to_wp": self.to_wp, "max_speed": self.max_speed, "label": self.label}


@dataclass
class Track:
    name: str
    centerline: np.ndarray          # (N, 2)
    half_width: float
    n_gates: int
    key: str = "custom"
    zones: list[Zone] = field(default_factory=list)
    briefing_text: str | None = None  # optional extra race-control text appended to the briefing
    left_wall: np.ndarray = field(init=False)      # (N, 2)
    right_wall: np.ndarray = field(init=False)     # (N, 2)
    wall_segments: np.ndarray = field(init=False)  # (2N, 4) x1,y1,x2,y2 (left wall first)
    gate_indices: np.ndarray = field(init=False)   # (n_gates,) centerline index of each gate
    gates: np.ndarray = field(init=False)          # (n_gates, 4) x1,y1,x2,y2 (left end, right end)
    gate_centers: np.ndarray = field(init=False)   # (n_gates, 2)
    tangents: np.ndarray = field(init=False)       # (N, 2) unit tangents
    arc: np.ndarray = field(init=False)            # (N+1,) arc length at each sample, arc[N] = length
    radii: np.ndarray = field(init=False)          # (N,) local turning radius (inf on straights)
    curvature: np.ndarray = field(init=False)      # (N,) signed 1/R, + = turning left

    def __post_init__(self) -> None:
        C = np.asarray(self.centerline, dtype=float)
        self.centerline = C
        self.zones = [_as_zone(z) for z in self.zones]
        T = geo.tangents_closed(C)
        self.tangents = T
        left_normal = np.column_stack([-T[:, 1], T[:, 0]])
        self.left_wall = C + left_normal * self.half_width
        self.right_wall = C - left_normal * self.half_width
        self.wall_segments = np.vstack(
            [_closed_segments(self.left_wall), _closed_segments(self.right_wall)]
        )
        # bounding boxes of the wall segments, used to cull collision and ray tests
        W = self.wall_segments
        self._wall_lox, self._wall_hix = np.minimum(W[:, 0], W[:, 2]), np.maximum(W[:, 0], W[:, 2])
        self._wall_loy, self._wall_hiy = np.minimum(W[:, 1], W[:, 3]), np.maximum(W[:, 1], W[:, 3])
        n = len(C)
        self.gate_indices = (np.arange(self.n_gates) * n // self.n_gates).astype(int)
        self.gates = np.column_stack(
            [self.left_wall[self.gate_indices], self.right_wall[self.gate_indices]]
        )
        self.gate_centers = C[self.gate_indices]
        # projection helpers: polyline pieces C[i] -> C[i+1], kept as flat 1-D arrays (fast per tick)
        self._piece = np.roll(C, -1, axis=0) - C
        self._piece_len = np.linalg.norm(self._piece, axis=1)
        self._cx, self._cy = C[:, 0].copy(), C[:, 1].copy()
        self._px, self._py = self._piece[:, 0].copy(), self._piece[:, 1].copy()
        self._inv_len2 = 1.0 / np.maximum(self._piece_len ** 2, geo.EPS)
        self.arc = np.concatenate([[0.0], np.cumsum(self._piece_len)])
        self._gate_arcs = [float(self.arc[i]) for i in self.gate_indices]   # plain floats: cheap per tick
        self.radii = geo.curvature_radius_closed(C)
        self.curvature = geo.signed_curvature_closed(C)

    # ------------------------------------------------------------------ info
    @property
    def n_points(self) -> int:
        return len(self.centerline)

    @property
    def length(self) -> float:
        return float(self.arc[-1])

    @property
    def start_position(self) -> np.ndarray:
        """One centerline sample before the start/finish gate."""
        return self.centerline[-1].copy()

    @property
    def start_heading(self) -> float:
        t = self.tangents[-1]
        return math.atan2(t[1], t[0])

    def min_turn_radius(self) -> float:
        return float(np.min(self.radii))

    def heading_at(self, index: int) -> float:
        """Track direction (radians, CCW from +x) at a centerline index."""
        t = self.tangents[int(index) % self.n_points]
        return math.atan2(t[1], t[0])

    def gate_heading(self, gate: int) -> float:
        return self.heading_at(self.gate_indices[gate])

    def gate_arc(self, gate: int) -> float:
        """Arc length from gate 0 to `gate` along the centerline."""
        return self._gate_arcs[gate % self.n_gates]

    def bounds(self) -> tuple[float, float, float, float]:
        allpts = np.vstack([self.left_wall, self.right_wall])
        return (
            float(allpts[:, 0].min()),
            float(allpts[:, 1].min()),
            float(allpts[:, 0].max()),
            float(allpts[:, 1].max()),
        )

    def segment_indices(self, gate: int) -> np.ndarray:
        """Centerline indices between `gate` (inclusive) and the next gate (exclusive)."""
        a = self.gate_indices[gate % self.n_gates]
        b = self.gate_indices[(gate + 1) % self.n_gates]
        if b > a:
            return np.arange(a, b)
        return np.concatenate([np.arange(a, self.n_points), np.arange(0, b)])

    def segment_min_radius(self, gate: int) -> float:
        """Tightest turning radius on the centerline between `gate` and the next gate."""
        return float(np.min(self.radii[self.segment_indices(gate)]))

    def segment_of_index(self, index: int) -> int:
        """Waypoint segment (gate before) that contains centerline `index`."""
        return int(np.searchsorted(self.gate_indices, int(index) % self.n_points, side="right") - 1)

    # ------------------------------------------------------------ projection
    def project(self, point) -> tuple[float, float, int]:
        """Project a point onto the centerline.

        Returns (s, offset, index): arc length `s` from gate 0 (0 <= s < length), signed
        lateral `offset` (NEGATIVE = left of the centerline, POSITIVE = right, the same
        "+ = right" convention as steering) and the nearest centerline index.
        One vectorised pass over all centerline pieces (~20 us for 360 points).
        """
        px, py = float(point[0]), float(point[1])
        rx, ry = px - self._cx, py - self._cy                         # (N,) sample -> point
        t = (rx * self._px + ry * self._py) * self._inv_len2         # position along each piece
        np.maximum(t, 0.0, out=t)
        np.minimum(t, 1.0, out=t)
        fx, fy = rx - t * self._px, ry - t * self._py                 # foot of the perpendicular -> point
        i = int(np.argmin(fx * fx + fy * fy))
        L = float(self._piece_len[i])
        ti = float(t[i])
        s = (float(self.arc[i]) + ti * L) % self.length
        cross = (self._px[i] * fy[i] - self._py[i] * fx[i]) / max(L, geo.EPS)   # + = left of travel
        index = i if ti < 0.5 else (i + 1) % self.n_points
        return s, float(-cross), int(index)

    def respawn_pose(self, point, segment: int) -> tuple[np.ndarray, float]:
        """Marshal rule: nearest centerline sample INSIDE waypoint segment `segment` and the
        track heading there. Staying inside the segment means no gate is skipped or repeated."""
        idx = self.segment_indices(segment)
        d = np.linalg.norm(self.centerline[idx] - np.asarray(point, dtype=float), axis=1)
        i = int(idx[int(np.argmin(d))])
        return self.centerline[i].copy(), self.heading_at(i)

    def raycast(self, origin, angles, max_dist: float) -> np.ndarray:
        """Distance to the walls along each ray angle (radians, CCW), capped at `max_dist`.

        Only walls whose bounding box comes within `max_dist` of the car can be hit, so
        the others are culled before the vectorised ray test.
        """
        o = np.asarray(origin, dtype=float)
        return geo.raycast_many(o, angles, self.walls_near(o - max_dist, o + max_dist, 0.0), max_dist)

    def walls_near(self, lo: np.ndarray, hi: np.ndarray, margin: float = 1.0) -> np.ndarray:
        """Wall segments whose bounding box overlaps the box [lo, hi] (+ margin)."""
        m = ((self._wall_hix >= lo[0] - margin) & (self._wall_lox <= hi[0] + margin)
             & (self._wall_hiy >= lo[1] - margin) & (self._wall_loy <= hi[1] + margin))
        return self.wall_segments[m]

    # ------------------------------------------------------------------ zones
    def zone_for_segment(self, segment: int) -> Zone | None:
        """The (strictest) speed-limit zone covering a waypoint segment, or None."""
        found = [z for z in self.zones if segment % self.n_gates in z.segments(self.n_gates)]
        return min(found, key=lambda z: z.max_speed) if found else None

    def zone_ranges(self) -> list[dict]:
        """Zones as centerline index ranges (for replays / viewers)."""
        return [
            {
                "from_index": int(self.gate_indices[z.from_wp % self.n_gates]),
                "to_index": int(self.gate_indices[z.to_wp % self.n_gates]),
                "max_speed": float(z.max_speed),
                "label": z.label,
            }
            for z in self.zones
        ]

    def briefing(self, cfg: PhysicsConfig = DEFAULT_CONFIG) -> str:
        """Natural-language race-control briefing, generated from the zones (+ custom text)."""
        lines = [f"Race control briefing - {self.name}."]
        if self.zones:
            lines += [z.describe() for z in self.zones]
            lines.append(
                f"Every tick that ends inside a zone faster than its limit (+{cfg.zone_speed_tolerance:g} tolerance) "
                f"costs +{cfg.zone_penalty_seconds:g}s."
            )
        else:
            lines.append("No speed restrictions today.")
        lines.append(f"Wall contact costs +{cfg.crash_penalty_seconds:g}s.")
        if self.briefing_text:
            lines.append(self.briefing_text.strip())
        return " ".join(lines)

    # ---------------------------------------------------------------- corners
    def corners(self, cfg: PhysicsConfig = DEFAULT_CONFIG) -> list[dict]:
        """Auto-detected corners: runs of centerline samples tighter than the flat-out radius.

        A corner is a run where v_max = sqrt(a_lat * R) < max_speed, split where the turn
        direction flips (chicanes give two corners). Numbered T1, T2, ... from the start line.
        """
        return detect_corners(self, cfg)

    # ------------------------------------------------------------------ map
    def to_map(self, cfg: PhysicsConfig = DEFAULT_CONFIG) -> dict:
        """JSON-serialisable description that drivers receive in plan(track)."""
        waypoints = []
        for g in range(self.n_gates):
            r = self.segment_min_radius(g)
            waypoints.append(
                {
                    "index": g,
                    "center": _r2(self.gate_centers[g]),
                    "track_heading_deg": round(math.degrees(self.gate_heading(g)), 1),
                    "min_turn_radius_until_next": None if math.isinf(r) else round(min(r, 999.0), 1),
                    "centerline_index": int(self.gate_indices[g]),
                }
            )
        return {
            "key": self.key,
            "name": self.name,
            "track_width": round(2 * self.half_width, 2),
            "half_width": round(float(self.half_width), 2),
            "lap_length": round(self.length, 1),
            "n_waypoints": self.n_gates,
            "start_position": _r2(self.start_position),
            "start_heading_deg": round(math.degrees(self.start_heading), 1),
            "waypoints": waypoints,
            "centerline": [_r2(p) for p in self.centerline],
            "left_wall": [_r2(p) for p in self.left_wall],
            "right_wall": [_r2(p) for p in self.right_wall],
            "corners": self.corners(cfg),
            "briefing": self.briefing(cfg),
            "notes": (
                "Waypoints must be crossed in ascending order; crossing waypoint 0 after all "
                "others completes a lap. centerline/left_wall/right_wall are closed loops sampled "
                "every few units (the last point connects back to the first); centerline[0] is on "
                "the start/finish line and the car starts at centerline[-1]. "
                "min_turn_radius_until_next is the tightest corner between this waypoint and the "
                "next. With max lateral acceleration a, the fastest speed that holds through a "
                "corner of radius R is sqrt(a * R). Driver-frame convention: + = right for every "
                "angle, offset and steering value."
            ),
        }

    # -------------------------------------------------------------- validity
    def validate(self, min_radius: float | None = None, min_gap: float = 3.0) -> list[str]:
        """Geometry problems (empty list = valid track).

        * every corner wider than 1.2 * half_width (otherwise the inner wall folds over),
          or wider than `min_radius` when given;
        * the walls do not intersect themselves or each other;
        * separate parts of the track keep `min_gap` units of grass between their walls;
        * the start line sits on a flat-out straight (radius > 100 for 25 units either side);
        * zones reference valid gates.
        """
        problems = []
        limit = max(1.2 * self.half_width, min_radius or 0.0)
        if self.min_turn_radius() <= limit:
            problems.append(f"min centerline radius {self.min_turn_radius():.1f} <= {limit:.1f}")
        if _walls_self_intersect(self):
            problems.append("walls intersect")
        gap = _min_gap_between_sections(self)
        if gap < min_gap:
            problems.append(f"track sections only {gap:.1f} units apart (< {min_gap:g})")
        r_start = self.start_straight_radius()
        if r_start < 100.0:
            problems.append(f"start/finish is not on a straight (radius {r_start:.0f} < 100)")
        for z in self.zones:
            if not (0 <= z.from_wp < self.n_gates and 0 <= z.to_wp < self.n_gates) or z.from_wp == z.to_wp:
                problems.append(f"bad zone {z}")
        return problems

    def start_straight_radius(self, span: float = 25.0) -> float:
        """Tightest radius within `span` units of the start/finish line."""
        k = max(1, int(round(span / (self.length / self.n_points))))
        idx = np.arange(-k, k + 1) % self.n_points
        return float(np.min(self.radii[idx]))


def _r2(p) -> list[float]:
    return [round(float(p[0]), 2), round(float(p[1]), 2)]


def _closed_segments(points: np.ndarray) -> np.ndarray:
    nxt = np.roll(points, -1, axis=0)
    return np.column_stack([points, nxt])


def _as_zone(z) -> Zone:
    if isinstance(z, Zone):
        return z
    if isinstance(z, dict):
        return Zone(int(z["from_wp"]), int(z["to_wp"]), float(z["max_speed"]), str(z.get("label", "Yellow flag")))
    return Zone(*z)


def _walls_self_intersect(track: Track) -> bool:
    """True if any two non-neighbouring wall pieces cross (vectorised, ~0.5 M pairs)."""
    W = track.wall_segments
    hit = geo.segments_intersect(W, W)
    n = track.n_points
    i = np.arange(2 * n)
    side, k = i // n, i % n
    same_side = side[:, None] == side[None, :]
    dk = np.abs(k[:, None] - k[None, :])
    neighbours = same_side & ((dk <= 1) | (dk >= n - 1))
    return bool((hit & ~neighbours).any())


def _min_gap_between_sections(track: Track) -> float:
    """Smallest wall-to-wall gap between parts of the track that are far apart along the lap.

    Two centerline samples more than 3 track widths apart along the lap but close in space
    belong to different sections; their walls are 2 * half_width closer than their centres.
    """
    C = track.centerline
    d = np.linalg.norm(C[:, None, :] - C[None, :, :], axis=2)
    a = track.arc[:-1]
    along = np.abs(a[:, None] - a[None, :])
    along = np.minimum(along, track.length - along)
    far = along > max(6.0 * track.half_width, math.pi * track.min_turn_radius() + 2 * track.half_width)
    if not far.any():
        return math.inf
    return float(d[far].min() - 2 * track.half_width)


def detect_corners(track: Track, cfg: PhysicsConfig = DEFAULT_CONFIG) -> list[dict]:
    """Find corners from curvature (see Track.corners)."""
    n = track.n_points
    r_flat = cfg.max_speed ** 2 / cfg.max_lateral_accel
    tight = track.radii < r_flat
    sign = np.sign(track.curvature)
    if not tight.any():
        return []
    if tight.all():
        start = 0
    else:
        start = int(np.flatnonzero(~tight)[0])        # begin scanning on a straight
    order = (start + np.arange(n)) % n
    runs: list[list[int]] = []
    cur: list[int] = []
    for i in order:
        if tight[i] and cur and sign[i] != sign[cur[-1]]:
            runs.append(cur)
            cur = []
        if tight[i]:
            cur.append(int(i))
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    # merge same-direction runs separated by a short, almost-flat-out gap (one long corner)
    merged: list[list[int]] = []
    for r in runs:
        if merged and sign[r[0]] == sign[merged[-1][-1]] and (r[0] - merged[-1][-1]) % n <= 4:
            gap = [(merged[-1][-1] + k) % n for k in range(1, (r[0] - merged[-1][-1]) % n)]
            merged[-1] = merged[-1] + gap + r
        else:
            merged.append(r)
    runs = [r for r in merged if len(r) >= 2]
    # number corners in lap order starting from the start/finish line
    runs.sort(key=lambda r: float(track.arc[r[0]]))
    corners = []
    for k, run in enumerate(runs):
        idx = np.array(run)
        apex = int(idx[int(np.argmin(track.radii[idx]))])
        r_min = float(track.radii[apex])
        corners.append(
            {
                "id": f"T{k + 1}",
                "start_index": int(run[0]),
                "apex_index": apex,
                "end_index": int(run[-1]),
                "min_radius": round(r_min, 1),
                "direction": "left" if track.curvature[apex] > 0 else "right",
                "v_max": round(min(cfg.max_speed, math.sqrt(cfg.max_lateral_accel * r_min)), 1),
                "from_wp": track.segment_of_index(run[0]),
                "to_wp": (track.segment_of_index(run[-1]) + 1) % track.n_gates,
            }
        )
    return corners


# ---------------------------------------------------------------- builders
def track_from_control_points(
    points: Iterable,
    *,
    name: str,
    key: str = "custom",
    scale: float = 1.0,
    half_width: float = 9.0,
    n_gates: int = 16,
    n_samples: int | None = None,
    zones: Iterable = (),
    briefing_text: str | None = None,
) -> Track:
    """Spline through control points -> equally spaced centerline -> Track.

    The first control point becomes the start/finish line (centerline[0]); put it in the
    middle of a straight.
    """
    pts = np.asarray(list(points), dtype=float) * scale
    spline = geo.catmull_rom_closed(pts, samples_per_segment=30)
    if n_samples is None:
        n_samples = max(64, int(round(geo.polyline_length(spline) / SAMPLE_SPACING)))
    center = geo.resample_closed(spline, n_samples)
    return Track(name=name, centerline=center, half_width=half_width, n_gates=n_gates, key=key,
                 zones=list(zones), briefing_text=briefing_text)


def track_from_turtle(
    commands: list[tuple],
    adjust: tuple[int, int],
    *,
    name: str,
    key: str = "custom",
    half_width: float = 9.0,
    n_gates: int = 16,
    zones: Iterable = (),
    briefing_text: str | None = None,
) -> Track:
    """Straights + arcs -> closed centerline -> Track (see SPRINT_COMMANDS for the format).

    The turns must add up to +-360 degrees; the two straights `adjust` (indices into
    `commands`, with different headings) absorb the remaining position gap so the loop closes.
    """
    end, headings = _turtle(commands)[1:]
    i, j = adjust
    A = np.array([[math.cos(headings[i]), math.cos(headings[j])],
                  [math.sin(headings[i]), math.sin(headings[j])]])
    di, dj = np.linalg.solve(A, -end)
    cmds = list(commands)
    cmds[i] = ("S", cmds[i][1] + di)
    cmds[j] = ("S", cmds[j][1] + dj)
    if cmds[i][1] <= 0 or cmds[j][1] <= 0:
        raise ValueError(f"cannot close the loop with straights {adjust}: {cmds[i]}, {cmds[j]}")
    pts = _turtle(cmds)[0][:-1]                       # last point == first point
    n = max(64, int(round(geo.polyline_length(pts) / SAMPLE_SPACING)))
    center = geo.resample_closed(pts, n)
    # one light smoothing pass blends straight -> arc transitions (a 2-sample curvature ramp)
    center = (np.roll(center, 1, axis=0) + 2 * center + np.roll(center, -1, axis=0)) / 4
    center = geo.resample_closed(center, n)
    return Track(name=name, centerline=center, half_width=half_width, n_gates=n_gates, key=key,
                 zones=list(zones), briefing_text=briefing_text)


def _turtle(commands: list[tuple], step: float = 0.5) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Trace turtle commands from the origin heading +x.

    Returns (points, end position, heading at the start of every command)."""
    x = y = h = 0.0
    pts = [(0.0, 0.0)]
    headings = []
    for c in commands:
        headings.append(h)
        if c[0] == "S":
            n = max(1, int(c[1] / step))
            for _ in range(n):
                x += math.cos(h) * c[1] / n
                y += math.sin(h) * c[1] / n
                pts.append((x, y))
        else:
            r, ang = float(c[1]), math.radians(c[2])
            sgn = 1.0 if c[0] == "L" else -1.0
            n = max(2, int(r * ang / step))
            chord = 2 * r * math.sin(ang / n / 2)
            for _ in range(n):
                hm = h + sgn * ang / n / 2
                x += chord * math.cos(hm)
                y += chord * math.sin(hm)
                h += sgn * ang / n
                pts.append((x, y))
    return np.array(pts), np.array([x, y]), headings


# The first control point is the start/finish line, placed mid-straight.
GRAND_PRIX_CONTROL_POINTS = [
    (150, 0), (300, 0), (420, 20),            # start/finish straight
    (500, 110),                               # sweeping right-hander
    (470, 220), (380, 250),                   # tightening turn
    (330, 340), (400, 430),                   # chicane / kink
    (300, 500), (160, 480),                   # back section
    (90, 380),                                # hairpin-ish
    (150, 290), (90, 200),                    # esses
    (-40, 170), (-60, 60),                    # final corners back onto the straight
    (0, 0),
]
DEFAULT_CONTROL_POINTS = GRAND_PRIX_CONTROL_POINTS   # legacy name
DEFAULT_SCALE = 0.8

# Tracks built from straights and arcs ("turtle" commands) have exactly known corner radii:
#   ("S", length) | ("L", radius, degrees) | ("R", radius, degrees)
# The two straights named in `adjust` are lengthened/shortened so the loop closes exactly.
# The lap starts at the first command, so the first and last commands are straights.

# Short and technical: hairpin, chicane, a second hairpin and a slow final corner.
SPRINT_COMMANDS = [
    ("S", 138),                               # start/finish straight (second half)
    ("L", 21.6, 90),                          # T1
    ("S", 132),
    ("L", 16.8, 180),                         # T2 hairpin
    ("S", 48),
    ("R", 36, 90),                            # T3
    ("S", 36),                                # (closing straight, heading west)
    ("R", 16.8, 60), ("L", 16.8, 60),         # T4-T5 chicane
    ("S", 24),
    ("R", 30, 90),                            # T6
    ("S", 48),
    ("L", 19.2, 120), ("S", 24), ("L", 42, 60),   # T7 hairpin that opens up
    ("S", 144),                               # (closing straight, heading south)
    ("L", 30, 90),                            # T8 onto the main straight
    ("S", 102),
]
SPRINT_ADJUST = (6, 15)

# Fast and flowing: long straights, big sweepers and a few corners that reward braking.
# Every corner is either clearly flat out (R >= 110: <= 73 % grip at full speed) or clearly
# a braking corner (R <= 60), never "almost flat out", so a sensible planner has margin.
SPEEDWAY_COMMANDS = [
    ("S", 253),                               # start/finish straight (second half)
    ("L", 115, 70),                           # T1 fast left (flat out)
    ("S", 92),
    ("R", 138, 40), ("S", 23), ("L", 57.5, 110),   # flat right kink into a long braking left
    ("S", 138),
    ("L", 34.5, 80),                          # medium left: the big braking zone
    ("S", 69),
    ("R", 57.5, 50),
    ("S", 230),                               # (closing straight)
    ("L", 115, 60), ("S", 23), ("R", 120, 40), ("S", 23), ("L", 51.75, 80),   # flowing esses
    ("S", 115),                               # (closing straight)
    ("L", 57.5, 90),                          # last corner
    ("S", 172.5),
]
SPEEDWAY_ADJUST = (10, 16)


def make_grand_prix_track(n_samples: int = 360, half_width: float = 9.0, n_gates: int = 16) -> Track:
    """The home circuit: a long straight, a hairpin, a chicane and a mix of corner radii."""
    return track_from_control_points(GRAND_PRIX_CONTROL_POINTS, name="Grand Prix Circuit", key="grand_prix",
                                     scale=DEFAULT_SCALE, half_width=half_width, n_gates=n_gates,
                                     n_samples=n_samples)


def make_default_track(n_samples: int = 360, half_width: float = 9.0, n_gates: int = 16) -> Track:
    """Legacy alias of make_grand_prix_track()."""
    return make_grand_prix_track(n_samples, half_width, n_gates)


def make_oval_track(n_samples: int = 240, half_width: float = 10.0, n_gates: int = 8) -> Track:
    """A simple oval, useful for smoke tests and warm-ups."""
    pts = [(0, 0), (200, 0), (280, 60), (200, 120), (0, 120), (-80, 60)]
    return track_from_control_points(pts, name="Oval", key="oval", half_width=half_width,
                                     n_gates=n_gates, n_samples=n_samples)


def make_sprint_track() -> Track:
    """Short and technical: two hairpins and a chicane, little room to use full speed."""
    return track_from_turtle(SPRINT_COMMANDS, SPRINT_ADJUST, name="Sprint Ring", key="sprint")


def make_speedway_track() -> Track:
    """Long and fast: big sweepers where carrying speed matters more than braking."""
    return track_from_turtle(SPEEDWAY_COMMANDS, SPEEDWAY_ADJUST, name="Speedway", key="speedway")


def make_briefing_demo_track() -> Track:
    """The Grand Prix circuit with one yellow-flag zone, announced only in the briefing text."""
    tr = make_grand_prix_track()
    tr.key = "briefing_demo"
    tr.name = "Grand Prix Circuit (yellow flag)"
    tr.zones = [Zone(9, 11, 15.0, "Yellow flag")]
    tr.briefing_text = "Marshals are recovering a car near the hairpin. Expect the flag for the whole session."
    return tr


# ------------------------------------------------------------ random tracks
RANDOM_LENGTH = (1300.0, 1700.0)
RANDOM_MIN_RADIUS = 12.0
RANDOM_SMOOTHING_PASSES = 4


def make_random_track(seed: int | str, half_width: float = 9.0, n_gates: int = 16) -> Track:
    """Procedural track, deterministic per seed (same seed -> same track on every machine).

    Envelope: lap 1300-1700 units, half_width 9, min centerline radius >= 12, 16 gates,
    no wall overlap, start/finish in the middle of the longest straight. Candidates are
    drawn from a seeded RNG until one passes Track.validate().
    """
    seed_int = _seed_to_int(seed)
    center = _random_centerline(seed_int, half_width, n_gates)
    return Track(name=f"Random #{seed}", centerline=center.copy(), half_width=half_width,
                 n_gates=n_gates, key=f"random:{seed}")


def _seed_to_int(seed: int | str) -> int:
    """Any seed -> a non-negative int for numpy: integers as they are (negative ones wrap to 64 bits),
    anything else (e.g. "1.5", "monza") by its CRC32."""
    s = str(seed).strip()
    try:
        value = int(s)
    except ValueError:
        return zlib.crc32(s.encode("utf-8"))
    return value if value >= 0 else value & 0xFFFFFFFFFFFFFFFF


@lru_cache(maxsize=64)
def _random_centerline(seed: int, half_width: float, n_gates: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    for _ in range(500):
        center = _random_candidate(rng)
        if center is None:
            continue
        tr = Track(name="candidate", centerline=center, half_width=half_width, n_gates=n_gates)
        if not tr.validate(min_radius=RANDOM_MIN_RADIUS):
            return center
    raise RuntimeError(f"could not generate a valid random track for seed {seed}")


def _random_candidate(rng: np.random.Generator) -> np.ndarray | None:
    """Random star-shaped loop -> spline -> scaled to a random length -> start on the longest straight."""
    n = int(rng.integers(8, 14))
    base = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
    ang = base + rng.uniform(-0.3, 0.3, n) * (2 * math.pi / n)
    rad = rng.uniform(0.45, 1.0, n)
    pts = np.column_stack([rad * np.cos(ang), rad * np.sin(ang)])
    if rng.random() < 0.5:
        pts = pts[::-1]                                   # clockwise half of the time
    spline = geo.catmull_rom_closed(pts, samples_per_segment=30)
    target = rng.uniform(*RANDOM_LENGTH)
    spline = spline * (target / geo.polyline_length(spline))
    n_samples = int(round(target / SAMPLE_SPACING))
    center = geo.resample_closed(spline, n_samples)
    # a Catmull-Rom spline only has a continuous direction; a few [1, 2, 1] smoothing passes
    # also make the curvature continuous, so corner radii change gradually (planner friendly)
    for _ in range(RANDOM_SMOOTHING_PASSES):
        center = (np.roll(center, 1, axis=0) + 2 * center + np.roll(center, -1, axis=0)) / 4
    center = geo.resample_closed(center, n_samples)
    if not RANDOM_LENGTH[0] <= geo.polyline_length(center) <= RANDOM_LENGTH[1]:
        return None
    # put the start/finish line 40 % of the way into the longest straight run
    radii = geo.curvature_radius_closed(center)
    straight = radii > 250.0
    if straight.all() or not straight.any():
        return None
    run_start, run_len = _longest_run(straight)
    if run_len * SAMPLE_SPACING < 120.0:
        return None
    start = (run_start + int(0.4 * run_len)) % len(center)
    center = np.roll(center, -start, axis=0)
    # normalise the pose: start line at the origin, start straight heading +x
    t = geo.tangents_closed(center)[0]
    a = -math.atan2(t[1], t[0])
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return (center - center[0]) @ rot.T


def _longest_run(mask: np.ndarray) -> tuple[int, int]:
    """Start index and length of the longest circular run of True values."""
    n = len(mask)
    first_false = int(np.flatnonzero(~mask)[0])
    best = (0, 0)
    cur_start, cur_len = 0, 0
    for k in range(n):
        i = (first_false + 1 + k) % n
        if mask[i]:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best[1]:
                best = (cur_start, cur_len)
        else:
            cur_len = 0
    return best


# ---------------------------------------------------------------- file tracks
def load_track_file(path: str | Path) -> Track:
    """Load a track from a Python file (the instructor's sealed mystery track).

    The file defines EITHER `make_track() -> Track` OR `CONTROL_POINTS` (list of (x, y))
    plus optional `NAME`, `ZONES` (list of Zone / (from_wp, to_wp, max_speed[, label]) /
    dicts), `BRIEFING` (extra briefing text), `SCALE`, `HALF_WIDTH`, `N_GATES`.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"track file not found: {path}")
    mod_name = f"agp_track_{zlib.crc32(str(path.resolve()).encode()):08x}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    key = f"file:{path.stem}"
    if hasattr(module, "make_track"):
        tr = module.make_track()
        if not isinstance(tr, Track):
            raise TypeError(f"{path}: make_track() must return a Track")
        if tr.key == "custom":
            tr.key = key
        return tr
    if not hasattr(module, "CONTROL_POINTS"):
        raise ValueError(f"{path}: define make_track() or CONTROL_POINTS")
    return track_from_control_points(
        module.CONTROL_POINTS,
        name=getattr(module, "NAME", path.stem),
        key=key,
        scale=float(getattr(module, "SCALE", 1.0)),
        half_width=float(getattr(module, "HALF_WIDTH", 9.0)),
        n_gates=int(getattr(module, "N_GATES", 16)),
        zones=getattr(module, "ZONES", ()),
        briefing_text=getattr(module, "BRIEFING", None),
    )


# ---------------------------------------------------------------- registry
TRACKS: dict[str, Callable[[], Track]] = {
    "grand_prix": make_grand_prix_track,
    "default": make_grand_prix_track,          # alias
    "oval": make_oval_track,
    "sprint": make_sprint_track,
    "speedway": make_speedway_track,
    "briefing_demo": make_briefing_demo_track,
}
PUBLIC_TRACKS = ("grand_prix", "oval", "sprint", "speedway", "briefing_demo")


def make_track(spec: str | Track | None = None) -> Track:
    """Build a track from a spec.

    * a registry key: "grand_prix" (alias "default"), "oval", "sprint", "speedway", "briefing_demo"
    * "random:<seed>": procedural track, deterministic per seed
    * "file:path.py" or a path ending in ".py": a track file (see load_track_file)
    * a Track instance is returned unchanged; None means "grand_prix"
    """
    if spec is None:
        return make_grand_prix_track()
    if isinstance(spec, Track):
        return spec
    s = str(spec).strip()
    if s in TRACKS:
        return TRACKS[s]()
    if s.startswith("random:"):
        return make_random_track(s.split(":", 1)[1])
    if s.startswith("file:"):
        return load_track_file(s.split(":", 1)[1])
    if s.endswith(".py"):
        return load_track_file(s)
    raise KeyError(f"unknown track {spec!r}; use one of {sorted(TRACKS)}, 'random:<seed>' or 'file:path.py'")
