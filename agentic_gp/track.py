"""Track geometry: centerline, walls and waypoint gates.

A track is defined by a handful of control points. A closed Catmull-Rom spline is
fitted through them, resampled to equal arc-length spacing, and offset left/right
by `half_width` to produce the two walls. Waypoint "gates" are segments spanning
the track from wall to wall at evenly spaced centerline indices. Gate 0 is the
start/finish line.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import geometry as geo


@dataclass
class Track:
    name: str
    centerline: np.ndarray          # (N, 2)
    half_width: float
    n_gates: int
    left_wall: np.ndarray = field(init=False)      # (N, 2)
    right_wall: np.ndarray = field(init=False)     # (N, 2)
    wall_segments: np.ndarray = field(init=False)  # (2N, 4)
    gate_indices: np.ndarray = field(init=False)   # (n_gates,)
    gates: np.ndarray = field(init=False)          # (n_gates, 4) x1,y1,x2,y2
    gate_centers: np.ndarray = field(init=False)   # (n_gates, 2)

    def __post_init__(self) -> None:
        C = np.asarray(self.centerline, dtype=float)
        T = geo.tangents_closed(C)
        left_normal = np.column_stack([-T[:, 1], T[:, 0]])
        self.left_wall = C + left_normal * self.half_width
        self.right_wall = C - left_normal * self.half_width
        self.wall_segments = np.vstack(
            [_closed_segments(self.left_wall), _closed_segments(self.right_wall)]
        )
        n = len(C)
        self.gate_indices = (np.arange(self.n_gates) * n // self.n_gates).astype(int)
        self.gates = np.column_stack(
            [self.left_wall[self.gate_indices], self.right_wall[self.gate_indices]]
        )
        self.gate_centers = C[self.gate_indices]

    # ------------------------------------------------------------------ info
    @property
    def n_points(self) -> int:
        return len(self.centerline)

    @property
    def length(self) -> float:
        return geo.polyline_length(self.centerline, closed=True)

    @property
    def start_position(self) -> np.ndarray:
        """One centerline sample before the start/finish gate."""
        return self.centerline[-1].copy()

    @property
    def start_heading(self) -> float:
        t = geo.tangents_closed(self.centerline)[-1]
        return math.atan2(t[1], t[0])

    def min_turn_radius(self) -> float:
        return float(np.min(geo.curvature_radius_closed(self.centerline)))

    def gate_heading(self, gate: int) -> float:
        t = geo.tangents_closed(self.centerline)[self.gate_indices[gate]]
        return math.atan2(t[1], t[0])

    def bounds(self) -> tuple[float, float, float, float]:
        allpts = np.vstack([self.left_wall, self.right_wall])
        return (
            float(allpts[:, 0].min()),
            float(allpts[:, 1].min()),
            float(allpts[:, 0].max()),
            float(allpts[:, 1].max()),
        )

    def segment_min_radius(self, gate: int) -> float:
        """Tightest turning radius on the centerline between `gate` and the next gate."""
        radii = geo.curvature_radius_closed(self.centerline)
        a = self.gate_indices[gate]
        b = self.gate_indices[(gate + 1) % self.n_gates]
        if b > a:
            idx = np.arange(a, b)
        else:
            idx = np.concatenate([np.arange(a, self.n_points), np.arange(0, b)])
        return float(np.min(radii[idx]))

    def to_map(self) -> dict:
        """JSON-serialisable description that agents may use for planning."""
        waypoints = []
        for g in range(self.n_gates):
            r = self.segment_min_radius(g)
            waypoints.append(
                {
                    "index": g,
                    "center": [round(float(v), 2) for v in self.gate_centers[g]],
                    "track_heading_deg": round(math.degrees(self.gate_heading(g)), 1),
                    "min_turn_radius_until_next": None if math.isinf(r) else round(min(r, 999.0), 1),
                }
            )
        return {
            "name": self.name,
            "track_width": round(2 * self.half_width, 2),
            "lap_length": round(self.length, 1),
            "n_waypoints": self.n_gates,
            "start_position": [round(float(v), 2) for v in self.start_position],
            "start_heading_deg": round(math.degrees(self.start_heading), 1),
            "waypoints": waypoints,
            "notes": (
                "Waypoints must be crossed in ascending order; crossing waypoint 0 after all "
                "others completes the lap. min_turn_radius_until_next is the tightest corner "
                "between this waypoint and the next. With max lateral acceleration a, the "
                "fastest speed that holds through a corner of radius R is sqrt(a * R)."
            ),
        }


def _closed_segments(points: np.ndarray) -> np.ndarray:
    nxt = np.roll(points, -1, axis=0)
    return np.column_stack([points, nxt])


# ---------------------------------------------------------------- factories
# The first control point is the start/finish line, placed mid-straight.
DEFAULT_CONTROL_POINTS = [
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
DEFAULT_SCALE = 0.8


def make_default_track(n_samples: int = 360, half_width: float = 9.0, n_gates: int = 16) -> Track:
    """The competition track: a loop with a long straight and a mix of corner radii."""
    pts = np.array(DEFAULT_CONTROL_POINTS, dtype=float) * DEFAULT_SCALE
    spline = geo.catmull_rom_closed(pts, samples_per_segment=30)
    center = geo.resample_closed(spline, n_samples)
    return Track(name="Grand Prix Circuit", centerline=center, half_width=half_width, n_gates=n_gates)


def make_oval_track(n_samples: int = 240, half_width: float = 10.0, n_gates: int = 8) -> Track:
    """A simple oval, useful for smoke tests and warm-ups."""
    pts = [(0, 0), (200, 0), (280, 60), (200, 120), (0, 120), (-80, 60)]
    spline = geo.catmull_rom_closed(np.array(pts, dtype=float), samples_per_segment=30)
    center = geo.resample_closed(spline, n_samples)
    return Track(name="Oval", centerline=center, half_width=half_width, n_gates=n_gates)


TRACKS = {"default": make_default_track, "oval": make_oval_track}
