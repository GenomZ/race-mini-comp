"""Small, dependency-light 2D geometry helpers (numpy only).

    from agentic_gp.sim import geometry as geo
    d = geo.raycast(np.zeros(2), 0.0, walls, 200.0)            # one ray
    ds = geo.raycast_many(np.zeros(2), angles, walls, 200.0)   # many rays at once
    hit = geo.first_hit(path, walls)                            # earliest wall hit along a polyline

Conventions
-----------
* Angles are radians, counter-clockwise positive, measured from the +x axis.
* "Left" of a heading is heading + pi/2.
* Segments are stored as arrays of shape (M, 4): x1, y1, x2, y2.
"""

from __future__ import annotations

import math

import numpy as np

EPS = 1e-9


def wrap_angle(a: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    return (a + math.pi) % (2 * math.pi) - math.pi


def unit(angle: float) -> np.ndarray:
    """Unit vector pointing along `angle`."""
    return np.array([math.cos(angle), math.sin(angle)])


def signed_angle_to(heading: float, vec: np.ndarray) -> float:
    """Signed angle from `heading` to direction of `vec` (positive = CCW = left)."""
    target = math.atan2(vec[1], vec[0])
    return wrap_angle(target - heading)


def raycast(origin: np.ndarray, angle: float, segments: np.ndarray, max_dist: float) -> float:
    """Distance from origin along `angle` to the nearest segment, or max_dist."""
    d = unit(angle)
    a = segments[:, 0:2]
    b = segments[:, 2:4]
    v1 = origin - a                    # (M,2)
    v2 = b - a                         # (M,2)
    v3 = np.array([-d[1], d[0]])       # perpendicular to ray
    denom = v2 @ v3                    # (M,)
    ok = np.abs(denom) > EPS
    t = np.full(len(segments), np.inf)
    s = np.full(len(segments), np.inf)
    cross = v2[:, 0] * v1[:, 1] - v2[:, 1] * v1[:, 0]
    t[ok] = cross[ok] / denom[ok]
    s[ok] = (v1[ok] @ v3) / denom[ok]
    hit = ok & (t >= 0.0) & (s >= 0.0) & (s <= 1.0)
    if not hit.any():
        return float(max_dist)
    return float(min(t[hit].min(), max_dist))


def raycast_many(origin: np.ndarray, angles: np.ndarray, segments: np.ndarray, max_dist: float) -> np.ndarray:
    """Vectorised `raycast` for several ray angles at once -> distances, shape (R,).

    Solves origin + t * d = a + s * (b - a) for every (ray, segment) pair; a hit needs
    t >= 0 (in front of the car) and 0 <= s <= 1 (on the segment).
    """
    ang = np.asarray(angles, dtype=float)
    if len(segments) == 0:
        return np.full(len(ang), float(max_dist))
    dx, dy = np.cos(ang)[:, None], np.sin(ang)[:, None]        # (R,1) ray directions
    ox, oy = float(origin[0]), float(origin[1])
    wx, wy = segments[:, 0] - ox, segments[:, 1] - oy           # (M,) origin -> segment start
    ex, ey = segments[:, 2] - segments[:, 0], segments[:, 3] - segments[:, 1]   # (M,) segment direction
    denom = dx * ey - dy * ex                                   # (R,M) cross(d, e)
    denom[np.abs(denom) <= EPS] = np.nan                        # parallel: never a hit
    t = (wx * ey - wy * ex) / denom                             # distance along the ray
    s = (wx * dy - wy * dx) / denom                             # position along the segment
    t[~((t >= 0.0) & (s >= 0.0) & (s <= 1.0))] = np.inf        # NaN compares False -> inf
    return np.minimum(t.min(axis=1), max_dist)


def segment_hits(p0: np.ndarray, p1: np.ndarray, segments: np.ndarray) -> np.ndarray:
    """Boolean mask of segments intersected by the segment p0 -> p1."""
    r = p1 - p0
    q = segments[:, 0:2]
    s = segments[:, 2:4] - q
    rxs = r[0] * s[:, 1] - r[1] * s[:, 0]
    qp = q - p0
    qpxr = qp[:, 0] * r[1] - qp[:, 1] * r[0]
    ok = np.abs(rxs) > EPS
    t = np.full(len(segments), np.inf)
    u = np.full(len(segments), np.inf)
    qpxs = qp[:, 0] * s[:, 1] - qp[:, 1] * s[:, 0]
    t[ok] = qpxs[ok] / rxs[ok]
    u[ok] = qpxr[ok] / rxs[ok]
    return ok & (t >= 0.0) & (t <= 1.0) & (u >= 0.0) & (u <= 1.0)


def first_hit(path: np.ndarray, segments: np.ndarray) -> tuple[int, float, np.ndarray] | None:
    """Earliest crossing of the polyline `path` (K+1, 2) with any of `segments` (M, 4).

    Returns (k, t, point): the path piece k (path[k] -> path[k+1]), the fraction t along
    it and the crossing point; None if the path touches no segment. All K x M pairs are
    tested in one vectorised pass (no Python loop over walls).
    """
    P = np.asarray(path, dtype=float)
    if len(segments) == 0:
        return None
    p0 = P[:-1]                                   # (K,2)
    r = P[1:] - p0                                # (K,2)
    rx, ry = r[:, 0:1], r[:, 1:2]                 # (K,1)
    sx, sy = segments[:, 2] - segments[:, 0], segments[:, 3] - segments[:, 1]   # (M,)
    qpx = segments[:, 0] - p0[:, 0:1]             # (K,M)
    qpy = segments[:, 1] - p0[:, 1:2]
    rxs = rx * sy - ry * sx                       # (K,M)
    rxs[np.abs(rxs) <= EPS] = np.nan              # parallel pieces never count as a hit
    t = (qpx * sy - qpy * sx) / rxs               # fraction along the path piece
    u = (qpx * ry - qpy * rx) / rxs               # fraction along the wall segment
    hit = (t >= 0.0) & (t <= 1.0) & (u >= 0.0) & (u <= 1.0)
    rows = np.flatnonzero(hit.any(axis=1))
    if len(rows) == 0:
        return None
    k = int(rows[0])
    tk = float(t[k][hit[k]].min())
    return k, tk, p0[k] + tk * r[k]


def segments_intersect(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise proper-intersection matrix (Ma, Mb) between two segment arrays (M, 4)."""
    p, r = a[:, None, 0:2], (a[:, 2:4] - a[:, 0:2])[:, None, :]
    q, s = b[None, :, 0:2], (b[:, 2:4] - b[:, 0:2])[None, :, :]
    rxs = r[..., 0] * s[..., 1] - r[..., 1] * s[..., 0]
    qp = q - p
    ok = np.abs(rxs) > EPS
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (qp[..., 0] * s[..., 1] - qp[..., 1] * s[..., 0]) / rxs
        u = (qp[..., 0] * r[..., 1] - qp[..., 1] * r[..., 0]) / rxs
    return ok & (t >= 0.0) & (t <= 1.0) & (u >= 0.0) & (u <= 1.0)


def crosses(p0: np.ndarray, p1: np.ndarray, segment: np.ndarray) -> bool:
    """True if the path p0 -> p1 crosses a single segment (x1, y1, x2, y2)."""
    return bool(segment_hits(p0, p1, segment.reshape(1, 4))[0])


def crosses_xy(x0: float, y0: float, x1: float, y1: float, seg) -> bool:
    """Scalar (pure Python, no numpy overhead) version of `crosses` for the per-substep gate check."""
    qx, qy = float(seg[0]), float(seg[1])
    sx, sy = float(seg[2]) - qx, float(seg[3]) - qy
    rx, ry = x1 - x0, y1 - y0
    rxs = rx * sy - ry * sx
    if abs(rxs) <= EPS:
        return False
    px, py = qx - x0, qy - y0
    t = (px * sy - py * sx) / rxs
    u = (px * ry - py * rx) / rxs
    return 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0


def catmull_rom_closed(points: np.ndarray, samples_per_segment: int = 20) -> np.ndarray:
    """Sample a closed Catmull-Rom spline through `points`."""
    P = np.asarray(points, dtype=float)
    n = len(P)
    out = []
    ts = np.linspace(0.0, 1.0, samples_per_segment, endpoint=False)
    for i in range(n):
        p0, p1, p2, p3 = P[(i - 1) % n], P[i], P[(i + 1) % n], P[(i + 2) % n]
        for t in ts:
            t2, t3 = t * t, t * t * t
            pt = 0.5 * (
                (2 * p1)
                + (-p0 + p2) * t
                + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                + (-p0 + 3 * p1 - 3 * p2 + p3) * t3
            )
            out.append(pt)
    return np.array(out)


def resample_closed(points: np.ndarray, n_samples: int) -> np.ndarray:
    """Resample a closed polyline to `n_samples` points equally spaced by arc length."""
    P = np.asarray(points, dtype=float)
    closed = np.vstack([P, P[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    targets = np.linspace(0.0, total, n_samples, endpoint=False)
    xs = np.interp(targets, cum, closed[:, 0])
    ys = np.interp(targets, cum, closed[:, 1])
    return np.column_stack([xs, ys])


def polyline_length(points: np.ndarray, closed: bool = True) -> float:
    """Total length of a polyline (closed: includes the last -> first piece)."""
    P = np.asarray(points, dtype=float)
    if closed:
        P = np.vstack([P, P[:1]])
    return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())


def tangents_closed(points: np.ndarray) -> np.ndarray:
    """Unit tangents of a closed polyline via central differences."""
    P = np.asarray(points, dtype=float)
    d = np.roll(P, -1, axis=0) - np.roll(P, 1, axis=0)
    norms = np.linalg.norm(d, axis=1, keepdims=True)
    return d / np.maximum(norms, EPS)


def signed_curvature_closed(points: np.ndarray) -> np.ndarray:
    """Signed curvature 1/R at each point of a closed polyline (positive = turning left / CCW)."""
    P = np.asarray(points, dtype=float)
    a, b, c = np.roll(P, 1, axis=0), P, np.roll(P, -1, axis=0)
    ab, bc, ca = np.linalg.norm(b - a, axis=1), np.linalg.norm(c - b, axis=1), np.linalg.norm(a - c, axis=1)
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    return 2.0 * cross / np.maximum(ab * bc * ca, EPS)


def curvature_radius_closed(points: np.ndarray) -> np.ndarray:
    """Approximate local turning radius at each point of a closed polyline."""
    P = np.asarray(points, dtype=float)
    a, b, c = np.roll(P, 1, axis=0), P, np.roll(P, -1, axis=0)
    ab, bc, ca = np.linalg.norm(b - a, axis=1), np.linalg.norm(c - b, axis=1), np.linalg.norm(a - c, axis=1)
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    area2 = np.abs(cross)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (ab * bc * ca) / np.maximum(2.0 * area2, EPS)
    return np.where(area2 < EPS, np.inf, r)
