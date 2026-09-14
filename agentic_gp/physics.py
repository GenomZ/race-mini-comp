"""Car dynamics: a kinematic bicycle model with a lateral grip limit.

State is (x, y, heading, speed). Heading is CCW-positive radians. Positive
steering turns the car to the RIGHT (clockwise) to match the tool API, where
-1 = full left and +1 = full right.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import PhysicsConfig


@dataclass
class CarState:
    x: float
    y: float
    heading: float
    speed: float = 0.0

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y])

    def copy(self) -> "CarState":
        return CarState(self.x, self.y, self.heading, self.speed)


def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def integrate(state: CarState, throttle: float, steering: float, cfg: PhysicsConfig, dt: float) -> CarState:
    """Advance the car by `dt` seconds. Returns a new state (does not mutate)."""
    throttle = clamp(float(throttle), -1.0, 1.0)
    steering = clamp(float(steering), -1.0, 1.0)
    v = state.speed

    # --- longitudinal ---------------------------------------------------
    if throttle >= 0:
        a = throttle * cfg.accel
    elif v > 1e-6:
        a = throttle * cfg.brake           # braking while moving forward
    else:
        a = throttle * cfg.accel * 0.5     # gentle reverse when (nearly) stopped
    a -= cfg.drag * v
    v_new = v + a * dt
    # Do not let braking flip the direction of travel within a single step.
    if v > 1e-6 and throttle < 0 and v_new < 0:
        v_new = 0.0
    v_new = clamp(v_new, -cfg.max_reverse_speed, cfg.max_speed)

    # --- lateral: steering = fraction of available grip -------------------
    v_mid = 0.5 * (v + v_new)
    yaw_rate = steering * available_yaw_rate(v_mid, cfg)   # rad/s, positive = right (CW)
    if abs(v_mid) < 1e-6:
        yaw_rate = 0.0                                      # a stationary car cannot turn
    heading_new = state.heading - yaw_rate * dt   # minus: positive steering = clockwise

    # --- position ---------------------------------------------------------
    h_mid = 0.5 * (state.heading + heading_new)
    x_new = state.x + v_mid * math.cos(h_mid) * dt
    y_new = state.y + v_mid * math.sin(h_mid) * dt
    return CarState(x_new, y_new, heading_new, v_new)


def available_yaw_rate(speed: float, cfg: PhysicsConfig) -> float:
    """Yaw rate (rad/s) produced by full steering at `speed`."""
    if abs(speed) < 1e-6:
        return cfg.max_yaw_rate
    return min(cfg.max_yaw_rate, cfg.max_lateral_accel / abs(speed))


def turn_radius(speed: float, steering: float, cfg: PhysicsConfig) -> float:
    """Radius of the circle the car follows at `speed` with constant `steering`."""
    yaw = abs(steering) * available_yaw_rate(speed, cfg)
    return math.inf if yaw < 1e-9 else abs(speed) / yaw


def max_corner_speed(radius: float, cfg: PhysicsConfig) -> float:
    """Fastest speed that the grip limit allows through a corner of `radius`."""
    if not math.isfinite(radius):
        return cfg.max_speed
    return min(cfg.max_speed, math.sqrt(cfg.max_lateral_accel * max(radius, 0.0)))
