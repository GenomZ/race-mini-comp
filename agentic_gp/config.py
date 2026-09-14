"""Physics and competition constants.

All units are abstract "track units" (roughly metres) and seconds.
Every number here is deliberately simple so students can reason about it.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class PhysicsConfig:
    # --- time -------------------------------------------------------------
    tick_seconds: float = 0.5      # virtual time that passes per DriveTool call
    substep_seconds: float = 0.05  # internal integration step

    # --- longitudinal dynamics -------------------------------------------
    max_speed: float = 40.0        # hard cap on forward speed
    max_reverse_speed: float = 6.0 # slow reverse so a stuck car can back off a wall
    accel: float = 12.0            # units/s^2 at full throttle
    brake: float = 25.0            # units/s^2 at full brake (while moving forward)
    drag: float = 0.15             # linear drag coefficient (a = -drag * v)

    # --- lateral dynamics ---------------------------------------------------
    # Steering commands a fraction of the available lateral grip:
    #   yaw_rate = steering * min(max_yaw_rate, max_lateral_accel / |speed|)
    # so the turn radius at speed v with steering s is  R = v^2 / (max_lateral_accel * |s|),
    # never tighter than v / max_yaw_rate. Faster => wider corners => speed matters.
    max_lateral_accel: float = 20.0  # units/s^2
    max_yaw_rate: float = 1.5        # rad/s, low-speed cap (~86 deg/s)

    # --- sensors ----------------------------------------------------------
    ray_max_distance: float = 200.0
    ray_angles_deg: tuple = (0, -20, 20, -45, 45, -90, 90)  # negative = left of heading

    # --- competition rules ------------------------------------------------
    crash_penalty_seconds: float = 5.0
    crash_pushback: float = 2.0    # how far the car is nudged back from a wall after a crash
    max_ticks: int = 500           # attempt ends (DNF) after this many DriveTool calls (250 s)

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_CONFIG = PhysicsConfig()
