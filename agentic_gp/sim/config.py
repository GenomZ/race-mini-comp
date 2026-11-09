"""Physics and competition constants.

    from agentic_gp.sim.config import PhysicsConfig
    cfg = PhysicsConfig(crash_rule="classic", max_ticks=300)

All units are abstract "track units" (roughly metres) and seconds.
Every number here is deliberately simple so students can reason about it.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

CRASH_RULES = ("marshal", "classic")


@dataclass(frozen=True)
class PhysicsConfig:
    # --- time -------------------------------------------------------------
    tick_seconds: float = 0.5      # virtual time that passes per drive() call
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
    # "marshal": after a wall hit the car is put back on the centerline (inside the current
    #   waypoint segment), pointing along the track, speed 0. A stopped car cannot turn, so
    #   without this a car facing a wall would crash again and again.
    # "classic": the car stops where it hit and is nudged `crash_pushback` units back (the
    #   original rule, kept for the "crash loop" demo in chapter 1).
    crash_rule: str = "marshal"
    crash_pushback: float = 2.0    # classic rule only: nudge back from the wall
    zone_penalty_seconds: float = 1.0   # per tick that ends inside a speed-limit zone too fast
    zone_speed_tolerance: float = 0.5   # speed may exceed the zone limit by this much
    max_ticks: int = 500           # per lap: an attempt of L laps is a DNF after max_ticks * L ticks

    def __post_init__(self) -> None:
        if self.crash_rule not in CRASH_RULES:
            raise ValueError(f"crash_rule must be one of {CRASH_RULES}, got {self.crash_rule!r}")

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_CONFIG = PhysicsConfig()
