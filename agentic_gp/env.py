"""The racing gym: RaceEnv.

    env = RaceEnv()
    obs = env.reset()
    obs = env.step(throttle=0.8, steering=-0.1)   # advances 0.5 virtual seconds
    ...
    env.attempts   # list of finished attempt summaries (lap time, crashes, ...)

The environment is deterministic. Every call to `step` is logged (JSONL) so that
agents can be debugged and replays plotted with `agentic_gp.render`.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np

from . import geometry as geo
from .config import PhysicsConfig, DEFAULT_CONFIG
from .physics import CarState, integrate, clamp
from .track import Track, make_default_track


@dataclass
class AttemptResult:
    attempt: int
    completed: bool
    lap_time: float | None          # sim time + penalties, None if DNF
    sim_time: float
    penalty_time: float
    crashes: int
    waypoints_passed: int
    ticks: int
    ended_by: str                   # "lap_complete" | "max_ticks" | "reset"

    def to_dict(self) -> dict:
        return asdict(self)


class RaceEnv:
    def __init__(
        self,
        track: Track | None = None,
        config: PhysicsConfig = DEFAULT_CONFIG,
        log_dir: str | Path | None = None,
        run_name: str | None = None,
        verbose: bool = False,
    ) -> None:
        self.track = track or make_default_track()
        self.cfg = config
        self.verbose = verbose
        self.attempts: list[AttemptResult] = []
        self.attempt_traces: list[dict] = []   # trajectory + crash points per finished attempt
        self.attempt_no = 0
        self._log_path: Path | None = None
        if log_dir is not None:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            name = run_name or time.strftime("run_%Y%m%d_%H%M%S")
            self._log_path = Path(log_dir) / f"{name}.jsonl"
        self._log_fh = None
        self._started = False
        self.reset()

    # ------------------------------------------------------------ lifecycle
    def reset(self) -> dict:
        """Put the car on the start line and start a new attempt.

        Resetting an attempt that has not been driven yet (tick == 0) simply
        re-places the car; it is not recorded as an attempt.
        """
        if self._started and not self.done and self.tick > 0:
            self._finish_attempt("reset")
        if not self._started or self.done or self.tick > 0:
            self.attempt_no += 1
        self._started = True
        sx, sy = self.track.start_position
        self.car = CarState(float(sx), float(sy), self.track.start_heading, 0.0)
        self.sim_time = 0.0
        self.penalty_time = 0.0
        self.crashes = 0
        self.tick = 0
        self.next_gate = 1 % self.track.n_gates
        self.gates_passed = 0
        self.lap_complete = False
        self.done = False
        self.ended_by: str | None = None
        self.trajectory: list[tuple[float, float, float]] = [(self.car.x, self.car.y, 0.0)]
        self.crash_points: list[tuple[float, float]] = []
        self.last_controls = (0.0, 0.0)
        self._log({"event": "reset", "attempt": self.attempt_no})
        return self.sensors()

    def close(self) -> None:
        if self._started and not self.done:
            self._finish_attempt("reset")
        if self._log_fh:
            self._log_fh.close()
            self._log_fh = None

    # ------------------------------------------------------------- stepping
    def step(self, throttle: float, steering: float) -> dict:
        """Apply controls for one tick (tick_seconds) and return the new observation."""
        if self.done:
            obs = self.sensors()
            obs["events"] = ["attempt_over: call reset_environment to start a new attempt"]
            return obs

        throttle = clamp(_as_float(throttle), -1.0, 1.0)
        steering = clamp(_as_float(steering), -1.0, 1.0)
        self.last_controls = (throttle, steering)
        events: list[str] = []
        n_sub = max(1, int(round(self.cfg.tick_seconds / self.cfg.substep_seconds)))
        dt = self.cfg.tick_seconds / n_sub
        crashed_this_tick = False

        for _ in range(n_sub):
            prev = self.car
            new = integrate(prev, 0.0 if crashed_this_tick else throttle, steering, self.cfg, dt)
            p0, p1 = prev.position, new.position
            if not crashed_this_tick and geo.segment_hits(p0, p1, self.track.wall_segments).any():
                crashed_this_tick = True
                self.crashes += 1
                self.penalty_time += self.cfg.crash_penalty_seconds
                self.crash_points.append((float(p1[0]), float(p1[1])))
                events.append(
                    f"crash: hit a wall, +{self.cfg.crash_penalty_seconds:g}s penalty, speed reset to 0"
                )
                # Nudge back from the wall so the car has room to steer away.
                back = p0 - geo.unit(prev.heading) * self.cfg.crash_pushback
                if not geo.segment_hits(p0, back, self.track.wall_segments).any():
                    p0 = back
                new = CarState(float(p0[0]), float(p0[1]), prev.heading, 0.0)
            else:
                gate = self.track.gates[self.next_gate]
                if geo.crosses(p0, p1, gate):
                    self.gates_passed += 1
                    events.append(f"waypoint_passed: {self.next_gate}")
                    if self.next_gate == 0:
                        self.lap_complete = True
                    self.next_gate = (self.next_gate + 1) % self.track.n_gates
            self.car = new
            self.sim_time += dt
            if self.lap_complete:
                break

        self.tick += 1
        self.trajectory.append((self.car.x, self.car.y, self.car.speed))

        if self.lap_complete:
            events.append(
                f"lap_complete: {self.lap_time:.2f}s (incl. {self.penalty_time:g}s penalties)"
            )
            self._finish_attempt("lap_complete")
        elif self.tick >= self.cfg.max_ticks:
            events.append("attempt_over: max ticks reached (DNF)")
            self._finish_attempt("max_ticks")

        obs = self.sensors()
        obs["events"] = events
        self._log(
            {
                "event": "step",
                "attempt": self.attempt_no,
                "throttle": throttle,
                "steering": steering,
                **obs,
            }
        )
        if self.verbose:
            print(
                f"[t={self.tick:3d} {self.sim_time:6.1f}s] thr={throttle:+.2f} str={steering:+.2f} "
                f"v={self.car.speed:5.1f} wp={self.next_gate:2d} {' | '.join(events)}"
            )
        return obs

    # -------------------------------------------------------------- sensing
    def sensors(self) -> dict:
        car = self.car
        pos = car.position
        rays = {}
        for deg in self.cfg.ray_angles_deg:
            # negative deg = left = CCW = heading + |deg|
            d = geo.raycast(pos, car.heading - math.radians(deg), self.track.wall_segments, self.cfg.ray_max_distance)
            rays[_ray_name(deg)] = round(d, 1)
        target = self.track.gate_centers[self.next_gate]
        vec = target - pos
        after = self.track.gate_centers[(self.next_gate + 1) % self.track.n_gates]
        vec2 = after - pos
        return {
            "speed": round(car.speed, 2),
            **rays,
            "distance_to_next_waypoint": round(float(np.linalg.norm(vec)), 1),
            "angle_to_next_waypoint": round(-math.degrees(geo.signed_angle_to(car.heading, vec)), 1),
            "distance_to_waypoint_after_next": round(float(np.linalg.norm(vec2)), 1),
            "angle_to_waypoint_after_next": round(-math.degrees(geo.signed_angle_to(car.heading, vec2)), 1),
            "next_waypoint_index": int(self.next_gate),
            "waypoints_passed": int(self.gates_passed),
            "waypoints_total": int(self.track.n_gates),
            "position": [round(car.x, 2), round(car.y, 2)],
            "heading_deg": round(math.degrees(geo.wrap_angle(car.heading)), 1),
            "sim_time": round(self.sim_time, 2),
            "penalty_time": round(self.penalty_time, 1),
            "crashes": int(self.crashes),
            "tick": int(self.tick),
            "ticks_remaining": int(max(0, self.cfg.max_ticks - self.tick)),
            "lap_complete": bool(self.lap_complete),
            "attempt_over": bool(self.done),
        }

    def track_map(self) -> dict:
        m = self.track.to_map()
        m["physics"] = {
            "tick_seconds": self.cfg.tick_seconds,
            "max_speed": self.cfg.max_speed,
            "accel": self.cfg.accel,
            "brake": self.cfg.brake,
            "max_lateral_accel": self.cfg.max_lateral_accel,
            "max_yaw_rate_rad_s": self.cfg.max_yaw_rate,
            "steering_model": (
                "yaw_rate = steering * min(max_yaw_rate, max_lateral_accel / speed); "
                "turn radius = speed^2 / (max_lateral_accel * |steering|)"
            ),
            "crash_penalty_seconds": self.cfg.crash_penalty_seconds,
            "max_ticks_per_attempt": self.cfg.max_ticks,
        }
        return m

    # ------------------------------------------------------------- results
    @property
    def lap_time(self) -> float:
        return self.sim_time + self.penalty_time

    @property
    def best_lap(self) -> float | None:
        times = [a.lap_time for a in self.attempts if a.completed and a.lap_time is not None]
        return min(times) if times else None

    def _finish_attempt(self, reason: str) -> None:
        self.done = True
        self.ended_by = reason
        res = AttemptResult(
            attempt=self.attempt_no,
            completed=self.lap_complete,
            lap_time=round(self.lap_time, 3) if self.lap_complete else None,
            sim_time=round(self.sim_time, 3),
            penalty_time=round(self.penalty_time, 3),
            crashes=self.crashes,
            waypoints_passed=self.gates_passed,
            ticks=self.tick,
            ended_by=reason,
        )
        self.attempts.append(res)
        self.attempt_traces.append(
            {
                "attempt": self.attempt_no,
                "trajectory": list(self.trajectory),
                "crash_points": list(self.crash_points),
                "result": res,
            }
        )
        self._log({"event": "attempt_end", **res.to_dict()})

    # -------------------------------------------------------------- logging
    def _log(self, record: dict[str, Any]) -> None:
        if self._log_path is None:
            return
        if self._log_fh is None:
            self._log_fh = open(self._log_path, "a", encoding="utf-8")
        self._log_fh.write(json.dumps(record) + "\n")
        self._log_fh.flush()

    @property
    def log_path(self) -> Path | None:
        return self._log_path


def _ray_name(deg: float) -> str:
    if deg == 0:
        return "distance_to_wall_front"
    side = "left" if deg < 0 else "right"
    return f"distance_to_wall_{side}_{abs(int(deg))}"


def _as_float(v: Any) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(f) or math.isinf(f):
        return 0.0
    return f
