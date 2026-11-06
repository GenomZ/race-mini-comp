"""The racing gym: RaceEnv (one car, one or more laps).

    env = RaceEnv(track="sprint", laps=3)
    obs = env.reset()
    while not obs["attempt_over"]:
        obs = env.step(throttle=0.8, steering=-0.1)   # advances 0.5 virtual seconds
    env.attempts[-1]            # AttemptResult: race time, lap times, crashes, ...
    env.attempt_traces[-1]      # {"attempt", "result", "frames", "crash_points"} for viewers/reports

The environment is deterministic: the same controls always give the same lap.
Driver-frame convention everywhere: + = right (steering, ray/waypoint angles, offset).
Every call to `step` can be logged as JSONL (`log_dir=...`).
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
from .track import Track, make_track

OBS_KEYS = (
    "speed", "distance_to_wall_front", "distance_to_wall_left_20", "distance_to_wall_right_20",
    "distance_to_wall_left_45", "distance_to_wall_right_45", "distance_to_wall_left_90",
    "distance_to_wall_right_90", "distance_to_next_waypoint", "angle_to_next_waypoint",
    "distance_to_waypoint_after_next", "angle_to_waypoint_after_next", "next_waypoint_index",
    "waypoints_passed", "waypoints_total", "position", "heading_deg", "offset_from_center", "segment",
    "progress", "lap", "laps_total", "race_progress", "sim_time", "penalty_time", "race_time",
    "crashes", "tick", "ticks_remaining", "finished", "attempt_over", "zone_speed_limit", "phase", "dt",
)  # + "events" (list of strings) on every reset/step result


@dataclass
class AttemptResult:
    attempt: int
    completed: bool                 # all laps finished
    race_time: float | None         # sim time + penalties, None if DNF
    lap_times: list[float]          # race-clock time of every completed lap (penalties included)
    best_lap: float | None
    sim_time: float
    penalty_time: float             # crashes + zone violations
    crashes: int
    waypoints_passed: int
    laps: int                       # laps planned for this attempt
    laps_completed: int
    ticks: int
    ended_by: str                   # "finished" | "max_ticks" | "reset" | "stopped"
    zone_violations: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class RaceEnv:
    def __init__(
        self,
        track: Track | str | None = None,
        config: PhysicsConfig = DEFAULT_CONFIG,
        laps: int = 1,
        log_dir: str | Path | None = None,
        run_name: str | None = None,
        verbose: bool = False,
    ) -> None:
        self.track = make_track(track)
        self.cfg = config
        self.laps = max(1, int(laps))
        self.verbose = verbose
        self.phase = "session"      # set by the runner: "practice" | "qualifying" | "race" | "session"
        self.attempts: list[AttemptResult] = []
        self.attempt_traces: list[dict] = []   # frames + crash points per finished attempt
        self.attempt_no = 0
        self._ray_angles = np.radians(np.asarray(config.ray_angles_deg, dtype=float))
        self._ray_names = [_ray_name(d) for d in config.ray_angles_deg]
        self._log_path: Path | None = None
        if log_dir is not None:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            name = run_name or time.strftime("run_%Y%m%d_%H%M%S")
            self._log_path = Path(log_dir) / f"{name}.jsonl"
        self._log_fh = None
        self._started = False
        self.reset()

    # ------------------------------------------------------------ lifecycle
    def reset(self, laps: int | None = None) -> dict:
        """Put the car on the start line and start a new attempt of `laps` laps.

        Resetting an attempt that has not been driven yet (tick == 0) simply
        re-places the car; it is not recorded as an attempt.
        """
        if self._started and not self.done and self.tick > 0:
            self._finish_attempt("reset")
        if not self._started or self.done or self.tick > 0:
            self.attempt_no += 1
        self._started = True
        if laps is not None:
            self.laps = max(1, int(laps))
        sx, sy = self.track.start_position
        self.car = CarState(float(sx), float(sy), self.track.start_heading, 0.0)
        self.sim_time = 0.0
        self.penalty_time = 0.0
        self.crashes = 0
        self.zone_violations = 0
        self.tick = 0
        self.next_gate = 1 % self.track.n_gates
        self.gates_passed = 0
        self.laps_completed = 0
        self.lap_times: list[float] = []
        self._lap_start_clock = 0.0
        self.finished = False
        self.done = False
        self.ended_by: str | None = None
        self.crash_points: list[tuple[float, float]] = []
        self.last_controls = (0.0, 0.0)
        self._update_position_info()
        self.frames: list[dict] = [self._frame([], False)]
        self._log({"event": "reset", "attempt": self.attempt_no, "laps": self.laps})
        obs = self.sensors()
        obs["events"] = []
        return obs

    def stop(self) -> dict:
        """End the current attempt now (ended_by="stopped"), e.g. on a UI stop button or a timeout."""
        events = []
        if self._started and not self.done:
            self._finish_attempt("stopped")
            events.append("attempt_over: stopped")
        obs = self.sensors()
        obs["events"] = events
        return obs

    def close(self) -> None:
        if self._started and not self.done:
            self._finish_attempt("stopped")
        if self._log_fh:
            self._log_fh.close()
            self._log_fh = None

    # ------------------------------------------------------------- stepping
    def step(self, throttle: float, steering: float) -> dict:
        """Apply controls for one tick (tick_seconds) and return the new observation."""
        if self.done:
            obs = self.sensors()
            obs["events"] = ["attempt_over: call reset to start a new attempt"]
            return obs

        cfg, track = self.cfg, self.track
        throttle = clamp(_as_float(throttle), -1.0, 1.0)
        steering = clamp(_as_float(steering), -1.0, 1.0)
        self.last_controls = (throttle, steering)
        events: list[str] = []
        n_sub = max(1, int(round(cfg.tick_seconds / cfg.substep_seconds)))
        dt = cfg.tick_seconds / n_sub

        # 1) Integrate the whole tick as if nothing were in the way, then find the first
        #    wall hit with ONE vectorised test of all substep moves against nearby walls.
        states = [self.car]
        for _ in range(n_sub):
            states.append(integrate(states[-1], throttle, steering, cfg, dt))
        path = np.array([(s.x, s.y) for s in states])
        walls = track.walls_near(path.min(axis=0), path.max(axis=0))
        hit = geo.first_hit(path, walls) if len(walls) else None
        crash_k = hit[0] if hit is not None else n_sub

        # 2) Replay the substeps in order: gates, laps, the crash.
        crashed = False
        final_lap_done = False
        for k in range(n_sub):
            if k == crash_k:
                crashed = True
                self._crash(states[k], hit[2], events)
            elif not crashed:
                prev, new = states[k], states[k + 1]
                self.car = new
                gate = track.gates[self.next_gate]
                if geo.crosses_xy(prev.x, prev.y, new.x, new.y, gate):
                    final_lap_done = self._pass_gate(events, self.sim_time + dt)
            # after a crash the car stands still (speed 0) for the rest of the tick
            self.sim_time += dt
            if final_lap_done:
                break

        # 3) Speed-limit zones: judged on where the tick ENDS (a tick that crosses gate 0 is judged in
        #    the new lap and booked to it). Not after the finish line: that lap is never driven.
        self._update_position_info()
        if not final_lap_done:
            self._check_zone(events)
        if final_lap_done:
            self._complete_lap(events, self.race_time, final=True)

        self.tick += 1
        if self.finished:
            events.append(
                f"finished: {self.laps} lap{'s' if self.laps != 1 else ''} in {self.race_time:.2f}s "
                f"(incl. {self.penalty_time:g}s penalties)"
            )
        self.frames.append(self._frame(events, crashed))
        if self.finished:
            self._finish_attempt("finished")
        elif self.tick >= self.max_ticks_total:
            events.append("attempt_over: max ticks reached (DNF)")
            self._finish_attempt("max_ticks")

        obs = self.sensors()
        obs["events"] = events
        self._log({"event": "step", "attempt": self.attempt_no, "throttle": throttle, "steering": steering, **obs})
        if self.verbose:
            print(
                f"[t={self.tick:3d} {self.sim_time:6.1f}s] thr={throttle:+.2f} str={steering:+.2f} "
                f"v={self.car.speed:5.1f} lap={obs['lap']} wp={self.next_gate:2d} {' | '.join(events)}"
            )
        return obs

    def _crash(self, prev: CarState, contact: np.ndarray, events: list[str]) -> None:
        """A wall hit during the move that starts at `prev`: penalty, speed 0, put the car back."""
        cfg = self.cfg
        self.crashes += 1
        self.penalty_time += cfg.crash_penalty_seconds
        self.crash_points.append((float(contact[0]), float(contact[1])))
        msg = f"crash: hit a wall, +{cfg.crash_penalty_seconds:g}s penalty, speed reset to 0"
        if cfg.crash_rule == "marshal":
            # Marshal rule: back onto the centerline, pointing along the track. Stays inside
            # the current waypoint segment so no gate is skipped or counted twice.
            pos, heading = self.track.respawn_pose(prev.position, self.segment)
            self.car = CarState(float(pos[0]), float(pos[1]), heading, 0.0)
            msg += ", marshals put the car back on the centerline"
        else:
            # Classic rule: stop and nudge back from the wall, heading unchanged.
            p0 = prev.position
            back = p0 - geo.unit(prev.heading) * cfg.crash_pushback
            if not geo.segment_hits(p0, back, self.track.wall_segments).any():
                p0 = back
            self.car = CarState(float(p0[0]), float(p0[1]), prev.heading, 0.0)
        events.append(msg)

    def _pass_gate(self, events: list[str], clock_sim: float) -> bool:
        """Count the gate just crossed. Returns True when that completed the LAST lap."""
        g = self.next_gate
        self.gates_passed += 1
        events.append(f"waypoint_passed: {g}")
        self.next_gate = (g + 1) % self.track.n_gates
        if g != 0:
            return False
        if self.laps_completed + 1 >= self.laps:
            return True                              # finished; lap time booked after the zone check
        self._complete_lap(events, clock_sim + self.penalty_time, final=False)
        return False

    def _complete_lap(self, events: list[str], clock: float, final: bool) -> None:
        lap_time = clock - self._lap_start_clock
        self.lap_times.append(lap_time)
        self.laps_completed += 1
        self._lap_start_clock = clock
        events.append(f"lap_complete: lap {self.laps_completed} in {lap_time:.2f}s")
        if final:
            self.finished = True
            self._update_position_info()

    def _check_zone(self, events: list[str]) -> None:
        zone = self.track.zone_for_segment(self.segment)
        if zone is None or self.car.speed <= zone.max_speed + self.cfg.zone_speed_tolerance:
            return
        self.zone_violations += 1
        self.penalty_time += self.cfg.zone_penalty_seconds
        events.append(
            f"zone_violation: speed {self.car.speed:.1f} > limit {zone.max_speed:g} ({zone.label}, "
            f"waypoints {zone.from_wp}-{zone.to_wp}), +{self.cfg.zone_penalty_seconds:g}s penalty"
        )

    # ---------------------------------------------------- position on track
    @property
    def segment(self) -> int:
        """Waypoint segment the car is on (the gate it passed last)."""
        return (self.next_gate - 1) % self.track.n_gates

    @property
    def lap(self) -> int:
        """1-based current lap (stays at laps after finishing)."""
        return min(self.laps_completed + 1, self.laps)

    def _update_position_info(self) -> None:
        """Projection onto the centerline -> offset and lap progress.

        Progress is measured inside the current waypoint segment and clamped to it, so it
        is 0 on the start grid (the car starts just behind gate 0), grows along the lap and
        never jumps ahead of or behind the gates actually passed.
        """
        tr = self.track
        s, offset, index = tr.project(self.car.position)
        self.offset = offset
        self.center_index = index
        if self.finished:
            self.progress = 1.0
            return
        seg = self.segment
        a = tr.gate_arc(seg)
        b = tr.length if seg + 1 == tr.n_gates else tr.gate_arc(seg + 1)
        ds = (s - a + tr.length / 2) % tr.length - tr.length / 2
        self.progress = (a + clamp(ds, 0.0, b - a)) / tr.length

    @property
    def race_progress(self) -> float:
        if self.finished:
            return float(self.laps)
        return self.laps_completed + self.progress

    # -------------------------------------------------------------- sensing
    def sensors(self) -> dict:
        car, tr = self.car, self.track
        pos = car.position
        # ray angle: + = right of heading = clockwise = heading - angle
        dists = tr.raycast(pos, car.heading - self._ray_angles, self.cfg.ray_max_distance)
        vec = tr.gate_centers[self.next_gate] - pos
        vec2 = tr.gate_centers[(self.next_gate + 1) % tr.n_gates] - pos
        zone = tr.zone_for_segment(self.segment)
        obs = {"speed": round(car.speed, 2)}
        obs.update({name: round(float(d), 1) for name, d in zip(self._ray_names, dists)})
        obs.update({
            "distance_to_next_waypoint": round(math.hypot(vec[0], vec[1]), 1),
            "angle_to_next_waypoint": round(-math.degrees(geo.signed_angle_to(car.heading, vec)), 1),
            "distance_to_waypoint_after_next": round(math.hypot(vec2[0], vec2[1]), 1),
            "angle_to_waypoint_after_next": round(-math.degrees(geo.signed_angle_to(car.heading, vec2)), 1),
            "next_waypoint_index": int(self.next_gate),
            "waypoints_passed": int(self.gates_passed),
            "waypoints_total": int(tr.n_gates),
            "position": [round(car.x, 2), round(car.y, 2)],
            "heading_deg": round(math.degrees(geo.wrap_angle(car.heading)), 1),
            "offset_from_center": round(self.offset, 2),
            "segment": int(self.segment),
            "progress": round(self.progress, 4),
            "lap": int(self.lap),
            "laps_total": int(self.laps),
            "race_progress": round(self.race_progress, 4),
            "sim_time": round(self.sim_time, 2),
            "penalty_time": round(self.penalty_time, 1),
            "race_time": round(self.race_time, 2),
            "crashes": int(self.crashes),
            "tick": int(self.tick),
            "ticks_remaining": int(max(0, self.max_ticks_total - self.tick)),
            "finished": bool(self.finished),
            "attempt_over": bool(self.done),
            "zone_speed_limit": None if zone is None else float(zone.max_speed),
            "phase": self.phase,
            "dt": self.cfg.tick_seconds,
        })
        return obs

    def track_map(self) -> dict:
        """track.to_map() + the physics constants + the number of laps (what plan() receives)."""
        m = self.track.to_map(self.cfg)
        m["physics"] = {
            "tick_seconds": self.cfg.tick_seconds,
            "max_speed": self.cfg.max_speed,
            "accel": self.cfg.accel,
            "brake": self.cfg.brake,
            "drag": self.cfg.drag,
            "max_lateral_accel": self.cfg.max_lateral_accel,
            "max_yaw_rate_rad_s": self.cfg.max_yaw_rate,
            "steering_model": (
                "yaw_rate = steering * min(max_yaw_rate, max_lateral_accel / speed); "
                "turn radius = speed^2 / (max_lateral_accel * |steering|); steering + = right"
            ),
            "crash_penalty_seconds": self.cfg.crash_penalty_seconds,
            "crash_rule": self.cfg.crash_rule,
            "zone_penalty_seconds": self.cfg.zone_penalty_seconds,
            "max_ticks_per_lap": self.cfg.max_ticks,
            "ray_angles_deg": list(self.cfg.ray_angles_deg),
            "ray_max_distance": self.cfg.ray_max_distance,
        }
        m["laps"] = self.laps
        return m

    # ------------------------------------------------------------- results
    @property
    def max_ticks_total(self) -> int:
        return self.cfg.max_ticks * self.laps

    @property
    def race_time(self) -> float:
        """Race clock = sim time + all penalties so far."""
        return self.sim_time + self.penalty_time

    @property
    def lap_time(self) -> float:
        """Legacy name of race_time."""
        return self.race_time

    @property
    def best_lap(self) -> float | None:
        """Fastest completed lap over all recorded attempts (None if no lap was completed)."""
        times = [t for a in self.attempts for t in a.lap_times]
        return min(times) if times else None

    def _frame(self, events: list[str], crashed: bool) -> dict:
        car = self.car
        throttle, steering = self.last_controls if self.tick > 0 else (0.0, 0.0)
        return {
            "tick": self.tick,
            "t": self.race_time,
            "sim_time": self.sim_time,
            "x": car.x,
            "y": car.y,
            "heading_deg": math.degrees(geo.wrap_angle(car.heading)),
            "speed": car.speed,
            "throttle": throttle,
            "steering": steering,
            "lap": self.lap,
            "segment": self.segment,
            "progress": self.progress,
            "race_progress": self.race_progress,
            "offset_from_center": self.offset,
            "events": list(events),
            "crashed": bool(crashed),
            "penalty_time": self.penalty_time,
            "zone_violation": any(e.startswith("zone_violation") for e in events),
        }

    def _finish_attempt(self, reason: str) -> None:
        self.done = True
        self.ended_by = reason
        res = AttemptResult(
            attempt=self.attempt_no,
            completed=self.finished,
            race_time=round(self.race_time, 3) if self.finished else None,
            lap_times=[round(t, 3) for t in self.lap_times],
            best_lap=round(min(self.lap_times), 3) if self.lap_times else None,
            sim_time=round(self.sim_time, 3),
            penalty_time=round(self.penalty_time, 3),
            crashes=self.crashes,
            waypoints_passed=self.gates_passed,
            laps=self.laps,
            laps_completed=self.laps_completed,
            ticks=self.tick,
            ended_by=reason,
            zone_violations=self.zone_violations,
        )
        self.attempts.append(res)
        self.attempt_traces.append(
            {
                "attempt": self.attempt_no,
                "result": res,
                "frames": self.frames,
                "crash_points": list(self.crash_points),
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
