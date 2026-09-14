"""Rule-based reference driver (organiser sanity check - NOT an agentic solution).

Used to prove the track is drivable and to give organisers a reference lap time.
Students may read it: it is the shape of a good *execution layer*.

    plan (once)   : get_track_map -> spline through the waypoint centers ->
                    curvature -> speed profile with a backward braking pass
    execute (tick): pure pursuit on the planned path + the speed profile +
                    ray-based wall safety, one `drive` call per tick

It uses nothing but the four tools (position/heading come from the sensors).
"""

from __future__ import annotations

import json
import math

import numpy as np

from .. import geometry as geo
from ..config import PhysicsConfig
from ..physics import available_yaw_rate, clamp
from ..tools import tools_by_name


class ReflexDriver:
    def __init__(self, track_map: dict, cfg: PhysicsConfig, aggression: float = 1.0, spacing: float = 3.0) -> None:
        self.cfg = cfg
        self.aggression = aggression
        centers = np.array([w["center"] for w in track_map["waypoints"]], dtype=float)
        spline = geo.catmull_rom_closed(centers, samples_per_segment=25)
        n = max(32, int(geo.polyline_length(spline) / spacing))
        self.path = geo.resample_closed(spline, n)                     # (n, 2)
        self.ds = geo.polyline_length(self.path) / n
        self.v_profile = self.build_speed_profile(self.path, self.ds, cfg, aggression)
        self.idx = 0
        self.recover_ticks = 0

    # ------------------------------------------------------------ planning
    @staticmethod
    def build_speed_profile(path: np.ndarray, ds: float, cfg: PhysicsConfig, aggression: float) -> np.ndarray:
        radius = geo.curvature_radius_closed(path)
        v = np.minimum(cfg.max_speed, np.sqrt(cfg.max_lateral_accel * np.minimum(radius, 1e6))) * aggression
        v = np.clip(v, 5.0, cfg.max_speed)
        n = len(v)
        for _ in range(2):                       # backward braking pass, twice for the wrap-around
            for i in reversed(range(n)):
                j = (i + 1) % n
                v[i] = min(v[i], math.sqrt(v[j] ** 2 + 2.0 * cfg.brake * ds))
        return v

    # ------------------------------------------------------------ execution
    def decide(self, obs: dict) -> tuple[float, float]:
        cfg = self.cfg
        speed = obs["speed"]
        pos = np.array(obs["position"], dtype=float)
        heading = math.radians(obs["heading_deg"])
        front = obs["distance_to_wall_front"]
        l20, r20 = obs["distance_to_wall_left_20"], obs["distance_to_wall_right_20"]
        l45, r45 = obs["distance_to_wall_left_45"], obs["distance_to_wall_right_45"]

        # --- recovery: stopped against a wall -> reverse towards the open side ---
        if self.recover_ticks > 0:
            self.recover_ticks -= 1
            return -1.0, (-1.0 if l45 > r45 else 1.0)
        if speed < 1.5 and min(front, l20, r20) < 4.0:
            self.recover_ticks = 1
            return -1.0, (-1.0 if l45 > r45 else 1.0)

        # --- localise on the planned path -----------------------------------------
        n = len(self.path)
        window = np.arange(self.idx - 5, self.idx + 25) % n
        d = np.linalg.norm(self.path[window] - pos, axis=1)
        if d.min() > 30.0:                        # lost (e.g. after a crash): full search
            window = np.arange(n)
            d = np.linalg.norm(self.path - pos, axis=1)
        self.idx = int(window[int(np.argmin(d))])

        # --- pure pursuit steering -------------------------------------------------
        lookahead = clamp(6.0 + 0.45 * abs(speed), 8.0, 22.0)
        target = self.path[(self.idx + int(round(lookahead / self.ds))) % n]
        vec = target - pos
        alpha = geo.signed_angle_to(heading, vec)              # +CCW = left
        L = float(np.linalg.norm(vec))
        curvature = 2.0 * math.sin(alpha) / max(L, 1e-6)
        yaw_needed = max(abs(speed), 4.0) * curvature          # rad/s, +left
        steer = -yaw_needed / available_yaw_rate(max(abs(speed), 4.0), cfg)   # tool: +right
        steer += 0.5 * max(0.0, 1.0 - l20 / 7.0) + 0.3 * max(0.0, 1.0 - l45 / 5.0)
        steer -= 0.5 * max(0.0, 1.0 - r20 / 7.0) + 0.3 * max(0.0, 1.0 - r45 / 5.0)
        steer = clamp(steer, -1.0, 1.0)

        # --- speed -------------------------------------------------------------------
        ahead = int(round(max(abs(speed), 5.0) * cfg.tick_seconds / self.ds))
        v_plan = float(np.min(self.v_profile[(self.idx + np.arange(0, ahead + 2)) % n]))
        margin = 4.0 + 1.5 * speed * cfg.tick_seconds
        v_front = math.sqrt(2.0 * cfg.brake * max(front - margin, 0.0))
        v_side = cfg.max_speed
        if min(l45, r45) < 5.0:
            v_side = 14.0
        if min(l20, r20) < 5.0:
            v_side = 10.0
        target_speed = clamp(min(v_plan, v_front, v_side), 5.0, cfg.max_speed)
        throttle = clamp((target_speed - speed) / (cfg.accel * cfg.tick_seconds), -1.0, 1.0)
        return throttle, steer


def run(env, tools, attempts: int = 1, verbose: bool = True, aggression: float = 1.0, **_) -> None:
    by = tools_by_name(tools)
    track_map = json.loads(by["get_track_map"].invoke({}))
    for _ in range(attempts):
        driver = ReflexDriver(track_map, env.cfg, aggression)
        obs = json.loads(by["reset_environment"].invoke({}))
        while not obs["attempt_over"]:
            throttle, steer = driver.decide(obs)
            obs = json.loads(by["drive"].invoke({"throttle": throttle, "steering": steer}))
            if verbose and obs["events"]:
                print(f"tick {obs['tick']:3d} t={obs['sim_time']:6.1f}s v={obs['speed']:5.1f} -> {obs['events']}")
