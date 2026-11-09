"""L3 Learner - a learning agent: drive, read the lap report, change ONE thing, keep what works.

Agent type (Russell & Norvig): LEARNING AGENT on top of the L2 planner.
    performance element = drive()            (the L2 planner + PathFollower)
    critic              = the lap report     (lap time, crashes, per-segment table)
    learning element    = reflect()          (keep-best, revert, update the knobs)
    problem generator   = one experiment per attempt (push the corners / trust the walls / ...)
Agentic concepts: MEMORY (self.history = episodic memory, self.best = what worked) and REFLECTION.

    plan(track) ──> knobs v1 ──> drive (attempt 1) ──> report ──┐
                      ^                                         │ reflect():
                      │   crash?  slow that segment + the one   │  - better than best? keep it
                      │           before (x0.85), more margin   │  - worse? revert to best,
                      │   clean?  next experiment from the best │    halve that experiment's step
                      └─────────── knobs v2, v3, ... <──────────┘  - next phase is qualifying/race?
                                                                     lock in the best and stop

How it works
- Knobs: a speed factor per waypoint segment (x the physics corner speed) and one "safety margin"
  (how much space the executor keeps free in front of the car; less margin = later braking).
- After every attempt: a crash slows down the crash segment and the one before it (x0.85) and adds
  margin; a clean lap is compared with the best lap so far (keep best / revert to best).
- Problem generator: three experiments in turn - trust the walls (less margin), push every
  segment, push the 4 slowest segments. An experiment that made things worse gets a smaller step next time
  (the sim is deterministic: repeating the same change would give the same bad lap).
- Lock-in: when next_phase is "qualifying" or "race", stop experimenting and drive the best knobs.
- No per-track numbers anywhere: it learns on whatever track it is given.

Run it:
    python race.py drive drivers/examples/l3_learner.py --attempts 4
    python race.py drive drivers/examples/l3_learner.py --track random:7 --attempts 6
    python race.py drive drivers/examples/l3_learner.py --attempts 6 --param racing_line=1   # instructor's best

Expected (0 crashes, a learning curve): grand_prix 52.5 -> 50.55 -> 49.3 -> 48.3 s in 4 attempts
(L2 stays at 52.5); with racing_line=1: 49.95 -> 46.3 -> 45.45 -> 44.95 s, 43.2 s after 6.

Try this
1. Remove the keep-best/revert lines: does the curve still go down, or does it wander?
2. --param crash_cut=1.0 (crashes are not punished) with --param push=1.3: watch it get greedy.
3. Write a rule that stops "the same crash twice in a row" (hint: every self.history entry has
   "crash_segments", the segments where that attempt crashed).
4. Add a 4th experiment to EXPERIMENTS (e.g. a longer pure-pursuit lookahead) and see if it helps.
"""

from __future__ import annotations

import copy

import numpy as np

from agentic_gp import driverkit as dk


class Driver:
    NAME = "Learner"
    AUTHOR = "Agentic GP"
    COLOR = "#e9c46a"
    KIND = "code"
    MOTTO = "Every lap teaches me something."
    ANATOMY = ["perceive", "act", "map", "plan", "memory", "reflect"]
    PARAMS = {
        "racing_line": 0,         # 0 = learn on the centerline (L2), 1 = on the racing line (L2b)
        "push": 1.10,             # "push" experiment: x speed factor on the segments it touches
        "margin_step": 1.0,       # "trust the walls" experiment: remove this much safety margin
        "start_margin": 3.0,      # safety margin of attempt 1 (driverkit default)
        "min_margin": 0.8,        # never brake later than this
        "crash_cut": 0.85,        # a crash slows that segment and the one before by this factor
        "max_factor": 1.6,        # speed factors stay within [0.5, max_factor]
    }
    EXPERIMENTS = ("trust walls", "push all", "push slowest")

    def __init__(self) -> None:
        self.follower: dk.PathFollower | None = None
        self.history: list[dict] = []        # episodic memory: one entry per attempt
        self.say_next: str | None = None

    # ------------------------------------------------------------------ plan: once per session
    def plan(self, track: dict) -> None:
        P = self.PARAMS
        self.phys = dk.physics_of(track)
        self.points = (dk.racing_line(track, 3.0) if P["racing_line"]
                       else np.asarray(track["centerline"], dtype=float))
        radii = dk.curvature_radii(self.points)
        self.grip_speed = np.minimum(self.phys["max_speed"],
                                     np.sqrt(self.phys["max_lateral_accel"] * np.minimum(radii, 1e9)))
        self.segment_of = dk.segment_ids(len(self.points), dk.waypoint_indices(track, self.points))
        n = len(track["waypoints"])
        self.knobs = {"factor": [1.0] * n, "margin": P["start_margin"]}
        self.steps = {"push all": P["push"] - 1.0, "trust walls": P["margin_step"], "push slowest": P["push"] - 1.0}
        self.best: dict | None = None        # {"knobs", "time", "attempt"}
        self.next_experiment = 0
        self.locked = False
        self.experiment = "baseline"
        self._build()
        self.say_next = f"Learner: baseline plan, margin {self.knobs['margin']:.1f}"

    def _build(self) -> None:
        """Knobs -> speed profile -> executor."""
        factor = np.array(self.knobs["factor"])[self.segment_of]
        v = np.clip(self.grip_speed * factor, 5.0, self.phys["max_speed"])
        v = dk.braking_pass(v, self.points, self.phys["brake"])
        margin = self.knobs["margin"]
        self.follower = dk.PathFollower(self.points, v, physics=self.phys, segments=self.segment_of,
                                        safety_margin=margin, blind_factor=min(1.0, margin / 3.0))

    # ------------------------------------------------------------------ drive: every tick
    def drive(self, obs: dict) -> dict:
        if self.follower is None:
            return {"throttle": 0.3, "steering": dk.clamp(obs["angle_to_next_waypoint"] / 30.0)}
        throttle, steering = self.follower.control(obs)
        say, self.say_next = self.say_next, None
        return {"throttle": throttle, "steering": steering, "say": say}

    # ------------------------------------------------------------------ reflect: after every attempt
    def reflect(self, report: dict) -> None:
        P = self.PARAMS
        time = report["race_time"] if report["completed"] else None
        self.history.append({"attempt": report["attempt"], "experiment": self.experiment,
                             "knobs": copy.deepcopy(self.knobs), "time": time, "crashes": report["crashes"],
                             "crash_segments": [c["segment"] for c in report.get("crash_log") or []]})

        # 1. CRITIC: is this the best lap so far? (a DNF never is)
        improved = time is not None and (self.best is None or time < self.best["time"])
        if improved:
            self.best = {"knobs": copy.deepcopy(self.knobs), "time": time, "attempt": report["attempt"]}
        elif self.experiment in self.steps:
            self.steps[self.experiment] *= 0.5            # that experiment hurt: smaller step next time

        # 2. LOCK-IN: qualifying / race next -> no more experiments, drive the best we know.
        if self.locked or report.get("next_phase") in ("qualifying", "race"):
            self.locked = True
            if self.best:
                self.knobs = copy.deepcopy(self.best["knobs"])
                self._build()
                self.say_next = f"Locked in: best plan from attempt {self.best['attempt']} ({self.best['time']:.2f} s)"
            return

        # 3. LEARN: start from the best knobs (revert), then react to crashes or run the next experiment.
        knobs = copy.deepcopy(self.best["knobs"] if self.best else self.knobs)
        n = len(knobs["factor"])
        if report["crashes"]:
            knobs = copy.deepcopy(self.knobs)             # fix the plan that crashed, segment by segment
            for crash in report["crash_log"]:
                for seg in (crash["segment"], (crash["segment"] - 1) % n):
                    knobs["factor"][seg] = max(0.5, knobs["factor"][seg] * P["crash_cut"])
            knobs["margin"] = min(P["start_margin"], knobs["margin"] + P["margin_step"])
            self.experiment = "crash fix"
            note = f"{report['crashes']} crash(es): slowing segments " + \
                   ", ".join(str(c["segment"]) for c in report["crash_log"][:4])
        else:
            self.experiment = self.EXPERIMENTS[self.next_experiment % len(self.EXPERIMENTS)]
            self.next_experiment += 1
            step = self.steps[self.experiment]
            if self.experiment == "push all":
                knobs["factor"] = [min(P["max_factor"], f * (1 + step)) for f in knobs["factor"]]
            elif self.experiment == "trust walls":
                knobs["margin"] = max(P["min_margin"], knobs["margin"] - step)
            else:                                          # push the 4 slowest segments of this lap
                slow = sorted(report["segments"], key=lambda s: s["time"], reverse=True)[:4]
                for s in slow:
                    knobs["factor"][s["segment"]] = min(P["max_factor"], knobs["factor"][s["segment"]] * (1 + step))
            if time is None:                               # no lap time (stalled, out of ticks, ...)
                result = f"did not finish ({report.get('ended_by') or 'no lap'})"
            elif improved:
                result = f"{time:.2f} s (new best)"
            else:
                result = f"{time:.2f} s (slower than best {self.best['time']:.2f}, reverted)"
            if time is None and self.best:
                result += f", reverted to the best ({self.best['time']:.2f} s)"
            note = f"{result}; next: {self.experiment}"
        self.knobs = knobs
        self._build()
        self.say_next = note
