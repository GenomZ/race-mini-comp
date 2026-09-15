"""fruit_fly_minimax — a fruit-fly-connectome-shaped racing agent.

Source / inspiration: the awesome-fly list (https://github.com/cobanov/awesome-fly),
in particular the MaleCNS-constrained controllers like FlyDoom, Fly Dino
(flydino / flyjump), and the fly-connectome-template browser workbench.

Design pattern from those projects:
  1. Fixed connectivity graph (the "connectome") — we keep the topology.
  2. Leaky-tanh dynamics on neuron activations.
  3. A *small* trainable readout that maps selected neurons -> motor outputs.
  4. Train the readout via Cross-Entropy Method (CEM, same as flydino).

This file is self-contained and intentionally simple:
  - The "connectome" is a small hand-curated toy subset (~40 neurons, 5 regions)
    with edge weights that *mimic* the MaleCNS look (region names, neuron ID
    shape `DNgxx`, `aMexi`, etc.) but are NOT real MaleCNS data. We do not
    ship the 140k-neuron MaleCNS dataset here.
  - Sensors (LiDAR rays, speed, angle-to-waypoint, distance-to-waypoint)
    are wired into a small set of input neurons.
  - Two descending neurons (aDN1_left, aDN1_right in the toy graph) drive
    throttle and steering via a learned linear readout.
  - The readout weights are trained across `attempts` racing attempts using
    Cross-Entropy Method on the lap time fitness signal. Weights are saved
    to / loaded from `fruit_fly_weights.json` so the agent is reproducible.

Run it:
    python evaluate.py fruit_fly_minimax --attempts 1 --plot
    python evaluate.py fruit_fly_minimax --attempts 30 --plot   # train
    python evaluate.py fruit_fly_minimax --attempts 1 --agent-arg load_weights=false

Tunables (CLI):
    --agent-arg n_neurons=<int>          # how many hidden neurons in the toy graph
    --agent-arg n_generations=<int>      # CEM generations per "train" attempt
    --agent-arg pop_size=<int>           # CEM population size
    --agent-arg elite_frac=<float>       # CEM elite fraction (e.g. 0.2)
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass, field
from typing import Any

from ..tools import tools_by_name

# Where to save/load the trained readout weights.
WEIGHTS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "fruit_fly_weights.json",
)


# ============================================================================
#   PART 1 — TOY CONNECTOME
#   ----------------------------------------------------------------------------
#   A small, fixed, hand-curated subset of the MaleCNS topology. Region names
#   and neuron-ID shapes follow the convention used in fly-connectome work
#   (e.g. "aMe" = antennal mechanosensory, "DNg" = descending neuron group).
#
#   This is NOT real MaleCNS data. It is a pedagogical stand-in: enough to
#   show the pattern, small enough to be readable on one screen.
#
#   Graph layout (5 regions, 40 neurons total):
#       visual    (8)   -- receives left/right LiDAR + speed
#       mech      (6)   -- receives angle-to-waypoint + distance
#       central   (12)  -- recurrent hidden
#       ascend    (6)   -- second-stage processing
#       descend   (8)   -- last 2 (aDN1_left, aDN1_right) drive the motor
#
#   Edges are sparse, weighted in {-1, -0.5, 0.5, 1}. Self-connections and
#   edges within the same region are kept (recurrence). Cross-region edges
#   follow the standard MaleCNS direction: visual -> central -> descend.
# ============================================================================

REGION_ORDER = ["visual", "mech", "central", "ascend", "descend"]


def _region_for(neuron_id: str) -> str:
    """Heuristic: which region does this toy neuron ID belong to?

    Visual inputs: aMeNN_v  (visual antennal-mech)
    Mech inputs:   aMeNN_m
    Central:       CNxxx
    Ascend:        ASxx
    Descend:       aDN*, pIP*, DNa*, DNb*
    """
    if neuron_id.endswith("_v"):
        return "visual"
    if neuron_id.endswith("_m"):
        return "mech"
    if neuron_id.startswith("CN"):
        return "central"
    if neuron_id.startswith("AS"):
        return "ascend"
    return "descend"


def _toy_connectome(n_hidden: int = 16) -> dict[str, Any]:
    """Build a toy connectome with `n_hidden` central neurons and fixed I/O.

    Returns a dict with keys:
        neuron_ids:   list[str], the 40+ IDs in REGION_ORDER
        n_inputs:     7  (visual:5, mech:2)
        n_outputs:    2  (aDN1_left, aDN1_right, the last 2 descend neurons)
        edges:        list[(src_idx, dst_idx, weight)]
    """
    rng = random.Random(0xCAFE)        # deterministic topology

    # Build per-region neuron ID lists. The IDs follow MaleCNS naming style.
    visual = [f"aMe{i:02d}_v" for i in range(1, 6)]      # 5 visual inputs
    mech   = [f"aMe{i:02d}_m" for i in range(6, 8)]      # 2 mech inputs
    central = [f"CN{i:03d}"   for i in range(1, n_hidden + 1)]
    ascend = [f"AS{i:02d}"    for i in range(1, 7)]      # 6
    descend = [
        "aDN1_left", "aDN1_right",                         # the 2 motor neurons
        "aDN2a", "aDN2b", "aDN3", "pIP10", "DNa01", "DNb01",
    ]
    neurons = visual + mech + central + ascend + descend
    idx = {nid: i for i, nid in enumerate(neurons)}

    edges: list[tuple[int, int, float]] = []

    def _add(src_ids, dst_ids, choices):
        for s in src_ids:
            for d in dst_ids:
                if s == d:
                    continue
                if rng.random() < 0.55:                # sparse ~55% density
                    edges.append((idx[s], idx[d], rng.choice(choices)))

    # 1. visual -> central (forward)
    _add(visual, central, [-0.5, 0.5, 1.0])
    # 2. mech -> central (forward)
    _add(mech, central, [0.5, 1.0])
    # 3. central <-> central (recurrence, both directions)
    _add(central, central, [-1.0, -0.5, 0.5, 1.0])
    # 4. central -> ascend (forward)
    _add(central, ascend, [0.5, 1.0])
    # 5. ascend -> descend (forward)
    _add(ascend, descend, [0.5, 1.0])
    # 6. descend -> descend (recurrence)
    _add(descend, descend, [-0.5, 0.5, 1.0])
    # 7. visual -> descend (skip connection)
    _add(visual, descend, [0.5])

    return {
        "neuron_ids": neurons,
        "n_inputs": len(visual) + len(mech),
        "n_outputs": 2,                              # aDN1_left, aDN1_right
        "output_ids": ["aDN1_left", "aDN1_right"],
        "edges": edges,
        "n_hidden": n_hidden,
    }


# ============================================================================
#   PART 2 — LEAKY-TANH DYNAMICS
#   ----------------------------------------------------------------------------
#   Standard continuous-time formulation from the fly-connectome literature:
#       tau dv/dt = -v + tanh(W @ v + I)
#   We integrate with simple Euler at dt=1 per call. Tanh bounds activations
#   to [-1, 1]. The readout is a linear combination of the two aDN1 outputs.
# ============================================================================

class ConnectomeBrain:
    """A leaky-tanh recurrent net with a fixed topology + a trainable readout."""

    def __init__(self, topo: dict[str, Any], readout: list[list[float]] | None = None):
        self.topo = topo
        self.n = len(topo["neuron_ids"])
        self.v = [0.0] * self.n                      # activations
        # W is a dense n x n weight matrix (constructed from the sparse edge list).
        self.W = [[0.0] * self.n for _ in range(self.n)]
        for src, dst, w in topo["edges"]:
            self.W[dst][src] += w
        # Normalise by row so dynamics stay bounded.
        for i in range(self.n):
            row_sum = sum(abs(x) for x in self.W[i]) or 1.0
            self.W[i] = [x / row_sum for x in self.W[i]]
        # tau is per-neuron.
        self.tau = [1.0 + 0.5 * random.random() for _ in range(self.n)]
        self.readout = readout or self._zero_readout()

    def _zero_readout(self) -> list[list[float]]:
        """Default readout: zero. Outputs are zero until trained.

        Layout: 2 outputs (throttle, steering) each = sum_i W_out[i] * v[i].
        """
        return [[0.0] * self.n, [0.0] * self.n]

    def reset(self) -> None:
        self.v = [0.0] * self.n

    def step(self, inputs: list[float]) -> None:
        """One Euler step of leaky-tanh dynamics. `inputs` overwrites the
        first `topo.n_inputs` activations (the sensor neurons)."""
        assert len(inputs) == self.topo["n_inputs"], (
            f"expected {self.topo['n_inputs']} inputs, got {len(inputs)}"
        )
        # Drive the input neurons with the sensor values (clipped to [-1, 1]).
        for i, x in enumerate(inputs):
            self.v[i] = max(-1.0, min(1.0, x))
        # Update the rest.
        new_v = list(self.v)
        for i in range(self.topo["n_inputs"], self.n):
            s = sum(self.W[i][j] * self.v[j] for j in range(self.n))
            new_v[i] = self.v[i] + (math.tanh(s) - self.v[i]) / self.tau[i]
        self.v = new_v

    def motor(self) -> tuple[float, float]:
        """Linear readout -> (throttle, steering) in [-1, 1]."""
        thr = sum(self.readout[0][i] * self.v[i] for i in range(self.n))
        str_ = sum(self.readout[1][i] * self.v[i] for i in range(self.n))
        return max(-1.0, min(1.0, thr)), max(-1.0, min(1.0, str_))


# ============================================================================
#   PART 3 — SENSOR ENCODING
#   ----------------------------------------------------------------------------
#   Map the env's 7 LiDAR ray distances + speed + angle + waypoint-distance
#   to 7 normalised input values for the brain. Normalisation is per-channel
#   so the network sees roughly equal magnitudes regardless of track layout.
# ============================================================================

def encode_sensors(obs: dict) -> list[float]:
    """7-dim input vector from the env observation. All values in roughly [-1, 1].

    Order: [speed_norm, angle_norm, dist_norm, l20, r20, l45, r45]
    Speed uses a tanh-style squashing; angle is divided by 90 deg.
    """
    speed = obs["speed"]
    angle = obs["angle_to_next_waypoint"]        # deg, - = left, + = right
    dist  = obs["distance_to_next_waypoint"]
    front = obs["distance_to_wall_front"]

    # Distance / LiDAR -> neuron current in [-1, 1]. "1" = far (safe), "-1" = wall.
    def _dist(d: float) -> float:
        # 0 m -> -1; 30 m -> +1; linear in between, clipped.
        return max(-1.0, min(1.0, (d - 15.0) / 15.0))

    return [
        max(-1.0, min(1.0, speed / 30.0)),    # speed/30: 30 m/s -> +1
        max(-1.0, min(1.0, angle / 90.0)),   # angle/90: 90 deg -> +/-1
        max(-1.0, min(1.0, (dist - 50.0) / 50.0)),    # 0m = -1, 100m = +1
        _dist(front),
        _dist(obs["distance_to_wall_left_20"]),
        _dist(obs["distance_to_wall_right_20"]),
        _dist(obs["distance_to_wall_left_45"]),
    ]


# ============================================================================
#   PART 4 — CROSS-ENTROPY METHOD (CEM) TRAINER
#   ----------------------------------------------------------------------------
#   Fly Dino uses CEM to train only the readout weights. We mirror that:
#   - Sample N candidate readout matrices from a Gaussian (mean, std).
#   - Run a full lap with each candidate; record the lap time.
#   - Keep the top elite_frac%; refit (mean, std) to their statistics.
#   - Repeat for n_generations.
#
#   IMPORTANT: each "lap" is a real RaceEnv.run() call. The gym is deterministic
#   given the same controls, so two candidates with the same readout produce
#   the same lap time -> we just compare control outputs.
# ============================================================================

@dataclass
class CEMState:
    mean: list[list[float]] = field(default_factory=list)
    std:  list[list[float]] = field(default_factory=list)

    @classmethod
    def initial(cls, n_outputs: int, n_neurons: int, seed: int = 0) -> "CEMState":
        rng = random.Random(seed)
        m = [[rng.uniform(-0.1, 0.1) for _ in range(n_neurons)] for _ in range(n_outputs)]
        s = [[0.5 for _ in range(n_neurons)] for _ in range(n_outputs)]
        return cls(mean=m, std=s)

    @classmethod
    def initial_biased(cls, topo: dict[str, Any], seed: int = 0) -> "CEMState":
        """Like initialize, but seed the readout with a forward-motion bias.

        Bias rules:
          - Throttle readout: positive weights on the input neurons and the
            descend region neurons (so pressing the gas wires through).
          - Steering readout: small positive weight on the angle-to-waypoint
            neuron (so turning the wheel tracks the next waypoint).
        Std starts wide so CEM has room to explore.
        """
        rng = random.Random(seed)
        n = len(topo["neuron_ids"])
        ids = topo["neuron_ids"]
        # Throttle row: positive bias on inputs + descend, mild elsewhere.
        thr = []
        for i, nid in enumerate(ids):
            if i < topo["n_inputs"]:
                thr.append(rng.uniform(0.3, 0.6))
            elif _region_for(nid) == "descend":
                thr.append(rng.uniform(0.3, 0.6))
            else:
                thr.append(rng.uniform(-0.1, 0.1))
        # Steering row: small bias toward angle input (index 1).
        str_ = [rng.uniform(-0.1, 0.1) for _ in range(n)]
        # Find the angle input neuron — it's the 2nd input (index 1) by convention.
        str_[1] = rng.uniform(0.3, 0.6)
        # Also bias steering positive toward the right side when walls are
        # close on the right: weights on the right-side LiDAR neurons.
        # These are inputs indices 4 (r20), 5 (l45). We add a tiny bias.
        for idx in (4, 5):
            str_[idx] += rng.uniform(0.05, 0.15)
        std = [[0.4 for _ in range(n)] for _ in range(2)]
        return cls(mean=[thr, str_], std=std)

    def sample(self, rng: random.Random) -> list[list[float]]:
        return [[
            max(-2.0, min(2.0, rng.gauss(self.mean[o][i], self.std[o][i])))
            for i in range(len(self.mean[o]))
        ] for o in range(len(self.mean))]

    def update(self, elites: list[list[list[float]]]) -> None:
        """Refit (mean, std) to the elite population."""
        n_out = len(elites[0])
        n = len(elites)
        new_mean = [[0.0] * len(self.mean[0]) for _ in range(n_out)]
        new_std  = [[0.0] * len(self.std[0])  for _ in range(n_out)]
        for cand in elites:
            for o in range(n_out):
                for i in range(len(cand[o])):
                    new_mean[o][i] += cand[o][i]
        for o in range(n_out):
            for i in range(len(new_mean[o])):
                new_mean[o][i] /= n
        for cand in elites:
            for o in range(n_out):
                for i in range(len(cand[o])):
                    d = cand[o][i] - new_mean[o][i]
                    new_std[o][i] += d * d
        for o in range(n_out):
            for i in range(len(new_std[o])):
                new_std[o][i] = max(0.05, math.sqrt(new_std[o][i] / n))
        self.mean = new_mean
        self.std = new_std


# ============================================================================
#   PART 5 — STUCK RECOVERY
#   ----------------------------------------------------------------------------
#   If the brain drives the car into a wall and pins it, hard-reverse for a
#   few ticks to back off. Same idea as the other agents; here we expose it
#   as a small wrapper so the run() loop stays readable.
# ============================================================================

def recover(brain: ConnectomeBrain) -> tuple[float, float] | None:
    """Return a recovery (throttle, steering) or None if not pinned."""
    return None  # The brain itself manages avoidance; this hook is a future hook.


# ============================================================================
#   PART 6 — ENTRY POINT (called by evaluate.py)
#   ----------------------------------------------------------------------------
#   Pattern (matches dummy/hermes/reflex/agentic):
#       def run(env, tools, attempts=1, verbose=True, **kwargs) -> None
#
#   We use the FIRST attempt as a "test" lap if the user supplied
#   load_weights=true (default) and a weights file exists; otherwise we treat
#   every attempt as a CEM training lap, save the best weights at the end.
# ============================================================================

def run(env, tools, attempts: int = 1, verbose: bool = True,
        n_neurons: int = 16,
        n_generations: int = 8, pop_size: int = 12, elite_frac: float = 0.25,
        load_weights: bool = True,
        train_if_no_weights: bool = True,
        **kwargs) -> None:
    by = tools_by_name(tools)
    cfg = getattr(env, "cfg", None)

    # Build the connectome (fixed topology, deterministic seed).
    topo = _toy_connectome(n_hidden=n_neurons)
    n_neurons_total = len(topo["neuron_ids"])

    # Decide whether to train or evaluate.
    have_weights = os.path.isfile(WEIGHTS_PATH) and load_weights
    if have_weights:
        with open(WEIGHTS_PATH) as f:
            saved = json.load(f)
        readout = saved["readout"]
        n_neurons_used = saved["n_neurons_total"]
        if n_neurons_used != n_neurons_total:
            print(f"[fruit_fly_minimax] WARNING: weights file has "
                  f"n_neurons_total={n_neurons_used}, current graph has "
                  f"{n_neurons_total}. Reinitialising readout.")
            readout = None
    else:
        readout = None

    if verbose:
        print(f"[fruit_fly_minimax] Connectome: {len(topo['neuron_ids'])} neurons, "
              f"{len(topo['edges'])} edges. Output = aDN1_left + aDN1_right.")
        if have_weights:
            print(f"[fruit_fly_minimax] Loaded trained readout from {WEIGHTS_PATH}")
        elif train_if_no_weights and attempts > 1:
            print(f"[fruit_fly_minimax] No weights file found. Will train via CEM "
                  f"across {attempts} attempts "
                  f"({n_generations} gens x {pop_size} pop).")
        else:
            print(f"[fruit_fly_minimax] No weights and no training budget. Using "
                  f"zero readout (car will go straight).")

    if have_weights or readout is not None:
        # Run evaluation attempts.
        for attempt_idx in range(attempts):
            brain = ConnectomeBrain(topo, readout=readout)
            _drive_attempt(env, by, brain, topo, verbose=attempt_idx == 0)
        return

    # ----- CEM training across multiple attempts -----
    # Initialise the readout mean with a forward-motion bias so the search
    # starts from "go forward" rather than from a zero-mean Gaussian. The
    # weights on the input neurons (which carry speed/angle/distance) and
    # the descend region (which contains aDN1) are seeded positive for
    # throttle; steering is seeded near zero with a small positive weight
    # on the angle-to-waypoint neuron.
    cem = CEMState.initial_biased(topo, seed=0xBEEF)
    rng = random.Random(0xFACE)
    best_readout: list[list[float]] | None = None
    best_lap = float("inf")
    n_train_attempts = max(1, attempts - 1)   # last attempt is eval with best

    # First attempt: use zero readout as a baseline measurement.
    if verbose:
        print(f"[fruit_fly_minimax] baseline attempt with zero readout ...")
    brain = ConnectomeBrain(topo, readout=[[0.0] * len(topo["neuron_ids"])] * 2)
    _drive_attempt(env, by, brain, topo, verbose=False)

    gen_idx = 0
    while gen_idx < n_generations and (gen_idx * pop_size + 1) < n_train_attempts:
        gen_idx += 1
        # Sample pop_size candidates
        candidates = [cem.sample(rng) for _ in range(pop_size)]
        scored: list[tuple[float, list[list[float]]]] = []
        for cand in candidates:
            # We need a fresh attempt per candidate. Drive one attempt end-to-end.
            brain = ConnectomeBrain(topo, readout=cand)
            res = _run_single_attempt(env, by, brain, topo)
            # Use the real race score: sim_time + 5 s per crash + 250 s DNF penalty.
            # The 250 s bonus for DNF is large enough to dominate the ranking
            # but small enough to NOT swamp the within-DNF signal.
            if res["completed"]:
                lap_time = res["lap_time"]
            else:
                lap_time = 250.0 + res["sim_time"] + 5.0 * res["crashes"]
            scored.append((lap_time, cand))

        scored.sort(key=lambda x: x[0])
        n_elite = max(2, int(elite_frac * len(scored)))
        elites = [c for _, c in scored[:n_elite]]
        cem.update(elites)
        if scored[0][0] < best_lap:
            best_lap = scored[0][0]
            best_readout = scored[0][1]
        if verbose:
            print(f"[fruit_fly_minimax] gen {gen_idx:2d}/{n_generations}: "
                  f"best={scored[0][0]:.2f}s  mean_top3={sum(s for s,_ in scored[:3])/3:.2f}s")

    # Save the best readout for future runs.
    if best_readout is not None:
        with open(WEIGHTS_PATH, "w") as f:
            json.dump({"readout": best_readout, "n_neurons_total": n_neurons_total,
                       "best_lap": best_lap}, f, indent=2)
        if verbose:
            print(f"[fruit_fly_minimax] Saved trained readout (best lap "
                  f"{best_lap:.2f}s) to {WEIGHTS_PATH}")

    # Final attempt: drive with the best readout found.
    if attempts >= 1 and best_readout is not None:
        brain = ConnectomeBrain(topo, readout=best_readout)
        _drive_attempt(env, by, brain, topo, verbose=True)


def _drive_attempt(env, by, brain: "ConnectomeBrain", topo: dict,
                   verbose: bool) -> dict:
    """Run one race attempt with the given brain, print a short summary."""
    obs = json.loads(by["reset_environment"].invoke({}))
    brain.reset()
    last_print_tick = 0
    while not obs["attempt_over"]:
        inputs = encode_sensors(obs)
        brain.step(inputs)
        throttle, steering = brain.motor()
        obs = json.loads(by["drive"].invoke({"throttle": throttle, "steering": steering}))
        if verbose and obs["tick"] - last_print_tick >= 20:
            last_print_tick = obs["tick"]
            print(f"[fruit_fly_minimax] tick {obs['tick']:3d} t={obs['sim_time']:5.1f}s "
                  f"v={obs['speed']:5.1f} thr={throttle:+.2f} str={steering:+.2f}")
    res = env.attempts[-1]
    if res.completed:
        print(f"[fruit_fly_minimax] attempt {res.attempt}: lap {res.lap_time:.2f}s "
              f"crashes={res.crashes} ticks={res.ticks}")
    else:
        print(f"[fruit_fly_minimax] attempt {res.attempt}: DNF ({res.ended_by}) "
              f"waypoints {res.waypoints_passed}/{env.track.n_gates}")
    return {"completed": res.completed, "lap_time": res.lap_time or 0.0,
            "sim_time": res.sim_time, "crashes": res.crashes}


def _run_single_attempt(env, by, brain: "ConnectomeBrain", topo: dict) -> dict:
    """Same as _drive_attempt but silent — used by the CEM inner loop."""
    obs = json.loads(by["reset_environment"].invoke({}))
    brain.reset()
    while not obs["attempt_over"]:
        inputs = encode_sensors(obs)
        brain.step(inputs)
        throttle, steering = brain.motor()
        obs = json.loads(by["drive"].invoke({"throttle": throttle, "steering": steering}))
    res = env.attempts[-1]
    return {"completed": res.completed, "lap_time": res.lap_time or 0.0,
            "sim_time": res.sim_time, "crashes": res.crashes}