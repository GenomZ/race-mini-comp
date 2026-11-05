"""Fly Broken - the Ch7 exercise: ONE row of this fly's WIRING table is wrong. Find it and fix it.

This is fly_toy.py (57 spiking neurons, see that file for how the brain works) with a single wiring
mistake. On the home track it drives most of the lap like the toy, then hits the LEFT wall twice
around T5, the hairpin (the crash log says "before T5" and "T5"):

    python race.py drive drivers/examples/fly_broken.py --verbose        (grand_prix: 93.05 s, 2 crashes)
    python race.py drive drivers/examples/fly_toy.py                      (grand_prix: 63.3 s, 0 crashes)

Rules of the exercise: change exactly ONE row of WIRING (no other code), and do not diff against
fly_toy.py - read the neurons instead.

Fixed = BOTH of these (a lucky weight that just avoids the hairpin is not a fix):
  a) the mirror self-check passes:   python drivers/examples/fly_broken.py      -> "SYMMETRIC"
  b) grand_prix in exactly 63.3 s with 0 crashes (briefing_demo 76.3 s): the brain has no noise, so
     the right fix reproduces the healthy fly's lap to the tick.

Hints (reveal one at a time)
1. Watch the replay going into T5 (or read the radio lines just before the two crashes). Which way
   does the car turn just before it hits, and on which side is the wall? (T5 is a LEFT hairpin: its
   inner wall is on the left.)
2. Just before each crash the radio names the loudest input neurons while the fly turns, e.g.
   "turning left; loudest inputs WALL_L45 ..., WALL_L20 ...". A wall on the LEFT is making the fly
   turn LEFT. (Fly Lab: the same neurons light up in the raster and in the per-tick debug rates.)
3. Run the self-check with details:  python drivers/examples/fly_broken.py -v
   Which wall stimulus does not give the mirror image of its right-hand twin?
4. Follow that wall neuron in WIRING, and its right-hand twin. Braitenberg: a wall on the left must
   excite the neuron that turns the car AWAY from it (crossed wiring). Which row is not crossed?
"""

from __future__ import annotations

import numpy as np

SPINAL_CORD = ""          # this fly has no engineered speed rule: every control comes from neurons

# ============================================================================ NEURONS
# cell type: (number of cells, tonic bias, role).  Bias is a constant input (in thresholds): a
# negative bias makes a neuron hard to fire (GF needs a strong looming signal).
NEURONS: dict[str, tuple[int, float, str]] = {
    # --- eyes and body sense (input neurons, driven by the encoder) ---------------------------------
    "LC10a_L":   (4, 0.0, "visual projection neuron: the target (next waypoint + look-ahead) is to the LEFT"),
    "LC10a_R":   (4, 0.0, "visual projection neuron: the target is to the RIGHT"),
    "WALL_L20":  (2, 0.0, "wall-ray neuron: wall near, 20 deg left"),
    "WALL_L45":  (2, 0.0, "wall-ray neuron: wall near, 45 deg left"),
    "WALL_L90":  (2, 0.0, "wall-ray neuron: wall near, 90 deg left"),
    "WALL_R20":  (2, 0.0, "wall-ray neuron: wall near, 20 deg right"),
    "WALL_R45":  (2, 0.0, "wall-ray neuron: wall near, 45 deg right"),
    "WALL_R90":  (2, 0.0, "wall-ray neuron: wall near, 90 deg right"),
    "FRONT":     (3, 0.0, "wall straight ahead is near"),
    "LPLC2":     (3, 0.0, "looming detector: the wall ahead grows fast (speed / distance)"),
    "FAST":      (3, 0.0, "proprioception: fires more the faster we go"),
    "STILL":     (2, 0.0, "proprioception: fires when the car is (almost) not moving"),
    # --- central brain ---------------------------------------------------------------------------
    "P1":        (2, 0.0, "arousal: tonic drive, the fly's 'mood' (PARAMS arousal)"),
    "AOTU019_L": (3, 0.0, "pursuit hub, left"),
    "AOTU019_R": (3, 0.0, "pursuit hub, right"),
    "AND_STUCK": (2, 0.0, "coincidence neuron: fires only for wall ahead AND standing still"),
    # --- descending neurons (the decoder reads these) ------------------------------------------
    "DNa02_L":   (4, 0.0, "descending neuron: turn LEFT"),
    "DNa02_R":   (4, 0.0, "descending neuron: turn RIGHT"),
    "P9":        (4, 0.0, "descending neuron: walk forward (throttle)"),
    "GF":        (2, -0.4, "Giant Fiber: escape / emergency brake"),
    "MDN":       (2, 0.0, "moonwalker descending neuron: walk backward (reverse)"),
}

# ============================================================================ WIRING
# (pre, post, weight, why).  weight = input (in thresholds) when `pre` fires at 100 Hz; < 0 inhibits.
WIRING: list[tuple[str, str, float, str]] = [
    # pursuit: chase the target with the SAME side
    ("LC10a_L",   "AOTU019_L", 1.5,  "target left -> left pursuit hub"),
    ("LC10a_R",   "AOTU019_R", 1.5,  "target right -> right pursuit hub"),
    ("P1",        "AOTU019_L", 0.5,  "arousal sharpens pursuit"),
    ("P1",        "AOTU019_R", 0.5,  "arousal sharpens pursuit"),
    ("AOTU019_L", "DNa02_L",   2.0,  "left hub -> turn left"),
    ("AOTU019_R", "DNa02_R",   2.0,  "right hub -> turn right"),
    # wall reflex: CROSSED (Braitenberg) - a wall on the left turns the car right
    ("WALL_L20",  "DNa02_L",   3.0,  "wall ahead-left"),
    ("WALL_L45",  "DNa02_R",   2.1,  "wall left"),
    ("WALL_L90",  "DNa02_R",   1.2,  "wall beside, left (a nudge)"),
    ("WALL_R20",  "DNa02_L",   3.0,  "wall ahead-right"),
    ("WALL_R45",  "DNa02_L",   2.1,  "wall right"),
    ("WALL_R90",  "DNa02_L",   1.2,  "wall beside, right (a nudge)"),
    ("DNa02_L",   "DNa02_R",  -0.8,  "mutual inhibition: one clear decision"),
    ("DNa02_R",   "DNa02_L",  -0.8,  "mutual inhibition: one clear decision"),
    # forward drive and the speed governor
    ("P1",        "P9",        2.4,  "arousal -> walk forward"),
    ("FAST",      "P9",       -1.0,  "speed governor: the faster, the less push"),
    ("FRONT",     "P9",       -1.5,  "wall ahead: ease off"),
    ("AOTU019_L", "P9",       -0.3,  "turning: slow down a little"),
    ("AOTU019_R", "P9",       -0.3,  "turning: slow down a little"),
    # escape: looming -> Giant Fiber -> brake
    ("LPLC2",     "GF",        1.5,  "looming -> Giant Fiber"),
    ("P1",        "GF",       -0.15, "arousal: a keen fly ignores small threats (the risk knob)"),
    ("GF",        "P9",       -3.0,  "escape freezes forward walking"),
    # stuck: coincidence of FRONT and STILL -> MDN -> back up
    ("FRONT",     "AND_STUCK", 0.6,  "wall ahead (alone: below threshold)"),
    ("STILL",     "AND_STUCK", 0.6,  "standing still (alone: below threshold)"),
    ("AND_STUCK", "MDN",       2.0,  "stuck -> moonwalk"),
    ("MDN",       "P9",       -3.0,  "backing up: stop walking forward"),
]

# ============================================================================ DECODER (Hz scales)
STEER_HZ = 250.0     # DNa02 R - L difference that means full lock
GO_HZ = 100.0        # P9 rate that means full throttle
BRAKE_HZ = 60.0      # GF rate that means full brake
BACK_HZ = 50.0       # MDN rate that means full reverse

# ============================================================================ ENCODER ranges
WALL_RANGE = {20: 18.0, 45: 13.5, 90: 9.0}  # wall-ray neuron starts firing when that ray is shorter than this
FRONT_RANGE = 25.0                          # FRONT starts firing when the wall ahead is closer than this

# ============================================================================ LIF constants
DT_MS = 1.0          # integration step
WINDOW_MS = 50.0     # brain time simulated per drive() call
TAU_M_MS = 10.0      # membrane time constant
TAU_S_MS = 5.0       # synaptic trace time constant
T_REF_MS = 2.0       # refractory period

TYPES = list(NEURONS)
SIZES = np.array([NEURONS[t][0] for t in TYPES])
STARTS = np.concatenate([[0], np.cumsum(SIZES)[:-1]])
N_NEURONS = int(SIZES.sum())                                        # 57
CELLS = {t: np.arange(s, s + n) for t, s, n in zip(TYPES, STARTS, SIZES)}
NEURON_NAMES = [f"{t}#{i + 1}" for t in TYPES for i in range(NEURONS[t][0])]


def build_weights(wiring, scale: dict | None = None) -> np.ndarray:
    """WIRING table -> N x N weight matrix W[post, pre] on the synaptic traces.

    A trace grows by 1 per spike and decays with TAU_S, so a cell firing at 100 Hz has a mean
    trace of 100 Hz * 5 ms = 0.5.  Dividing by 0.5 * n_pre makes `w` mean "thresholds of input per
    100 Hz of the pre-synaptic cell type".  `scale` multiplies rows by (pre, post) or by name."""
    w_mat = np.zeros((N_NEURONS, N_NEURONS))
    for pre, post, w, *_ in wiring:
        k = (scale or {}).get((pre, post), 1.0)
        n_pre = len(CELLS[pre])
        w_mat[np.ix_(CELLS[post], CELLS[pre])] += k * w / (0.5 * n_pre)
    return w_mat


def _seed(value) -> int | None:
    """SCRAMBLE param -> seed (None, "", "none", False = this wiring)."""
    text = str(value).strip().lower()
    return None if text in ("none", "", "false", "off", "real") else int(float(text))


class Driver:
    NAME = "Fly Broken"
    AUTHOR = "Agentic GP"
    COLOR = "#e76f51"
    KIND = "fly"
    MOTTO = "One synapse away from greatness."
    ANATOMY = ["perceive", "act", "spiking"]
    SPINAL_CORD = SPINAL_CORD
    PARAMS = {
        "arousal": 2.2,       # P1 drive: the fly's cruising "mood" (higher = faster)
        "wall_gain": 1.0,     # x every WALL_* -> DNa02 synapse
        "lc10a_gain": 1.0,    # how hard the target's bearing drives LC10a
        "crossed": True,      # False = wall neurons wired to the SAME side (Ch7 exercise)
        "mdn": True,          # False = remove the moonwalker (Ch7 exercise)
        "lookahead": 0.5,     # how far LC10a's "target" blends toward the waypoint after next (0 = none)
        "SCRAMBLE": None,     # None = this wiring; 0, 1, 2 ... = scrambled control with that seed
    }
    WIRING = WIRING           # fly_broken.py swaps in its own table

    def __init__(self) -> None:
        p = self.PARAMS
        wiring = list(self.WIRING)
        if not p.get("crossed", True):                       # exercise: uncross the wall wiring
            swap = {"DNa02_L": "DNa02_R", "DNa02_R": "DNa02_L"}
            wiring = [(a, swap[b], w, why) if a.startswith("WALL_") else (a, b, w, why)
                      for a, b, w, why in wiring]
        if not p.get("mdn", True):                           # exercise: remove MDN
            wiring = [r for r in wiring if "MDN" not in (r[0], r[1])]
        seed = _seed(p.get("SCRAMBLE"))
        if seed is not None:                                 # the control: same rows, random targets
            posts = np.random.default_rng(seed).permutation([b for _, b, *_ in wiring])
            wiring = [(a, str(b), w, "scrambled") for (a, _, w, _), b in zip(wiring, posts)]
        scale = {(a, b): float(p.get("wall_gain", 1.0)) for a, b, *_ in wiring if a.startswith("WALL_")}
        self.W = build_weights(wiring, scale)
        self.bias = np.concatenate([[NEURONS[t][1]] * NEURONS[t][0] for t in TYPES])
        # cells of one type differ a little (0.85..1.15 x input): a smooth population rate code
        self.gain = np.concatenate([np.linspace(0.85, 1.15, n) if n > 1 else [1.0] for n in SIZES])
        self.reset()

    def reset(self) -> None:
        self.v = np.zeros(N_NEURONS)              # membrane potential (threshold = 1)
        self.trace = np.zeros(N_NEURONS)          # synaptic traces
        self.refractory = np.zeros(N_NEURONS)     # ms left
        self.rates = {t: 0.0 for t in TYPES}
        self.last_spikes: list[tuple[float, int]] = []   # (t_ms in window, neuron index): the raster
        self.t_ms = 0.0
        self.last_say: str | None = None          # the radio starts fresh too

    def plan(self, track: dict) -> None:
        self.reset()                              # a fresh brain for every session

    def reflect(self, report: dict) -> None:
        self.reset()                              # ... and for every attempt: same start, same spikes

    # ------------------------------------------------------------------ ENCODER: sensors -> input current
    def encode(self, obs: dict) -> np.ndarray:
        """Sensor -> input current per input neuron (1.0 = exactly at threshold). Smooth, no if/else."""
        p = self.PARAMS
        g = lambda key, default=200.0: float(obs.get(key, default))
        clip = np.clip
        speed = g("speed", 0.0)
        front = max(g("distance_to_wall_front"), 0.5)
        # where to look: the next waypoint, blending in the one after it when we are close
        a1, d1 = g("angle_to_next_waypoint", 0.0), g("distance_to_next_waypoint")
        a2 = g("angle_to_waypoint_after_next", 0.0)
        blend = float(clip((25.0 - d1) / 20.0, 0.0, p.get("lookahead", 0.5)))
        aim = (1 - blend) * a1 + blend * a2
        current = {
            "LC10a_L": 1.0 + p["lc10a_gain"] * float(clip(-aim / 30.0, 0.0, 3.0)),
            "LC10a_R": 1.0 + p["lc10a_gain"] * float(clip(aim / 30.0, 0.0, 3.0)),
            "FRONT": 2.5 * float(clip(1.0 - front / FRONT_RANGE, 0.0, 1.0)),
            "LPLC2": 2.0 * float(clip(speed / front, 0.0, 3.0)),     # 1 / time-to-contact
            "FAST": 0.8 + 2.0 * float(clip(speed / 40.0, 0.0, 1.0)),
            "STILL": 2.0 * float(clip(1.0 - abs(speed) / 3.0, 0.0, 1.0)),
            "P1": float(p["arousal"]),
        }
        for side, key in (("L", "left"), ("R", "right")):
            for deg, rng in WALL_RANGE.items():
                near = 1.0 - g(f"distance_to_wall_{key}_{deg}") / rng
                current[f"WALL_{side}{deg}"] = 2.2 * float(clip(near, 0.0, 1.0))
        i_ext = self.bias.copy()
        for t, c in current.items():
            i_ext[CELLS[t]] += c
        return i_ext

    # ------------------------------------------------------------------ BRAIN: simulate the LIF network
    def simulate(self, i_ext: np.ndarray, ms: float = WINDOW_MS) -> np.ndarray:
        """Run the network for `ms` of brain time; return the firing rate of every neuron (Hz)."""
        steps = int(round(ms / DT_MS))
        decay_v = np.exp(-DT_MS / TAU_M_MS)
        decay_s = np.exp(-DT_MS / TAU_S_MS)
        counts = np.zeros(N_NEURONS)
        spikes = []
        v, tr, ref = self.v, self.trace, self.refractory
        for k in range(steps):
            current = self.gain * (i_ext + self.W @ tr)
            v = current + (v - current) * decay_v           # exact exponential step
            v[ref > 0] = 0.0
            ref = np.maximum(ref - DT_MS, 0.0)
            fired = v >= 1.0
            v[fired] = 0.0
            ref[fired] = T_REF_MS
            tr = tr * decay_s + fired
            counts += fired
            spikes.extend((k * DT_MS, int(i)) for i in np.flatnonzero(fired))
        self.v, self.trace, self.refractory = v, tr, ref
        self.last_spikes = spikes
        self.t_ms += steps * DT_MS
        return counts / (steps * DT_MS / 1000.0)

    # ------------------------------------------------------------------ DECODER: rates -> controls
    def decode(self, rates: dict) -> tuple[float, float]:
        steering = (rates["DNa02_R"] - rates["DNa02_L"]) / STEER_HZ
        throttle = rates["P9"] / GO_HZ - rates["GF"] / BRAKE_HZ - rates["MDN"] / BACK_HZ
        return float(np.clip(throttle, -1, 1)), float(np.clip(steering, -1, 1))

    def drive(self, obs: dict) -> dict:
        hz = self.simulate(self.encode(obs))
        self.rates = {t: float(hz[CELLS[t]].mean()) for t in TYPES}
        throttle, steering = self.decode(self.rates)
        return {"throttle": throttle, "steering": steering, "say": self._radio(),
                "debug": {t: round(r) for t, r in self.rates.items()}}

    # ------------------------------------------------------------------ radio: what the neurons are doing
    def _radio(self) -> str | None:
        """Speak when the dominant neural event changes (viewers show every new line)."""
        r = self.rates
        steer = (r["DNa02_R"] - r["DNa02_L"]) / STEER_HZ
        turn = "right" if steer > 0 else "left"
        loud = sorted((t for t in TYPES if t.startswith(("WALL_", "LC10a"))), key=lambda t: -r[t])[:2]
        why = (f"DNa02 L {r['DNa02_L']:.0f} / R {r['DNa02_R']:.0f} Hz: turning {turn}; loudest inputs "
               + ", ".join(f"{t} {r[t]:.0f} Hz" for t in loud))
        if r["MDN"] > 20:
            topic = "MDN"
            text = (f"MDN {r['MDN']:.0f} Hz: moonwalking off the wall "
                    f"(AND_STUCK {r['AND_STUCK']:.0f} Hz = FRONT AND STILL)")
        elif r["GF"] > 20:
            topic = "GF"
            text = f"GF fired {r['GF']:.0f} Hz: wall looming (LPLC2 {r['LPLC2']:.0f} Hz), brake! {why}"
        elif abs(steer) >= 0.05:
            topic, text = turn, why
        else:
            topic, text = None, None
        if topic == self.last_say:
            return None
        self.last_say = topic
        return text

    def raster(self) -> list[tuple[float, str]]:
        """Last window's spikes as (t_ms, neuron name), for a raster plot in the Fly Lab."""
        return [(t, NEURON_NAMES[i]) for t, i in self.last_spikes]


# ============================================================================ self-check: is the wiring mirror-symmetric?
def mirror_check(driver_cls=Driver, verbose: bool = False) -> bool:
    """Show the fly a target, and walls at 20/45/90 deg, on the LEFT and then on the RIGHT (open road,
    speed 15). A healthy fly steers by exactly opposite amounts, and AWAY from every wall."""
    far = {"speed": 15.0, "distance_to_wall_front": 200.0, "angle_to_next_waypoint": 0.0,
           "distance_to_next_waypoint": 80.0, **{f"distance_to_wall_{k}_{d}": 200.0
                                                 for k in ("left", "right") for d in WALL_RANGE}}

    def steer(change: dict) -> float:
        d = driver_cls()
        for _ in range(3):
            d.drive({**far, **change})
        return d.decode(d.rates)[1]

    stimuli = [("target 30 deg", {"angle_to_next_waypoint": -30.0}, {"angle_to_next_waypoint": 30.0}, None)]
    stimuli += [(f"wall at {deg} deg", {f"distance_to_wall_left_{deg}": 0.4 * rng},
                 {f"distance_to_wall_right_{deg}": 0.4 * rng}, +1) for deg, rng in WALL_RANGE.items()]
    healthy = True
    for name, left, right, away in stimuli:
        s_left, s_right = steer(left), steer(right)
        ok = abs(s_left + s_right) <= 0.02 and (away is None or s_left > 0.05)
        healthy &= ok
        if verbose:
            print(f"  {name:14s} left {s_left:+.2f}  right {s_right:+.2f}  {'ok' if ok else 'NOT a mirror image / not away'}")
    return healthy


if __name__ == "__main__":
    import sys
    verbose = "-v" in sys.argv
    print("Mirror self-check (steering + = right):" if verbose else
          "Mirror self-check (add -v to see each stimulus; steering + = right):")
    print("SYMMETRIC - now race it on grand_prix" if mirror_check(verbose=verbose)
          else "NOT SYMMETRIC - check the WIRING")
