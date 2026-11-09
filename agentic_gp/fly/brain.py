"""FlyBrain: a FlyWire-wired spiking network that drives the car.

    from agentic_gp.fly import FlyBrain
    brain = FlyBrain.from_default(scramble_seed=None)    # None = real wiring, int = scrambled control
    controls, snapshot = brain.step(obs)                  # controls = {"throttle": .., "steering": ..}

Each ``step`` turns the car's sensors into input spike rates on real FlyWire neuron
types, simulates ``window_ms`` (100 ms) of the network, and reads steering off DNa02
(descending) and AOTU019 (a central-brain hub ONE synapse after LC10a), braking off MDN and
an emergency stop off the DNp01 giant fiber.  What is engineered and what comes from the
connectome:

=====================  ==========================================  =====================
stage                  what                                        source
=====================  ==========================================  =====================
eyes (encoder)         bearing -> LC10a (small-object detector),   ENGINEERED mapping
                       dimmed when a wall hides the waypoint;      (rates are our
                       wall rays -> LC16; 1 / time to contact ->   choice; smooth, no
                       LPLC2 + LC4 (looming detectors, graded)     if/else on sensors)
reflex (declared)      "antenna touch" (front wall < 3) -> MDN     ENGINEERED shortcut
                       directly; off when touch_hz = 0             into a read neuron
brain                  4,494 neurons, 407k connections, synapse    CONNECTOME (FlyWire
                       counts as weights, Shiu et al. LIF model    v783, Shiu 2024)
readout (decoder)      steering = DNa02 R - L (descending) + 0.5 x ENGINEERED gains on
                       AOTU019 R - L (central hub; carries most    CONNECTOME-driven
                       of the steering in races); brake = MDN;     neurons
                       emergency = DNp01 giant fiber >= 50 Hz
"spinal cord"          throttle = speed controller using brake     ENGINEERED, said
                       and steering; no neuron drives speed        openly
=====================  ==========================================  =====================

How much of this is the deep circuit?  :func:`~agentic_gp.fly.connectome.shallow` keeps only
the direct input -> readout synapses (566 of 407,139) and still drives most of a lap (README.md,
"What the controls show"): most of the steering is the LC10a -> AOTU019 relay.

Sign conventions match the race: ``angle_to_next_waypoint`` > 0 means the waypoint is to
the RIGHT, and ``steering`` > 0 turns RIGHT.  A target on the right drives the right
LC10a, which excites the right DNa02 (the fly turns toward it).

Controls (scrambled, lesioned, mirrored brains) keep the DEFAULT readout parameters, which
were measured on the real brain.  Do not ``calibrate()`` a control: calibration would
rescale its noise into a steering signal.
"""
from __future__ import annotations

import math
import time
import warnings
from dataclasses import asdict, dataclass, field, replace

import numpy as np

from .connectome import Circuit, lesioned, load_circuit, mirrored, scrambled
from .connectome import shallow as shallow_circuit
from .lif import LIFNetwork, LIFParams

__all__ = ["FlyBrain", "EncoderParams", "ReadoutParams", "calibrate", "wall_distance_at",
           "INPUT_GROUPS", "INPUT_TYPES", "READOUT_GROUPS", "CONTROL_PATH"]

INPUT_GROUPS = ("LC10a_L", "LC10a_R", "LC16_L", "LC16_R", "LPLC2_L", "LPLC2_R", "LC4_L", "LC4_R")
INPUT_TYPES = ("LC10a", "LC16", "LPLC2", "LC4")
READOUT_GROUPS = ("DNa02_L", "DNa02_R", "AOTU019_L", "AOTU019_R", "AOTU025_L", "AOTU025_R",
                  "DNa01_L", "DNa01_R", "DNa03_L", "DNa03_R", "MDN_L", "MDN_R", "DNp01_L", "DNp01_R")
#: the car's wall rays, from left to right: (angle in degrees, + = right, observation key)
RAYS = ((-90, "distance_to_wall_left_90"), (-45, "distance_to_wall_left_45"), (-20, "distance_to_wall_left_20"),
        (0, "distance_to_wall_front"), (20, "distance_to_wall_right_20"), (45, "distance_to_wall_right_45"),
        (90, "distance_to_wall_right_90"))
_FAR = 1e9
#: FlyBrain methods on the sensor -> control path: no if/else on sensors (Fly-class rule 2, tested)
CONTROL_PATH = ("bearing_rates", "visibility", "looming", "encode", "reflex", "decode", "steer_terms", "step")


@dataclass
class EncoderParams:
    """Sensor -> input spike rates (Hz).  Engineered starting values (research notes, section 8)."""

    # bearing -> LC10a (fruitfly-lab encoder, re-implemented)
    bearing_max_hz: float = 150.0      # peak rate of an LC10a cell
    bearing_sigma: float = 0.3         # eccentricity tuning width (0..1 = 0..90 deg)
    binocular_deg: float = 6.0         # the far eye fades out linearly between 0 and this bearing
    bearing_clip_deg: float = 90.0     # a target behind the car = maximal peripheral drive
    # occlusion: a wall closer than the waypoint, in the waypoint's direction, hides it
    occlusion: bool = True
    occluded_gain: float = 0.3         # LC10a drive multiplier when the waypoint is hidden
    occlusion_ramp: float = 5.0        # fully hidden once the wall is this much closer than the waypoint
    occlusion_cone_deg: tuple = (20.0, 45.0)   # only targets roughly ahead can be hidden (fades out between)
    # wall rays -> LC16 per side: rate = wall_max_hz * max(1 - distance / range) over that side's rays
    wall_max_hz: float = 200.0
    wall_range_front: float = 30.0     # front ray drives BOTH sides (-> MDN brake)
    wall_range: dict = field(default_factory=lambda: {20: 18.0, 45: 12.0, 90: 7.0})
    # looming: 1 / time to contact (speed / front ray: how fast the wall ahead grows) -> LPLC2 + LC4
    # on both sides, GRADED (no sensor threshold): 0 Hz at a time to contact of ttc_onset_s or more,
    # rising linearly in 1/TTC to loom_hz at ttc_full_s or less; faded in between min_speed and
    # 2 x min_speed (a parked car does not loom).
    loom_hz: float = 100.0
    ttc_onset_s: float = 0.9
    ttc_full_s: float = 0.7
    min_speed: float = 1.0
    # "antenna touch" REFLEX: a wall closer than touch_range on the front ray drives MDN directly
    # (rate = touch_hz * (1 - distance / touch_range)).  ENGINEERED shortcut, and the only input that
    # goes straight into a neuron the decoder reads (Fly-class: a declared REFLEX, see README.md): the
    # extracted circuit has no mechanosensory pathway, but in the fly touch in front does trigger
    # backward walking.
    # Without it a car parked nose-to-wall creeps forward into the wall (pursuit suppresses LC16 -> MDN);
    # with it the car backs up (96 nose-to-wall starts at speed 0 on 3 tracks: crash within 8 s
    # 56% -> 16%).  Race trade-off (fly_connectome.py, 1 lap): under the default "marshal" crash rule a
    # touch costs only 5 s and the marshal re-points the car, while backing up cannot steer around a
    # wall that hides the waypoint, so laps get SLOWER (sprint 79 -> 165 s, grand_prix 119 -> 190 s).
    # Under the "classic" rule it ends the crash loop (75-77 -> 3-5 crashes per attempt).  Set
    # touch_hz=0 to switch it off (fly_connectome.py does that unless crash_rule is "classic").
    touch_range: float = 3.0
    touch_hz: float = 100.0


@dataclass
class ReadoutParams:
    """Neuron rates (Hz) -> controls.  Defaults were measured on the REAL brain (``calibrate``);
    per-side gains because the left side fires ~1.2-1.7x more than the right."""

    dna02_gain_left: float = 150.0     # DNa02_L rate that means "full left"
    dna02_gain_right: float = 130.0
    aotu019_weight: float = 0.5        # AOTU019 carries the central field, where DNa02 is silent
    aotu019_gain_left: float = 175.0
    aotu019_gain_right: float = 175.0
    steer_offset: float = 0.0          # raw steer with the target dead ahead (subtracted)
    steer_gain: float = 1.0            # steering = clip(steer_gain * raw); < 1 calms the 0.5 s tick overshoot
    mdn_full_brake_hz: float = 30.0    # mean MDN rate that means "full brake / back up"
    dnp01_min_hz: float = 50.0         # mean giant-fiber rate that triggers the emergency stop
    # (real brain: 0 Hz without looming, ~140 Hz with it; scrambled brains: up to ~30 Hz of noise)
    # "spinal cord" (engineered, no neuron drives forward speed in this circuit)
    cruise_speed: float = 16.0
    turn_slowdown: float = 0.5         # target speed *= 1 - turn_slowdown * |steering|
    reverse_speed: float = 6.0         # MDN is the "moonwalker": full brake = back up at this speed
    throttle_gain: float = 0.3         # throttle = gain * (target speed - speed)
    emergency_throttle: float = -1.0


def _f(obs: dict, key: str, default: float) -> float:
    try:
        x = float(obs.get(key, default))
    except (TypeError, ValueError):
        return default
    return float(np.where(np.isnan(x), default, x))          # NaN -> default, without a branch


def wall_distance_at(obs: dict, bearing_deg: float) -> float:
    """Distance to the wall in direction ``bearing_deg`` (+ = right), interpolated between the
    car's 7 rays (missing rays count as far away)."""
    angles = [a for a, _ in RAYS]
    dist = [_f(obs, k, _FAR) for _, k in RAYS]
    return float(np.interp(float(np.clip(bearing_deg, -90.0, 90.0)), angles, dist))


class FlyBrain:
    """Encoders + LIF network on a :class:`Circuit` + readout.  See the module docstring."""

    def __init__(self, circuit: Circuit, encoder_params: EncoderParams | None = None,
                 readout_params: ReadoutParams | None = None, lif_params: LIFParams | None = None,
                 window_ms: float = 100.0, seed: int | None = 0, backend: str = "auto"):
        self.circuit = circuit
        self.encoder = encoder_params or EncoderParams()
        self.readout = readout_params or ReadoutParams()
        self.lif_params = lif_params or LIFParams()
        self.window_ms = float(window_ms)
        self.net = LIFNetwork(circuit.n, circuit.pre, circuit.post, circuit.weight_mv(self.lif_params.w_syn),
                              self.lif_params, seed=seed, backend=backend)
        self.idx = {g: circuit.group(g) for g in INPUT_GROUPS + READOUT_GROUPS}
        missing = [g for g in ("LC10a_L", "LC10a_R", "DNa02_L", "DNa02_R") if self.idx[g].size == 0]
        if missing:
            raise ValueError(f"circuit lacks required groups {missing}")
        self._silenced = circuit.silenced_index()
        self._is_member = {}
        for g in READOUT_GROUPS:
            m = np.zeros(circuit.n, bool)
            m[self.idx[g]] = True
            self._is_member[g] = m
        self._lc10a_pref = {s: np.nan_to_num(circuit.pref_ecc[self.idx[f"LC10a_{s}"]], nan=0.5) for s in "LR"}
        self.last_rates = np.zeros(circuit.n)
        self.last_spikes = None
        self.last_window_start_ms = 0.0

    @classmethod
    def from_default(cls, scramble_seed: int | None = None, lesion=(), mirror: bool | str = False,
                     path=None, scramble_method: str = "degree", shallow: bool = False, **kwargs) -> "FlyBrain":
        """The classroom brain.  ``scramble_seed=int`` -> scrambled control (``scramble_method``
        "degree" or the stricter "strength", see :func:`~agentic_gp.fly.connectome.scrambled`);
        ``shallow=True`` -> keep only the direct input -> readout synapses (the second control);
        ``lesion=["DNa02"]`` silences groups; ``mirror=True`` swaps every left/right label (a
        sanity check: it must still steer toward targets), ``mirror="eyes"`` swaps only the input
        neurons ("crossed eyes": it should steer away).  Controls keep the default (real-brain)
        readout parameters."""
        c = load_circuit(path)
        if scramble_seed is not None:
            c = scrambled(c, scramble_seed, method=scramble_method)
        if shallow:
            c = shallow_circuit(c, INPUT_GROUPS, READOUT_GROUPS)
        if lesion:
            c = lesioned(c, lesion)
        if mirror == "eyes":
            c = mirrored(c, cell_types=INPUT_TYPES)
        elif mirror:
            c = mirrored(c)
        return cls(c, **kwargs)

    # ------------------------------------------------------------ encoder
    def bearing_rates(self, bearing_deg: float, strength: float = 1.0, out: np.ndarray | None = None) -> np.ndarray:
        """LC10a input rates for a target at ``bearing_deg`` (+ = right), scaled by ``strength``.

        Each LC10a cell is tuned to an eccentricity (0 = ahead, 1 = 90 deg).  The eye on the
        target's side is driven fully; the other eye only near the midline, fading out linearly
        by ``binocular_deg`` (so steering changes smoothly through 0)."""
        e = self.encoder
        rates = np.zeros(self.circuit.n) if out is None else out
        az = float(np.clip(bearing_deg, -e.bearing_clip_deg, e.bearing_clip_deg))
        ecc = min(abs(az) / 90.0, 1.0)
        binoc = max(e.binocular_deg, 1e-9)
        # eye weight, smooth in the bearing: 1 on the target's side, fading to 0 over binocular_deg
        # on the other side (no if/else on the sensor: Fly-class rule 2)
        eye = {"R": float(np.clip(1.0 + az / binoc, 0.0, 1.0)), "L": float(np.clip(1.0 - az / binoc, 0.0, 1.0))}
        for s in "LR":
            w = strength * eye[s]
            pref = self._lc10a_pref[s]
            r = w * e.bearing_max_hz * np.exp(-((pref - ecc) ** 2) / (2 * e.bearing_sigma ** 2))
            cells = self.idx[f"LC10a_{s}"]
            rates[cells] = np.maximum(rates[cells], r)
        return rates

    def visibility(self, obs: dict) -> float:
        """How visible the waypoint is: 0 when it lies roughly ahead (``occlusion_cone_deg``) and a
        wall in its direction is at least ``occlusion_ramp`` closer than it, 1 in plain view or
        when the distance is unknown (linear in between).  Off to the side it counts as visible:
        the car's few rays cannot tell an inside corner from a wall in the way."""
        e = self.encoder
        wp = _f(obs, "distance_to_next_waypoint", 0.0)      # unknown -> 0: no wall can be in front of it
        bearing = _f(obs, "angle_to_next_waypoint", 0.0)
        inner, outer = e.occlusion_cone_deg
        cone = float(np.clip((outer - abs(bearing)) / max(outer - inner, 1e-9), 0.0, 1.0))
        hidden = float(np.clip((wp - wall_distance_at(obs, bearing)) / max(e.occlusion_ramp, 1e-9), 0.0, 1.0))
        return 1.0 - float(bool(e.occlusion)) * cone * hidden

    def looming(self, obs: dict) -> float:
        """LPLC2 / LC4 input rate (Hz), graded in 1 / time to contact with the wall ahead."""
        e = self.encoder
        front = max(_f(obs, "distance_to_wall_front", _FAR), 0.1)
        speed = _f(obs, "speed", 0.0)
        expansion = speed / front                                  # 1 / time to contact (1/s)
        lo, hi = 1.0 / max(e.ttc_onset_s, 1e-9), 1.0 / max(e.ttc_full_s, 1e-9)
        ramp = float(np.clip((expansion - lo) / max(hi - lo, 1e-9), 0.0, 1.0))
        moving = float(np.clip(speed / max(e.min_speed, 1e-9) - 1.0, 0.0, 1.0))
        return e.loom_hz * ramp * moving

    def encode(self, obs: dict) -> tuple[np.ndarray, dict]:
        """Sensors -> input rate per INPUT neuron (Hz), plus a small summary for the snapshot.
        Only sensory types are driven here (LC10a, LC16, LPLC2, LC4); see :meth:`reflex`."""
        e = self.encoder
        bearing = _f(obs, "angle_to_next_waypoint", 0.0)
        vis = self.visibility(obs)
        rates = self.bearing_rates(bearing, strength=e.occluded_gain + (1.0 - e.occluded_gain) * vis)
        front = _f(obs, "distance_to_wall_front", _FAR)
        loom_hz = self.looming(obs)
        info = {"bearing_deg": bearing, "visibility": vis}
        for s, side in (("L", "left"), ("R", "right")):
            prox = max(0.0, 1.0 - front / e.wall_range_front)
            for deg, rng in e.wall_range.items():
                prox = max(prox, 1.0 - _f(obs, f"distance_to_wall_{side}_{deg}", _FAR) / rng)
            wall_hz = e.wall_max_hz * min(prox, 1.0)
            rates[self.idx[f"LC16_{s}"]] = wall_hz
            rates[self.idx[f"LPLC2_{s}"]] = loom_hz
            rates[self.idx[f"LC4_{s}"]] = loom_hz
            info[f"LC16_{s}"] = wall_hz
            info[f"loom_{s}"] = loom_hz
        for s in "LR":
            cells = self.idx[f"LC10a_{s}"]
            info[f"LC10a_{s}"] = float(rates[cells].mean()) if cells.size else 0.0
        return rates, info

    # ------------------------------------------------------------ declared reflex
    def reflex(self, obs: dict, rates: np.ndarray, info: dict) -> None:
        """REFLEX (declared, engineered): "antenna touch" - a wall closer than ``touch_range`` on the
        front ray drives MDN directly.  The ONLY input into a neuron the decoder reads; switched off
        with ``EncoderParams(touch_hz=0)`` (fly_connectome.py: on only under the classic crash rule)."""
        e = self.encoder
        front = _f(obs, "distance_to_wall_front", _FAR)
        touch_hz = e.touch_hz * float(np.clip(1.0 - front / max(e.touch_range, 1e-9), 0.0, 1.0))
        for s in "LR":
            rates[self.idx[f"MDN_{s}"]] = np.maximum(rates[self.idx[f"MDN_{s}"]], touch_hz)
        info["touch_MDN"] = touch_hz

    # ------------------------------------------------------------ readout
    def group_rates(self, rates: np.ndarray) -> dict[str, float]:
        return {g: float(rates[i].mean()) if i.size else 0.0 for g, i in self.idx.items() if g in READOUT_GROUPS}

    def steer_terms(self, gr: dict) -> dict[str, float]:
        r = self.readout
        dn = gr["DNa02_R"] / r.dna02_gain_right - gr["DNa02_L"] / r.dna02_gain_left
        ao = r.aotu019_weight * (gr["AOTU019_R"] / r.aotu019_gain_right - gr["AOTU019_L"] / r.aotu019_gain_left)
        return {"DNa02": dn, "AOTU019": ao, "offset": -r.steer_offset, "raw": dn + ao - r.steer_offset}

    def decode(self, gr: dict, speed: float) -> dict:
        """Readout-group rates (Hz) and the car's speed -> motor decision."""
        r = self.readout
        st = self.steer_terms(gr)
        steering = float(np.clip(r.steer_gain * st["raw"], -1.0, 1.0))
        brake = float(np.clip(0.5 * (gr["MDN_L"] + gr["MDN_R"]) / r.mdn_full_brake_hz, 0.0, 1.0))
        dnp01_hz = 0.5 * (gr["DNp01_L"] + gr["DNp01_R"])
        emergency = dnp01_hz >= r.dnp01_min_hz
        # ---- "spinal cord": engineered speed rule (the circuit has no forward-drive neuron) ----
        target = (r.cruise_speed * (1.0 - brake) * (1.0 - r.turn_slowdown * abs(steering))
                  - r.reverse_speed * brake)
        stop = float(emergency)                       # decoded from DNp01 (a neuron), not from a sensor
        throttle = stop * r.emergency_throttle + (1.0 - stop) * float(np.clip(r.throttle_gain * (target - speed), -1, 1))
        return {"steering": steering, "throttle": throttle, "brake": brake, "emergency": bool(emergency),
                "dnp01_hz": dnp01_hz, "target_speed": target, "steer_terms": st}

    # --------------------------------------------------------------- step
    def _run(self, rates: np.ndarray, ms: float):
        if self._silenced.size:
            rates = rates.copy()
            rates[self._silenced] = 0.0          # lesioned cells get no sensory input either
        return self.net.run(ms, rates)

    def step(self, obs: dict) -> tuple[dict, dict]:
        """One car tick: encode -> simulate ``window_ms`` -> decode.

        Returns ``(controls, snapshot)``; ``controls = {"throttle": float, "steering": float}``,
        ``snapshot`` is a JSON-friendly dict of inputs, neuron rates, the motor decision and
        ``spikes`` (readout-group spike times in ms from the window start, for a raster).
        The full rate vector and spikes of the window are kept in ``last_rates`` / ``last_spikes``
        (absolute brain time; the window began at ``last_window_start_ms``).
        """
        t0 = time.perf_counter()
        rates_in, info = self.encode(obs)
        self.reflex(obs, rates_in, info)
        self.last_window_start_ms = self.net.t_ms
        rates, spikes = self._run(rates_in, self.window_ms)
        self.last_rates, self.last_spikes = rates, spikes
        gr = self.group_rates(rates)
        out_spikes = {}
        for g in READOUT_GROUPS:                  # few cells each: cheap, JSON-friendly raster
            m = self._is_member[g][spikes.idx]
            out_spikes[g] = (spikes.t_ms[m] - self.last_window_start_ms).round(2).tolist()
        dnp01 = len(out_spikes["DNp01_L"]) + len(out_spikes["DNp01_R"])
        motor = self.decode(gr, _f(obs, "speed", 0.0))
        controls = {"throttle": motor["throttle"], "steering": motor["steering"]}
        snapshot = {
            "t_ms": self.net.t_ms, "window_ms": self.window_ms, "circuit": self.circuit.label,
            "inputs": info, "rates": gr, "dnp01_spikes": dnp01,
            "motor": {k: v for k, v in motor.items() if k != "steer_terms"},
            "steer_terms": motor["steer_terms"], "spikes": out_spikes, "n_spikes": int(spikes.idx.size),
            "compute_ms": (time.perf_counter() - t0) * 1e3,
        }
        return controls, snapshot

    def reset(self) -> None:
        """Network back to rest (call between races)."""
        self.net.reset()
        self.last_window_start_ms = 0.0

    # ------------------------------------------------------ experiments
    def stimulate(self, group_hz: dict[str, float], ms: float = 500.0, warmup_ms: float = 100.0) -> dict[str, float]:
        """Electrophysiology: drive groups at fixed rates from rest, return readout rates (Hz).
        Resets the network (do not call mid-race)."""
        rates = np.zeros(self.circuit.n)
        for g, hz in group_hz.items():
            rates[self.circuit.group(g)] = hz
        return self.group_rates(self.measure(rates, ms, warmup_ms)[0])

    def bearing_response(self, bearing_deg: float, ms: float = 500.0, warmup_ms: float = 100.0) -> dict:
        """Readout rates and steering for a target at ``bearing_deg`` (no walls).  Resets the network."""
        gr = self.group_rates(self.measure(self.bearing_rates(bearing_deg), ms, warmup_ms)[0])
        gr["steer"] = float(np.clip(self.steer_terms(gr)["raw"], -1, 1))
        return gr

    def measure(self, rates: np.ndarray, ms: float = 500.0, warmup_ms: float = 100.0):
        """From rest, run ``warmup_ms`` then ``ms`` at fixed input ``rates``; returns the
        ``(rates_hz, spikes)`` of the measured part and leaves the network reset."""
        self.net.reset()
        if warmup_ms > 0:
            self._run(rates, warmup_ms)
        out = self._run(rates, ms)
        self.reset()
        return out

    def _measure(self, rates, ms, warmup_ms):     # kept for older notebooks: readout-group rates only
        return self.group_rates(self.measure(rates, ms, warmup_ms)[0])

    def calibrate(self, apply: bool = True, **kwargs) -> ReadoutParams:
        """Per-side steering gains measured on THIS brain (see :func:`calibrate`).  Real brain only."""
        p = calibrate(self, **kwargs)
        if apply:
            self.readout = p
        return p

    def params(self) -> dict:
        return {"encoder": asdict(self.encoder), "readout": asdict(self.readout), "lif": asdict(self.lif_params),
                "window_ms": self.window_ms, "circuit": self.circuit.label, "backend": self.net.backend}


def calibrate(brain: FlyBrain, ms: float = 500.0, warmup_ms: float = 100.0, central_deg: float = 10.0,
              floor_hz: float = 20.0, max_offset: float = 0.3) -> ReadoutParams:
    """Measure per-side readout gains so left and right targets give symmetric steering.

    * ``dna02_gain_left/right``  = DNa02_L at -90 deg / DNa02_R at +90 deg (full turn);
    * ``aotu019_gain_left/right`` = AOTU019_L at -central_deg / AOTU019_R at +central_deg;
    * ``steer_offset``           = remaining raw steering with the target dead ahead,
      clipped to +-``max_offset``.

    A response below ``floor_hz`` is not a steering signal: that gain is NOT changed and a
    warning is issued (this is what happens on a scrambled brain: do not calibrate controls).
    The network is reset; other readout fields are kept.
    """
    left, right = brain.bearing_response(-90.0, ms, warmup_ms), brain.bearing_response(90.0, ms, warmup_ms)
    cl, cr = brain.bearing_response(-central_deg, ms, warmup_ms), brain.bearing_response(central_deg, ms, warmup_ms)
    measured = {"dna02_gain_left": left["DNa02_L"], "dna02_gain_right": right["DNa02_R"],
                "aotu019_gain_left": cl["AOTU019_L"], "aotu019_gain_right": cr["AOTU019_R"]}
    weak = {k: round(v, 1) for k, v in measured.items() if v < floor_hz}
    if weak:
        warnings.warn(f"calibrate({brain.circuit.label}): responses below {floor_hz} Hz {weak}; those gains are "
                      "left unchanged. This brain does not steer toward targets (scrambled or lesioned?): keep the "
                      "real brain's readout parameters for controls.", RuntimeWarning, stacklevel=2)
    p = replace(brain.readout, steer_offset=0.0, **{k: v for k, v in measured.items() if v >= floor_hz})
    old, brain.readout = brain.readout, p
    try:
        raw = float(brain.steer_terms(brain.bearing_response(0.0, ms, warmup_ms))["raw"])
    finally:
        brain.readout = old
    if abs(raw) > max_offset:
        warnings.warn(f"calibrate({brain.circuit.label}): steering offset {raw:+.2f} with the target ahead clipped "
                      f"to +-{max_offset}", RuntimeWarning, stacklevel=2)
    return replace(p, steer_offset=float(np.clip(raw, -max_offset, max_offset)))
