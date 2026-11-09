"""Fly Connectome - 4,494 real fruit-fly neurons from FlyWire drive the car (Ch7 part B).

The wiring is NOT designed by us: it is the steering circuit of an adult fruit fly, read from the
FlyWire whole-brain connectome (v783: real neurons, real synapse counts as weights), simulated with the
Shiu et al. 2024 spiking (LIF) model in agentic_gp.fly. We build the eyes (encoder) and the readout
(decoder), and we read TWO steering cell types - one of them is not a descending neuron:

    bearing to waypoint -> LC10a L/R --1 synapse--> AOTU019 L/R (central hub)  --+
                                     --(connectome)--> DNa02 L/R (descending) ---+-> steering
    wall rays           -> LC16 L/R  --(connectome)--> MDN "moonwalker"          -> brake / reverse
    wall looming fast   -> LPLC2+LC4 --(connectome)--> DNp01 giant fiber         -> emergency stop

    steering = 0.4 x [ DNa02 R/130 - L/150  +  0.5 x (AOTU019 R - L)/175 ]   (rates in Hz)

In races about three quarters of the steering signal comes from the AOTU019 term: AOTU019 sits ONE
synapse after the LC10a input, and DNa02 is silent for targets within ~20 deg of straight ahead.

What is engineered and what is connectome (honesty table, also in agentic_gp/fly/README.md):
    encoder (sensor -> input spike rates)   ENGINEERED (rates are our choice; smooth, input neurons only)
    brain (4,494 neurons, 407k synapses)    CONNECTOME (FlyWire v783, CC BY-NC 4.0, Shiu 2024 model)
    decoder (rates -> steering/brake/stop)  ENGINEERED gains on connectome-driven neurons; steering
                                            read from DNa02 (descending) AND AOTU019 (central hub)
    SPINAL_CORD speed rule (below)          ENGINEERED, declared: no neuron in this circuit drives speed
    REFLEX "antenna touch" -> MDN           ENGINEERED, declared; on only under the classic crash rule

The CONTROLS (what does the wiring contribute?):
    --param SCRAMBLE=3     same neurons, same number of connections per neuron, random targets
    --param SHALLOW=true   keep only the 566 direct input -> readout connections (0.14%), drop the rest
If the real fly beats the scrambled flies, the WIRING (not the neuron count or the encoder) carries
the skill. If the shallow fly is nearly as good, that skill is mostly the direct LC10a -> AOTU019 /
LC10a -> DNa02 synapses, not the deep circuit. (Both are true - see MEASURED.)

Fly-class rule (Fly Cup; full text in agentic_gp/fly/README.md): controls must be decoded from neuron
activity. Sensors enter only through the encoder, as smooth functions, into input neurons only. An
engineered rule that also reads the car's speed is allowed only inside a section marked SPINAL_CORD
and declared in SPINAL_CORD = "..."; a contact reflex into a read neuron only if declared in REFLEX.

Run it:
    python race.py drive drivers/examples/fly_connectome.py --track oval
    python race.py drive drivers/examples/fly_connectome.py --track oval --param SCRAMBLE=3
    python race.py drive drivers/examples/fly_connectome.py --track oval --param SHALLOW=true
    python race.py drive drivers/examples/fly_connectome.py --param READOUT=DN          (DNa02 only)
    python race.py drive drivers/examples/fly_connectome.py --param LESION=DNa02        (or AOTU019, LC10a)
    python race.py drive drivers/examples/fly_connectome.py --param MIRROR=eyes         (or MIRROR=all)
    python race.py drive drivers/examples/fly_connectome.py --param SCRAMBLE=0 --param SCRAMBLE_METHOD=strength
    python race.py drive drivers/examples/fly_connectome.py --crash-rule classic          (TOUCH auto-enables)

TOUCH (the REFLEX: a wall < 3 units ahead drives MDN, so a car parked nose-to-wall backs up) is on
only under the classic crash rule by default ("auto"). Under the default marshal rule a wall touch
costs 5 s and the marshal re-points the car, while backing up cannot steer around a wall that hides
the waypoint, so it makes laps slower (see MEASURED).

Needs the circuit file data/fly/steering_circuit_783.npz (instructor: python -m agentic_gp.fly.extract).
Each drive() simulates 100 ms of brain time: about 20-50 ms of compute with numpy (5-20 ms with the
optional numba). Brain noise is seeded (PARAMS seed), so a run is reproducible.

Try this
1. Real vs SCRAMBLE=0..4 on oval: who finishes, how many crashes? (Fly Lab: the tuning curve.)
2. Now SHALLOW=true: how much of the real fly's lead survives with 0.14% of its synapses?
3. READOUT=DN reads only the descending neurons. LESION=DNa02 vs LESION=AOTU019: which one hurts
   more, and is that a fact about the fly or about OUR decoder (which reads both)?
4. Predict, then run: MIRROR=eyes swaps only the eyes (input neurons) - which way will it steer?
   MIRROR=all swaps every label, eyes AND readout: it must still chase the target (a sanity check);
   its lap time only measures the left/right differences in the data.
5. Read the radio: which neurons fired just before a crash?

MEASURED (1 lap, default PARAMS, brain seed 0, marshal crash rule; time incl. 5 s per crash)
    track          real          SHALLOW       scrambled seeds 0-3 ("degree")   scrambled seed 4
    grand_prix     117.1 s, 2    132.2 s, 3    292-364 s, 25-39 crashes          DNF (circles)
    oval            49.2 s, 0     64.7 s, 2    126-163 s,  9-17 crashes          DNF
    sprint          78.6 s, 1    101.8 s, 3    211-263 s, 20-29 crashes          DNF
    speedway       122.8 s, 0    125.1 s, 0    285-351 s, 21-34 crashes          DNF
    briefing_demo  134.1 s, 2    151.2 s, 3    302-375 s, 25-39 crashes          DNF
  Brain noise (seeds 1-3, real wiring): oval 49.1-50.5 s / 0, grand_prix 111.3-124.8 s / 2-3.
  Seed 4 steers right all the time (AOTU019_R >> L) and drives in circles. Other controls:
    READOUT=DN (descending neurons only): oval 79.8 s/3, grand_prix 231.3 s/14, sprint 127.7 s/5,
          speedway 247.7 s/16
    oval: LESION=DNa02 58.0 s/1, LESION=AOTU019 140.3 s/6, LESION=LC10a DNF (blind),
          MIRROR=all 52.2 s/0, MIRROR=eyes 452.2 s/42 (steers away), SCRAMBLE_METHOD=strength
          seeds 0-2 122.6-164.1 s/9-14
    grand_prix: LESION=DNa02 138.2 s/5, AOTU019 303.9 s/18, LC10a DNF, MIRROR=all 111.4 s/1,
          MIRROR=eyes DNF (31 crashes), strength 269.9-338.8 s/24-32
    TOUCH=true under the marshal rule: sprint 78.6 -> 86.8 s, grand_prix 117.1 -> 120.3 s.
    --crash-rule classic: oval 49.2 s/0; sprint and grand_prix DNF with TOUCH (13 and 9 crashes);
          without it sprint 339.3 s/37 and grand_prix DNF (65 crashes).
  In races the AOTU019 term is 77% (oval) / 79% (grand_prix) of the steering signal.
  With the emergency stop switched off for every brain, real 49.3 s / 114.6 s vs scrambled the same
  as above: the gap is steering, not braking.
  For comparison: Fly Toy 63.3 s (grand_prix) / 29.0 s (oval); L2 Planner 52.5 s / 26 s.
  drive() 25-33 ms per tick with numpy (4 races in parallel on a laptop).

What the controls show (Ch7 discussion):
  * Real beats SCRAMBLE by 2.3-3.3x: with the same neurons, connection counts, encoder and decoder,
    shuffling the wiring breaks pursuit. The skill lives in the wiring.
  * SHALLOW keeps 566 of 407,139 connections (the direct LC10a/LC16/LPLC2/LC4 -> readout ones) and is
    only 2-31% slower than the full brain. So most of that skill is a ONE-synapse relay, mainly
    LC10a -> AOTU019; the other ~4,400 neurons add a little (oval 64.7 -> 49.2 s), not the bulk.
  * LESION=DNa02 costs little and LESION=AOTU019 costs a lot because OUR decoder reads AOTU019 (with
    READOUT=DN the DNa02-only fly is 1.6-2x slower). That is a fact about our readout, not a finding
    about how a real fly steers.
  * MIRROR=all is a sanity check (same synapses, eyes and readout both relabelled): it must chase the
    target; the small time difference is the left/right asymmetry of the data. MIRROR=eyes is the
    real prediction test: crossed eyes steer away from every target.
What it does NOT prove: that a real fly steers this way (our encoder, readout gains, speed rule and
the 2D car are engineered), or that left/right differences are biological (they may come from
reconstruction and proofreading).
"""

from __future__ import annotations

import sys

import numpy as np

from agentic_gp.fly import CircuitNotFoundError, EncoderParams, FlyBrain

SPINAL_CORD = ("throttle = 0.3 * (target - speed); target = cruise * (1 - brake) * (1 - 0.5 * |steering|) "
               "- 6 * brake; brake = MDN rate / 30 Hz; full brake when DNp01 >= 50 Hz")
REFLEX = ("antenna touch: front wall closer than 3 units -> MDN at up to 100 Hz "
          "(PARAMS TOUCH; 'auto' = on only under the classic crash rule)")

# (the radio shows at most 160 characters: the whole hint fits)
MISSING_HINT = ("No data/fly/steering_circuit_783.npz: ask your instructor for that one file "
                "(instructor: python -m agentic_gp.fly.extract). The fly coasts.")
_YES = ("1", "true", "yes", "on")


def _seed_or_none(value) -> int | None:
    """SCRAMBLE param -> scramble seed: None/False/"none"/"" = real wiring, True = 0, int = that seed."""
    if value is None or value is False:
        return None
    if value is True:
        return 0
    text = str(value).strip().lower()
    if text in ("", "none", "null", "false", "no", "off", "real"):
        return None
    if text in ("true", "yes", "on"):
        return 0
    return int(float(text))


def _groups(value) -> list[str]:
    """LESION param -> list of cell types: "DNa02,AOTU019" or ["DNa02"]."""
    if not value:
        return []
    if isinstance(value, str):
        return [g.strip() for g in value.replace(";", ",").split(",") if g.strip()]
    return [str(g) for g in value]


def _mirror(value) -> bool | str:
    """MIRROR param -> False ("off"), True ("all"/"true": every label) or "eyes" (input neurons only)."""
    text = str(value).strip().lower()
    return "eyes" if text == "eyes" else text in _YES + ("all",)


class Driver:
    NAME = "Fly Connectome"
    AUTHOR = "Agentic GP"
    COLOR = "#8338ec"
    KIND = "fly"
    MOTTO = "Real fly wiring; we built only the eyes and the readout."
    ANATOMY = ["perceive", "act", "spiking"]
    SPINAL_CORD = SPINAL_CORD
    REFLEX = REFLEX
    PARAMS = {
        "SCRAMBLE": None,             # None = real FlyWire wiring; 0, 1, 2 ... = scrambled control with that seed
        "SCRAMBLE_METHOD": "degree",  # "degree" (keeps connection counts) or "strength" (also input synapse totals)
        "SHALLOW": False,             # True = keep only the direct input -> readout synapses (second control)
        "READOUT": "DN+AOTU019",      # steering read from DNa02 + AOTU019; "DN" = descending neurons only
        "LESION": "",                 # silence cell types, e.g. "DNa02" or "AOTU019,AOTU025"
        "MIRROR": "off",              # "all" = swap every left/right label (sanity check); "eyes" = inputs only
        "TOUCH": "auto",              # REFLEX antenna touch -> MDN: "auto" = only under the classic crash rule
        "steer_gain": 0.4,            # decoder: steering = 0.4 x (DNa02 + AOTU019 terms); 1.0 overshoots every tick
        "cruise_speed": 20.0,         # spinal cord: cruising speed with no brake and no turn
        "seed": 0,                    # brain noise (Poisson input) seed
    }

    def __init__(self) -> None:
        self.brain: FlyBrain | None = None
        self.error: str | None = None
        self.snapshots: list[dict] = []          # per tick of the LATEST attempt, for the Fly Lab (in-process runs)
        self.attempt = 0
        self._fresh = False                      # set by reflect(): the next drive() starts a new snapshot list
        self.last_topic: str | None = None
        self.said_error = False

    # ------------------------------------------------------------------ plan: build the brain once
    def plan(self, track: dict) -> None:
        p = self.PARAMS
        rule = str((track.get("physics") or {}).get("crash_rule", "marshal"))
        touch = str(p.get("TOUCH", "auto")).lower()
        touch_on = rule == "classic" if touch == "auto" else touch in _YES   # see TOUCH in the docstring
        try:
            self.brain = FlyBrain.from_default(
                scramble_seed=_seed_or_none(p.get("SCRAMBLE")), lesion=_groups(p.get("LESION")),
                mirror=_mirror(p.get("MIRROR", "off")), shallow=str(p.get("SHALLOW", False)).lower() in _YES,
                scramble_method=str(p.get("SCRAMBLE_METHOD", "degree")),
                encoder_params=EncoderParams() if touch_on else EncoderParams(touch_hz=0.0),
                seed=int(p.get("seed", 0)))
            self.brain.readout.cruise_speed = float(p.get("cruise_speed", 20.0))
            self.brain.readout.steer_gain = float(p.get("steer_gain", 0.4))
            if str(p.get("READOUT", "DN+AOTU019")).strip().upper() == "DN":
                self.brain.readout.aotu019_weight = 0.0      # descending neurons only
            self.error = None
        except CircuitNotFoundError:
            self.brain, self.error = None, MISSING_HINT
            print(MISSING_HINT, file=sys.stderr)
        self.snapshots, self.attempt, self.last_topic, self._fresh = [], 0, None, False

    def reflect(self, report: dict) -> None:
        if self.brain is not None:
            self.brain.reset()                   # a rested brain (clock back to 0) for the next attempt
        self._fresh = True                       # ... so the snapshots restart with it (at the next drive)
        self.attempt += 1
        self.last_topic = None                   # and the radio speaks again

    # ------------------------------------------------------------------ drive: one brain window per tick
    def drive(self, obs: dict) -> dict:
        if self.brain is None:
            if self.error is None:               # plan() was never called (e.g. a bare drive() test)
                self.plan({})
            if self.brain is None:
                say = None if self.said_error else self.error
                self.said_error = True
                return {"throttle": 0.0, "steering": 0.0, "say": say}
        if self._fresh:                          # first tick of a new attempt: one brain clock per list
            self.snapshots, self._fresh = [], False
        _, snap = self.brain.step(obs)           # encoder -> 100 ms of LIF on the connectome -> decoder
        snap["attempt"] = self.attempt
        motor = snap["motor"]
        throttle = self.spinal_cord(motor, float(obs.get("speed", 0.0)))
        self.snapshots.append(snap)
        if len(self.snapshots) > 3000:
            del self.snapshots[:1000]
        rates = snap["rates"]
        debug = {k: round(v) for k, v in rates.items() if k[:5] in ("DNa02", "AOTU0", "MDN_L", "MDN_R", "DNp01")}
        debug.update({"LC10a_L": round(snap["inputs"]["LC10a_L"]), "LC10a_R": round(snap["inputs"]["LC10a_R"]),
                      "brake": round(motor["brake"], 2), "visibility": round(snap["inputs"]["visibility"], 2),
                      "steer_DNa02": round(snap["steer_terms"]["DNa02"], 2),
                      "steer_AOTU019": round(snap["steer_terms"]["AOTU019"], 2),
                      "compute_ms": round(snap["compute_ms"], 1)})
        return {"throttle": throttle, "steering": motor["steering"], "say": self._radio(snap), "debug": debug}

    # ====================================================================== SPINAL_CORD
    # The ONLY engineered control rule (declared in SPINAL_CORD above). It reads neuron outputs
    # (steering from DNa02/AOTU019, brake from MDN, emergency from DNp01) and the car's own speed
    # (proprioception). It never reads walls or waypoints: those reach the car only through neurons.
    def spinal_cord(self, motor: dict, speed: float) -> float:
        r = self.brain.readout
        brake, steering = motor["brake"], motor["steering"]
        target = r.cruise_speed * (1.0 - brake) * (1.0 - r.turn_slowdown * abs(steering)) - r.reverse_speed * brake
        normal = float(np.clip(r.throttle_gain * (target - speed), -1.0, 1.0))
        stop = float(motor["emergency"])         # DNp01 (giant fiber) above threshold: 1.0, else 0.0
        return stop * r.emergency_throttle + (1.0 - stop) * normal
    # ====================================================================== end SPINAL_CORD

    # ------------------------------------------------------------------ radio: describe the neural state
    def _radio(self, snap: dict) -> str | None:
        r, m, inp = snap["rates"], snap["motor"], snap["inputs"]
        dn_l, dn_r = r["DNa02_L"], r["DNa02_R"]
        ao_l, ao_r = r["AOTU019_L"], r["AOTU019_R"]
        if m["emergency"]:
            topic = "DNp01"
            text = (f"Giant fiber DNp01 {m['dnp01_hz']:.0f} Hz: looming wall "
                    f"(LPLC2/LC4 input {inp['loom_L']:.0f} Hz), emergency stop!")
        elif m["brake"] > 0.5:
            topic = "MDN"
            mdn, lc16 = 0.5 * (r["MDN_L"] + r["MDN_R"]), max(inp["LC16_L"], inp["LC16_R"])
            text = f"MDN moonwalker {mdn:.0f} Hz: braking (LC16 wall {lc16:.0f} Hz)"
        elif inp["visibility"] < 0.3:
            topic = "hidden"
            text = "Waypoint hidden behind a wall: LC10a dimmed, pursuit weak"
        elif abs(m["steering"]) > 0.15:
            side = "right" if m["steering"] > 0 else "left"
            topic = side
            text = (f"turning {side}: AOTU019 L {ao_l:.0f} / R {ao_r:.0f} Hz, "
                    f"DNa02 L {dn_l:.0f} / R {dn_r:.0f} Hz")
        else:
            topic, text = None, None
        if topic == self.last_topic:
            return None
        self.last_topic = topic
        return text
