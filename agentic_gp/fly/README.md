# `agentic_gp.fly`: a fruit-fly brain that drives the car

This package supports chapter 7, "Fly Brains", part B. It takes 4,494 real neurons from the
FlyWire adult fruit-fly connectome, with their real synapse counts, and simulates them with the
Shiu et al. 2024 spiking model. The simulated brain steers the car. A scrambled copy of the same
wiring serves as the scientific control, and a "shallow" copy (only the direct input -> readout
connections) shows how much of the skill is a one-synapse relay.

The toy spiking fly from part A is not in this package. It is a self-contained driver file,
`drivers/examples/fly_toy.py`.

Before you show it: the data is **derived from FlyWire** and licensed CC BY-NC 4.0, so it may be
used only for non-commercial teaching. Read [ATTRIBUTION.md](ATTRIBUTION.md), which also says what
to do for a paid event.

## Using it from a driver

```python
from agentic_gp.fly import FlyBrain

brain = FlyBrain.from_default(scramble_seed=None)   # None = real wiring; 0..4 = scrambled controls
controls, snapshot = brain.step(obs)                # once per car tick (simulates 100 ms of brain)
env.step(**controls)                                # controls = {"throttle": t, "steering": s}
```

`obs` uses the race sensor fields:

- `angle_to_next_waypoint`: degrees, + = right;
- `distance_to_next_waypoint`: used to tell whether a wall hides the waypoint;
- `distance_to_wall_front` and `distance_to_wall_{left,right}_{20,45,90}`;
- `speed`.

Missing fields are treated as "far away" or 0; without `distance_to_next_waypoint` the waypoint
always counts as visible. `steering` > 0 turns right, as in the race.

The returned `snapshot` is a JSON-friendly dict:

| Key | Contents |
|---|---|
| `inputs` | Encoder output: `bearing_deg`, `visibility` (0–1), `LC10a_L/R`, `LC16_L/R`, `loom_L/R` in Hz, and the declared reflex `touch_MDN`. |
| `rates` | Readout neurons in Hz: DNa02, AOTU019, AOTU025, DNa01, DNa03, MDN and DNp01, each L and R. |
| `spikes` | Spike times of the readout neurons, in ms from the window start, for a raster. |
| `motor` | `steering`, `throttle`, `brake`, `emergency`, `dnp01_hz`, `target_speed`. |
| `steer_terms` | The steering command split into its DNa02 and AOTU019 parts and the offset. |
| other | `t_ms`, `window_ms`, `circuit`, `dnp01_spikes`, `n_spikes`, `compute_ms`. |

`brain.last_rates` (all 4,494 neurons) and `brain.last_spikes` hold the full window, in absolute
brain time. The window started at `brain.last_window_start_ms`.

Variants for the exercises:

```python
FlyBrain.from_default(scramble_seed=3)                            # same neurons, same connection counts, random wiring
FlyBrain.from_default(scramble_seed=3, scramble_method="strength") # also keeps each neuron's input synapse totals
FlyBrain.from_default(shallow=True)                               # keep only the 566 direct input -> readout connections
FlyBrain.from_default(lesion=["DNa02"])                           # silence a cell type: no synapses in or out, never fires
FlyBrain.from_default(mirror=True)                                # swap every left/right label (sanity check, see below)
FlyBrain.from_default(mirror="eyes")                              # swap only the input neurons: "crossed eyes"
brain.calibrate()                                                 # per-side gains measured on THIS brain (~5 s); real brain only
FlyBrain(circuit, EncoderParams(occlusion=False), ReadoutParams(cruise_speed=12))
```

**Controls keep the real brain's readout.** The `ReadoutParams` defaults were measured on the real
brain. Do not `calibrate()` a scrambled or lesioned brain: it would turn noise into a steering
signal. `calibrate()` now refuses: any gain whose response is below 20 Hz is left unchanged, the
offset is clipped to ±0.3, and it issues a `RuntimeWarning`.

**What the lesions do** (lateralization index at ±15/30/60/90°, default readout; 1 = always turns
fully toward the target). These numbers describe OUR readout, which reads AOTU019 as well as DNa02:
a lesion of AOTU019 hurts most because the decoder listens to it, not because the fly is known to
steer through it.

| Lesion | Index | What happens |
|---|---|---|
| none | 0.68 | |
| DNa02 | 0.46 | Large turns are gone; AOTU019 still gives about ±0.5 |
| AOTU019 | 0.14 | Central targets now steer the wrong way; only ±90° still works |
| AOTU025 | 0.66 | No effect: it is not read out |
| LC10a | 0.00 | The fly is blind to the target |

## What is connectome and what is engineered

| Stage | What | Source |
|---|---|---|
| Eyes (encoder) | Bearing → LC10a (small-object detector): eccentricity tuning (σ = 0.3, up to 150 Hz); the far eye fades out linearly between 0° and 6°. If a wall in the waypoint's direction is closer than the waypoint, LC10a drive drops to 30% (only for targets within about 20–45° of straight ahead). Wall rays → LC16 on each side (`200 Hz × (1 − d/range)`; the front ray drives both sides). Looming, graded: 1 / time to contact (speed / front ray) → LPLC2 + LC4, 0 Hz at 0.9 s or more, rising linearly in 1/TTC to 100 Hz at 0.7 s, faded in between speed 1 and 2. All smooth: no if/else on sensors, and only input neurons are driven. | **Engineered.** Each LC10a cell's preferred eccentricity is estimated from the connectome. |
| Reflex (declared) | "Antenna touch": a wall closer than 3 units on the front ray drives MDN directly (up to `touch_hz` 100 Hz). The only input into a neuron the decoder reads; the circuit has no mechanosensory pathway. Off unless `touch_hz > 0` (`fly_connectome.py`: on only under the classic crash rule). | **Engineered**, declared as a `REFLEX` |
| Brain | 4,494 neurons and 407,139 connections, with signed synapse counts × 0.275 mV. Shiu et al. LIF model: τm 20 ms, τsyn 5 ms, 1.8 ms delay, 2.2 ms refractory, on Brian2's step schedule. | **Connectome** (FlyWire v783) and a published model |
| Readout | Steering = `DNa02_R/130 − DNa02_L/150 + 0.5·(AOTU019_R/175 − AOTU019_L/175)`. DNa02 is a descending neuron; **AOTU019 is a central-brain hub one synapse after LC10a**, and in races its term is 77–79% of the steering signal. Brake = mean MDN ("moonwalker") / 30 Hz. Emergency stop = mean DNp01 (giant fiber) rate of at least 50 Hz. `steer_gain` (library default 1.0) scales the steering; the race driver uses 0.4, because 1.0 overshoots on every 0.5 s tick. `aotu019_weight=0` (driver: `READOUT=DN`) reads the descending neurons only. | **Engineered gains** on connectome-driven neurons |
| "Spinal cord" | `target speed = cruise × (1 − brake) × (1 − 0.5·|steering|) − 6 × brake`, so a strong MDN backs the car up (cruise 16 in the library, 20 in `fly_connectome.py`). Throttle = 0.3 × (target − speed), or −1 during an emergency stop. | **Engineered, and we say so openly.** No neuron in this circuit drives forward speed. |

Points to say out loud:

- **Most steering is read from AOTU019, one synapse after the LC10a input.** For targets within
  about 20° of straight ahead, DNa02 is silent and only AOTU019 shows the side. In races the AOTU019
  term is 77% (oval) / 79% (grand_prix) of the steering signal; reading DNa02 alone (`READOUT=DN`)
  is 1.6–2× slower. And the `shallow` circuit, with only the 566 direct input → readout
  connections, is only 2–31% slower than the full brain: most of the skill is that one-synapse relay.
- **Pursuit suppresses retreat.** While LC10a is driven at full strength, the LC16 → MDN brake is
  silent. With a waypoint ahead and LC16 at 187 Hz, MDN fires 0/0 Hz; at 20% LC10a drive it fires
  23/24.5 Hz. So the brake only works when the waypoint is hidden or dimmed, which is why the
  encoder dims LC10a when a wall is in the way.
- **Wall avoidance through the circuit is weak.** LC16 on one side drives the opposite DNa02 to at
  most about 19–25 Hz, which is 0.15–0.2 of full lock. The waypoints do most of the steering.
- **Left and right differ.** The left side fires about 1.2–1.7× more than the right. That may be
  real asymmetry or may come from how the dataset was reconstructed and proofread. `calibrate()`
  measures per-side gains so the car steers symmetrically.
- **Throttle uses `speed`** in the engineered speed rule. That is allowed in the Fly class only
  because it is declared (see "The Fly-class rule" below).
- **Parked nose-to-wall, the car can back up (declared reflex).** Pursuit silences LC16 → MDN, so
  without help a car stopped 2 units from a wall drove forward into it. The "antenna touch" reflex
  drives MDN directly when the front wall is closer than 3 units. Test: car placed nose-to-wall at
  speed 0 in 96 spots on three tracks, crash within 8 s in 58% of spots without it, 18% with it.
  Under the default marshal crash rule it does not pay in a race (a touch costs 5 s and the marshal
  re-points the car, but backing up cannot steer around a wall that hides the waypoint): sprint
  78.6 → 86.8 s, grand_prix 117.1 → 120.3 s. So the driver turns it on only for
  `crash_rule="classic"` (`TOUCH="auto"`); there it cuts the crashes (sprint 37 → 13, grand_prix
  65 → 9 per attempt), but the brain still does not finish sprint or grand_prix under that rule.
  `EncoderParams(touch_hz=0)` switches it off. Because it feeds a decoded neuron directly, a driver
  that uses it must declare it in `REFLEX = "..."` (Fly-class rule 4).

## Measured responses

All runs below use dt = 0.25 ms, the default, and the default readout.

**Bearing tuning curve.** Rates are in Hz, measured over 500 ms after 100 ms of warm-up.
`steer_raw` is not clipped.

| Bearing (°) | −90 | −60 | −45 | −30 | −15 | 0 | +15 | +30 | +45 | +60 | +90 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| DNa02_L | 156 | 132 | 102 | 34 | 0 | 0 | 0 | 0 | 4 | 0 | 0 |
| DNa02_R | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 66 | 108 | 136 |
| AOTU019_L | 48 | 136 | 156 | 166 | 176 | 172 | 30 | 2 | 0 | 0 | 0 |
| AOTU019_R | 0 | 0 | 0 | 0 | 0 | 162 | 188 | 184 | 172 | 148 | 44 |
| steer_raw | −1.18 | −1.27 | −1.13 | −0.70 | −0.50 | −0.03 | +0.45 | +0.54 | +0.97 | +1.25 | +1.17 |

Steering changes smoothly through the midline: −0.49 at −5.9°, −0.07 at −2°, +0.12 at +2° and
+0.41 at +5.9°.

The lateralization index is the mean of sign(bearing) × clip(steer_raw), over the 13 bearings from
−90° to +90° in 15° steps, excluding 0°:

| Wiring | Index | Largest \|steer_raw\| |
|---|---|---|
| Real | **0.85** | 1.39 |
| Scrambled, `degree`, seeds 0–4 | −0.01, −0.09, −0.00, −0.00, −0.05 | 0.12–0.26 |
| Scrambled, `strength`, seeds 0–4 | +0.10, +0.03, −0.04, +0.09, +0.02 | 0.15–0.51 |

`calibrate()` on the real brain takes about 6 s and gives DNa02 gains 156 L / 132 R, AOTU019
gains 174 L / 188 R and an offset of −0.06.

**Brake and escape modules.** Rates are in Hz, measured over 1 s.

| Input | Response |
|---|---|
| LC16 on both sides at 100 / 150 / 200 Hz | MDN L/R 3/6, 21.5/24, 32/34 |
| LC16 on the left only, 150 Hz | DNa02_R 19, DNa02_L 0, MDN 0: a turn away from the wall, no brake |
| LPLC2 + LC4 on both sides, 100 Hz | DNp01 L/R 123 / 161, MDN about 6 |
| Graded looming in races (grand_prix + sprint) | input 1–50 Hz: DNp01 10–20 Hz, 1 stop in 11 ticks; 50–100 Hz: DNp01 100–105 Hz, 3 of 3; 100 Hz: 130 Hz, 20 of 20 |
| No input | Every neuron at 0 |

The giant fiber is silent without looming in the real brain. In scrambled brains its noise reaches
about 30 Hz, which is below the 50 Hz emergency threshold. Scrambled brains also lose the
looming → DNp01 response (0–28 Hz instead of about 140 Hz), so they never make an emergency stop.
That is not what makes them slow: with the emergency stop switched off for every brain, the real
fly still does oval 49.3 s / grand_prix 114.6 s and the scrambled ones exactly their times below.

**Driving with the race driver** `drivers/examples/fly_connectome.py` (decoder `steer_gain` 0.4,
cruise 20, graded looming; one attempt, 1 lap, brain seed 0, race time includes crash penalties):

| Brain | grand_prix | oval | sprint | speedway | briefing_demo |
|---|---|---|---|---|---|
| Real | **117.1 s, 2** | **49.2 s, 0** | **78.6 s, 1** | **122.8 s, 0** | **134.1 s, 2** |
| Real, brain seeds 1–3 | 111.3–124.8 s, 2–3 | 49.1–50.5 s, 0 | | | |
| Shallow (566 direct connections) | 132.2 s, 3 | 64.7 s, 2 | 101.8 s, 3 | 125.1 s, 0 | 151.2 s, 3 |
| Real, `READOUT=DN` (DNa02 only) | 231.3 s, 14 | 79.8 s, 3 | 127.7 s, 5 | 247.7 s, 16 | |
| Scrambled `degree` seeds 0–3 | 292–364 s, 25–39 | 126–163 s, 9–17 | 211–263 s, 20–29 | 285–351 s, 21–34 | 302–375 s, 25–39 |
| Scrambled seed 4 | DNF (circles) | DNF | DNF | DNF | DNF |
| Lesion DNa02 / AOTU019 / LC10a | 138 s, 5 / 304 s, 18 / DNF | 58 s, 1 / 140 s, 6 / DNF | | | |
| Mirrored, all labels | 111.4 s, 1 | 52.2 s, 0 | | | |
| Mirrored, eyes only | DNF, 31 | 452.2 s, 42 | | | |
| Scrambled `strength` seeds 0–2 | 270–339 s, 24–32 | 123–164 s, 9–14 | | | |

### What the controls show

- **Scrambled:** with the same neurons, connection counts, encoder and decoder, the real wiring is
  2.3–3.3× faster. The skill lives in the wiring.
- **Shallow:** most of that skill is the direct input → readout connections (mainly LC10a →
  AOTU019). The other ~4,400 neurons make the fly 2–31% faster, not 3× faster. Do not say "4,494
  neurons drive the car" without this.
- **Lesions and `READOUT=DN`:** lesioning DNa02 costs little and AOTU019 a lot because our decoder
  reads AOTU019. This is a property of our readout, not a finding about the fly.
- **Mirror:** `mirror=True` relabels inputs and readout alike over identical synapses, so the fly must
  still chase the target. It is a sanity check, and its lap time only reflects the left/right
  asymmetry of the data and of the per-side readout gains. `mirror="eyes"` swaps only the inputs and
  the fly steers away from every target, as predicted.

The decoder's steering gain matters most (3 brain seeds each, real wiring, cruise 16; measured
before the looming input became graded):
`steer_gain` 1.0 gives oval 88–102 s and grand_prix 181–223 s; 0.4 gives 60 s and 131–139 s; 0.2
is slower again (62–64 s, 162–173 s). Cruise 20 with 0.4 gives the table above.

**Driving with the library defaults** (`steer_gain` 1.0, cruise 16; the Stage 1 measurement, before
the looming input became graded):

| Brain | oval | sprint | grand_prix |
|---|---|---|---|
| Real | 102.4 s, 4 crashes | 139.7 s, 6 crashes | 223.0 s, 9 crashes |
| Real, `EncoderParams(occlusion=False)` | 108.2 s, 5 crashes | 177.1 s, 10 crashes | 200.3 s, 8 crashes |
| Scrambled seeds 0 / 1 / 2 | 173 / 206 / 177 s, 16–20 crashes | 247 / 391 / 298 s, 24–41 crashes | 326 / 513 / 405 s, 31–53 crashes |

`viz.drive_summary` shows why the scrambled cars lose. They never trigger an emergency stop. On 41–69%
of off-centre ticks they barely steer (|steering| < 0.1), and they steer away from the waypoint
about as often as toward it. The real brain steers toward it on 82–89% of off-centre ticks.

Stronger braking settings (`ReadoutParams(mdn_full_brake_hz=20)` or `EncoderParams(occluded_gain=0.2)`)
back the car off walls better, but in these tests the car then stalled on two of the three tracks.

## The race drivers and the Fly-class rule

| File | What |
|---|---|
| `drivers/examples/fly_toy.py` | Part A. A self-contained 57-neuron LIF toy (numpy only) with a readable `NEURONS` + `WIRING` table: LC10a → AOTU019 → DNa02 pursuit, crossed wall-ray → DNa02 reflex, P1 arousal, P9 forward drive with a FAST speed governor, LPLC2 → Giant Fiber brake, and the FRONT AND STILL coincidence neuron → MDN. Deterministic; grand_prix 63.3 s, oval 29.0 s, clean on 17 tracks. `SCRAMBLE` control (WIRING targets shuffled); `python drivers/examples/fly_toy.py` runs a mirror-symmetry self-check. |
| `drivers/examples/fly_broken.py` | The Ch7 exercise: the toy with ONE wrong wall row. Hits the left wall twice around T5 (the hairpin) on grand_prix (93.05 s); graded hints in its docstring. Fixed = the mirror self-check passes AND grand_prix is exactly the toy's 63.3 s, 0 crashes. The answer is not written anywhere students read: instructors diff the file against `fly_toy.py`. |
| `drivers/examples/fly_connectome.py` | Part B. `FlyBrain` on the FlyWire circuit; `PARAMS` `SCRAMBLE` (seed), `SCRAMBLE_METHOD`, `SHALLOW`, `READOUT`, `LESION`, `MIRROR` (`off`/`all`/`eyes`), `TOUCH`, `steer_gain`, `cruise_speed`, `seed`. Declares `SPINAL_CORD` and `REFLEX`. Radio messages describe the neural state; a missing circuit file gives a clear message and the car coasts. |

**The Fly-class rule** (Fly Cup and the "Biologically Honest" ribbon):

1. Throttle and steering must be decoded from neuron activity (firing rates or spikes), with a fixed
   decoder.
2. Sensors may enter only through the encoder: sensor → input current or spike rate, using smooth
   functions. No `if`/`else` on sensors in the control path. The encoder drives only INPUT
   (sensory) neurons: a neuron the decoder reads must get no encoder input, so there is always at
   least one synapse between a sensor and a control.
3. Exception 1, `SPINAL_CORD`: an engineered rule may combine neuron outputs with the car's own
   `speed` (proprioception), but only inside a section marked `SPINAL_CORD` in the file, and only if
   the driver declares it in a one-line `SPINAL_CORD = "..."` attribute. It must never read walls,
   waypoints or the track map. `fly_toy.py` declares `SPINAL_CORD = ""`; `fly_connectome.py`
   declares its speed rule.
4. Exception 2, `REFLEX`: a contact reflex (a wall closer than a few units) may drive a decoded neuron
   directly, only if declared in `REFLEX = "..."`. Bearings, waypoints and the map never may.
   `fly_connectome.py` declares its antenna touch → MDN (on only under the classic crash rule).
5. A Fly-class finisher must beat its own scrambled control (`PARAMS["SCRAMBLE"]`) to win Best Fly
   Brain. `fly_toy.py` shows how: shuffle the targets of the WIRING rows.

**Checking.** `python -m agentic_gp.fly.rules drivers/my_fly.py` lints a file
(`agentic_gp.fly.rules.check_fly_rules(path)` from Python). It warns about a branch on anything read
from `obs` outside the radio and the `SPINAL_CORD` section (the brain's own `self.…` calls stop the
trace, since their outputs are neurons), a `SPINAL_CORD` section that reads walls or waypoints, a
missing `SPINAL_CORD` declaration or section, and a missing `SCRAMBLE` param. It is a lint. Stewards
still read the encoder and the `SPINAL_CORD` / `REFLEX` sections in the pit interview, because "no
encoder input into decoded neurons" cannot be checked statically. `tests/test_fly_drivers.py` runs
the lint on the three example files and the same branch check on the `FlyBrain` control path
(`brain.CONTROL_PATH`).

## Simulator

`LIFNetwork` follows Brian2's default schedule in every step: integrate the neurons that are not
refractory, check the threshold, deliver the synaptic input and Poisson kicks that arrive at this
step (dropped while refractory), then reset. The delay is `round(1.8/dt)` steps. A neuron ignores
input until `int(2.2/dt)` steps after its spike. Integration is exact, not Euler.

The test suite checks it spike for spike against an independent step-by-step implementation of
that schedule. Against the reviewer's Brian2-order reference on the full circuit, rates differ by
about 3–4 Hz on average over about 330 active cells. That is the size of Poisson noise, because the
two use different random streams.

## Speed

Times are per 100 ms simulated window on a Ryzen 7 5800H laptop (16 threads). The measurements
were taken while other heavy jobs kept the machine busy, so the wall-clock times are 3–4× the CPU
times.

| Setup | CPU per window | Wall per window |
|---|---|---|
| numpy, dt 0.25, one eye or race-like input | about 45 ms (35–47 ms over several runs) | 100–200 ms under load |
| numpy, dt 0.25, silent | under 16 ms | 13–57 ms |
| numpy, dt 0.1 | about 80 ms | 270–470 ms under load |
| numba (optional), dt 0.25, race-like input | about 9 ms | 44 ms under load |

On an idle machine, an earlier version with the same per-step work measured about 6 ms silent and
25 ms with one eye driven (wall clock). `numba` gives identical spikes; the tests check this when
numba is installed. A driver's `drive()` call may take up to 2 s, so either backend fits easily.

## Fly Lab helpers (`agentic_gp.fly.viz`)

These helpers return plain data and do not import Streamlit.

```python
from agentic_gp.fly import load_circuit
from agentic_gp.fly import viz
c = load_circuit()
viz.circuit_table(c)                         # rows per cell type and side: neurons, synapses in and out, inhibitory share
viz.sankey_data(c, min_synapses=50)          # {"nodes": [...], "links": [{"source", "target", "value", "sign", ...}]}
sw = viz.bearing_sweep(c, angles=range(-90, 91, 15), scramble_seeds=range(5), progress=cb)   # about 30–90 s
viz.lateralization(sw["real"], sw["angles"]) # 0.85 real, about 0 scrambled
viz.stimulate(c, {"LC16": 150}, groups=["MDN_L", "MDN_R"])
viz.raster(brain.last_spikes, c, ["LC10a_L", "AOTU019_L", "DNa02_L"], t0_ms=brain.last_window_start_ms)
viz.raster_from_snapshots(snapshots)         # readout spikes across a race, synced to the ticks
viz.drive_summary(snapshots)                 # emergency / braking / toward / away / weak_steer shares
```

Cache `bearing_sweep` with `st.cache_data`. If the circuit file is missing, `load_circuit()` raises
`CircuitNotFoundError` with instructions; show them in the UI.

## Instructor steps

1. **Before T−7: confirm the event is non-commercial.** See ATTRIBUTION.md. If it is not, skip the
   connectome part of the chapter.
2. **T−7 days: build the circuit.** Run the extractor once. It needs `pyarrow` and `pandas`, which
   only the extractor uses:

   ```
   pip install pyarrow pandas
   python -m agentic_gp.fly.extract
   ```

   - It downloads two pinned inputs into `data/flywire/` (about 132 MB): `Connectivity_783.parquet`
     from Shiu et al. at commit `91bdd1e` and the annotations TSV at tag `v3.2.0`.
   - It checks their sha256.
   - It writes `data/fly/steering_circuit_783.npz` (1.1 MB) and the `.json` sidecar. The sidecar
     records the group sizes, the recipe, the source URLs, the checksums, the license and the build
     time.
   - The `.npz` is byte-for-byte reproducible. The sidecar records both its file hash and a content
     hash of the arrays (`output.content_sha256`).
   - It takes about 10–20 s once the inputs are present.
   - If you already have the input files, add `--reuse-from <folder>`. Files are matched by size and
     verified by hash.
   - `data/` is git-ignored by `data/.gitignore`, so the data is never committed.
3. **Check it.** Run `pytest tests/test_fly_brain.py`, which takes about 20 s. The circuit tests
   check sizes, steering toward the target, the scrambled controls, lesions, the brake, the
   emergency stop and the absence of false stops.
4. **Hand it out.** Give students the `.npz` and `.json` files. They go in `data/fly/` in each
   student's copy, or anywhere if `AGENTIC_GP_FLY_CIRCUIT` points to the `.npz`. This is allowed
   only for **non-commercial** teaching, with ATTRIBUTION.md included.
5. **Optional speed-up.** Run `pip install numba` on the race machine.

A `fly` extra in `pyproject.toml` (`numba`, `pyarrow`, `pandas`) would make steps 2 and 5 a single
install; that file belongs to the core package.

## Future work: MaleCNS v1.0 (CC BY 4.0)

If the non-commercial clause is a problem, use the Janelia MaleCNS v1.0 connectome. It is licensed
CC BY 4.0 and can be downloaded without a login from
`storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/`. It has the same type
names: LC10a, AOTU019/025, DNa01–11, MDN, DNp01, LPLC2, LC4 and LC16.

The extractor does not support it yet (`--source malecns` is not implemented). It would need to:

- filter to `Traced` bodies;
- derive signs from `consensus_nt`;
- re-tune `w_syn`, because the Shiu value was fit on FlyWire.

## Files

| File | Contents |
|---|---|
| `lif.py` | `LIFNetwork(n, pre, post, weight_mv, params=LIFParams(), seed=0, backend="auto")` with `.run(ms, input_rates_hz) -> (rates_hz, Spikes(t_ms, idx))`, `.reset()` and `.t_ms`. Uses Brian2's schedule, exact integration and a delay ring with batched delivery. |
| `connectome.py` | `Circuit` (with `.group("LC10a_L")`), `load_circuit(path=None)`, `scrambled(c, seed, method="degree"\|"strength")`, `shallow(c, inputs, outputs)`, `lesioned(c, groups)`, `mirrored(c, cell_types=None)`, `save_circuit` and `content_sha256`. |
| `extract.py` | The instructor tool above. |
| `brain.py` | `FlyBrain` (`encode` → `reflex` → LIF → `decode`), `EncoderParams`, `ReadoutParams`, `calibrate(brain)`, `wall_distance_at(obs, deg)` and `CONTROL_PATH`. |
| `rules.py` | `check_fly_rules(path)`: the Fly-class lint (also `python -m agentic_gp.fly.rules FILE`). |
| `viz.py` | Data for the Fly Lab page. |
| `drivers/examples/fly_*.py` | The race drivers (see above); tests in `tests/test_fly_drivers.py`. |
