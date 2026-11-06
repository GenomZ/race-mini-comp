"""Data helpers for the Streamlit "Fly Lab" (no Streamlit or plotting imports here).

Everything returns plain lists / dicts / numpy arrays, ready for ``st.dataframe``,
plotly, Altair or a JSON payload:

* :func:`circuit_table`    - neurons and synapses per cell type and side;
* :func:`sankey_data`      - cell type x side flow graph (nodes + links) for a Sankey;
* :func:`bearing_sweep`    - the "electrophysiology" experiment: put a target at each
                             bearing, record DNa02 / AOTU019, real wiring vs scrambled seeds;
* :func:`lateralization`   - one number per sweep: how strongly steering follows the bearing;
* :func:`stimulate`        - drive named groups at fixed rates, read the output neurons;
* :func:`raster`           - spike times of selected groups (from ``FlyBrain.last_spikes``
                             or a ``LIFNetwork.run``), and :func:`raster_from_snapshots` for
                             a race replay;
* :func:`drive_summary`    - why a car did what it did over a race: emergency stops, braking,
                             steering toward / away from the waypoint (e.g. real vs scrambled).

Example::

    from agentic_gp.fly import load_circuit
    from agentic_gp.fly.viz import bearing_sweep, lateralization
    sweep = bearing_sweep(load_circuit(), angles=range(-90, 91, 15), scramble_seeds=range(5))
    sweep["real"]["DNa02_L"]                     # Hz at each angle
    lateralization(sweep["real"], sweep["angles"])  # ~0.8 for real wiring, ~0 for scrambled
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from .connectome import Circuit, scrambled

__all__ = ["KEY_TYPES", "circuit_table", "sankey_data", "bearing_sweep", "lateralization",
           "stimulate", "raster", "raster_from_snapshots", "drive_summary"]

#: cell types with a role in the car (inputs, hubs, outputs), in display order
KEY_TYPES = ("LC10a", "LC16", "LPLC2", "LC4", "AOTU019", "AOTU025",
             "DNa01", "DNa02", "DNa03", "DNa04", "DNa11", "MDN", "DNp01")
SWEEP_GROUPS = ("DNa02_L", "DNa02_R", "AOTU019_L", "AOTU019_R", "AOTU025_L", "AOTU025_R")
_SIDE_SHORT = {"left": "L", "right": "R", "center": "C"}


def _label(cell_type: str, side: str, key_types) -> str:
    base = cell_type if cell_type in key_types else "other"
    return f"{base}_{_SIDE_SHORT.get(side, '?')}"


def circuit_table(circuit: Circuit) -> list[dict]:
    """One row per (cell type, side): neurons, synapses in/out, inhibitory share of output.

    Sorted by neuron count; key types (inputs/outputs of the car) are flagged ``key=True``."""
    labels = np.char.add(np.char.add(circuit.cell_type.astype(str), "|"), circuit.side.astype(str))
    uniq, inv = np.unique(labels, return_inverse=True)
    syn = np.abs(circuit.syn).astype(np.int64)
    n_cells = np.bincount(inv, minlength=uniq.size)
    out_syn = np.bincount(inv[circuit.pre], weights=syn, minlength=uniq.size)
    in_syn = np.bincount(inv[circuit.post], weights=syn, minlength=uniq.size)
    inh = np.bincount(inv[circuit.pre], weights=syn * (circuit.syn < 0), minlength=uniq.size)
    rows = []
    for k, lab in enumerate(uniq.tolist()):
        t, s = lab.split("|", 1)
        rows.append({"cell_type": t, "side": s, "neurons": int(n_cells[k]), "synapses_out": int(out_syn[k]),
                     "synapses_in": int(in_syn[k]),
                     "inhibitory_out": round(float(inh[k] / out_syn[k]), 3) if out_syn[k] else 0.0,
                     "key": t in KEY_TYPES})
    rows.sort(key=lambda r: (-r["key"], -r["neurons"], r["cell_type"], r["side"]))
    return rows


def sankey_data(circuit: Circuit, key_types: Iterable[str] = KEY_TYPES, min_synapses: int = 50,
                include_other: bool = True) -> dict:
    """Flow graph by cell type and side, for a Sankey diagram.

    Neurons of ``key_types`` keep their name ("LC10a_L"); all others are pooled per side
    ("other_L").  Links sum synapses between groups; links below ``min_synapses`` are dropped.
    Returns ``{"nodes": [names], "links": [{"source": i, "target": j, "value": synapses,
    "sign": +1/-1 (majority), "excitatory": n, "inhibitory": n}]}``.  Self-links are dropped.
    """
    key_types = tuple(key_types)
    labels = np.array([_label(t, s, key_types) for t, s in zip(circuit.cell_type.tolist(), circuit.side.tolist())])
    order = sorted(set(labels.tolist()), key=lambda x: (x.startswith("other"),
                   key_types.index(x.rsplit("_", 1)[0]) if x.rsplit("_", 1)[0] in key_types else 99, x))
    if not include_other:
        order = [x for x in order if not x.startswith("other")]
    pos = {nm: i for i, nm in enumerate(order)}
    code = np.array([pos.get(x, -1) for x in labels.tolist()])
    a, b = code[circuit.pre], code[circuit.post]
    ok = (a >= 0) & (b >= 0) & (a != b)
    m = len(order)
    pair = a[ok] * m + b[ok]
    syn = circuit.syn[ok].astype(np.int64)
    exc = np.bincount(pair, weights=np.clip(syn, 0, None), minlength=m * m)
    inh = np.bincount(pair, weights=np.clip(-syn, 0, None), minlength=m * m)
    links = []
    for p in np.flatnonzero(exc + inh >= max(min_synapses, 1)):
        links.append({"source": int(p // m), "target": int(p % m), "value": int(exc[p] + inh[p]),
                      "sign": 1 if exc[p] >= inh[p] else -1, "excitatory": int(exc[p]), "inhibitory": int(inh[p])})
    links.sort(key=lambda d: -d["value"])
    return {"nodes": order, "links": links}


def _group_means(brain, rates: np.ndarray, groups) -> dict:
    c = brain.circuit
    return {g: float(rates[c.group(g)].mean()) if c.group(g).size else 0.0 for g in groups}


def stimulate(circuit: Circuit, group_hz: dict, ms: float = 500.0, warmup_ms: float = 100.0,
              groups: Iterable[str] | None = None, seed: int = 0, backend: str = "auto") -> dict:
    """Drive groups at fixed Poisson rates (e.g. ``{"LC16": 150}``) from rest and return the
    mean rate (Hz) of each of ``groups`` (default: the brain's readout groups)."""
    from .brain import FlyBrain, READOUT_GROUPS

    brain = FlyBrain(circuit, seed=seed, backend=backend)
    rates = np.zeros(circuit.n)
    for g, hz in group_hz.items():
        rates[circuit.group(g)] = hz
    r, _ = brain.measure(rates, ms, warmup_ms)
    return _group_means(brain, r, READOUT_GROUPS if groups is None else groups)


def bearing_sweep(circuit: Circuit, angles: Iterable[float] = tuple(range(-90, 91, 15)),
                  scramble_seeds: Iterable[int] = (), ms: float = 500.0, warmup_ms: float = 100.0,
                  groups: Iterable[str] = SWEEP_GROUPS, seed: int = 0, backend: str = "auto",
                  progress=None, scramble_method: str = "degree", **brain_kwargs) -> dict:
    """Tuning curve: place a target at each bearing (no walls) and record output neurons.

    ``circuit`` is the real wiring; each seed in ``scramble_seeds`` adds a scrambled copy
    (``scramble_method`` "degree" or "strength"; same input mapping, same readout).
    ``progress(done, total)`` is called after every point (for a progress bar).  Returns::

        {"angles": [...],
         "real":      {"DNa02_L": [...Hz], ..., "steer_raw": [...]},
         "scrambled": {seed: {...same keys...}},
         "ms": ms, "label": circuit.label}

    ``steer_raw`` is the brain's readout (default :class:`ReadoutParams`, i.e. the real brain's
    gains, or ``readout_params=`` in ``brain_kwargs``) before clipping, the same for every
    wiring so the curves are comparable.
    """
    from .brain import FlyBrain

    angles = [float(a) for a in angles]
    seeds = list(scramble_seeds)
    groups = tuple(groups)
    total = len(angles) * (1 + len(seeds))
    done = 0

    def one(c: Circuit) -> dict:
        nonlocal done
        brain = FlyBrain(c, seed=seed, backend=backend, **brain_kwargs)
        cols = {g: [] for g in groups}
        cols["steer_raw"] = []
        for a in angles:
            r, _ = brain.measure(brain.bearing_rates(a), ms, warmup_ms)
            for g, v in _group_means(brain, r, groups).items():
                cols[g].append(v)
            cols["steer_raw"].append(float(brain.steer_terms(brain.group_rates(r))["raw"]))
            done += 1
            if progress is not None:
                progress(done, total)
        return cols

    out = {"angles": angles, "real": one(circuit), "scrambled": {}, "ms": ms, "label": circuit.label}
    for s in seeds:
        out["scrambled"][s] = one(scrambled(circuit, s, method=scramble_method))
    return out


def lateralization(curve: dict, angles: Iterable[float] | None = None, key: str = "steer_raw",
                   dead_zone_deg: float = 6.0) -> float:
    """How well steering follows the target side, in [-1, 1].

    Mean of ``sign(bearing) * clip(steer_raw, -1, 1)`` over bearings outside the binocular
    dead zone.  +1: always turns toward the target at full strength; 0: flat or random;
    negative: turns away.  ``curve`` is ``sweep["real"]`` or ``sweep["scrambled"][seed]``;
    pass ``angles`` when the curve does not carry them (``bearing_sweep`` output does at top level).
    """
    if angles is None:
        angles = curve.get("angles")
        if angles is None:
            raise ValueError("pass angles=sweep['angles']")
    a = np.asarray(list(angles), dtype=float)
    s = np.clip(np.asarray(curve[key], dtype=float), -1, 1)
    m = np.abs(a) >= dead_zone_deg
    return float(np.mean(np.sign(a[m]) * s[m])) if m.any() else 0.0


def raster(spikes, circuit: Circuit, groups: Iterable[str], t0_ms: float = 0.0) -> dict:
    """Spike raster for selected groups.

    ``spikes`` is a :class:`~agentic_gp.fly.lif.Spikes` (``brain.last_spikes`` or the second
    value of ``LIFNetwork.run``), whose times are absolute brain time.  Returns
    ``{group: {"t_ms": [...], "row": [...], "n": cells}}`` where ``row`` numbers the group's
    neurons 0..n-1 and times are relative to ``t0_ms``; pass
    ``t0_ms=brain.last_window_start_ms`` for times within the last window."""
    t = np.asarray(spikes.t_ms, dtype=float)
    idx = np.asarray(spikes.idx)
    out = {}
    for g in groups:
        cells = circuit.group(g)
        rank = np.full(circuit.n, -1, np.int64)
        rank[cells] = np.arange(cells.size)
        m = rank[idx] >= 0 if idx.size else np.zeros(0, bool)
        out[g] = {"t_ms": (t[m] - t0_ms).round(3).tolist(), "row": rank[idx[m]].tolist(), "n": int(cells.size)}
    return out


def raster_from_snapshots(snapshots: Iterable[dict], groups: Iterable[str] | None = None) -> dict:
    """Concatenate the per-tick readout spikes stored in ``FlyBrain.step`` snapshots
    (``snapshot["spikes"]``) into one raster on the brain's clock: ``{group: [t_ms, ...]}``
    plus ``"ticks": [window start ms, ...]`` for a cursor synced to the car replay."""
    out: dict = {"ticks": []}
    for snap in snapshots:
        t0 = float(snap.get("t_ms", 0.0)) - float(snap.get("window_ms", 0.0))
        out["ticks"].append(t0)
        for g, ts in (snap.get("spikes") or {}).items():
            if groups is not None and g not in groups:
                continue
            out.setdefault(g, []).extend(t0 + x for x in ts)
    return out


def drive_summary(snapshots: Iterable[dict], dead_zone_deg: float = 6.0) -> dict:
    """What the brain did over a list of ``FlyBrain.step`` snapshots (one race, one brain).

    Returns shares of ticks: ``emergency`` (giant-fiber stop), ``braking`` (MDN brake > 0.5),
    ``toward`` / ``away`` (steering sign agrees / disagrees with the waypoint side, for
    |bearing| >= ``dead_zone_deg`` and |steering| >= 0.1), ``weak_steer`` (|steering| < 0.1
    although the waypoint is off-centre), plus means of ``throttle``, ``abs_steering``,
    ``brake``, ``visibility`` and ``ticks``.  Use it to show WHY a scrambled car loses: by
    stopping, by not steering, or by steering the wrong way."""
    snaps = list(snapshots)
    if not snaps:
        return {"ticks": 0}
    m = [s.get("motor", {}) for s in snaps]
    bearing = np.array([float(s.get("inputs", {}).get("bearing_deg", 0.0)) for s in snaps])
    steer = np.array([float(x.get("steering", 0.0)) for x in m])
    off = np.abs(bearing) >= dead_zone_deg
    moving = off & (np.abs(steer) >= 0.1)
    agree = np.sign(steer) == np.sign(bearing)
    n_off = max(int(off.sum()), 1)
    return {
        "ticks": len(snaps),
        "emergency": float(np.mean([bool(x.get("emergency", False)) for x in m])),
        "braking": float(np.mean([float(x.get("brake", 0.0)) > 0.5 for x in m])),
        "toward": float((moving & agree).sum() / n_off),
        "away": float((moving & ~agree).sum() / n_off),
        "weak_steer": float((off & ~moving).sum() / n_off),
        "throttle": float(np.mean([float(x.get("throttle", 0.0)) for x in m])),
        "abs_steering": float(np.mean(np.abs(steer))),
        "brake": float(np.mean([float(x.get("brake", 0.0)) for x in m])),
        "visibility": float(np.mean([float(s.get("inputs", {}).get("visibility", 1.0)) for s in snaps])),
    }
